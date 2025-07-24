"""Unified training module for Focoos models.

This module provides a simplified and unified training implementation that combines
the functionality of the original FocoosTrainer and the engine Trainer classes.
"""

import json
import os
import time
import weakref
from collections.abc import Mapping
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch import GradScaler, autocast

from focoos.data.datasets.map_dataset import MapDataset
from focoos.data.loaders import build_detection_test_loader, build_detection_train_loader
from focoos.hub.focoos_hub import FocoosHUB
from focoos.models.focoos_model import BaseModelNN
from focoos.nn.layers.norm import FrozenBatchNorm2d
from focoos.ports import ArtifactName, HubSyncLocalTraining, ModelInfo, ModelStatus, Task, TrainerArgs, TrainingInfo
from focoos.processor.base_processor import Processor
from focoos.trainer.checkpointer import Checkpointer
from focoos.trainer.evaluation.evaluator import inference_on_dataset
from focoos.trainer.evaluation.get_eval import get_evaluator
from focoos.trainer.evaluation.utils import print_csv_format
from focoos.trainer.events import EventStorage, get_event_storage
from focoos.trainer.hooks import hook
from focoos.trainer.hooks.early_stop import EarlyStopException, EarlyStoppingHook
from focoos.trainer.hooks.metrics_json_writer import JSONWriter
from focoos.trainer.hooks.metrics_printer import CommonMetricPrinter
from focoos.trainer.hooks.sync_to_hub import SyncToHubHook
from focoos.trainer.hooks.tensorboard_writer import TensorboardXWriter
from focoos.trainer.hooks.visualization import VisualizationHook
from focoos.trainer.solver import ema
from focoos.trainer.solver.build import build_lr_scheduler, build_optimizer
from focoos.utils.distributed.dist import comm, create_ddp_model
from focoos.utils.env import seed_all_rng
from focoos.utils.logger import capture_all_output, get_logger
from focoos.utils.metrics import is_json_compatible, parse_metrics
from focoos.utils.system import get_system_info

# Mapping of task types to their primary evaluation metrics
TASK_METRICS = {
    Task.DETECTION.value: "bbox/AP",
    Task.SEMSEG.value: "sem_seg/mIoU",
    Task.INSTANCE_SEGMENTATION.value: "segm/AP",
    Task.CLASSIFICATION.value: "classification/Accuracy",
    Task.KEYPOINT.value: "keypoints/AP",
    # Task.PANOPTIC_SEGMENTATION.value: "panoptic_seg/PQ",
}

logger = get_logger("FocoosTrainer")


class FocoosTrainer:
    def train(
        self,
        args: TrainerArgs,
        model: BaseModelNN,
        processor: Processor,
        model_info: ModelInfo,
        data_train: MapDataset,
        data_val: MapDataset,
        hub: Optional[FocoosHUB] = None,
    ):
        """Train the model using the configured settings."""
        self.args = args
        self.model = model
        self.processor = processor
        self.model_info = model_info
        self.data_train = data_train
        self.data_val = data_val
        self.resume = args.resume
        self.finished = False

        self.args.run_name = self.args.run_name.strip()
        # Setup logging and environment
        self.output_dir = os.path.join(self.args.output_dir, self.args.run_name)
        revision = 1
        original_run_name = self.args.run_name
        while os.path.exists(self.output_dir):
            logger.warning(f"Trainer Output directory {self.output_dir} already exists. Creating a new one...")
            self.args.run_name = original_run_name + f"-{revision}"
            self.output_dir = os.path.join(self.args.output_dir, self.args.run_name)
            revision += 1
        if comm.is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)

        with capture_all_output(log_path=os.path.join(self.output_dir, "log.txt"), rank=comm.get_local_rank()):
            logger.info(f"📁 Run name: {self.args.run_name} | Output dir: {self.output_dir}")

            logger.debug("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))
            get_system_info().pprint()

            seed_all_rng(None if self.args.seed < 0 else self.args.seed + comm.get_rank())
            torch.backends.cudnn.benchmark = False

            if self.args.ckpt_dir:
                self.ckpt_dir = self.args.ckpt_dir
                if comm.is_main_process():
                    os.makedirs(self.ckpt_dir, exist_ok=True)
                    logger.info(f"[Checkpoints directory] {self.ckpt_dir}")
            else:
                self.ckpt_dir = self.output_dir

            # Setup model and data
            self._setup_model_and_data(model, processor, model_info, data_train, data_val)

            assert self.data_train is not None, "Train dataset is required for training"

            # Setup Optimizer
            optim = build_optimizer(
                name=self.args.optimizer,
                learning_rate=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                model=model,
                weight_decay_norm=self.args.weight_decay_norm,
                weight_decay_embed=self.args.weight_decay_embed,
                backbone_multiplier=self.args.backbone_multiplier,
                decoder_multiplier=self.args.decoder_multiplier,
                head_multiplier=self.args.head_multiplier,
                clip_gradients=self.args.clip_gradients,
                extra=self.args.optimizer_extra,
            )
            scheduler = build_lr_scheduler(
                name=self.args.scheduler,
                max_iters=self.args.max_iters,
                optimizer=optim,
                extra=self.args.scheduler_extra,
            )

            # Setup dataset
            train_loader = build_detection_train_loader(
                dataset=self.data_train,
                total_batch_size=args.batch_size,
                num_workers=args.workers,
            )

            # Handle Multi-GPU Training
            model = create_ddp_model(
                model,
                broadcast_buffers=self.args.ddp_broadcast_buffers,
                find_unused_parameters=self.args.ddp_find_unused,
            )  # type: ignore

            # Setup Trainer
            trainer_loop = TrainerLoop(
                model=model,
                processor=self.processor,
                dataloader=train_loader,
                optimizer=optim,
                amp=args.amp_enabled,
                clip_gradient=args.clip_gradients,
                gather_metric_period=args.gather_metric_period,
                zero_grad_before_forward=args.zero_grad_before_forward,
            )

            # Setup Checkpointer
            checkpointer = Checkpointer(
                model,  # type: ignore
                save_dir=self.ckpt_dir,
                trainer=trainer_loop,
                **ema.get_ema_checkpointer(model) if args.ema_enabled else {},
            )

            if self.args.sync_to_hub:
                self._setup_hub(hub)

            self._register_hooks(trainer_loop, model, checkpointer, optim, scheduler, args)

            # Load checkpoint if needed
            if self.checkpoint:
                checkpointer.resume_or_load(path=self.checkpoint, resume=args.resume)
            else:
                checkpointer.resume_or_load(path="", resume=args.resume)

            if self.args.resume and checkpointer.has_checkpoint():
                # The checkpoint stores the training iteration that just finished, thus we start
                # at the next iteration
                start_iter = trainer_loop.iter + 1
            else:
                start_iter = 0

            output_lines = [
                f"🚀 Starting training from iteration {start_iter}/{self.args.max_iters} ",
                f"📁 output_dir: {self.output_dir}",
                "========== 🔧 Main Hyperparameters 🔧 ==========",
                f" - max_iter: {self.args.max_iters}",
                f" - batch_size: {self.args.batch_size}",
                f" - learning_rate: {self.args.learning_rate}",
                " - resolution: !TODO",
                f" - optimizer: {self.args.optimizer}",
                f" - scheduler: {self.args.scheduler}",
                f" - weight_decay: {self.args.weight_decay}",
                f" - ema_enabled: {self.args.ema_enabled}",
                f" - amp_enabled: {self.args.amp_enabled}",
                "================================================",
            ]
            logger.info("\n".join(output_lines))

            self._update_training_info_and_dump(ModelStatus.TRAINING_RUNNING)
            trainer_loop.train(start_iter=start_iter, max_iter=args.max_iters)
            self.finished = True
            self.finish()
            if self.args.sync_to_hub:
                self.remote_model.sync_local_training_job(
                    local_training_info=HubSyncLocalTraining(
                        status=ModelStatus.TRAINING_COMPLETED,
                        iterations=self.args.max_iters,
                        training_info=self.model_info.training_info,
                    ),
                    dir=self.output_dir,
                    upload_artifacts=[
                        ArtifactName.WEIGHTS,
                        ArtifactName.METRICS,
                    ],
                )

    def eval(
        self,
        args: TrainerArgs,
        model: BaseModelNN,
        processor: Processor,
        model_info: ModelInfo,
        data_val: MapDataset,
        restore_best: bool = False,
        save_json: bool = True,
    ):
        """Run model evaluation on test set.

        Args:
            restore_best: Whether to restore best checkpoint before testing

        Returns:
            dict: Evaluation metrics
        """
        self.args = args
        self.output_dir = os.path.join(self.args.output_dir, f"{self.args.run_name.strip()}_eval")
        if comm.is_main_process() and save_json:
            os.makedirs(self.output_dir, exist_ok=True)
        original_weights_uri = model_info.weights_uri
        self._setup_model_and_data(model, processor, model_info, None, data_val)
        self.model_info.weights_uri = original_weights_uri
        if comm.is_main_process() and save_json:
            self.model_info.dump_json(os.path.join(self.output_dir, "model_info.json"))
        model = create_ddp_model(self.model)  # type: ignore

        if restore_best:
            # Setup Checkpointer to recover trained model or load from scratch
            checkpointer = Checkpointer(
                model=model,  # type: ignore
                save_dir=self.output_dir,
                **ema.get_ema_checkpointer(model) if args.ema_enabled else {},
            )
            checkpointer.resume_or_load(path=self.checkpoint or "", resume=args.resume)
            if args.ema_enabled:
                ema.apply_model_ema(model)

        eval_result = self._do_eval(model)

        if comm.get_rank() == 0:
            key, value = TASK_METRICS[self.task.value].split("/")
            raw_metrics = _add_prefix(eval_result[key], key)
            if (
                self.model_info.val_metrics is None
                or raw_metrics[TASK_METRICS[self.task.value]]
                > self.model_info.val_metrics[TASK_METRICS[self.task.value]]
            ):
                self.model_info.val_metrics = raw_metrics
                self.model_info.dump_json(os.path.join(self.output_dir, "model_info.json"))
                _metrics = [raw_metrics]
                if save_json:
                    with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                        f.write(json.dumps(_metrics))

        return eval_result

    def _setup_hub(self, hub: Optional[FocoosHUB] = None):
        if comm.is_main_process() and self.args.sync_to_hub:
            hub = hub or FocoosHUB()
            self.remote_model = hub.new_model(self.model_info)

            self.model_info.ref = self.remote_model.ref
            logger.info(f"Model {self.model_info.name} created in hub with ref {self.model_info.ref}")

        """Setup logging and environment variables."""

    def _setup_model_and_data(
        self,
        model: BaseModelNN,
        processor: Processor,
        model_info: ModelInfo,
        data_train: Optional[MapDataset],
        data_val: MapDataset,
    ):
        """Setup model and data."""
        # Setup Model
        self.model = model
        self.processor = processor.train()
        self.model_info = model_info
        if self.model_info.name != self.args.run_name:
            self.model_info.name = self.args.run_name
        self.model_info.weights_uri = os.path.join(self.output_dir, "model_final.pth")
        self.checkpoint = self.args.init_checkpoint
        # Setup data
        self.data_train = data_train
        self.data_val = data_val

        # Get task and num_classes from validation dataset
        self.num_classes = data_val.dataset.metadata.num_classes
        self.task = data_val.dataset.metadata.task

        # Apply model modifications
        if self.args.freeze_bn:
            self.model = FrozenBatchNorm2d.convert_frozen_batchnorm(self.model)

        # Setup DDP if needed
        if comm.get_world_size() > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.to(self.args.device)

        # Setup EMA if enabled
        if self.args.ema_enabled and not hasattr(self.model, "ema_state"):
            ema.build_model_ema(self.model)

        # Setup evaluator
        self.data_evaluator = get_evaluator(dataset_dict=self.data_val.dataset, task=self.task)

        if data_train:
            logger.info(
                f"📊 [TRAIN DATASET: {len(data_train)} samples] {str(data_train.dataset.metadata)} | "
                f"[Train augmentations] {data_train.mapper.augmentations}"
            )
        # Log dataset info
        logger.info(
            f"📊 [VALIDATION DATASET: {len(data_val)} samples] Classes: {data_val.dataset.metadata.num_classes} | "
            f"Augmentations: {data_val.mapper.augmentations} | "
            f"Evaluator: {type(self.data_evaluator)} 🔍"
        )

        # Save metadata
        if comm.get_rank() == 0:
            self.model_info.dump_json(os.path.join(self.output_dir, "model_info.json"))

    def _store_model(self, save_file):
        """Store model weights to file.

        Args:
            save_file: Path to save model
        """
        data = {}
        if self.args.ema_enabled:
            ema.apply_model_ema(self.model)
        data["model"] = self.model.state_dict()
        save_file = os.path.join(self.output_dir, save_file)
        logger.info("Saving final model to {}".format(save_file))
        torch.save(data, save_file)
        self.model_info.weights_uri = save_file

    def _restore_best_model(self, name: str = "model_best.pth"):
        """Restore best model from checkpoint.

        Args:
            name: Checkpoint filename

        Returns:
            bool: Whether restore was successful
        """
        best_path = os.path.join(self.ckpt_dir, name)
        if os.path.exists(best_path):
            state_dict = torch.load(best_path, weights_only=True)
            self.model.load_state_dict(state_dict["model"])
            if self.args.ema_enabled and "ema_state" in state_dict:
                self.model.ema_state.load_state_dict(state_dict["ema_state"])  # type: ignore
            return True
        return False

    def finish(self):
        """Clean up and finalize training/testing."""
        if comm.get_rank() == 0:
            logger.info("🏁 End of training.")
            # save model to model_final.pth - if EMA, store it.

            if self.finished:
                restored = self._restore_best_model()
                if restored:
                    logger.info("Restored best model from checkpoint.")
                    if self.args.ema_enabled:
                        ema.apply_model_ema(self.model, save_current=True)
                    os.remove(os.path.join(self.ckpt_dir, "model_best.pth"))
                self._store_model("model_final.pth")
            try:
                parsed_metrics = parse_metrics(os.path.join(self.output_dir, "metrics.json"))
                parsed_metrics.valid_metrics = []
                parsed_metrics.train_metrics = []
                self.model_info.val_metrics = parsed_metrics.best_valid_metric
            except Exception as e:
                logger.warning(f"Error parsing metrics.json: {e}")
                pass

            self._update_training_info_and_dump(ModelStatus.TRAINING_COMPLETED, str(self.args.max_iters))

    def _do_eval(self, model):
        """Internal method to evaluate model.

        Args:
            model: Model to evaluate

        Returns:
            dict: Evaluation metrics
        """
        data_loader = build_detection_test_loader(
            self.data_val,
            num_workers=self.args.workers,
        )

        ret = inference_on_dataset(
            model,
            processor=self.processor,
            data_loader=data_loader,
            evaluator=self.data_evaluator,
        )
        print_csv_format(ret)
        return ret

    def _val(self):
        """Run model evaluation on validation set.

        Returns:
            dict: Evaluation metrics
        """
        if self.args.ema_enabled:
            logger.info("🔍 Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(self.model):
                eval_res = self._do_eval(self.model)
        else:
            logger.info("🔍 Run evaluation without EMA.")
            eval_res = self._do_eval(self.model)

        if comm.get_rank() == 0:
            key, value = TASK_METRICS[self.task.value].split("/")
            storage = get_event_storage()
            iteration = storage.iteration
            raw_metrics = _add_prefix(eval_res[key], key)
            raw_metrics["iteration"] = iteration

            if (
                self.model_info.val_metrics is None
                or raw_metrics[TASK_METRICS[self.task.value]]
                > self.model_info.val_metrics[TASK_METRICS[self.task.value]]
            ):
                self.model_info.val_metrics = raw_metrics
                self.model_info.updated_at = datetime.now().isoformat()
                logger.info(f"✨ New best validation metric: {raw_metrics[TASK_METRICS[self.task.value]]}")
        return eval_res

    def _register_hooks(self, trainer, model, checkpointer, optim, scheduler, args):
        """Register hooks for the trainer.

        Args:
            trainer: The trainer instance
            model: The model
            checkpointer: The checkpointer
            optim: The optimizer
            scheduler: The learning rate scheduler
            args: Training arguments
        """
        trainer.register_hooks(
            [
                hook.IterationTimer(),
                hook.LRScheduler(optimizer=optim, scheduler=scheduler),
                (
                    ema.EMAHook(
                        model,
                        decay=args.ema_decay,
                        warmup=args.ema_warmup if not args.resume else 0,
                        device=args.device,
                    )
                    if args.ema_enabled
                    else None
                ),
                hook.EvalHook(args.eval_period, lambda: self._val()),
                (  # this should be after the eval hook to print the metrics
                    hook.PeriodicWriter(
                        [
                            CommonMetricPrinter(args.max_iters),
                            JSONWriter(
                                os.path.join(self.ckpt_dir, "metrics.json"),
                            ),
                            TensorboardXWriter(self.output_dir),
                        ],
                        period=args.log_period,
                    )
                    if comm.is_main_process()
                    else None
                ),
                EarlyStoppingHook(
                    enabled=args.early_stop,
                    eval_period=args.eval_period,
                    patience=args.patience,
                    val_metric=TASK_METRICS[self.task.value],
                    mode="max",
                ),
            ]
        )

        if comm.is_main_process():
            trainer.register_hooks(
                [
                    hook.BestCheckpointer(
                        checkpointer=checkpointer,
                        eval_period=args.eval_period,
                        val_metric=TASK_METRICS[self.task.value],
                        mode="max",
                    ),
                    hook.PeriodicCheckpointer(
                        checkpointer,
                        period=args.checkpointer_period,
                        max_to_keep=args.checkpointer_max_to_keep,
                    ),
                    VisualizationHook(
                        model=self.model,  # type: ignore
                        processor=self.processor,
                        dataset=self.data_val,
                        period=self.args.eval_period,
                        n_sample=self.args.samples,
                        output_dir=self.output_dir,
                    ),
                ]
            )
            if self.args.sync_to_hub and self.remote_model:
                trainer.register_hooks(
                    [
                        SyncToHubHook(
                            remote_model=self.remote_model,
                            model_info=self.model_info,
                            output_dir=self.output_dir,
                            sync_period=60,
                        ),
                    ]
                )

    def _update_training_info_and_dump(self, new_status: ModelStatus, detail: Optional[str] = None):
        self.model_info.status = new_status
        self.model_info.updated_at = datetime.now().isoformat()
        if self.model_info.training_info is None:
            self.model_info.training_info = TrainingInfo()

        self.model_info.training_info.main_status = new_status
        if self.model_info.training_info.status_transitions is None:
            self.model_info.training_info.status_transitions = []
        if self.model_info.training_info.main_status != new_status:
            self.model_info.training_info.main_status = new_status

        if new_status in [ModelStatus.TRAINING_ERROR, ModelStatus.TRAINING_COMPLETED]:
            self.model_info.training_info.end_time = datetime.now().isoformat()

        if new_status == ModelStatus.TRAINING_ERROR:
            self.model_info.training_info.failure_reason = detail

        self.model_info.training_info.status_transitions.append(
            dict(
                status=new_status,
                timestamp=datetime.now().isoformat(),
                detail=detail,
            )
        )
        if comm.is_main_process():
            self.model_info.dump_json(os.path.join(self.output_dir, ArtifactName.INFO))


class TrainerLoop:
    """Unified training loop implementation.

    This class implements the core training loop functionality, combining
    the features of SimpleTrainer and Trainer with AMP support.
    """

    def __init__(
        self,
        model,
        processor,
        dataloader,
        optimizer,
        amp=False,
        clip_gradient: float = 0.0,
        grad_scaler=None,
        gather_metric_period=1,
        zero_grad_before_forward=False,
    ):
        """Initialize the trainer loop.

        Args:
            model: The model to train
            dataloader: The data loader
            optimizer: The optimizer
            amp: Whether to use automatic mixed precision
            clip_gradient: Gradient clipping value
            grad_scaler: Gradient scaler for AMP
            gather_metric_period: How often to gather metrics
            zero_grad_before_forward: Whether to zero gradients before forward pass
        """
        self._hooks = []
        self.iter = 0
        self.start_iter = 0
        self.max_iter = 0
        self.storage = None
        self.skip_optimizer_step = False  # Flag to skip optimizer step

        # Set model to training mode
        model.train()

        self.model = model
        self.processor = processor
        self.data_loader = dataloader
        self._data_loader_iter_obj = None
        self.optimizer = optimizer
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward

        # AMP setup
        if amp:
            if grad_scaler is None:
                # the init_scale avoids the first step to be too large
                # and the scheduler.step() warning
                grad_scaler = GradScaler(init_scale=2**10)
            self.grad_scaler = grad_scaler
            self.amp = amp
            self.precision = torch.float16
        else:
            self.amp = False
            self.precision = torch.float32

        # Gradient clipping
        self.clip_gradient = clip_gradient

    def register_hooks(self, hooks):
        """Register hooks for the trainer.

        Args:
            hooks: List of hooks to register
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """Train the model.

        Args:
            start_iter: Starting iteration
            max_iter: Maximum iteration
        """
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.iter += 1
            except EarlyStopException as e:
                logger.info(f"🚨 Early stopping triggered: {e}")
            except Exception as e:
                logger.error(f"🚨 Exception during training: {e}")
                raise e
            finally:
                # Verifica se c'è stata un'eccezione prima di eseguire after_train
                self.after_train()

    def before_train(self):
        """Called before training starts."""
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        """Called after training ends."""
        if self.storage is not None:
            self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        """Called before each training step."""
        if self.storage is not None:
            self.storage.iter = self.iter
        for h in self._hooks:
            h.before_step()

    def after_backward(self):
        """Called after backward pass."""
        for h in self._hooks:
            h.after_backward()

    def after_step(self):
        """Called after each training step."""
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        """Run a single training step."""
        assert self.model.training, "[UnifiedTrainerLoop] model was changed to eval mode!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        if self.amp:
            assert torch.cuda.is_available(), "[UnifiedTrainerLoop] CUDA is required for AMP training!"
            with autocast(enabled=self.amp, dtype=self.precision, device_type="cuda"):
                # we need to have preprocess data here
                images, targets = self.processor.preprocess(data, dtype=self.precision, device=self.model.device)
                loss_dict = self.model(images, targets).loss
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())
        else:
            images, targets = self.processor.preprocess(data, dtype=self.precision, device=self.model.device)
            loss_dict = self.model(images, targets).loss
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()  # type: ignore
            if self.clip_gradient > 0.0:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
        else:
            losses.backward()  # type: ignore
            if self.clip_gradient > 0.0:
                self.clip_grads(self.model.parameters())

        self.after_backward()
        self._write_metrics(loss_dict, data_time)

        if self.amp:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

    @property
    def _data_loader_iter(self):
        """Get the data loader iterator."""
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    def clip_grads(self, params):
        """Clip gradients.

        Args:
            params: Parameters to clip gradients for

        Returns:
            float: Total norm of the parameters
        """
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=self.clip_gradient)
        return 0.0

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        iter: Optional[int] = None,
    ) -> None:
        """Write metrics to storage.

        Args:
            loss_dict: Dictionary of losses
            data_time: Time taken by data loading
            prefix: Prefix for metric names
            iter: Current iteration
        """
        logger = get_logger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                self.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception as e:
                logger.warning(f"🚨 skipping write metrics due to exception: {e}")

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        cur_iter: int,
        prefix: str = "",
    ) -> None:
        """Write metrics to storage.

        Args:
            loss_dict: Dictionary of losses
            data_time: Time taken by data loading
            cur_iter: Current iteration
            prefix: Prefix for metric names
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics among all workers for logging
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])  # type: ignore
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}  # type: ignore
            total_losses_reduced = sum(metrics_dict.values())  # type: ignore
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={cur_iter}!\nloss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter)
            if len(metrics_dict) > 1:
                storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        """Get the state dict of the trainer.

        Returns:
            dict: State dict
        """
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state  # type: ignore
        ret["optimizer"] = self.optimizer.state_dict()
        if self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()  # type: ignore
        return ret

    def load_state_dict(self, state_dict):
        """Load the state dict of the trainer.

        Args:
            state_dict: State dict to load
        """
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)  # type: ignore
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.amp and "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])  # type: ignore


def _add_prefix(metric, key):
    """Add prefix to metric keys.

    Args:
        metric: Metric dictionary
        key: Prefix to add

    Returns:
        dict: Metric dictionary with prefix
    """
    return {f"{key}/{k}": v for k, v in metric.items() if is_json_compatible(v)}


def run_train(
    train_args: TrainerArgs,
    data_train: MapDataset,
    data_val: MapDataset,
    image_model: BaseModelNN,
    processor: Processor,
    model_info: ModelInfo,  # type: ignore  # noqa: F821
    hub: Optional[FocoosHUB] = None,
):
    """Run model training.

    Args:
        train_args: Training configuration
        data_train: Training dataset
        data_val: Validation dataset
        image_model: Model to train
        metadata: Model metadata/configuration
        rank: Rank of the process
    Returns:
        tuple: (trained model, updated metadata)
    """
    trainer = FocoosTrainer()
    trainer.train(
        args=train_args,
        model=image_model,
        processor=processor,
        model_info=model_info,
        data_train=data_train,
        data_val=data_val,
        hub=hub,
    )

    return image_model, model_info


def run_eval(
    train_args: TrainerArgs,
    data_val: MapDataset,
    image_model: BaseModelNN,
    processor: Processor,
    model_info: ModelInfo,
    save_json: bool = True,
):
    trainer = FocoosTrainer()
    trainer.eval(
        args=train_args,
        model=image_model,
        processor=processor,
        model_info=model_info,
        data_val=data_val,
        save_json=save_json,
    )

    return image_model, model_info
