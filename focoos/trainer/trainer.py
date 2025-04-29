"""Unified training module for Focoos models.

This module provides a simplified and unified training implementation that combines
the functionality of the original FocoosTrainer and the engine Trainer classes.
"""

import logging
import os
import time
import weakref
from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
from torch import GradScaler, autocast

from focoos.data.datasets.map_dataset import MapDataset
from focoos.data.loaders import build_detection_test_loader, build_detection_train_loader
from focoos.models.fai_model import BaseModelNN
from focoos.nn.layers.norm import FrozenBatchNorm2d
from focoos.ports import ModelInfo, Task, TrainerArgs
from focoos.trainer.checkpointer import DetectionCheckpointer
from focoos.trainer.evaluation.evaluator import inference_on_dataset
from focoos.trainer.evaluation.get_eval import get_evaluator
from focoos.trainer.evaluation.utils import print_csv_format
from focoos.trainer.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter, get_event_storage
from focoos.trainer.hooks import hook
from focoos.trainer.hooks.early_stop import EarlyStoppingHook
from focoos.trainer.hooks.visualization import VisualizationHook
from focoos.trainer.solver import ema
from focoos.trainer.solver.build import build_lr_scheduler, build_optimizer
from focoos.utils.distributed.dist import comm, create_ddp_model
from focoos.utils.env import collect_env_info, seed_all_rng
from focoos.utils.logger import add_file_logging, get_logger

# Mapping of task types to their primary evaluation metrics
task_metrics = {
    Task.DETECTION.value: "bbox/AP",
    Task.SEMSEG.value: "sem_seg/mIoU",
    Task.INSTANCE_SEGMENTATION.value: "segm/AP",
    # Task.PANOPTIC_SEGMENTATION.value: "panoptic_seg/PQ",
}

logger = get_logger("trainer")


class FocoosTrainer:
    def __init__(
        self,
        args: TrainerArgs,
        model: BaseModelNN,
        model_info: ModelInfo,
        data_val: MapDataset,
        data_train: Optional[MapDataset] = None,
    ):
        """Initialize the trainer.

        Args:
            args: Training configuration
            model: Model to train/evaluate
            metadata: Model metadata/configuration
            data_val: Validation dataset
            data_train: Optional training dataset
        """
        self.args = args
        self.resume = args.resume
        self.finished = False

        # Setup logging and environment
        self._setup_logging()

        # Setup model and data
        self._setup_model_and_data(model, model_info, data_train, data_val)

        # Setup training components
        self._setup_training_components()

    def _setup_logging(self):
        """Setup logging and environment variables."""
        self.output_dir = os.path.join(self.args.output_dir, self.args.run_name)
        if comm.is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)

        add_file_logging(logger=logger, verbose=True, output=self.output_dir, rank=comm.get_local_rank())
        logger.info(f"Output dir: {self.output_dir}")

        logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))
        logger.debug("Environment info:\n" + collect_env_info())
        seed_all_rng(None if self.args.seed < 0 else self.args.seed + comm.get_rank())
        torch.backends.cudnn.benchmark = False

        if self.args.ckpt_dir:
            self.ckpt_dir = self.args.ckpt_dir
            if comm.is_main_process():
                os.makedirs(self.ckpt_dir, exist_ok=True)
                logger.info(f"[CKPT DIR] {self.ckpt_dir}")
        else:
            self.ckpt_dir = self.output_dir

    def _setup_model_and_data(self, model, model_info, data_train, data_val):
        """Setup model and data."""
        # Setup Model
        self.model = model
        self.model_info = model_info
        self.checkpoint = self.args.init_checkpoint

        self.metric = None
        self.best_metric = None

        # Setup data
        self.data_train = data_train
        self.data_val = data_val

        # Get task and num_classes from validation dataset
        self.num_classes = data_val.dataset.metadata.num_classes
        self.task = data_val.dataset.metadata.task

        # Apply model modifications
        if self.args.freeze_bn:
            self.model = FrozenBatchNorm2d.convert_frozen_batchnorm(self.model)
        elif self.args.freeze_bn_bkb:
            self.model.pixel_decoder.backbone = FrozenBatchNorm2d.convert_frozen_batchnorm(
                self.model.pixel_decoder.backbone
            )

        if self.args.reset_classifier:
            self.model.reset_classifier()

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
                f"📊 [TRAIN DATASET {len(data_train)}] {str(data_train.dataset.metadata)} | "
                f"[Train augmentations] {data_train.mapper.augmentations}"
            )
        # Log dataset info
        logger.info(
            f"📊 [VALIDATION INFO] Classes: {data_val.dataset.metadata.num_classes} | "
            f"Dataset: {len(data_val)} {str(data_val.dataset.metadata)} | "
            f"Augmentations: {data_val.mapper.augmentations} | "
            f"Evaluator: {type(self.data_evaluator)} 🔍"
        )

        # Save metadata
        if comm.get_rank() == 0:
            self.model_info.dump_json(os.path.join(self.output_dir, "model_info.json"))

    def _setup_training_components(self):
        """Setup training components like optimizer, scheduler, etc."""
        # This will be called during train() method
        pass

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
        self.model_info.weights_uri = os.path.abspath(save_file)

    def _restore_best_model(self, name: str = "model_best.pth"):
        """Restore best model from checkpoint.

        Args:
            name: Checkpoint filename

        Returns:
            bool: Whether restore was successful
        """
        best_path = os.path.join(self.ckpt_dir, name)
        if os.path.exists(best_path):
            state_dict = torch.load(best_path)
            self.model.load_state_dict(state_dict["model"])
            if self.args.ema_enabled and "ema_state" in state_dict:
                self.model.ema_state.load_state_dict(state_dict["ema_state"])
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

            self.model_info.dump_json(os.path.join(self.output_dir, "model_info.json"))

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
                res = self._do_eval(self.model)
        else:
            logger.info("🔍 Run evaluation without EMA.")
            res = self._do_eval(self.model)

        if comm.get_rank() == 0:
            key, value = task_metrics[self.task.value].split("/")
            self.metric = _add_prefix(res[key], key)
            if (
                self.model_info.val_metrics is None
                or self.metric[task_metrics[self.task.value]]
                > self.model_info.val_metrics[task_metrics[self.task.value]]
            ):
                self.model_info.val_metrics = self.metric
        return res

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
                                force_close=True,
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
                    val_metric=task_metrics[self.task.value],
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
                        val_metric=task_metrics[self.task.value],
                        mode="max",
                    ),
                    hook.PeriodicCheckpointer(
                        checkpointer,
                        period=args.checkpointer_period,
                        max_to_keep=args.checkpointer_max_to_keep,
                    ),
                    VisualizationHook(
                        model=self.model,  # type: ignore
                        dataset=self.data_val,
                        period=self.args.vis_period,
                        # postprocessing=self.post_processor,
                        n_sample=self.args.samples,
                    ),
                ]
            )

    def train(self):
        """Train the model using the configured settings."""
        args = self.args
        model = self.model

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
        )

        # Setup Trainer
        trainer = TrainerLoop(
            model=model,
            dataloader=train_loader,
            optimizer=optim,
            amp=args.amp_enabled,
            clip_gradient=args.clip_gradients,
            gather_metric_period=args.gather_metric_period,
            zero_grad_before_forward=args.zero_grad_before_forward,
        )

        # Setup Checkpointer
        checkpointer = DetectionCheckpointer(
            model,
            save_dir=self.ckpt_dir,
            trainer=trainer,
            **ema.get_ema_checkpointer(model) if args.ema_enabled else {},
        )

        self._register_hooks(trainer, model, checkpointer, optim, scheduler, args)

        # Load checkpoint if needed
        if self.checkpoint:
            checkpointer.resume_or_load(path=self.checkpoint, resume=args.resume)
        else:
            checkpointer.resume_or_load(path="", resume=args.resume)

        if self.args.resume and checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            start_iter = trainer.iter + 1
        else:
            start_iter = 0

        trainer.train(start_iter=start_iter, max_iter=args.max_iters)
        self.finished = True
        self.finish()

    def test(self, restore_best: bool = False):
        """Run model evaluation on test set.

        Args:
            restore_best: Whether to restore best checkpoint before testing

        Returns:
            dict: Evaluation metrics
        """
        args = self.args
        model = self.model

        model = create_ddp_model(model)

        if restore_best:
            # Setup Checkpointer to recover trained model or load from scratch
            checkpointer = DetectionCheckpointer(
                model=model,
                save_dir=self.output_dir,
                **ema.get_ema_checkpointer(model) if args.ema_enabled else {},
            )
            checkpointer.resume_or_load(path=self.checkpoint or "", resume=args.resume)
            if args.ema_enabled:
                ema.apply_model_ema(model)

        res = self._do_eval(model)
        if comm.get_rank() == 0:
            key, value = task_metrics[self.task.value].split("/")
            self.metric = _add_prefix(res[key], key)
            if (
                self.model_info.val_metrics is None
                or self.metric[task_metrics[self.task.value]]
                > self.model_info.val_metrics[task_metrics[self.task.value]]
            ):
                self.model_info.val_metrics = self.metric

        self.finished = True
        self.finish()
        return res


class TrainerLoop:
    """Unified training loop implementation.

    This class implements the core training loop functionality, combining
    the features of SimpleTrainer and Trainer with AMP support.
    """

    def __init__(
        self,
        model,
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

        # Set model to training mode
        model.train()

        self.model = model
        self.data_loader = dataloader
        self._data_loader_iter_obj = None
        self.optimizer = optimizer
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward

        # AMP setup
        if amp:
            if grad_scaler is None:
                grad_scaler = GradScaler()
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
        logger.info("🚀 Starting training from iteration {}".format(start_iter))

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
            except Exception as e:
                logger.error(f"Exception during training: {e}")
                raise e
            finally:
                self.after_train()

    def before_train(self):
        """Called before training starts."""
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        """Called after training ends."""
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        """Called before each training step."""
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
                loss_dict = self.model(data).loss
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())
        else:
            loss_dict = self.model(data).loss
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_gradient > 0.0:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
        else:
            losses.backward()
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
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                self.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

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
    return {f"{key}/{k}": v for k, v in metric.items()}
