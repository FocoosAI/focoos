"""
FocoosLightningModule - PyTorch Lightning wrapper for Focoos models.

This module enables training Focoos models using PyTorch Lightning's Trainer.
It provides:
- Training step with loss computation
- Validation step with evaluation metrics
- Optimizer and scheduler configuration
- Automatic mixed precision support
- Compact metric logging similar to original trainer
"""

import datetime
import time
from collections import deque
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from torch.optim import Optimizer

from focoos.models.base_model import BaseModelNN
from focoos.ports import DatasetEntry, ModelInfo, Task, TrainerArgs
from focoos.processor.base_processor import Processor
from focoos.trainer.evaluation.detection_lightning_evaluation import DetectionLightningEvaluator
from focoos.trainer.evaluation.sem_seg_evaluation import SemSegEvaluator
from focoos.trainer.solver.build import build_lr_scheduler, build_optimizer
from focoos.utils.logger import get_logger

logger = get_logger("FocoosLightningModule")


class FocoosLightningModule(L.LightningModule):
    """
    PyTorch Lightning Module for Focoos models.

    This module wraps a Focoos BaseModelNN and provides Lightning-compatible
    training and validation logic with compact metric logging.

    Args:
        model: The Focoos BaseModelNN to train
        processor: The Focoos processor for preprocessing
        model_info: Model metadata and configuration
        train_args: Training configuration arguments

    Example:
        >>> from focoos.model_manager import ModelManager
        >>> from focoos.trainer.lightning_module import FocoosLightningModule
        >>> model = ModelManager.get("fai-detr-s-coco", num_classes=80)
        >>> lightning_model = FocoosLightningModule(model=model.model, processor=model.processor, model_info=model.model_info, train_args=train_args)
    """

    def __init__(
        self,
        model: BaseModelNN,
        processor: Processor,
        model_info: ModelInfo,
        train_args: TrainerArgs,
    ):
        super().__init__()
        # Model components
        self.model = model
        self.processor = processor
        self.model_info = model_info
        self.train_args = train_args
        self.task = model_info.task

        # Save hyperparameters (excluding complex objects)
        self.save_hyperparameters(ignore=["model", "processor", "model_info"])

        # Validation tracking
        self.val_metric_key = self._get_task_metric_key()
        self.best_val_metric = 0.0
        self.validation_step_outputs: List[Dict[str, Any]] = []
        self.validation_step_inputs: List[DatasetEntry] = []
        self.evaluator: Optional[Any] = None  # Lazy initialization on first validation

        # Timing tracking for data_time metric
        self._step_end_time: Optional[float] = None
        self._batch_start_time: Optional[float] = None

        # Metric smoothing (window for averaging, automatically limited by maxlen)
        self._window_size = 20
        self._time_history: deque = deque(maxlen=self._window_size)
        self._data_time_history: deque = deque(maxlen=self._window_size)

        logger.info(f"🚀 FocoosLightningModule initialized for task: {self.task}")
        logger.info(f"📊 Tracking validation metric: {self.val_metric_key}")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def flops_per_batch(self) -> Optional[float]:
        """
        Return the number of FLOPs (floating point operations) per batch.
        Used by ThroughputMonitor for detailed performance metrics.

        Returns:
            None (FLOPS calculation not yet implemented)
        """
        # TODO: Implement FLOPS calculation based on model architecture
        return None

    def get_model(self) -> BaseModelNN:
        """
        Get the underlying Focoos model.

        Returns:
            The BaseModelNN instance wrapped by this Lightning module
        """
        return self.model

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_task_metric_key(self) -> str:
        """Get the primary validation metric key for the current task."""
        task_metrics = {
            Task.DETECTION.value: "bbox/AP",
            Task.SEMSEG.value: "sem_seg/mIoU",
            Task.INSTANCE_SEGMENTATION.value: "sem_seg/mIoU",
            Task.CLASSIFICATION.value: "classification/F1",
            Task.KEYPOINT.value: "keypoints/AP",
        }
        return task_metrics.get(self.task.value, "val/loss")

    def _format_eta(self, iter_time: float, current_iter: int) -> str:
        """
        Calculate and format ETA (Estimated Time to completion).

        Args:
            iter_time: Average time per iteration
            current_iter: Current iteration number

        Returns:
            Formatted ETA string (e.g., "1:23:45")
        """
        remaining_iters = self.train_args.max_iters - current_iter
        eta_seconds = iter_time * remaining_iters
        return str(datetime.timedelta(seconds=int(eta_seconds)))

    def _print_evaluation_results(self, metrics: Dict[str, Any], task_name: str):
        """
        Print evaluation results in a compact, tabular, and elegant format.

        Args:
            metrics: Dictionary of metric names and values from evaluator
            task_name: Name of the task (e.g., "bbox", "segm", "sem_seg")
        """
        if not metrics:
            return

        # Separate main metrics from per-class metrics
        main_metrics = {}
        class_metrics = {}

        for metric_name, metric_value in metrics.items():
            if metric_value is None or metric_value != metric_value:  # Skip None and NaN
                continue

            if metric_name.startswith("AP-") or metric_name.startswith("AR-"):
                # Per-class metric
                class_metrics[metric_name] = metric_value
            else:
                # Main metric
                main_metrics[metric_name] = metric_value

        # Build compact output
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"📊 {task_name.upper()} VALIDATION")
        logger.info("=" * 70)

        if main_metrics:
            # Print main metrics in compact rows of 4 (already in 0-100 scale from evaluator)
            metrics_list = list(main_metrics.items())
            for i in range(0, len(metrics_list), 4):
                row = metrics_list[i : i + 4]
                row_str = " | ".join([f"{name:6s} {value:5.2f}" for name, value in row])
                logger.info(f"{row_str}")

        # Print per-class summary only if many classes
        if class_metrics and len(class_metrics) > 10:
            ap_metrics = {k: v for k, v in class_metrics.items() if k.startswith("AP-")}
            if ap_metrics:
                sorted_ap = sorted(ap_metrics.items(), key=lambda x: x[1], reverse=True)
                logger.info("-" * 70)
                # Show top-3 and bottom-3 in compact format (already in 0-100 scale)
                top3 = [f"{name[3:]:>12s}:{val:5.1f}" for name, val in sorted_ap[:3]]
                bot3 = [f"{name[3:]:>12s}:{val:5.1f}" for name, val in sorted_ap[-3:]]
                logger.info(f"Top-3: {' '.join(top3)}")
                logger.info(f"Low-3: {' '.join(bot3)}")

        # Print best metric on same line (values already in 0-100 scale)
        metric_key = self.val_metric_key.split("/")[-1] if "/" in self.val_metric_key else self.val_metric_key
        current_value = main_metrics.get(metric_key, 0.0)
        best_value = self.best_val_metric
        logger.info("=" * 70)
        logger.info(f"🏆 {metric_key}: {current_value:.2f}  |  Best: {best_value:.2f}")
        logger.info("=" * 70)
        logger.info("")

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, images: torch.Tensor, targets: Optional[list] = None):
        """
        Forward pass through the model.

        Args:
            images: Input images tensor [B, C, H, W]
            targets: Optional list of target dictionaries for training

        Returns:
            Model output
        """
        if targets is not None:
            return self.model(images, targets)
        return self.model(images)

    # =========================================================================
    # TRAINING LOGIC
    # =========================================================================

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Track when batch loading starts for data_time calculation."""
        self._batch_start_time = time.perf_counter()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Execute one training step.

        Args:
            batch: Batch data from dataloader
            batch_idx: Index of the batch

        Returns:
            Loss tensor for backpropagation
        """
        batch_size = len(batch)

        # Calculate data loading time (time between end of previous step and start of this batch)
        if self._step_end_time is not None and self._batch_start_time is not None:
            data_time = self._batch_start_time - self._step_end_time
            self._data_time_history.append(data_time)
            self.log("data_time", data_time, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

        # Preprocess data using the processor
        images, targets = self.processor.preprocess(
            batch,  # type: ignore
            dtype=torch.float16 if self.train_args.amp_enabled else torch.float32,
            device=self.device,
        )

        # Forward pass
        output = self.model(images, targets)
        loss_dict = output.loss

        # Handle different loss formats
        if isinstance(loss_dict, torch.Tensor):
            total_loss = loss_dict
            self.log("total_loss", total_loss, prog_bar=True, batch_size=batch_size)
            # Log epoch-averaged loss for progress bar
            self.log("total_loss_epoch", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        else:
            # Sum all losses
            total_loss = sum(loss_dict.values())  # type: ignore

            # Log individual losses (no prefix)
            for loss_name, loss_value in loss_dict.items():
                self.log(f"{loss_name}", loss_value, prog_bar=False, batch_size=batch_size)

            self.log("total_loss", total_loss, prog_bar=True, batch_size=batch_size, enable_graph=True)
            # Log epoch-averaged loss for progress bar
            self.log("total_loss_epoch", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        # Calculate and log step time
        step_end_time = time.perf_counter()
        if self._batch_start_time is not None:
            step_time = step_end_time - self._batch_start_time
            self._time_history.append(step_time)
            self.log("time", step_time, prog_bar=False, batch_size=batch_size)

        # Log GPU memory usage to progress bar in MB with fixed format
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            # Use format that shows as "XXXXM" in progress bar
            self.log("mem_MB", int(max_mem_mb), prog_bar=True, batch_size=batch_size)

        # Record end time for next iteration's data_time calculation
        self._step_end_time = step_end_time

        # Print compact metrics periodically (like CommonMetricPrinter)
        current_iter = self.global_step + 1
        if current_iter % self.train_args.log_period == 0:
            # Collect current metrics
            loss_value = total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
            metrics_to_print = {"total_loss": loss_value}
            if not isinstance(loss_dict, torch.Tensor):
                for k, v in loss_dict.items():
                    metrics_to_print[k] = v.item()

            # Add learning rate if available
            if self.optimizers():
                optimizer = self.optimizers()
                if isinstance(optimizer, Optimizer):
                    metrics_to_print["lr"] = optimizer.param_groups[0]["lr"]

        return total_loss  # type: ignore

    def on_train_epoch_end(self):
        """Called at the end of each training epoch to log learning rate."""
        if self.optimizers():
            optimizer = self.optimizers()
            if isinstance(optimizer, Optimizer):
                lr = optimizer.param_groups[0]["lr"]
                self.log("train/learning_rate", lr, on_epoch=True)

    # =========================================================================
    # VALIDATION LOGIC
    # =========================================================================

    def validation_step(self, batch: List[DatasetEntry], batch_idx: int) -> Dict[str, Any]:
        """
        Execute one validation step.

        Args:
            batch: Batch of DatasetEntry objects
            batch_idx: Index of the batch

        Returns:
            Dictionary with predictions
        """
        # Store inputs for evaluation
        self.validation_step_inputs.extend(batch)

        # Preprocess and run inference
        images, _ = self.processor.preprocess(
            batch,  # type: ignore
            dtype=torch.float16 if self.train_args.amp_enabled else torch.float32,
            device=self.device,
        )

        # Forward pass (no targets for validation)
        with torch.no_grad():
            outputs = self.model(images)

        # Postprocess outputs for evaluation (converts raw model outputs to DatasetEntry format)
        outputs = self.processor.eval_postprocess(outputs, batch)

        # Store outputs for evaluation (extend, not append, since outputs is a list of predictions)
        self.validation_step_outputs.extend(outputs)

        # Return predictions (Lightning will automatically gather these)
        return {"predictions": outputs}

    def on_validation_epoch_start(self):
        """Clear accumulated data from previous validation epoch."""
        self.validation_step_inputs.clear()
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        """
        Compute and log validation metrics at the end of validation epoch.

        This method:
        1. Gathers all predictions from validation steps
        2. Creates task-specific evaluator
        3. Computes metrics (AP, mIoU, etc.)
        4. Logs metrics to Lightning logger
        5. Clears accumulated data to prevent memory leaks
        """
        # Skip if no validation data was processed (e.g., during sanity check)
        if len(self.validation_step_inputs) == 0 or len(self.validation_step_outputs) == 0:
            if not getattr(self.trainer, "sanity_checking", False):
                logger.warning(
                    f"No validation data to evaluate. Inputs: {len(self.validation_step_inputs)}, Outputs: {len(self.validation_step_outputs)}"
                )
            return

        logger.info(f"📊 Computing metrics on {len(self.validation_step_inputs)} validation samples")

        # Get the validation dataset from the trainer's datamodule
        datamodule = getattr(self.trainer, "datamodule", None)
        if not datamodule or not hasattr(datamodule, "val_dataset"):
            logger.warning("Could not access validation dataset for metrics computation")
            return

        val_dict_dataset = datamodule.val_dataset.dict_dataset

        # Compute metrics based on task
        try:
            self._run_evaluation(val_dict_dataset)
        finally:
            # CRITICAL: Clear accumulated data to prevent memory leaks
            # This ensures GPU tensors are released after evaluation
            self.validation_step_inputs.clear()
            self.validation_step_outputs.clear()

    def _run_evaluation(self, val_dict_dataset):
        """
        Unified evaluation function for all tasks.

        Args:
            val_dict_dataset: Validation dataset dictionary
        """
        # Initialize evaluator on first use (lazy initialization)
        if self.evaluator is None:
            if self.task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
                task_name = "segm" if self.task == Task.INSTANCE_SEGMENTATION else "bbox"
                self.evaluator = DetectionLightningEvaluator(
                    dataset_dict=val_dict_dataset,
                    task=task_name,
                    distributed=False,  # Lightning handles distribution
                )
            elif self.task == Task.SEMSEG:
                self.evaluator = SemSegEvaluator(
                    dataset_dict=val_dict_dataset,
                    distributed=False,  # Lightning handles distribution
                )
            else:
                logger.warning(f"No evaluator available for task: {self.task}")
                return
        self.evaluator.reset()
        # Task-specific preprocessing
        if self.task == Task.SEMSEG:
            # Add sem_seg_file_name from dataset to inputs
            for inp in self.validation_step_inputs:
                if hasattr(inp, "image_id") and inp.image_id is not None:
                    dataset_item = val_dict_dataset[inp.image_id]
                    if "sem_seg_file_name" in dataset_item:
                        inp.sem_seg_file_name = dataset_item["sem_seg_file_name"]  # type: ignore

        # Process all accumulated predictions
        self.evaluator.process(self.validation_step_inputs, self.validation_step_outputs)  # type: ignore

        # Evaluate and get metrics
        metrics = self.evaluator.evaluate()

        # Determine task name for logging
        if self.task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
            task_name = "segm" if self.task == Task.INSTANCE_SEGMENTATION else "bbox"
            primary_metric = "AP"
        elif self.task == Task.SEMSEG:
            task_name = "sem_seg"
            primary_metric = "mIoU"
        else:
            return

        # Log metrics and print results
        if metrics and task_name in metrics:
            task_metrics = metrics[task_name]

            # Print formatted results
            self._print_evaluation_results(task_metrics, task_name)

            # Log metrics to Lightning logger
            for metric_name, metric_value in task_metrics.items():
                if metric_value is not None and metric_value == metric_value:  # Check for NaN
                    # For per-category metrics (AP-class_name), log under separate namespace
                    if metric_name.startswith("AP-"):
                        class_name = metric_name[3:]  # Remove "AP-" prefix
                        sanitized_name = class_name.replace(" ", "_").replace("/", "_")
                        self.log(
                            f"{task_name}/{sanitized_name}",
                            metric_value,
                            on_epoch=True,
                        )
                    else:
                        # Log main metrics
                        self.log(
                            f"{task_name}/{metric_name}",
                            metric_value,
                            on_epoch=True,
                            prog_bar=(metric_name == primary_metric),
                        )

            # Track best metric
            if self.val_metric_key in self.trainer.callback_metrics:
                metric_value = self.trainer.callback_metrics[self.val_metric_key]
                # Convert to float if it's a tensor
                current_val_metric = (
                    metric_value.item() if isinstance(metric_value, torch.Tensor) else float(metric_value)
                )
                if current_val_metric > self.best_val_metric:
                    self.best_val_metric = current_val_metric

    # =========================================================================
    # OPTIMIZER & SCHEDULER CONFIGURATION
    # =========================================================================

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Build optimizer using Focoos solver
        optimizer = build_optimizer(
            name=self.train_args.optimizer,
            learning_rate=self.train_args.learning_rate,
            weight_decay=self.train_args.weight_decay,
            model=self.model,
            weight_decay_norm=self.train_args.weight_decay_norm,
            weight_decay_embed=self.train_args.weight_decay_embed,
            backbone_multiplier=self.train_args.backbone_multiplier,
            decoder_multiplier=self.train_args.decoder_multiplier,
            head_multiplier=self.train_args.head_multiplier,
            clip_gradients=self.train_args.clip_gradients,
        )

        # Build scheduler using Focoos scheduler
        scheduler = build_lr_scheduler(
            name=self.train_args.scheduler,
            max_iters=self.train_args.max_iters,
            optimizer=optimizer,
        )

        # Return configuration for Lightning
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,
            },
        }
