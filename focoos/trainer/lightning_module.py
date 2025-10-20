"""
FocoosLightningModule - PyTorch Lightning wrapper for Focoos models.
This module enables training Focoos models using PyTorch Lightning's Trainer.
"""

from typing import Any, Dict, List, Optional

import lightning as L
import torch
from torch.optim import Optimizer

from focoos.models.base_model import BaseModelNN
from focoos.ports import DatasetEntry, ModelInfo, Task, TrainerArgs
from focoos.processor.base_processor import Processor
from focoos.trainer.evaluation.detection_evaluation import DetectionEvaluator
from focoos.trainer.evaluation.sem_seg_evaluation import SemSegEvaluator
from focoos.trainer.solver.build import build_lr_scheduler, build_optimizer
from focoos.utils.logger import get_logger

logger = get_logger("FocoosLightningModule")


class FocoosLightningModule(L.LightningModule):
    """
    PyTorch Lightning Module for Focoos models.

    This module wraps a Focoos BaseModelNN and provides Lightning-compatible
    training and validation logic. It handles:
    - Training step with loss computation
    - Validation step with metrics
    - Optimizer and scheduler configuration
    - Automatic mixed precision support

    Args:
        model: The Focoos BaseModelNN to train
        processor: The Focoos processor for preprocessing
        model_info: Model metadata and configuration
        train_args: Training configuration arguments

    Example:
        >>> from focoos.model_manager import ModelManager
        >>> from focoos.data.lightning import FocoosLightningDataModule
        >>> from focoos.trainer.lightning_module import FocoosLightningModule
        >>> # Load model and setup
        >>> model = ModelManager.get("fai-detr-s-coco", num_classes=80)
        >>> lightning_model = FocoosLightningModule(model=model.model, processor=model.processor, model_info=model.model_info, train_args=train_args)
        >>> # Train with Lightning Trainer
        >>> trainer = L.Trainer(max_epochs=10, accelerator="gpu")
        >>> trainer.fit(lightning_model, datamodule=data_module)
    """

    def __init__(
        self,
        model: BaseModelNN,
        processor: Processor,
        model_info: ModelInfo,
        train_args: TrainerArgs,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model_info = model_info
        self.train_args = train_args
        self.task = model_info.task

        # Save hyperparameters (excluding complex objects)
        self.save_hyperparameters(ignore=["model", "processor", "model_info"])

        # Track best validation metric
        self.best_val_metric = 0.0

        # Determine the primary metric based on task
        self.val_metric_key = self._get_task_metric_key()

        # Validation accumulation
        self.validation_step_outputs: List[Dict[str, Any]] = []
        self.validation_step_inputs: List[DatasetEntry] = []

        logger.info(f"🚀 FocoosLightningModule initialized for task: {self.task}")
        logger.info(f"📊 Tracking validation metric: {self.val_metric_key}")

    def _get_task_metric_key(self) -> str:
        """Get the primary metric key for the current task."""
        task_metrics = {
            Task.DETECTION.value: "val/bbox_AP",
            Task.SEMSEG.value: "val/sem_seg_mIoU",
            Task.INSTANCE_SEGMENTATION.value: "val/segm_AP",
            Task.CLASSIFICATION.value: "val/classification_F1",
            Task.KEYPOINT.value: "val/keypoints_AP",
        }
        return task_metrics.get(self.task.value, "val/loss")

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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step - computes loss on a batch.

        Args:
            batch: Batch dictionary from the dataloader
            batch_idx: Index of the batch

        Returns:
            Loss tensor for backpropagation
        """
        # Preprocess data using the processor
        images, targets = self.processor.preprocess(
            batch,  # type: ignore
            dtype=torch.float16 if self.train_args.amp_enabled else torch.float32,
            device=self.device,
        )

        # Forward pass
        output = self.model(images, targets)
        loss_dict = output.loss

        # Get batch size for logging
        batch_size = len(batch)

        # Handle different loss formats
        if isinstance(loss_dict, torch.Tensor):
            total_loss = loss_dict
            self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        else:
            # Sum all losses
            total_loss = sum(loss_dict.values())  # type: ignore

            # Log individual losses
            for loss_name, loss_value in loss_dict.items():
                self.log(f"train/{loss_name}", loss_value, on_step=True, on_epoch=True, batch_size=batch_size)

            self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return total_loss  # type: ignore

    def validation_step(self, batch: List[DatasetEntry], batch_idx: int) -> Dict[str, Any]:
        """
        Validation step - generates predictions and accumulates for metrics calculation.

        Args:
            batch: List of DatasetEntry from the validation dataloader
            batch_idx: Index of the batch

        Returns:
            Dictionary with validation outputs
        """
        # Preprocess data (without targets for eval mode)
        images, _ = self.processor.preprocess(
            batch,  # type: ignore
            dtype=torch.float32,
            device=self.device,
        )

        # Forward pass WITHOUT targets to get predictions (model is in eval mode)
        with torch.no_grad():
            output = self.model(images, [])

            # Use eval_postprocess to get predictions in the right format
            predictions = self.processor.eval_postprocess(output, batch)

            # Accumulate for metric calculation at epoch end
            if predictions:
                for inp, pred in zip(batch, predictions):
                    self.validation_step_inputs.append(inp)
                    self.validation_step_outputs.append(pred)

        return {}

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Build optimizer using Focoos optimizer builder
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
            extra=self.train_args.optimizer_extra,
        )

        # Build scheduler using Focoos scheduler builder
        scheduler = build_lr_scheduler(
            name=self.train_args.scheduler,
            max_iters=self.train_args.max_iters,
            optimizer=optimizer,
            extra=self.train_args.scheduler_extra,
        )

        logger.info(f"🔧 Optimizer: {self.train_args.optimizer}, Scheduler: {self.train_args.scheduler}")

        # Return optimizer and scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update scheduler per step
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        self.model.train()
        self.processor.train()

    def on_validation_epoch_start(self):
        """Called at the start of each validation epoch."""
        # Set model to eval mode to get predictions
        # (FAIDetr returns empty tensors in train mode to save memory)
        self.model.eval()
        self.processor.train()  # Keep processor in train mode to handle DatasetEntry format with targets

        # Clear accumulated data from previous epoch/sanity check
        self.validation_step_inputs.clear()
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log learning rate
        if self.optimizers():
            optimizer = self.optimizers()
            if isinstance(optimizer, Optimizer):
                lr = optimizer.param_groups[0]["lr"]
                self.log("train/learning_rate", lr, on_epoch=True)

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch - compute COCO metrics."""
        # Skip if no validation data was processed (e.g., during sanity check)
        if len(self.validation_step_inputs) == 0 or len(self.validation_step_outputs) == 0:
            # Don't log warning during sanity check (trainer.sanity_checking is True)
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
        if self.task in [Task.DETECTION, Task.INSTANCE_SEGMENTATION]:
            # Create detection/instance segmentation evaluator
            task_name = "segm" if self.task == Task.INSTANCE_SEGMENTATION else "bbox"
            evaluator = DetectionEvaluator(
                dataset_dict=val_dict_dataset,
                task=task_name,
                distributed=False,  # Lightning handles distribution
            )

            # Process all accumulated predictions
            evaluator.process(self.validation_step_inputs, self.validation_step_outputs)

            # Evaluate and get metrics
            metrics = evaluator.evaluate()

            # Log metrics
            if metrics and task_name in metrics:
                task_metrics = metrics[task_name]
                for metric_name, metric_value in task_metrics.items():
                    if metric_value is not None and metric_value == metric_value:  # Check for NaN
                        self.log(
                            f"val/{task_name}_{metric_name}",
                            metric_value,
                            on_epoch=True,
                            prog_bar=(metric_name == "AP"),
                        )

                # Track best metric
                if self.val_metric_key in self.trainer.callback_metrics:
                    current_val_metric = self.trainer.callback_metrics[self.val_metric_key].item()
                    if current_val_metric > self.best_val_metric:
                        self.best_val_metric = current_val_metric
                        logger.info(f"✨ New best validation metric: {current_val_metric:.4f}")

        elif self.task == Task.SEMSEG:
            # Create semantic segmentation evaluator
            evaluator = SemSegEvaluator(
                dataset_dict=val_dict_dataset,
                distributed=False,  # Lightning handles distribution
            )
            evaluator.reset()  # Initialize confusion matrix

            # Add sem_seg_file_name from dataset to inputs before evaluation
            for inp in self.validation_step_inputs:
                # Get the original dataset entry by image_id
                if hasattr(inp, "image_id") and inp.image_id is not None:
                    dataset_item = val_dict_dataset[inp.image_id]
                    if "sem_seg_file_name" in dataset_item:
                        inp.sem_seg_file_name = dataset_item["sem_seg_file_name"]  # type: ignore

            # Process all accumulated predictions
            # DatasetEntry with sem_seg_file_name is compatible with SemanticSegmentationDatasetEntry
            evaluator.process(self.validation_step_inputs, self.validation_step_outputs)  # type: ignore

            # Evaluate and get metrics
            metrics = evaluator.evaluate()

            # Log metrics
            if metrics and "sem_seg" in metrics:
                task_metrics = metrics["sem_seg"]
                for metric_name, metric_value in task_metrics.items():
                    if metric_value is not None and metric_value == metric_value:  # Check for NaN
                        self.log(
                            f"val/sem_seg_{metric_name}",
                            metric_value,
                            on_epoch=True,
                            prog_bar=(metric_name == "mIoU"),
                        )

                # Track best metric
                if self.val_metric_key in self.trainer.callback_metrics:
                    current_val_metric = self.trainer.callback_metrics[self.val_metric_key].item()
                    if current_val_metric > self.best_val_metric:
                        self.best_val_metric = current_val_metric
                        logger.info(f"✨ New best validation metric: {current_val_metric:.4f}")

    def on_train_start(self):
        """Called at the start of training."""
        logger.info("🚀 Starting training...")
        logger.info(f"📊 Max iterations: {self.train_args.max_iters}")
        logger.info(f"📊 Batch size: {self.train_args.batch_size}")
        logger.info(f"📊 Learning rate: {self.train_args.learning_rate}")

    def on_train_end(self):
        """Called at the end of training."""
        logger.info("🏁 Training completed!")
        logger.info(f"✨ Best validation metric: {self.best_val_metric:.4f}")

    def get_model(self) -> BaseModelNN:
        """Get the wrapped Focoos model."""
        return self.model

    def get_processor(self) -> Processor:
        """Get the Focoos processor."""
        return self.processor

    def get_model_info(self) -> ModelInfo:
        """Get the model metadata."""
        return self.model_info
