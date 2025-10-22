# Copyright (c) FocoosAI
"""
Lightning callback for writing metrics to JSON file.

This callback mimics the behavior of the old JSONWriter hook, saving
training and validation metrics to a JSON file.
"""

import json
import os
from typing import Any, Dict, List

from lightning.pytorch.callbacks import Callback

from focoos.utils.logger import get_logger

logger = get_logger("MetricsJSONWriter")


class MetricsJSONWriter(Callback):
    """
    Lightning callback that writes metrics to a JSON file.

    This callback saves all training and validation metrics to a JSON file
    in the output directory, compatible with the old trainer format.

    Args:
        output_dir: Directory where to save the metrics.json file
        log_period: Save metrics every log_period steps (default: 20)
        filename: Name of the JSON file (default: "metrics.json")
    """

    def __init__(self, output_dir: str, log_period: int = 20, filename: str = "metrics.json"):
        super().__init__()
        self.output_dir = output_dir
        self.log_period = log_period
        self.filepath = os.path.join(output_dir, filename)
        self.metrics_history: List[Dict[str, Any]] = []

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"MetricsJSONWriter initialized. Saving to: {self.filepath} every {log_period} steps")

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        """Called when a training batch ends."""
        # Collect training metrics from the current step
        if trainer.callback_metrics:
            if (trainer.global_step + 1) % self.log_period == 0:
                step_metrics = {}

                # Add iteration number
                step_metrics["iteration"] = trainer.global_step

                # Add training metrics ONLY (no validation metrics)
                for key, value in trainer.callback_metrics.items():
                    # Skip validation metrics (they have different prefixes and should only be saved on validation_end)
                    if (
                        key.startswith("val/")
                        or key.startswith("bbox/")
                        or key.startswith("segm/")
                        or key.startswith("sem_seg/")
                    ):
                        continue

                    # Only include training metrics
                    if key.startswith("loss_") or key.startswith("total_loss") or key == "data_time" or key == "time":
                        # Remove '_step' suffix for compatibility
                        metric_name = key.replace("_step", "")
                        try:
                            step_metrics[metric_name] = float(value)
                        except (TypeError, ValueError):
                            pass  # Skip non-scalar values

                # Only save if we have training metrics
                if len(step_metrics) > 1:  # More than just iteration
                    # Check if this iteration already exists
                    existing_idx = next(
                        (
                            i
                            for i, m in enumerate(self.metrics_history)
                            if m.get("iteration") == step_metrics["iteration"]
                        ),
                        None,
                    )

                    if existing_idx is not None:
                        # Update existing entry
                        self.metrics_history[existing_idx].update(step_metrics)
                    else:
                        # Add new entry
                        self.metrics_history.append(step_metrics)

                        self._save_metrics()

    def on_validation_end(self, trainer, pl_module):
        """Called when validation ends."""
        # Add validation metrics to the current iteration entry
        if trainer.callback_metrics:
            val_metrics = {}

            # Collect validation metrics (with any validation-related prefix)
            for key, value in trainer.callback_metrics.items():
                if (
                    key.startswith("val/")
                    or key.startswith("bbox/")
                    or key.startswith("segm/")
                    or key.startswith("sem_seg/")
                ):
                    # Keep the full path for validation metrics
                    try:
                        val_metrics[key] = float(value)
                    except (TypeError, ValueError):
                        pass  # Skip non-scalar values

            if val_metrics:
                # Get the current iteration metrics or create a new entry
                current_iter = trainer.global_step
                existing_idx = next(
                    (i for i, m in enumerate(self.metrics_history) if m.get("iteration") == current_iter), None
                )

                if existing_idx is not None:
                    # Update existing entry with validation metrics
                    self.metrics_history[existing_idx].update(val_metrics)
                else:
                    # Create new entry with just validation metrics
                    val_metrics["iteration"] = current_iter
                    self.metrics_history.append(val_metrics)

        # Save to file after each validation
        self._save_metrics()

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        # Final save
        self._save_metrics()
        logger.info(f"✅ Metrics saved to {self.filepath}")

    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
