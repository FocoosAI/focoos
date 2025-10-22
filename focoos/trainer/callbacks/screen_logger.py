# Copyright (c) FocoosAI
"""
Lightning callback for printing metrics to screen.

This callback mimics the behavior of CommonMetricPrinter, showing
training progress with ETA, losses, time metrics, and learning rate.
"""

import datetime
import time
from collections import deque
from typing import Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback

from focoos.utils.logger import get_logger

logger = get_logger("metrics")


class ScreenLogger(Callback):
    """
    Lightning callback that prints training metrics to the screen.

    Prints common metrics including iteration, losses, time, data_time,
    learning rate, ETA, and memory usage. Similar to CommonMetricPrinter
    from the original trainer.

    Args:
        max_iter: Maximum number of training iterations (for ETA calculation)
        log_period: Print metrics every log_period steps (default: 20)
        window_size: Window size for smoothing metrics (default: 20)
    """

    def __init__(self, max_iter: int, log_period: int = 20, window_size: int = 20):
        super().__init__()
        self.max_iter = max_iter
        self.log_period = log_period
        self.window_size = window_size

        # History tracking for smoothing
        self.time_history: deque = deque(maxlen=window_size)
        self.data_time_history: deque = deque(maxlen=window_size)
        self.loss_history: Dict[str, deque] = {}

        # For ETA calculation
        self._last_write_step: Optional[int] = None
        self._last_write_time: Optional[float] = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when a training batch ends."""
        # Only log at log_period intervals
        if (trainer.global_step + 1) % self.log_period != 0:
            return

        metrics = trainer.callback_metrics
        iteration = trainer.global_step

        # Collect time metrics
        if "time" in metrics:
            self.time_history.append(float(metrics["time"]))
        if "data_time" in metrics:
            self.data_time_history.append(float(metrics["data_time"]))

        # Collect loss metrics
        for key, value in metrics.items():
            if "loss" in key:
                if key not in self.loss_history:
                    self.loss_history[key] = deque(maxlen=self.window_size)
                self.loss_history[key].append(float(value))

        # Calculate ETA
        eta_string = self._get_eta(iteration)

        # Get learning rate
        try:
            optimizer = trainer.optimizers[0] if trainer.optimizers else None
            if optimizer:
                lr = optimizer.param_groups[0]["lr"]
                lr_str = f"{lr:.5g}"
            else:
                lr_str = "N/A"
        except (IndexError, KeyError):
            lr_str = "N/A"

        # Get memory usage
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            memory_str = f"max_mem: {max_mem_mb:.0f}M"
        else:
            memory_str = ""

        # Build losses string (with median smoothing)
        losses_str = "  ".join(
            [f"{key}: {self._median(values):.4g}" for key, values in self.loss_history.items() if len(values) > 0]
        )

        # Build time metrics string
        time_metrics = []
        if len(self.time_history) > 0:
            avg_time = sum(self.time_history) / len(self.time_history)
            time_metrics.append(f"time: {avg_time:.4f}")
        if len(self.data_time_history) > 0:
            avg_data_time = sum(self.data_time_history) / len(self.data_time_history)
            time_metrics.append(f"data_time: {avg_data_time:.4f}")
        time_metrics_str = "  ".join(time_metrics)

        # Build final message
        log_parts = []
        if eta_string:
            log_parts.append(f"eta: {eta_string}")
        log_parts.append(f"iter: {iteration}")
        if losses_str:
            log_parts.append(losses_str)
        if time_metrics_str:
            log_parts.append(time_metrics_str)
        log_parts.append(f"lr: {lr_str}")
        if memory_str:
            log_parts.append(memory_str)

        logger.info("  ".join(log_parts))

    def _get_eta(self, iteration: int) -> Optional[str]:
        """Calculate ETA based on time per iteration."""
        if iteration >= self.max_iter:
            return None

        # Use time history median if available
        if len(self.time_history) > 0:
            median_time = self._median(self.time_history)
            eta_seconds = median_time * (self.max_iter - iteration - 1)
            return str(datetime.timedelta(seconds=int(eta_seconds)))

        # Fallback: estimate based on last write
        if self._last_write_step is not None and self._last_write_time is not None:
            elapsed = time.perf_counter() - self._last_write_time
            steps_done = iteration - self._last_write_step
            if steps_done > 0:
                time_per_step = elapsed / steps_done
                eta_seconds = time_per_step * (self.max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                self._last_write_step = iteration
                self._last_write_time = time.perf_counter()
                return eta_string

        # First call
        self._last_write_step = iteration
        self._last_write_time = time.perf_counter()
        return None

    @staticmethod
    def _median(values):
        """Calculate median of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        return sorted_values[n // 2]
