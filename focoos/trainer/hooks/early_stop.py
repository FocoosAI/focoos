import logging

from focoos.trainer.hooks.base import HookBase


class EarlyStopException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class EarlyStoppingHook(HookBase):
    def __init__(
        self,
        enabled: bool,
        eval_period: int,
        patience: int,
        val_metric: str,
        mode: str = "max",
    ):
        """
        Initializes the EarlyStoppingHook.

        This hook is designed to monitor a specific validation metric during the training process
        and stop training when no improvement is observed in the metric for a specified number of
        iterations. This is particularly useful for preventing overfitting by halting training
        once the model's performance on the validation set no longer improves.

        Args:
            eval_period (int): The frequency (in iterations) at which the validation metric is evaluated.
                               For example, if `eval_period` is 100, the validation metric will be checked
                               every 100 iterations.
            patience (int): Number of consecutive evaluations with no improvement after which training will be stopped.
                            For example, if `patience` is set to 5, training will stop if the validation metric does not
                            improve for 5 consecutive evaluations.
            val_metric (str): The name of the validation metric to monitor. This should correspond to one of the metrics
                              calculated during the validation phase, such as "accuracy", "loss", etc.
            mode (str, optional): One of "min" or "max". This parameter dictates the direction of improvement
                                  for the validation metric. In "min" mode, the training will stop when the monitored
                                  quantity (e.g., loss) stops decreasing. In "max" mode, training will stop when
                                  the monitored quantity (e.g., accuracy) stops increasing. Defaults to "max".

        """
        self.enabled = enabled
        self.patience = patience
        self.val_metric = val_metric
        self.mode = mode
        self.best_metric = None
        self.num_bad_epochs = 0
        self._period = eval_period
        self._logger = logging.getLogger(__name__)

    def after_step(self):
        next_iter = self.trainer.iter + 1

        if self._period > 0 and next_iter % self._period == 0 and next_iter != self.trainer.max_iter and self.enabled:
            metric_tuple = self.trainer.storage.latest().get(self.val_metric)

            if metric_tuple is None:
                return
            else:
                current_metric, metric_iter = metric_tuple

            if (
                self.best_metric is None
                or (self.mode == "max" and current_metric > self.best_metric)
                or (self.mode == "min" and current_metric < self.best_metric)
            ):
                self.best_metric = current_metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                self._logger.info(f"{self.num_bad_epochs}/{self.patience} without improvements..")

            if self.num_bad_epochs >= self.patience:
                self.trainer.storage.put_scalar("early_stopping", True)
                raise EarlyStopException("Early Stopping Exception to stop the training..")
