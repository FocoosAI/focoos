import torch

from focoos.trainer.hooks.hook import HookBase
from focoos.utils.logger import get_logger

logger = get_logger("NaNGradientCheckHook")


class NaNGradientCheckHook(HookBase):
    """
    A hook that checks for NaN gradients after backward pass.
    If NaN gradients are detected, it can either:
    1. Simply log the occurrence
    2. Skip the optimization step to prevent NaN parameter updates
    3. Replace NaN gradients with zeros to maintain training
    """

    def __init__(self, fix_nans=True, skip_update=False, verbose=True):
        """
        Args:
            fix_nans (bool): Whether to replace NaN gradients with zeros
            skip_update (bool): Whether to skip optimization step if NaNs detected
            verbose (bool): Whether to log when NaN gradients are found
        """
        self.fix_nans = fix_nans
        self.skip_update = skip_update
        self.verbose = verbose
        self.nan_count = 0
        self._last_detected_nan = False

    def after_backward(self):
        """
        Check for NaN gradients after backward pass and before optimizer step.
        """
        trainer = self.trainer
        model = trainer.model

        # Check if any gradient is NaN
        has_nan = False
        nan_params = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan = True
                    nan_params.append(name)
                    if self.fix_nans:
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)

        if has_nan:
            self.nan_count += 1
            self._last_detected_nan = True

            if self.verbose:
                logger.warning(
                    f"NaN gradients detected at iteration={trainer.iter}! "
                    f"Affected params: {len(nan_params)}, "
                    f"Total incidents: {self.nan_count}"
                )

                if len(nan_params) < 10:  # Only log if there aren't too many NaN parameters
                    logger.warning(f"Parameters with NaN gradients: {nan_params}")
        else:
            self._last_detected_nan = False

    def after_step(self):
        """
        If skip_update is True and NaNs were detected, signal to skip the update.
        """
        if self.skip_update and self._last_detected_nan:
            # Skip the optimization step if NaN gradients were detected
            logger.warning(f"Skipping parameter update at iteration={self.trainer.iter} due to NaN gradients.")

    def state_dict(self):
        return {"nan_count": self.nan_count}

    def load_state_dict(self, state_dict):
        self.nan_count = state_dict["nan_count"]
