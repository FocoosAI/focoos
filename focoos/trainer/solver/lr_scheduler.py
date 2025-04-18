# Copyright (c) Focoos AI S.r.L.
import math
from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import LRScheduler

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIX ME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


class BaseLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return self.base_lrs


class WarmupPolyLR(BaseLRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    Reference: https://github.com/tensorflow/models/blob/21b73d22f3ed05b650e85ac50849408dd36de32e/research/deeplab/utils/train_utils.py#L337  # noqa
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 1.0,
        warmup_iters: int = 0,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, max_iters, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if math.pow((1.0 - self.last_epoch / self.max_iters), self.power) < self.constant_ending:
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iters), self.power)
            for base_lr in self.base_lrs
        ]


class WarmupMultiStepLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        milestones: List[float] = [],
        gamma: float = 0.1,
        warmup_factor: float = 1.0,
        warmup_iters: int = 0,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}",
                milestones,
            )
        self.milestones = [int(m * max_iters) for m in milestones]
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, max_iters, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 1.0,
        warmup_iters: int = 0,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, max_iters, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]


def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
