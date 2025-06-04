import collections.abc
from itertools import repeat

import torch.nn as nn


# Layer/Module Helpers
# Hacked together by / Copyright 2020 Ross Wightman (TIMM library)
# From torch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect implementation for EfficientNet networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper.
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956

    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode
        scale_by_keep: Whether to scale the kept paths to maintain sum

    Returns:
        Tensor with paths dropped
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# Copyright 2020 TIMM library
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initialize DropPath module.

        Args:
            drop_prob: Probability of dropping a path
            scale_by_keep: Whether to scale the kept paths to maintain sum
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Apply drop path to input tensor.

        Args:
            x: Input tensor

        Returns:
            Tensor with paths dropped
        """
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        """Return string representation of module parameters."""
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"
