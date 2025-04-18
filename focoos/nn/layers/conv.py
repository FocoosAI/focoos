import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import weight_init

from focoos.utils.env import TORCH_VERSION

from .norm import FrozenBatchNorm2d, get_norm


def check_if_dynamo_compiling():
    if TORCH_VERSION >= (2, 1):
        from torch._dynamo import is_compiling

        return is_compiling()
    else:
        return False


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            # Dynamo doesn't support context managers yet
            is_dynamo_compiling = check_if_dynamo_compiling()
            if not is_dynamo_compiling:
                with warnings.catch_warnings(record=True):
                    if x.numel() == 0 and self.training:
                        # https://github.com/pytorch/pytorch/issues/12013
                        assert not isinstance(self.norm, torch.nn.SyncBatchNorm), (
                            "SyncBatchNorm does not support empty inputs!"
                        )

        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class DepthwiseSeparableConv2d(nn.Module):
    """
    A kxk depthwise convolution + a 1x1 convolution.

    In :paper:`xception`, norm & activation are applied on the second conv.
    :paper:`mobilenet` uses norm & activation on both convs.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        dilation=1,
        *,
        norm1=None,
        activation1=None,
        norm2=None,
        activation2=None,
    ):
        """
        Args:
            norm1, norm2 (str or callable): normalization for the two conv layers.
            activation1, activation2 (callable(Tensor) -> Tensor): activation
                function for the two conv layers.
        """
        super().__init__()
        self.depthwise = Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=not norm1,
            norm=get_norm(norm1, in_channels),
            activation=activation1,
        )
        self.pointwise = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=not norm2,
            norm=get_norm(norm2, out_channels),
            activation=activation2,
        )

        # default initialization
        weight_init.c2_msra_fill(self.depthwise)
        weight_init.c2_msra_fill(self.pointwise)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
