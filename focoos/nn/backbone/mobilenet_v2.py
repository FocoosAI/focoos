from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn

from focoos.nn.layers.conv import Conv2d
from focoos.nn.layers.norm import get_norm

from .base import BackboneConfig, BaseBackbone


class InvertedResidual(nn.Module):
    """InvertedResidual block for MobileNetV2.

    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        act (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').

    Returns:
        Tensor: The output tensor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
        dilation=1,
        norm="BN",
        activation=None,
        **kwargs,
    ):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2], f"stride must in [1, 2]. But received {stride}."
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    bias="",
                    norm=get_norm(norm, hidden_dim),
                    activation=activation,
                    **kwargs,
                )
            )
        layers.extend(
            [
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden_dim,
                    bias="",
                    norm=get_norm(norm, hidden_dim),
                    activation=activation,
                    **kwargs,
                ),
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias="",
                    norm=get_norm(norm, out_channels),
                    activation=activation,
                    **kwargs,
                ),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@dataclass
class MobileNetV2Config(BackboneConfig):
    """MobileNetV2 configuration"""

    in_chans: int = 3
    widen_factor: float = 1.0
    strides: Tuple[int, ...] = (1, 2, 2, 2, 1, 2, 1)
    dilations: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1)
    frozen_stages: int = -1
    norm: str = "BN"
    model_type: str = "mobilenet_v2"


class MobileNetV2(BaseBackbone):
    """MobileNetV2 backbone.

    This backbone is the implementation of
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <https://arxiv.org/abs/1801.04381>`_.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        strides (Sequence[int], optional): Strides of the first block of each
            layer. If not specified, default config in ``arch_setting`` will
            be used.
        dilations (Sequence[int]): Dilation of each layer.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    arch_settings = [
        [1, 16, 1],
        [6, 24, 2],
        [6, 32, 3],
        [6, 64, 4],
        [6, 96, 3],
        [6, 160, 3],
        [6, 320, 1],
    ]

    def __init__(
        self,
        config: MobileNetV2Config,
    ):
        super().__init__(config)
        self.widen_factor = config.widen_factor
        self.strides = config.strides
        self.dilations = config.dilations
        assert len(config.strides) == len(config.dilations) == len(self.arch_settings)

        if config.frozen_stages not in range(-1, 7):
            raise ValueError(f"frozen_stages must be in range(-1, 7). But received {config.frozen_stages}")
        self.frozen_stages = config.frozen_stages
        self.norm = config.norm
        self.act = nn.functional.relu6

        self.in_channels = int(32 * config.widen_factor)

        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self._out_features = ["res2", "res3", "res4", "res5"]

        self.conv1 = Conv2d(
            in_channels=config.in_chans,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias="",
            norm=get_norm(config.norm, self.in_channels),
            activation=self.act,
        )

        self.layers = []
        self.layer_to_res = {
            "layer2": "res2",
            "layer3": "res3",
            "layer5": "res4",
            "layer7": "res5",
        }
        tot_stride = 1

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            tot_stride = tot_stride * stride
            dilation = self.dilations[i]
            out_channels = int(channel * self.widen_factor)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                expand_ratio=expand_ratio,
            )
            layer_name = f"layer{i + 1}"
            if layer_name in self.layer_to_res:
                res_block = self.layer_to_res[layer_name]
                self._out_feature_strides[res_block] = tot_stride
                self._out_feature_channels[res_block] = out_channels
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def make_layer(self, out_channels, num_blocks, stride, dilation, expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.

        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            dilation (int): Dilation of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidual(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1,
                    norm=self.norm,
                    activation=self.act,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        outs = {}
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if layer_name in self.layer_to_res:
                res_block = self.layer_to_res[layer_name]
                if res_block in self._out_features:
                    outs[res_block] = x

        return outs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
