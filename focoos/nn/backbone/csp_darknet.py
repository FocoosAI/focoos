# Copyright (c) MMPose authors.
#
# Reimplemented from MMPose: https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/csp_darknet.py.

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.utils.logger import get_logger

logger = get_logger("Backbone")


class ConvModule(nn.Module):
    """Conv module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The kernel size of the convolution.
        stride (int): The stride of the convolution.
        padding (int): The padding of the convolution.
        dilation (int): The dilation of the convolution.
        bias (bool): Whether to use bias.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        momentum=0.03,
        eps=0.001,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum, eps=eps)
        self.activate = nn.SiLU()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class DarknetBottleneck(nn.Module):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size of the convolution.
            Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 1)
        self.conv2 = ConvModule(hidden_channels, out_channels, 3, stride=1, padding=1)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish')
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        channel_attention: bool = False,
    ) -> None:
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)

        self.blocks = nn.Sequential(
            *[DarknetBottleneck(mid_channels, mid_channels, 1.0, add_identity) for _ in range(num_blocks)]
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
    ):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(5, 9, 13),
    ):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1, stride=1)
        self.poolings = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(conv2_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@dataclass
class CSPConfig(BackboneConfig):
    size: Literal["small", "medium", "large"] = "small"
    model_type: str = "csp_darknet"


CONFIGS = {
    "small": {
        "arch_setting": [
            [32, 64, 1, True, False],
            [64, 128, 3, True, False],
            [128, 256, 3, True, False],
            [256, 512, 1, False, True],
        ],
        "url": "https://public.focoos.ai/pretrained_models/backbones/csp_darknet_small.pth",
    },
    "medium": {
        "arch_setting": [
            [48, 96, 2, True, False],
            [96, 192, 6, True, False],
            [192, 384, 6, True, False],
            [384, 768, 2, False, True],
        ],
        "url": "https://public.focoos.ai/pretrained_models/backbones/csp_darknet_medium.pth",
    },
    "large": {
        "arch_setting": [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 1024, 3, False, True],
        ],
        "url": "https://public.focoos.ai/pretrained_models/backbones/csp_darknet_large.pth",
    },
}


class CSPDarknet(BaseBackbone):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Reimplemented from MMPose: https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/csp_darknet.py

    Args:
        config: Configuration object containing model parameters

    """

    def __init__(
        self,
        config: CSPConfig,
    ):
        super().__init__(config)
        arch_setting = CONFIGS[config.size]["arch_setting"]

        self.out_indices = [1, 2, 3, 4]
        self.frozen_stages = -1
        self.norm_eval = False

        self.stem = Focus(
            3,
            int(arch_setting[0][0]),
            kernel_size=3,
        )
        self.layers = ["stem"]

        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            stage = []
            conv_layer = ConvModule(in_channels, out_channels, 3, stride=2, padding=1)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=[5, 9, 13],
                )
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
            )
            stage.append(csp_layer)
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

        self.out_features = ["res2", "res3", "res4", "res5"]
        self.out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self.out_feature_channels = {
            "res2": arch_setting[0][1],
            "res3": arch_setting[1][1],
            "res4": arch_setting[2][1],
            "res5": arch_setting[3][1],
        }
        if config.use_pretrained:
            if config.backbone_url:
                state = torch.hub.load_state_dict_from_url(config.backbone_url)
                self.load_state_dict(state)
                logger.info(f"Loaded pretrained weights from {config.backbone_url}")
            else:
                state = torch.hub.load_state_dict_from_url(CONFIGS[config.size]["url"])
                self.load_state_dict(state)
                logger.info(f"Loaded pretrained weights from {CONFIGS[config.size]['url']}")

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)

            if i in self.out_indices:
                outs.append(x)
        return {"res2": outs[0], "res3": outs[1], "res4": outs[2], "res5": outs[3]}
