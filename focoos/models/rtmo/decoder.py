from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from focoos.models.rtmo.transformer import DetrTransformerEncoder, SinePositionalEncoding
from focoos.nn.backbone.base import ShapeSpec
from focoos.nn.backbone.csp_darknet import ConvModule


class ChannelMapper(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        num_outs: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)

        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    in_channel, out_channels, kernel_size, bias=bias, padding=(kernel_size - 1) // 2
                                ),
                            ),
                            ("bn", nn.BatchNorm2d(out_channels)),
                        ]
                    )
                )
            )

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        return tuple(outs)


class ProjectionConv(nn.Module):
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
        momentum=0.1,
        eps=1e-3,
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

    def forward(self, x):
        return self.bn(self.conv(x))


class RepVGGBlock(nn.Module):
    """A block in RepVGG architecture, supporting optional normalization in the
    identity branch.

    This block consists of 3x3 and 1x1 convolutions, with an optional identity
    shortcut branch that includes normalization.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): The stride of the block. Defaults to 1.
        padding (int): The padding of the block. Defaults to 1.
        dilation (int): The dilation of the block. Defaults to 1.
        groups (int): The groups of the block. Defaults to 1.
        padding_mode (str): The padding mode of the block. Defaults to 'zeros'.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to dict(type='BN').
        act_cfg (dict): The config dict for activation layers.
            Defaults to dict(type='ReLU').
        without_branch_norm (bool): Whether to skip branch_norm.
            Defaults to True.
        init_cfg (dict): The config dict for initialization. Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        without_branch_norm: bool = True,
    ):
        super(RepVGGBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # judge if input shape and output shape are the same.
        # If true, add a normalized identity shortcut.
        self.branch_norm = None
        if out_channels == in_channels and stride == 1 and padding == dilation and not without_branch_norm:
            self.branch_norm = nn.BatchNorm2d(in_channels)

        self.branch_3x3 = ProjectionConv(
            self.in_channels,
            self.out_channels,
            3,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
            momentum=0.1,
            eps=1e-5,
        )

        self.branch_1x1 = ProjectionConv(
            self.in_channels, self.out_channels, 1, groups=self.groups, momentum=0.1, eps=1e-5
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the RepVGG block.

        The output is the sum of 3x3 and 1x1 convolution outputs,
        along with the normalized identity branch output, followed by
        activation.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        if self.branch_norm is None:
            branch_norm_out = 0
        else:
            branch_norm_out = self.branch_norm(x)

        out = self.branch_3x3(x) + self.branch_1x1(x) + branch_norm_out

        out = self.act(out)

        return out


class CSPRepLayer(nn.Module):
    """CSPRepLayer, a layer that combines Cross Stage Partial Networks with
    RepVGG Blocks.

    Args:
        in_channels (int): Number of input channels to the layer.
        out_channels (int): Number of output channels from the layer.
        num_blocks (int): The number of RepVGG blocks to be used in the layer.
            Defaults to 3.
        widen_factor (float): Expansion factor for intermediate channels.
            Determines the hidden channel size based on out_channels.
            Defaults to 1.0.
        norm_cfg (dict): Configuration for normalization layers.
            Defaults to Batch Normalization with trainable parameters.
        act_cfg (dict): Configuration for activation layers.
            Defaults to SiLU (Swish) with in-place operation.
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1, widen_factor: float = 1.0):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * widen_factor)
        self.conv1 = ConvModule(in_channels, hidden_channels, kernel_size=1, momentum=0.1, eps=1e-5)
        self.conv2 = ConvModule(in_channels, hidden_channels, kernel_size=1, momentum=0.1, eps=1e-5)

        self.bottlenecks = nn.Sequential(*[RepVGGBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = ConvModule(hidden_channels, out_channels, kernel_size=1, momentum=0.1, eps=1e-5)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class HybridEncoder(nn.Module):
    def __init__(
        self,
        shape_specs: Dict[str, ShapeSpec],
        transformer_embed_dims: int = 256,
        transformer_num_heads: int = 8,
        transformer_feedforward_channels: int = 1024,
        transformer_dropout: float = 0.0,
        transformer_encoder_layers: int = 1,
        csp_layers: int = 1,
        hidden_dim: int = 256,
        use_encoder_idx: List[int] = [2],
        pe_temperature: int = 10000,
        widen_factor: float = 1.0,
        spe_learnable: bool = False,
        output_indices: Optional[List[int]] = [1, 2],  # 1/16 and 1/32
    ):
        super(HybridEncoder, self).__init__()
        self.input_channels = ["res3", "res4", "res5"]
        self.in_channels = [shape_specs[inc].channels for inc in self.input_channels]
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = transformer_encoder_layers
        self.pe_temperature = pe_temperature
        self.output_indices = output_indices

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:
            self.input_proj.append(
                ProjectionConv(in_channel, hidden_dim, kernel_size=1, padding=0, momentum=0.1, eps=1e-5)
            )

        # encoder transformer
        if len(use_encoder_idx) > 0:
            pos_enc_dim = self.hidden_dim // 2
            self.encoder = nn.ModuleList(
                [
                    DetrTransformerEncoder(
                        num_layers=transformer_encoder_layers,
                        embed_dims=transformer_embed_dims,
                        num_heads=transformer_num_heads,
                        feedforward_channels=transformer_feedforward_channels,
                        ffn_drop=transformer_dropout,
                    )
                    for _ in range(len(use_encoder_idx))
                ]
            )

        self.sincos_pos_enc = SinePositionalEncoding(
            pos_enc_dim, learnable=spe_learnable, temperature=self.pe_temperature, spatial_dim=2
        )

        # top-down fpn
        lateral_convs = list()
        fpn_blocks = list()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            lateral_convs.append(ConvModule(hidden_dim, hidden_dim, 1, 1, momentum=0.1, eps=1e-5))
            fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=csp_layers, widen_factor=widen_factor))
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.fpn_blocks = nn.ModuleList(fpn_blocks)

        # bottom-up pan
        downsample_convs = list()
        pan_blocks = list()
        for idx in range(len(self.in_channels) - 1):
            downsample_convs.append(ConvModule(hidden_dim, hidden_dim, 3, stride=2, padding=1, momentum=0.1, eps=1e-5))
            pan_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_blocks=csp_layers, widen_factor=widen_factor))
        self.downsample_convs = nn.ModuleList(downsample_convs)
        self.pan_blocks = nn.ModuleList(pan_blocks)

        self.projector = ChannelMapper([hidden_dim, hidden_dim], hidden_dim, kernel_size=1, num_outs=2)

    def forward(self, inputs: dict[str, Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        feats = [inputs[i] for i in self.input_channels]

        proj_feats = [self.input_proj[i](feats[i]) for i in range(len(self.input_channels))]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1).contiguous()

                if torch.onnx.is_in_onnx_export():
                    pos_enc = getattr(self, f"pos_enc_{i}")
                else:
                    pos_enc = self.sincos_pos_enc(size=(h, w))
                    pos_enc = pos_enc.transpose(-1, -2).reshape(1, h * w, -1)
                memory = self.encoder[i](src_flatten, query_pos=pos_enc, key_padding_mask=None)

                proj_feats[enc_ind] = memory.permute(0, 2, 1).contiguous().view([-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(feat_high, size=feat_low.shape[2:], mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], axis=1))  # type: ignore
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)  # Conv
            out = self.pan_blocks[idx](  # CSPRepLayer
                torch.cat([downsample_feat, feat_high], axis=1)
            )  # type: ignore
            outs.append(out)

        if self.output_indices is not None:
            outs = [outs[i] for i in self.output_indices]

        if self.projector is not None:
            outs = self.projector(outs)

        return tuple(outs)
