from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from focoos.nn.layers.misc import DropPath
from focoos.nn.layers.norm import LayerNorm
from focoos.utils.logger import get_logger

from .base import BackboneConfig, BaseBackbone, ShapeSpec

logger = get_logger("Backbone")


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


CONFIGS = {
    "atto": {
        "depths": [2, 2, 6, 2],
        "embed_dims": [40, 80, 160, 320],
        "url": "https://public.focoos.ai/pretrained_models/backbones/convnextv2_atto.pth",
    },
    "femto": {
        "depths": [2, 2, 6, 2],
        "embed_dims": [48, 96, 192, 384],
        "url": "https://public.focoos.ai/pretrained_models/backbones/convnextv2_femto.pth",
    },
    "pico": {
        "depths": [2, 2, 6, 2],
        "embed_dims": [64, 128, 256, 512],
        "url": "https://public.focoos.ai/pretrained_models/backbones/convnextv2_pico.pth",
    },
    "nano": {
        "depths": [2, 2, 8, 2],
        "embed_dims": [80, 160, 320, 640],
        "url": "https://public.focoos.ai/pretrained_models/backbones/convnextv2_nano.pth",
    },
    "tiny": {
        "depths": [3, 3, 9, 3],
        "embed_dims": [96, 192, 384, 768],
        "url": "https://public.focoos.ai/pretrained_models/backbones/convnextv2_tiny.pth",
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "embed_dims": [128, 256, 512, 1024],
        "url": "https://public.focoos.ai/pretrained_models/backbones/convnextv2_base.pth",
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "embed_dims": [192, 384, 768, 1536],
        "url": "https://public.focoos.ai/pretrained_models/backbones/convnextv2_large.pth",
    },
}


@dataclass
class ConvNeXtV2Config(BackboneConfig):
    """ConvNeXt V2 configuration"""

    model_type: str = "convnextv2"
    model_size: Optional[str] = "atto"
    drop_path_rate: float = 0.0
    depths: Optional[Tuple[int, ...]] = None
    embed_dims: Optional[Tuple[int, ...]] = None


class ConvNeXtV2(BaseBackbone):
    """ConvNeXt V2

    Args:
        config: Configuration object containing model parameters
    """

    def __init__(self, config: ConvNeXtV2Config):
        super().__init__(config)
        in_chans = 3

        if config.model_size:
            depths = CONFIGS[config.model_size]["depths"]
            dims = CONFIGS[config.model_size]["embed_dims"]
            backbone_url = config.backbone_url or CONFIGS[config.model_size]["url"]
        else:
            backbone_url = config.backbone_url
            depths = config.depths
            dims = config.embed_dims
            assert depths is not None and dims is not None, (
                "depths and embed_dims must be provided if model_size is not provided"
            )
        drop_path_rate = config.drop_path_rate

        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        if config.use_pretrained and backbone_url:
            state = torch.hub.load_state_dict_from_url(backbone_url)
            self.load_state_dict(state)
            logger.info(f"Load ConvNeXtV2{config.model_size} state_dict")

        self.out_features = ["res2", "res3", "res4", "res5"]

        self.out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self.out_feature_channels = {
            "res2": dims[0],
            "res3": dims[1],
            "res4": dims[2],
            "res5": dims[3],
        }

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs["res{}".format(i + 2)] = x
        return outs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.out_feature_channels[name],
                stride=self.out_feature_strides[name],
            )
            for name in self.out_features
        }
