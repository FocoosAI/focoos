import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from torch.nn import init

from focoos.utils.logger import get_logger

from .base import BackboneConfig, BaseBackbone

logger = get_logger("Backbone")


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    bias=False,
                ),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                    )
                )
            else:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                    )
                )

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                    )
                )
            else:
                self.conv_list.append(
                    ConvX(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                    )
                )

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):  # type: ignore
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


@dataclass
class STDCConfig(BackboneConfig):
    in_chans: int = 3
    base: int = 64  # from json: "base": 64
    layers: List[int] = field(default_factory=lambda: [4, 5, 3])  # from json: "layers": [4, 5, 3]
    out_features: List[str] = field(default_factory=lambda: ["res2", "res3", "res4", "res5"])  # from json
    model_type: str = "stdc"
    block_num: int = 4
    block_type: str = "cat"
    backbone_url: Optional[str] = None
    size: Optional[Literal["nano", "small", "large"]] = None
    use_conv_last: bool = False


class STDC(BaseBackbone):
    def __init__(self, config: STDCConfig):
        super().__init__(config)
        if config.size is not None:
            if config.size == "small":
                config.backbone_url = "https://public.focoos.ai/pretrained_models/backbones/stdc_small.pth"
                layers = [2, 2, 2]
                base = 64
                block_num = 4
                block_type = "cat"
            elif config.size == "large":
                config.backbone_url = "https://public.focoos.ai/pretrained_models/backbones/stdc_large.pth"
                layers = [4, 5, 3]
                base = 64
                block_num = 4
                block_type = "cat"
            elif config.size == "nano":
                config.backbone_url = "https://public.focoos.ai/pretrained_models/backbones/stdc_nano.pth"
                layers = [2, 2, 2]
                base = 32
                block_num = 4
                block_type = "cat"
            else:
                raise ValueError(f"Invalid size: {config.size}. The size should be small, large or nano.")
            if config.layers and layers != config.layers:
                logger.warning(f"Layers must be {layers} if size is {config.size}, provided {config.layers} not used.")
            if config.base and base != config.base:
                logger.warning(f"Base must be {base} if size is {config.size}, provided {config.base} not used.")
            if config.block_num and block_num != config.block_num:
                logger.warning(
                    f"Block num must be {block_num} if size is {config.size}, provided {config.block_num} not used."
                )
            if config.block_type and block_type != config.block_type:
                logger.warning(
                    f"Block type must be {block_type} if size is {config.size}, provided {config.block_type} not used."
                )
        else:
            if not config.layers or not config.base or not config.block_num or not config.block_type:
                raise ValueError("Layers, base, block_num and block_type must be provided if size is not provided.")
            layers = config.layers
            base = config.base
            block_num = config.block_num
            block_type = config.block_type

        if block_type == "cat":
            block = CatBottleneck
        elif block_type == "add":
            block = AddBottleneck

        if config.layers != [2, 2, 2] and config.layers != [4, 5, 3]:
            raise ValueError(f"Invalid layers: {config.layers}. The layers should be [2, 2, 2] or [4, 5, 3].")

        self.in_chans = config.in_chans
        self.features = self._make_layers(base, layers, block_num, block)

        if layers == [2, 2, 2]:
            self.out_ids = 1, 3, 5, 7

        elif layers == [4, 5, 3]:
            self.out_ids = 1, 5, 10, 13

        if config.use_pretrained and config.backbone_url:
            state = torch.hub.load_state_dict_from_url(config.backbone_url)
            self.load_state_dict(state)
            logger.info("Load STDC state_dict")

        self.out_features = config.out_features

        self.out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self.out_feature_channels = {
            "res2": base,
            "res3": base * 4,
            "res4": base * 8,
            "res5": base * 16,
        }

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(self.in_chans, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 1)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            2,
                        )
                    )
                else:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 2)),
                            base * int(math.pow(2, i + 2)),
                            block_num,
                            1,
                        )
                    )

        return nn.Sequential(*features)

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_ids:
                outs.append(x)
        outs = {f"res{i + 2}": outs[i] for i in range(len(outs))}
        return outs
