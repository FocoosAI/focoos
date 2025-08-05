from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.nn.layers.block import C2f
from focoos.nn.layers.conv import ConvNormLayerDarknet
from focoos.utils.logger import get_logger

logger = get_logger(__name__)

DARKNET_SIZES = {
    "n": {"depth": [1, 2, 2, 1], "width": [3, 16, 32, 64, 128, 256]},
    "s": {"depth": [1, 2, 2, 1], "width": [3, 32, 64, 128, 256, 512]},
    "m": {"depth": [2, 4, 4, 2], "width": [3, 48, 96, 192, 384, 576]},
    "l": {"depth": [3, 6, 6, 3], "width": [3, 64, 128, 256, 512, 512]},
    "x": {"depth": [3, 6, 6, 3], "width": [3, 80, 160, 320, 640, 640]},
}

DARKNET_PRETRAINED_WEIGHTS = {
    "n": "https://public.focoos.ai/pretrained_models/backbones/darknet_n.pth",
    "s": "https://public.focoos.ai/pretrained_models/backbones/darknet_s.pth",
    "m": "https://public.focoos.ai/pretrained_models/backbones/darknet_m.pth",
    "l": "https://public.focoos.ai/pretrained_models/backbones/darknet_l.pth",
    "x": "https://public.focoos.ai/pretrained_models/backbones/darknet_x.pth",
}


@dataclass
class C2fDarkNetConfig(BackboneConfig):
    size: Literal["n", "s", "m", "l", "x"] = "m"
    model_type: Literal["c2f_darknet"] = "c2f_darknet"


class C2fDarkNet(BaseBackbone):
    def __init__(self, config: C2fDarkNetConfig):
        super().__init__(config)

        self.width = DARKNET_SIZES[config.size]["width"]
        self.depth = DARKNET_SIZES[config.size]["depth"]
        self.activation = nn.SiLU(inplace=True)

        self.p1 = torch.nn.Sequential(
            ConvNormLayerDarknet(ch_in=self.width[0], ch_out=self.width[1], kernel_size=3, stride=2, padding=1)
        )
        self.p2 = torch.nn.Sequential(
            ConvNormLayerDarknet(ch_in=self.width[1], ch_out=self.width[2], kernel_size=3, stride=2, padding=1),
            C2f(ch_in=self.width[2], ch_out=self.width[2], shortcut=True, n=self.depth[0]),
        )
        self.p3 = torch.nn.Sequential(
            ConvNormLayerDarknet(ch_in=self.width[2], ch_out=self.width[3], kernel_size=3, stride=2, padding=1),
            C2f(ch_in=self.width[3], ch_out=self.width[3], shortcut=True, n=self.depth[1]),
        )
        self.p4 = torch.nn.Sequential(
            ConvNormLayerDarknet(ch_in=self.width[3], ch_out=self.width[4], kernel_size=3, stride=2, padding=1),
            C2f(ch_in=self.width[4], ch_out=self.width[4], shortcut=True, n=self.depth[2]),
        )
        self.p5 = torch.nn.Sequential(
            ConvNormLayerDarknet(ch_in=self.width[4], ch_out=self.width[5], kernel_size=3, stride=2, padding=1),
            C2f(ch_in=self.width[5], ch_out=self.width[5], shortcut=True, n=self.depth[3]),
        )

        if config.use_pretrained:
            if config.backbone_url:
                state = torch.hub.load_state_dict_from_url(config.backbone_url)
                self.load_state_dict(state)
                logger.info(f"Loaded pretrained weights from {config.backbone_url}")
            else:
                state = torch.hub.load_state_dict_from_url(DARKNET_PRETRAINED_WEIGHTS[config.size])
                # state = torch.load(DARKNET_PRETRAINED_WEIGHTS[config.size])
                self.load_state_dict(state)
                logger.info(f"Loaded pretrained weights from {DARKNET_PRETRAINED_WEIGHTS[config.size]}")

        # Define feature names, strides, and channels
        self.out_features = ["res2", "res3", "res4", "res5"]
        self.out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self.out_feature_channels = {
            "res2": self.width[2],
            "res3": self.width[3],
            "res4": self.width[4],
            "res5": self.width[5],
        }

    def forward_features(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p2, p3, p4, p5

    def forward(self, x):
        p2, p3, p4, p5 = self.forward_features(x)
        return {
            "res2": p2,
            "res3": p3,
            "res4": p4,
            "res5": p5,
        }


if __name__ == "__main__":
    for size in ["n", "s", "m", "l", "x"]:
        input_tensor = torch.ones(1, 3, 224, 224).float()
        back = C2fDarkNet(C2fDarkNetConfig(size=size, use_pretrained=True))
        model_out = back.forward(input_tensor)
        print([(k, o.shape) for k, o in model_out.items()])
