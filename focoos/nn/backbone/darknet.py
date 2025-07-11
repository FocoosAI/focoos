from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.nn.layers.block import C2f
from focoos.nn.layers.conv import ConvNormLayer

DARKNET_SIZES = {
    "n": {"depth": [1, 2, 2, 1], "width": [3, 16, 32, 64, 128, 256]},
    "s": {"depth": [1, 2, 2, 1], "width": [3, 32, 64, 128, 256, 512]},
    "m": {"depth": [2, 4, 4, 2], "width": [3, 48, 96, 192, 384, 576]},
    "l": {"depth": [3, 6, 6, 3], "width": [3, 64, 128, 256, 512, 512]},
    "x": {"depth": [3, 6, 6, 3], "width": [3, 80, 160, 320, 640, 640]},
}

ACTIVATION_MAPPING = {
    "silu": nn.SiLU(inplace=True),
    "relu": nn.ReLU(inplace=True),
    "leaky_relu": nn.LeakyReLU(inplace=True),
    "gelu": nn.GELU(),
}


@dataclass
class DarkNetConfig(BackboneConfig):
    size: Literal["n", "s", "m", "l", "x"] = "m"
    model_type: Literal["darknet"] = "darknet"
    activation: Literal["silu", "relu", "leaky_relu", "gelu"] = "silu"


class DarkNet(BaseBackbone):
    def __init__(self, config: DarkNetConfig):
        super().__init__(config)

        self.width = DARKNET_SIZES[config.size]["width"]
        self.depth = DARKNET_SIZES[config.size]["depth"]
        self.activation = ACTIVATION_MAPPING[config.activation]

        p1 = [
            ConvNormLayer(
                ch_in=self.width[0], ch_out=self.width[1], kernel_size=3, stride=2, padding=1, act=self.activation
            )
        ]
        p2 = [
            ConvNormLayer(
                ch_in=self.width[1], ch_out=self.width[2], kernel_size=3, stride=2, padding=1, act=self.activation
            ),
            C2f(ch_in=self.width[2], ch_out=self.width[2], shortcut=True, n=self.depth[0], activation=self.activation),
        ]
        p3 = [
            ConvNormLayer(
                ch_in=self.width[2], ch_out=self.width[3], kernel_size=3, stride=2, padding=1, act=self.activation
            ),
            C2f(ch_in=self.width[3], ch_out=self.width[3], shortcut=True, n=self.depth[1], activation=self.activation),
        ]
        p4 = [
            ConvNormLayer(
                ch_in=self.width[3], ch_out=self.width[4], kernel_size=3, stride=2, padding=1, act=self.activation
            ),
            C2f(ch_in=self.width[4], ch_out=self.width[4], shortcut=True, n=self.depth[2], activation=self.activation),
        ]
        p5 = [
            ConvNormLayer(
                ch_in=self.width[4], ch_out=self.width[5], kernel_size=3, stride=2, padding=1, act=self.activation
            ),
            C2f(ch_in=self.width[5], ch_out=self.width[5], shortcut=True, n=self.depth[3], activation=self.activation),
        ]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)
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
    for activation in ["silu", "relu", "leaky_relu", "gelu"]:
        for size in ["n", "s", "m", "l", "x"]:
            input_tensor = torch.ones(1, 3, 224, 224).float()
            back = DarkNet(DarkNetConfig(size=size, activation=activation))
            model_out = back.forward(input_tensor)
            print([(k, o.shape) for k, o in model_out.items()])
