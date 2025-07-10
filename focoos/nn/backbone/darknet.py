from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
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


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, channels_in, channels_out, shortcut=True, groups=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(channels_out * e)  # hidden channels
        self.cv1 = ConvNormLayer(ch_in=channels_in, ch_out=c_, kernel_size=k[0], stride=1)
        self.cv2 = ConvNormLayer(ch_in=c_, ch_out=channels_out, kernel_size=k[1], stride=1)
        self.add = shortcut and channels_in == channels_out

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, ch_in, ch_out, n=1, shortcut=False, groups=1, e=0.5, activation=nn.SiLU(inplace=True)):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(ch_out * e)  # hidden channels
        self.cv1 = ConvNormLayer(ch_in=ch_in, ch_out=2 * self.c, kernel_size=1, stride=1, act=activation)
        self.cv2 = ConvNormLayer(
            ch_in=(2 + n) * self.c, ch_out=ch_out, kernel_size=1, stride=1, act=activation
        )  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, groups, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DarkNet(BaseBackbone):
    def __init__(self, config: DarkNetConfig):
        super().__init__(config)

        width = DARKNET_SIZES[config.size]["width"]
        depth = DARKNET_SIZES[config.size]["depth"]
        activation = ACTIVATION_MAPPING[config.activation]

        p1 = [ConvNormLayer(ch_in=width[0], ch_out=width[1], kernel_size=3, stride=2, padding=1, act=activation)]
        p2 = [
            ConvNormLayer(ch_in=width[1], ch_out=width[2], kernel_size=3, stride=2, padding=1, act=activation),
            C2f(ch_in=width[2], ch_out=width[2], shortcut=True, n=depth[0], activation=activation),
        ]
        p3 = [
            ConvNormLayer(ch_in=width[2], ch_out=width[3], kernel_size=3, stride=2, padding=1, act=activation),
            C2f(ch_in=width[3], ch_out=width[3], shortcut=True, n=depth[1], activation=activation),
        ]
        p4 = [
            ConvNormLayer(ch_in=width[3], ch_out=width[4], kernel_size=3, stride=2, padding=1, act=activation),
            C2f(ch_in=width[4], ch_out=width[4], shortcut=True, n=depth[2], activation=activation),
        ]
        p5 = [
            ConvNormLayer(ch_in=width[4], ch_out=width[5], kernel_size=3, stride=2, padding=1, act=activation),
            C2f(ch_in=width[5], ch_out=width[5], shortcut=True, n=depth[3], activation=activation),
        ]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)
        # Define feature names, strides, and channels
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": width[2],
            "res3": width[3],
            "res4": width[4],
            "res5": width[5],
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
