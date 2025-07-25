import torch
import torch.nn as nn

from focoos.nn.layers.conv import ConvNormLayerDarknet


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvNormLayerDarknet(ch_in=c1, ch_out=c_, kernel_size=1, stride=1)
        self.cv2 = ConvNormLayerDarknet(ch_in=c_ * 4, ch_out=c2, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, ch_in, ch_out, n=1, shortcut=False, groups=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(ch_out * e)  # hidden channels
        self.cv1 = ConvNormLayerDarknet(ch_in=ch_in, ch_out=2 * self.c, kernel_size=1, stride=1)
        self.cv2 = ConvNormLayerDarknet(
            ch_in=(2 + n) * self.c, ch_out=ch_out, kernel_size=1, stride=1
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


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, channels_in, channels_out, shortcut=True, groups=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(channels_out * e)  # hidden channels
        self.cv1 = ConvNormLayerDarknet(ch_in=channels_in, ch_out=c_, kernel_size=k[0], stride=1)
        self.cv2 = ConvNormLayerDarknet(ch_in=c_, ch_out=channels_out, kernel_size=k[1], stride=1)
        self.add = shortcut and channels_in == channels_out

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
