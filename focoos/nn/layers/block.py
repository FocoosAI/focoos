import torch
import torch.nn as nn

from focoos.nn.layers.conv import ConvNormLayerDarknet


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) block. from yolov5"""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        hidden_c = c1 // 2
        self.cv1 = ConvNormLayerDarknet(ch_in=c1, ch_out=hidden_c, kernel_size=1, stride=1)
        self.cv2 = ConvNormLayerDarknet(ch_in=hidden_c * 4, ch_out=c2, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class C2f(nn.Module):
    """Alternate C2f block with chunk/split options."""

    def __init__(self, ch_in: int, ch_out: int, n: int = 1, shortcut: bool = False, groups: int = 1, e: float = 0.5):
        super().__init__()
        self.hidden_channels = int(ch_out * e)
        self.cv1 = ConvNormLayerDarknet(ch_in=ch_in, ch_out=2 * self.hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvNormLayerDarknet(ch_in=(2 + n) * self.hidden_channels, ch_out=ch_out, kernel_size=1, stride=1)
        self.m = nn.ModuleList(
            Bottleneck(self.hidden_channels, self.hidden_channels, shortcut, groups, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward with chunk."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        shortcut: bool = True,
        groups: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(channels_out * e)
        self.cv1 = ConvNormLayerDarknet(ch_in=channels_in, ch_out=hidden_channels, kernel_size=k[0], stride=1)
        self.cv2 = ConvNormLayerDarknet(ch_in=hidden_channels, ch_out=channels_out, kernel_size=k[1], stride=1)
        self.add = shortcut and channels_in == channels_out

    def forward(self, x):
        """Apply bottleneck with optional residual."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
