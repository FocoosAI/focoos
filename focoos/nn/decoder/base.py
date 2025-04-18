# Copyright (c) Facebook, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod
from typing import Dict

import torch.nn as nn
from torch.nn import functional as F

from focoos.nn.backbone.base import Backbone
from focoos.nn.layers.conv import Conv2d

__all__ = ["PixelDecoder", "BasePixelDecoder"]


class PixelDecoder(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self, backbone: Backbone, out_dim: int, feat_dim: int):
        """
        backbone: basic backbones to extract features from images
        feat_dim: number of output channels for the intermediate conv layers.
        out_dim: number of output channels for the final conv layer.
        norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        self.backbone = backbone
        self.input_shape = sorted(backbone.output_shape().items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in self.input_shape]
        # starting from "res2" to "res5"
        self.in_channels = [v.channels for k, v in self.input_shape]
        self.in_strides = [v.stride for k, v in self.input_shape]
        self.out_dim = out_dim
        self.feat_dim = feat_dim

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return self.backbone.size_divisibility

    @property
    def padding_constraints(self) -> Dict[str, int]:
        """
        This property is a generalization of size_divisibility. Some backbones and training
        recipes require specific padding constraints, such as enforcing divisibility by a specific
        integer (e.g., FPN) or padding to a square (e.g., ViTDet with large-scale jitter
        in :paper:vitdet). `padding_constraints` contains these optional items like:
        {
            "size_divisibility": int,
            "square_size": int,
        }
        `size_divisibility` will read from here if presented and `square_size` indicates the
        square padding size if `square_size` > 0.
        """
        return self.backbone.padding_constraints

    @abstractmethod
    def forward_features(self, features):
        """
        This should return two values:
            - Output features: Tensor [BxHxWxOut_Dim]
            - Multiscale features: List [ Tensor [BxH_ixW_ixFeat_Dim] ] where H_i, W_i may be different.
                If different scales, please provide high resolution last (e.g. Res5, Res4, Res3)
        """
        pass

    def forward(self, images):
        features = self.backbone(images)
        return self.forward_features(features)


class BasePixelDecoder(PixelDecoder):
    def __init__(self, backbone: Backbone, out_dim: int, feat_dim: int):
        """
        backbone: basic backbones to extract features from images
        feat_dim: number of output channels for the intermediate conv layers.
        out_dim: number of output channels for the final conv layer.
        norm (str or callable): normalization for all conv layers
        """
        super().__init__(backbone, out_dim, feat_dim)
        self.in_channels = [v.channels for k, v in self.input_shape]

        self.proj4 = Conv2d(self.in_channels[3], feat_dim, kernel_size=1, bias=False, activation=F.relu)
        self.conv_out = Conv2d(feat_dim, out_dim, kernel_size=1, bias=False)

    def forward_features(self, features):
        """
        This should return two values:
            - Output features: Tensor [BxHxWxOut_Dim]
            - Multiscale features: List [ Tensor [BxH_ixW_ixFeat_Dim] ] where H_i, W_i may be different.
        """
        [c1, c2, c3, c4] = [features[f] for f in self.in_features]
        feat = self.proj4(c4)
        out = F.interpolate(
            self.conv_out(feat),
            size=c1.size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        # interpolate then map to two dimensions. Return multi-scale-features where usually is 8, 16, 32 strides
        return out, [feat, feat, feat]
