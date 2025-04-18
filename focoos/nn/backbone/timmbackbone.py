import timm
import torch.nn as nn
from timm.models.layers import convert_sync_batchnorm

from focoos.utils.distributed import comm

from .base import Backbone


class TimmBackbone(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
    ):
        super().__init__()

        assert model_name in timm.list_models(), (
            f"{model_name} is not included in timm."
            f"Please use a model included in timm. "
            "Use timm.list_models() for the complete list."
        )

        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, exportable=True)
        if comm.get_world_size() > 1:
            self.model = convert_sync_batchnorm(self.model)

        self.feature_stride = self.model.feature_info.reduction()
        self.feature_channels = self.model.feature_info.channels()

    def forward(self, x):
        o = self.model(x)
        out = {"res2": o[-4], "res3": o[-3], "res4": o[-2], "res5": o[-1]}

        return out


class D2timm(TimmBackbone, Backbone):
    def __init__(self, name, pretrained, out_features):
        model_name = name
        pretrained = pretrained

        super().__init__(
            model_name,
            pretrained,
        )

        self._out_features = out_features

        self._out_feature_strides = {
            "res2": self.feature_stride[-4],
            "res3": self.feature_stride[-3],
            "res4": self.feature_stride[-2],
            "res5": self.feature_stride[-1],
        }
        self._out_feature_channels = {
            "res2": self.feature_channels[-4],
            "res3": self.feature_channels[-3],
            "res4": self.feature_channels[-2],
            "res5": self.feature_channels[-1],
        }
