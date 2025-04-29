from dataclasses import dataclass

from timm.layers import convert_sync_batchnorm

from focoos.utils.distributed import comm

from .base import BackboneConfig, BaseBackbone


@dataclass
class TimmBackboneConfig(BackboneConfig):
    model_name: str = ""
    pretrained: bool = False
    model_type: str = "timm"


class TimmBackbone(BaseBackbone):
    def __init__(
        self,
        config: TimmBackboneConfig,
    ):
        super().__init__(config)

        try:
            import timm
        except ImportError:
            raise ImportError("timm is not installed. Please install it with `pip install timm`.")

        assert config.model_name in timm.list_models(), (
            f"{config.model_name} is not included in timm."
            f"Please use a model included in timm. "
            "Use timm.list_models() for the complete list."
        )

        self.model = timm.create_model(
            config.model_name, pretrained=config.pretrained, features_only=True, exportable=True
        )
        if comm.get_world_size() > 1:
            self.model = convert_sync_batchnorm(self.model)

        self.feature_stride = self.model.feature_info.reduction()
        self.feature_channels = self.model.feature_info.channels()

        self._out_features = ["res2", "res3", "res4", "res5"]

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

    def forward(self, x):
        o = self.model(x)
        out = {"res2": o[-4], "res3": o[-3], "res4": o[-2], "res5": o[-1]}

        return out
