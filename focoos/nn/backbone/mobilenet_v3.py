from torchvision.models.mobilenetv3 import (
    MobileNetV3,
    _mobilenet_v3_conf,
)

from .base import Backbone


class D2MobileNetV3(MobileNetV3, Backbone):
    def __init__(
        self,
        size="small",
        width_mult=1.0,
        dilated=False,
        out_features=("res2", "res3", "res4", "res5"),
    ):
        assert size in [
            "small",
            "large",
        ], f"MobileNetv3 can only be small or large. Size is {size}"

        inverted_residual_setting, last_channel = _mobilenet_v3_conf(
            "mobilenet_v3_" + size, width_mult=width_mult, dilated=dilated
        )

        super().__init__(inverted_residual_setting, last_channel, 1)

        self.layer2res = {}
        self._out_feature_channels = {}
        if size == "small":
            self.layer2res[2] = "res2"
            self._out_feature_channels["res2"] = self.features[2].out_channels
            self.layer2res[4] = "res3"
            self._out_feature_channels["res3"] = self.features[4].out_channels
            self.layer2res[9] = "res4"
            self._out_feature_channels["res4"] = self.features[9].out_channels
            self.layer2res[12] = "res5"
            self._out_feature_channels["res5"] = self.features[12].out_channels
        else:
            self.layer2res[4] = "res2"
            self._out_feature_channels["res2"] = self.features[4].out_channels
            self.layer2res[7] = "res3"
            self._out_feature_channels["res3"] = self.features[7].out_channels
            self.layer2res[13] = "res4"
            self._out_feature_channels["res4"] = self.features[13].out_channels
            self.layer2res[16] = "res5"
            self._out_feature_channels["res5"] = self.features[16].out_channels

        if dilated:
            self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 8, "res5": 8}
        else:
            self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_features = out_features

    def forward(self, x):
        outs = {}
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.layer2res:
                outs[self.layer2res[idx]] = x

        return outs
