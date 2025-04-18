import torch

from focoos.nn.backbone.base import Backbone
from focoos.nn.layers.block import C2f
from focoos.nn.layers.yolo_conv import Conv

# [depth, widths, max_channels]
# versions = {
#     "n": [
#         [3, 6, 6, 3],
#         0.25,
#         1024,
#     ],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#     "s": [
#         1,
#         0.50,
#         1024,
#     ],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#     "m": [
#         2,
#         0.75,
#         768,
#     ],  # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#     "l": [
#         3,
#         1.00,
#         512,
#     ],  # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#     "x": [
#         3,
#         1.25,
#         512,
#     ],  # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
# }


class DarkNet(Backbone):
    def __init__(self, depth, width):
        super().__init__()

        p1 = [Conv(width[0], width[1], 3, 2, 1)]
        p2 = [
            Conv(width[1], width[2], 3, 2, 1),
            C2f(width[2], width[2], shortcut=True, n=depth[0]),
        ]
        p3 = [
            Conv(width[2], width[3], 3, 2, 1),
            C2f(width[3], width[3], shortcut=True, n=depth[1]),
        ]
        p4 = [
            Conv(width[3], width[4], 3, 2, 1),
            C2f(width[4], width[4], shortcut=True, n=depth[2]),
        ]
        p5 = [
            Conv(width[4], width[5], 3, 2, 1),
            C2f(width[5], width[5], shortcut=True, n=depth[3]),
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
    input_tensor = torch.ones(1, 3, 640, 640).float()
    versions = {
        "n": [[1, 2, 2, 1], [3, 16, 32, 64, 128, 256]],
        "s": [[1, 2, 2, 1], [3, 32, 64, 128, 256, 512]],
        "m": [[2, 4, 4, 2], [3, 48, 96, 192, 384, 576]],
        "l": [[3, 6, 6, 3], [3, 64, 128, 256, 512, 512]],
        "x": [[3, 6, 6, 3], [3, 80, 160, 320, 640, 640]],
    }
    v = "x"
    back = DarkNet(*versions.get(v), f"yolov8{v}.pt")
    model_out = back.forward(input_tensor)
    print([(k, o.shape) for k, o in model_out.items()])
