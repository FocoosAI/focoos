from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.nn.layers.block import C2f
from focoos.nn.layers.yolo_conv import Conv


@dataclass
class DarkNetConfig(BackboneConfig):
    """Configuration for DarkNet backbone.

    Args:
        depth: List of depth values for each stage
        width: List of width values for each stage
        pretrained_weights_path: Path to pretrained weights file
        model_type: Type of model
    """

    depth: Tuple[int, ...] = (1, 2, 2, 1)
    width: Tuple[int, ...] = (3, 32, 64, 128, 256, 512)
    pretrained_weights_path: str = ""
    model_type: str = "darknet"


class DarkNet(BaseBackbone):
    """DarkNet backbone architecture used in YOLOv8.

    This backbone consists of multiple stages with increasing feature dimensions
    and decreasing spatial resolution.
    """

    def __init__(
        self,
        config: DarkNetConfig,
    ):
        super().__init__(config)

        # Extract configuration parameters
        depth = config.depth
        width = config.width
        pretrained_weights_path = config.pretrained_weights_path

        # Define network stages
        self.stem = self._create_stem(width[0], width[1])
        self.stage1 = self._create_stage(width[1], width[2], depth[0])
        self.stage2 = self._create_stage(width[2], width[3], depth[1])
        self.stage3 = self._create_stage(width[3], width[4], depth[2])
        self.stage4 = self._create_stage(width[4], width[5], depth[3])

        # Load pretrained weights if provided
        if pretrained_weights_path:
            self._load_pretrained_weights(pretrained_weights_path)

        # Define feature names, strides, and channels
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {
            "res2": width[2],
            "res3": width[3],
            "res4": width[4],
            "res5": width[5],
        }

    def _create_stem(self, in_channels, out_channels):
        """Create the stem of the network."""
        return Conv(in_channels, out_channels, 3, 2, 1)

    def _create_stage(self, in_channels, out_channels, depth):
        """Create a stage of the network with downsampling and C2f blocks."""
        return nn.Sequential(
            Conv(in_channels, out_channels, 3, 2, 1),
            C2f(out_channels, out_channels, shortcut=True, n=depth),
        )

    def _load_pretrained_weights(self, weights_path):
        """Load pretrained weights from file."""
        if weights_path.endswith(".pt"):
            state_dict = torch.load(weights_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {weights_path}")

    def forward_features(self, x):
        """Forward pass through the backbone, returning features from each stage."""
        x = self.stem(x)
        stage1_output = self.stage1(x)
        stage2_output = self.stage2(stage1_output)
        stage3_output = self.stage3(stage2_output)
        stage4_output = self.stage4(stage3_output)
        return stage1_output, stage2_output, stage3_output, stage4_output

    def forward(self, x):
        """Forward pass returning a dictionary of features."""
        stage1_output, stage2_output, stage3_output, stage4_output = self.forward_features(x)
        return {
            "res2": stage1_output,
            "res3": stage2_output,
            "res4": stage3_output,
            "res5": stage4_output,
        }


# Legacy implementation for backward compatibility
class LegacyDarkNet(BaseBackbone):
    """Legacy implementation of DarkNet for backward compatibility."""

    def __init__(
        self,
        depth=[1, 2, 2, 1],
        width=[3, 32, 64, 128, 256, 512],
        pretrained_weights_path="",
    ):
        config = DarkNetConfig(depth=depth, width=width, pretrained_weights_path=pretrained_weights_path)
        super().__init__(config)

        # Define network stages using the old structure
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

        # Load pretrained weights if provided
        if pretrained_weights_path:
            if pretrained_weights_path.endswith(".pt"):
                state_dict = torch.load(pretrained_weights_path, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {pretrained_weights_path}")

    def forward(self, x):
        """Forward pass through the network stages."""
        outputs = {}
        x = self.p1(x)
        x = self.p2(x)
        outputs["res2"] = x
        x = self.p3(x)
        outputs["res3"] = x
        x = self.p4(x)
        outputs["res4"] = x
        x = self.p5(x)
        outputs["res5"] = x
        return outputs

    @staticmethod
    def convert_legacy_checkpoint(legacy_checkpoint_path, output_path=None):
        """
        Convert a checkpoint from LegacyDarkNet format to DarkNet format.

        Args:
            legacy_checkpoint_path (str): Path to the legacy checkpoint file
            output_path (str, optional): Path to save the converted checkpoint.
                If None, will use the legacy path with '_converted' suffix.

        Returns:
            str: Path to the converted checkpoint file
        """
        import os

        # Load legacy checkpoint
        legacy_state_dict = torch.load(legacy_checkpoint_path, map_location="cpu")
        if "model" in legacy_state_dict:
            legacy_state_dict = legacy_state_dict["model"]

        # Create mapping from legacy keys to new keys
        new_state_dict = {}

        # Mapping dictionary for layer name conversion
        mapping = {
            "backbone.stem": "p1",
            "backbone.dark2": "p2",
            "backbone.dark3": "p3",
            "backbone.dark4": "p4",
            "backbone.dark5": "p5",
        }

        for key, value in legacy_state_dict.items():
            # Skip non-backbone parameters
            if not any(key.startswith(prefix) for prefix in mapping.keys()):
                continue

            # Replace the prefix with the new one
            for old_prefix, new_prefix in mapping.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix)
                    new_state_dict[new_key] = value
                    break

        # Set default output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(legacy_checkpoint_path)
            output_path = f"{base}_converted{ext}"

        # Save the converted checkpoint
        torch.save(new_state_dict, output_path)
        print(f"Converted checkpoint saved to {output_path}")

        return output_path


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
    back = LegacyDarkNet(*versions.get(v), pretrained_weights_path=f"yolov8{v}.pt")
    model_out = back.forward(input_tensor)
    print([(k, o.shape) for k, o in model_out.items()])
    back.convert_legacy_checkpoint(f"yolov8{v}.pt", f"yolov8{v}_converted.pt")
