import pytest
import torch

from focoos.model_manager import ConfigBackboneManager
from focoos.nn.backbone.base import BackboneConfig, ShapeSpec
from focoos.nn.backbone.build import load_backbone

# List of all backbone types with their minimum required config
BACKBONE_CONFIGS = {
    "resnet": [{"model_type": "resnet", "use_pretrained": False, "depth": 18}],
    "stdc": [{"model_type": "stdc", "use_pretrained": False, "base": 64, "layers": [4, 5, 3]}],
    "swin": [{"model_type": "swin", "use_pretrained": False}],
    "mobilenet_v2": [{"model_type": "mobilenet_v2", "use_pretrained": False}],
    "convnextv2": [{"model_type": "convnextv2", "use_pretrained": False}],
    "csp_darknet": [
        {"model_type": "csp_darknet", "use_pretrained": False, "size": size} for size in ["small", "medium", "large"]
    ],
}

# Different input sizes to test
INPUT_SIZES = [
    (1, 3, 224, 224),  # Standard size
    (1, 3, 384, 384),  # Larger size
    (2, 3, 224, 224),  # Batch size 2
]


def test_build_function():
    """Test the backbone build function."""
    for backbone_type, config_dicts in BACKBONE_CONFIGS.items():
        for config_dict in config_dicts:
            config = ConfigBackboneManager.from_dict(config_dict)

            # Test that the backbone can be built
            backbone = load_backbone(config)
            assert backbone is not None, f"Failed to build backbone of type {backbone_type}"


@pytest.mark.parametrize("backbone_type", BACKBONE_CONFIGS.keys())
def test_backbone_initialization(backbone_type):
    """Test that each backbone can be initialized."""
    for config_dict in BACKBONE_CONFIGS[backbone_type]:
        config = ConfigBackboneManager.from_dict(config_dict)

        # Initialize the backbone
        backbone = load_backbone(config)

        assert backbone is not None, f"Failed to initialize backbone of type {backbone_type}"
        assert isinstance(backbone.config, BackboneConfig), "Backbone config should be an instance of BackboneConfig"


@pytest.mark.parametrize("backbone_type", BACKBONE_CONFIGS.keys())
@pytest.mark.parametrize("input_size", INPUT_SIZES)
def test_backbone_forward(backbone_type, input_size):
    """Test that each backbone can process a forward pass with different input sizes."""
    for config_dict in BACKBONE_CONFIGS[backbone_type]:
        config = ConfigBackboneManager.from_dict(config_dict)

        # Initialize the backbone
        backbone = load_backbone(config)

        # Create a random input tensor
        x = torch.rand(*input_size)

        # Switch to eval mode to avoid batch norm issues
        backbone.eval()

        # Forward pass
        with torch.no_grad():
            outputs = backbone(x)

        # Check that outputs is a dictionary
        assert isinstance(outputs, dict), f"Backbone {backbone_type} output should be a dictionary"

        # Check that each output is a tensor
        for name, tensor in outputs.items():
            assert isinstance(tensor, torch.Tensor), f"Output {name} should be a tensor"

            # Check that the batch dimension is preserved
            assert tensor.shape[0] == input_size[0], f"Output {name} should have batch size {input_size[0]}"


def test_output_shapes():
    """Test that the output_shape method returns the expected shapes."""
    for backbone_type, config_dicts in BACKBONE_CONFIGS.items():
        for config_dict in config_dicts:
            config = ConfigBackboneManager.from_dict(config_dict)

            # Initialize the backbone
            backbone = load_backbone(config)

            # Get output shapes
            shapes = backbone.output_shape()

            # Check that shapes is a dictionary
            assert isinstance(shapes, dict), f"output_shape for {backbone_type} should return a dictionary"

            # Check each shape specification
            for name, shape_spec in shapes.items():
                assert hasattr(shape_spec, "channels"), f"Shape spec for {name} should have channels attribute"
                assert hasattr(shape_spec, "stride"), f"Shape spec for {name} should have stride attribute"


def test_invalid_backbone_type():
    """Test that trying to build a backbone with an invalid type raises a ValueError."""
    config_dict = {"model_type": "invalid_backbone", "use_pretrained": False}

    with pytest.raises(ValueError, match="Backbone invalid_backbone not supported"):
        ConfigBackboneManager.from_dict(config_dict)


def test_size_divisibility():
    """Test that size_divisibility property works."""
    for backbone_type, config_dicts in BACKBONE_CONFIGS.items():
        for config_dict in config_dicts:
            config = ConfigBackboneManager.from_dict(config_dict)
            backbone = load_backbone(config)

            # Should be an integer
            assert isinstance(backbone.size_divisibility, int)


def test_padding_constraints():
    """Test that padding_constraints property works."""
    for backbone_type, config_dicts in BACKBONE_CONFIGS.items():
        for config_dict in config_dicts:
            config = ConfigBackboneManager.from_dict(config_dict)
            backbone = load_backbone(config)

            # Should return a dict
            constraints = backbone.padding_constraints
            assert isinstance(constraints, dict)


def test_output_shape():
    """Test that output_shape property works."""
    for backbone_type, config_dicts in BACKBONE_CONFIGS.items():
        for config_dict in config_dicts:
            config = ConfigBackboneManager.from_dict(config_dict)
            backbone = load_backbone(config)

            # Should return a dict
            shapes = backbone.output_shape()
            assert isinstance(shapes, dict)

            # Check each shape specification
            for name, shape_spec in shapes.items():
                assert isinstance(shape_spec, ShapeSpec)
                assert hasattr(shape_spec, "channels")
                assert hasattr(shape_spec, "stride")
            print(backbone_type, shapes)


if __name__ == "__main__":
    test_build_function()
    for backbone_type in BACKBONE_CONFIGS.keys():
        test_backbone_initialization(backbone_type)
        for input_size in INPUT_SIZES:
            test_backbone_forward(backbone_type, input_size)
    test_output_shapes()
    test_invalid_backbone_type()
    test_size_divisibility()
    test_padding_constraints()
    test_output_shape()
