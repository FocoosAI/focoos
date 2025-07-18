from unittest.mock import MagicMock, patch

import pytest

from focoos.model_registry.model_registry import ModelRegistry
from focoos.ports import ModelFamily, ModelInfo, ModelStatus, Task


class TestModelRegistry:
    """Test suite for the ModelRegistry class"""

    def test_list_models_returns_all_pretrained_models(self):
        """Check that list_models returns all pretrained models"""
        expected_models = [
            "fai-detr-l-obj365",
            "fai-detr-l-coco",
            "fai-detr-m-coco",
            "fai-mf-l-ade",
            "fai-mf-m-ade",
            "fai-mf-l-coco-ins",
            "fai-mf-m-coco-ins",
            "fai-mf-s-coco-ins",
            "bisenetformer-m-ade",
            "bisenetformer-l-ade",
            "bisenetformer-s-ade",
            "yoloxpose-s-coco",
        ]

        models = ModelRegistry.list_models()

        assert isinstance(models, list)
        assert len(models) == len(expected_models)
        for model in expected_models:
            assert model in models

    def test_exists_returns_true_for_valid_pretrained_model(self):
        """Check that exists returns True for a valid pretrained model"""
        assert ModelRegistry.exists("fai-detr-l-coco") is True
        assert ModelRegistry.exists("fai-mf-l-ade") is True

    def test_exists_returns_false_for_invalid_model(self):
        """Check that exists returns False for a non-existent model"""
        assert ModelRegistry.exists("modello-inesistente") is False
        assert ModelRegistry.exists("") is False
        assert ModelRegistry.exists("invalid-model-name") is False

    @patch("focoos.ports.ModelInfo.from_json")
    def test_get_model_info_with_pretrained_model(self, mock_from_json: MagicMock):
        """Check that get_model_info correctly loads a pretrained model"""
        # Setup mock ModelInfo
        mock_model_info = ModelInfo(
            name="fai-detr-l-coco",
            model_family=ModelFamily.DETR,
            classes=["person", "car", "dog"],
            im_size=640,
            task=Task.DETECTION,
            config={"num_classes": 3},
            status=ModelStatus.TRAINING_COMPLETED,
        )
        mock_from_json.return_value = mock_model_info

        # Test
        result = ModelRegistry.get_model_info("fai-detr-l-coco")

        # Verification
        assert result == mock_model_info
        mock_from_json.assert_called_once()
        # Check that it was called with the correct path
        call_args = mock_from_json.call_args[0][0]
        assert call_args.endswith("fai-detr-l-coco.json")

    @patch("os.path.exists")
    @patch("focoos.ports.ModelInfo.from_json")
    def test_get_model_info_with_custom_model_path(self, mock_from_json: MagicMock, mock_exists: MagicMock):
        """Check that get_model_info loads a model from a custom path"""
        # Setup
        custom_path = "/path/to/custom_model.json"
        mock_exists.return_value = True
        mock_model_info = ModelInfo(
            name="custom-model",
            model_family=ModelFamily.DETR,
            classes=["class1", "class2"],
            im_size=512,
            task=Task.DETECTION,
            config={"num_classes": 2},
        )
        mock_from_json.return_value = mock_model_info

        # Test
        result = ModelRegistry.get_model_info(custom_path)

        # Verification
        assert result == mock_model_info
        mock_exists.assert_called_once_with(custom_path)
        mock_from_json.assert_called_once_with(custom_path)

    @patch("os.path.exists")
    def test_get_model_info_raises_error_for_nonexistent_custom_path(self, mock_exists: MagicMock):
        """Check that get_model_info raises an error for a non-existent custom path"""
        # Setup
        custom_path = "/path/to/nonexistent_model.json"
        mock_exists.return_value = False

        # Test and verification
        with pytest.raises(ValueError, match=f"⚠️ Model {custom_path} not found"):
            ModelRegistry.get_model_info(custom_path)

        mock_exists.assert_called_once_with(custom_path)

    def test_get_model_info_raises_error_for_nonexistent_pretrained_model(self):
        """Check that get_model_info raises an error for a non-existent pretrained model"""
        nonexistent_model = "modello-inesistente"

        with pytest.raises(ValueError, match=f"⚠️ Model {nonexistent_model} not found"):
            ModelRegistry.get_model_info(nonexistent_model)

    @patch("focoos.ports.ModelInfo.from_json")
    def test_get_model_info_handles_json_loading_error(self, mock_from_json: MagicMock):
        """Check that get_model_info correctly handles JSON loading errors"""
        # Setup
        mock_from_json.side_effect = FileNotFoundError("File not found")

        # Test and verification
        with pytest.raises(FileNotFoundError):
            ModelRegistry.get_model_info("fai-detr-l-coco")

    def test_pretrained_models_dictionary_structure(self):
        """Check that the _pretrained_models dictionary has the correct structure"""
        pretrained_models = ModelRegistry._pretrained_models

        assert isinstance(pretrained_models, dict)
        assert len(pretrained_models) > 0

        for model_name, model_path in pretrained_models.items():
            # Check that the model name is a non-empty string
            assert isinstance(model_name, str)
            assert len(model_name) > 0

            # Check that the path is a string ending with .json
            assert isinstance(model_path, str)
            assert model_path.endswith(".json")
            assert model_name in model_path

    @pytest.mark.parametrize(
        "model_name,expected_exists",
        [
            ("fai-detr-l-coco", True),
            ("fai-detr-m-coco", True),
            ("fai-mf-l-ade", True),
            ("bisenetformer-s-ade", True),
            ("nonexistent-model", False),
            ("", False),
            ("fai-detr-xl-coco", False),  # Non-existent variant
        ],
    )
    def test_exists_parametrized(self, model_name: str, expected_exists: bool):
        """Parametrized test to check the existence of various models"""
        assert ModelRegistry.exists(model_name) == expected_exists

    def test_registry_path_configuration(self):
        """Check that REGISTRY_PATH is correctly configured"""
        from focoos.model_registry.model_registry import REGISTRY_PATH

        assert isinstance(REGISTRY_PATH, str)
        assert len(REGISTRY_PATH) > 0
        assert "model_registry" in REGISTRY_PATH

    @patch("focoos.utils.logger.get_logger")
    def test_logger_warning_on_model_not_found(self, mock_get_logger: MagicMock):
        """Check that a warning is logged when a model is not found"""
        nonexistent_model = "modello-inesistente"

        with pytest.raises(ValueError):
            ModelRegistry.get_model_info(nonexistent_model)
