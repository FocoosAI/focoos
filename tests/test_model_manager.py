import importlib
import os
from dataclasses import dataclass
from typing import Any, Type
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from focoos.model_manager import ConfigBackboneManager, ModelManager
from focoos.models.focoos_model import BaseModelNN, FocoosModel
from focoos.ports import ModelConfig, ModelFamily, ModelInfo, Task


@pytest.fixture
def mock_model_config():
    """Fixture to provide a mock ModelConfig."""
    return ModelConfig(
        num_classes=10,
    )


@pytest.fixture
def mock_model_info():
    """Fixture to provide a mock ModelInfo."""
    model_info = ModelInfo(
        name="test-model",
        model_family=ModelFamily.DETR,
        classes=["class1", "class2"],
        im_size=640,
        task=Task.DETECTION,
        config={},
    )
    return model_info


@pytest.fixture
def mock_focoos_model():
    """Fixture to provide a mock get_infer_model method."""
    return MagicMock(spec=FocoosModel)


@pytest.fixture
def functional_model_manager():
    """Fixture to provide a functional ModelManager for testing."""
    original_models_map = ModelManager._models_family_map.copy()

    # Clear mappings for test
    ModelManager._models_family_map = {}

    yield ModelManager  # Provide clean ModelManager to test

    # Restore original mappings after test
    ModelManager._models_family_map = original_models_map


@pytest.fixture
def clean_model_manager(mock_focoos_model: FocoosModel):
    """Fixture to provide a clean ModelManager for testing.

    This fixture saves the original model mappings, clears them for the test,
    and restores them afterward regardless of test outcome.
    """
    # Save original mappings
    original_models_map = ModelManager._models_family_map.copy()
    original_from_model_info = ModelManager._from_model_info
    original_from_local_dir = ModelManager._from_local_dir
    original_from_hub = ModelManager._from_hub

    # Clear mappings for test
    ModelManager._models_family_map = {}
    ModelManager._from_model_info = MagicMock(return_value=mock_focoos_model)
    ModelManager._from_local_dir = MagicMock(return_value=mock_focoos_model)
    ModelManager._from_hub = MagicMock(return_value=(mock_focoos_model, MagicMock()))

    yield ModelManager  # Provide clean ModelManager to test

    # Restore original mappings after test
    ModelManager._models_family_map = original_models_map
    ModelManager._from_model_info = original_from_model_info
    ModelManager._from_local_dir = original_from_local_dir
    ModelManager._from_hub = original_from_hub


@pytest.fixture
def dirty_model_manager(mock_focoos_model: FocoosModel):
    """Fixture to provide a clean ModelManager for testing.

    This fixture saves the original model mappings, clears them for the test,
    and restores them afterward regardless of test outcome.
    """
    # Save original mappings
    original_models_map = ModelManager._models_family_map.copy()
    original_from_model_info = ModelManager._from_model_info
    original_from_local_dir = ModelManager._from_local_dir
    original_from_hub = ModelManager._from_hub
    original_register_model = ModelManager.register_model

    # Clear mappings for test
    # Create a mock model loader function
    def mock_model_loader() -> Type[BaseModelNN]:
        # This would return a model class in real code
        return BaseModelNN

    ModelManager._models_family_map = {ModelFamily.DETR.value: mock_model_loader}
    ModelManager._from_model_info = MagicMock(return_value=mock_focoos_model)
    ModelManager._from_local_dir = MagicMock(return_value=mock_focoos_model)
    ModelManager._from_hub = MagicMock(return_value=(mock_focoos_model, MagicMock()))
    ModelManager.register_model = MagicMock()

    yield ModelManager  # Provide clean ModelManager to test

    # Restore original mappings after test
    ModelManager._models_family_map = original_models_map
    ModelManager._from_model_info = original_from_model_info
    ModelManager._from_local_dir = original_from_local_dir
    ModelManager._from_hub = original_from_hub
    ModelManager.register_model = original_register_model


def test_register_model(clean_model_manager):
    """Test that ModelManager.register_model correctly registers a model loader."""

    # Create a mock model loader function
    def mock_model_loader() -> Type[BaseModelNN]:
        # This would return a model class in real code
        return BaseModelNN

    # Test model family
    test_family = ModelFamily.DETR

    # Register the mock model loader
    clean_model_manager.register_model(test_family, mock_model_loader)

    # Check that the model loader was registered
    assert test_family.value in clean_model_manager._models_family_map
    assert clean_model_manager._models_family_map[test_family.value] == mock_model_loader

    # Test retrieving the model class
    model_class = clean_model_manager._models_family_map[test_family.value]()
    assert model_class == BaseModelNN


def test_get_with_model_info_without_kwargs(clean_model_manager, mock_model_info):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""
    model = clean_model_manager.get(name="test-model", model_info=mock_model_info)
    clean_model_manager._from_model_info.assert_called_once_with(model_info=mock_model_info, config=None)
    assert isinstance(model, FocoosModel)


def test_get_with_model_info_with_kwargs(clean_model_manager, mock_model_info):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""
    model = clean_model_manager.get(name="test-model", model_info=mock_model_info, pluto="test-pluto")
    clean_model_manager._from_model_info.assert_called_once_with(
        model_info=mock_model_info, config=None, pluto="test-pluto"
    )
    assert isinstance(model, FocoosModel)


def test_get_with_model_info_with_config(clean_model_manager, mock_model_config, mock_model_info):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""
    model = clean_model_manager.get(name="test-model", model_info=mock_model_info, config=mock_model_config)
    clean_model_manager._from_model_info.assert_called_once_with(model_info=mock_model_info, config=mock_model_config)
    assert isinstance(model, FocoosModel)


def test_get_with_get_model_hub(clean_model_manager):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""
    mock_hub = MagicMock()
    model = clean_model_manager.get(name="hub://test-model", hub=mock_hub)
    clean_model_manager._from_hub.assert_called_once_with(hub_uri="hub://test-model", hub=mock_hub, cache=True)
    assert isinstance(model, FocoosModel)


def test_get_with_get_model_local_dir(clean_model_manager, mock_model_config):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""
    # Setup mocks so that ModelRegistry.exists returns False to trigger local dir path
    with MagicMock() as mock_registry:
        mock_registry.exists.return_value = False
        clean_model_manager._from_local_dir.return_value = MagicMock(spec=ModelInfo)

        model = clean_model_manager.get(name="test-model")
        clean_model_manager._from_local_dir.assert_called_once_with(name="test-model", models_dir=None, config=None)
        clean_model_manager._from_model_info.assert_called_once()
        assert isinstance(model, FocoosModel)


def test_get_with_get_model_local_dir_with_config(clean_model_manager, mock_model_config):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""
    # Setup mocks so that ModelRegistry.exists returns False to trigger local dir path
    with MagicMock() as mock_registry:
        mock_registry.exists.return_value = False
        clean_model_manager._from_local_dir.return_value = MagicMock(spec=ModelInfo)

        model = clean_model_manager.get(name="test-model", config=mock_model_config)
        clean_model_manager._from_local_dir.assert_called_once_with(
            name="test-model", models_dir=None, config=mock_model_config
        )
        clean_model_manager._from_model_info.assert_called_once()
        assert isinstance(model, FocoosModel)


def test_get_with_get_model_local_dir_with_model_dir(clean_model_manager, mock_model_config):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""
    # Setup mocks so that ModelRegistry.exists returns False to trigger local dir path
    with MagicMock() as mock_registry:
        mock_registry.exists.return_value = False
        clean_model_manager._from_local_dir.return_value = MagicMock(spec=ModelInfo)

        model = clean_model_manager.get(name="test-model", models_dir="test-models-dir")
        clean_model_manager._from_local_dir.assert_called_once_with(
            name="test-model", models_dir="test-models-dir", config=None
        )
        clean_model_manager._from_model_info.assert_called_once()
        assert isinstance(model, FocoosModel)


def test_get_with_get_model_registry(mocker: MockerFixture, clean_model_manager, mock_model_info):
    """Test that ModelManager.get_infer_model correctly retrieves a model."""

    mocker.patch("focoos.model_manager.ModelRegistry.exists", return_value=True)
    mocker.patch("focoos.model_manager.ModelRegistry.get_model_info", return_value=mock_model_info)
    model = clean_model_manager.get(name="test-model")
    clean_model_manager._from_model_info.assert_called_once_with(model_info=mock_model_info, config=None)
    assert isinstance(model, FocoosModel)


def test_ensure_family_registered_already_registered(mocker: MockerFixture, dirty_model_manager):
    patched_function = mocker.patch("focoos.model_manager.importlib.import_module", return_value=MagicMock())
    dirty_model_manager._ensure_family_registered(ModelFamily.DETR)
    dirty_model_manager.register_model.assert_not_called()
    patched_function.assert_not_called()


def test_ensure_family_registered_not_registered(mocker: MockerFixture, clean_model_manager):
    fake_family_module = MagicMock()
    fake_family_module._register = MagicMock()
    mocker.patch("focoos.model_manager.importlib.import_module", return_value=fake_family_module)
    clean_model_manager._ensure_family_registered(ModelFamily.DETR)
    fake_family_module._register.assert_called_once()


def test_from_model_info_with_model_registry(mocker: MockerFixture, functional_model_manager, mock_model_info):
    # Mock the necessary components
    mock_model_class = MagicMock()
    mock_nn_model = MagicMock()
    mock_model_class.return_value = mock_nn_model

    # Mock the _ensure_family_registered method
    functional_model_manager._ensure_family_registered = MagicMock()

    # Mock the _models_family_map to return our mock_model_class
    functional_model_manager._models_family_map = {mock_model_info.model_family.value: lambda: mock_model_class}

    # Mock ConfigManager
    mock_config = MagicMock()
    mocker.patch("focoos.model_manager.ConfigManager.from_dict", return_value=mock_config)

    # Call the method
    result = functional_model_manager._from_model_info(model_info=mock_model_info)

    # Assertions
    functional_model_manager._ensure_family_registered.assert_called_once_with(mock_model_info.model_family)
    mock_model_class.assert_called_once_with(mock_config)
    assert isinstance(result, FocoosModel)
    assert result.model == mock_nn_model
    assert result.model_info == mock_model_info


def test_from_model_info_with_custom_config(
    mocker: MockerFixture, functional_model_manager, mock_focoos_model, mock_model_info, mock_model_config
):
    # Mock the necessary components
    mocker.patch("focoos.model_manager.FocoosModel", return_value=mock_focoos_model)
    mock_model_class = MagicMock()
    mock_nn_model = MagicMock()
    mock_model_class.return_value = mock_nn_model

    # Mock the _ensure_family_registered method
    mocker.patch.object(functional_model_manager, "_ensure_family_registered")

    # Mock the _models_family_map to return our mock_model_class
    functional_model_manager._models_family_map = {mock_model_info.model_family.value: lambda: mock_model_class}

    # Call the method with custom config
    result = functional_model_manager._from_model_info(model_info=mock_model_info, config=mock_model_config)

    # Assertions
    functional_model_manager._ensure_family_registered.assert_called_once_with(mock_model_info.model_family)
    mock_model_class.assert_called_once_with(mock_model_config)
    assert isinstance(result, FocoosModel)


def test_from_model_info_with_weights(mocker: MockerFixture, functional_model_manager, mock_model_info):
    # Mock the necessary components
    mock_model_class = MagicMock()
    mock_nn_model = MagicMock()
    mock_model_class.return_value = mock_nn_model
    mock_focoos_model = MagicMock(spec=FocoosModel)
    mock_focoos_model.load_weights = MagicMock()

    # Set weights_uri
    mock_model_info.weights_uri = "model_weights.pth"

    # Mock the _ensure_family_registered method
    mocker.patch.object(functional_model_manager, "_ensure_family_registered")

    # Mock the FocoosModel creation
    mocker.patch("focoos.model_manager.FocoosModel", return_value=mock_focoos_model)

    # Mock the _models_family_map to return our mock_model_class
    mock_model_class_function = MagicMock()
    mock_model_class_function.return_value = mock_model_class
    functional_model_manager._models_family_map = {mock_model_info.model_family.value: mock_model_class_function}

    # Mock ConfigManager - removed ArtifactsManager as it doesn't exist
    mock_config = MagicMock()
    mocker.patch("focoos.model_manager.ConfigManager.from_dict", return_value=mock_config)

    # Call the method
    result = functional_model_manager._from_model_info(model_info=mock_model_info)

    # Assertions
    mock_model_class_function.assert_called_once()
    mock_model_class.assert_called_once()
    # Note: load_weights is not called in the current implementation
    # The weights loading is handled differently now
    assert result == mock_focoos_model


def test_from_model_info_unsupported_family(functional_model_manager, mock_model_info):
    # Create a mock model family that's not in the _models_family_map
    # Using a different model family than what's in the maps
    unsupported_family = list(ModelFamily)[0]  # Get first enum value
    mock_model_info.model_family = unsupported_family

    # Mock the _ensure_family_registered method to do nothing
    functional_model_manager._ensure_family_registered = MagicMock()

    # Clear the _models_family_map
    functional_model_manager._models_family_map = {}

    # Call the method and expect a ValueError
    with pytest.raises(ValueError, match=f"Model {unsupported_family} not supported"):
        functional_model_manager._from_model_info(model_info=mock_model_info)


def test_from_local_dir_success(mocker: MockerFixture, functional_model_manager, mock_model_info):
    # Mock os.path functions
    mocker.patch("os.path.exists", return_value=True)

    # Mock ModelInfo.from_json
    mocker.patch("focoos.ports.ModelInfo.from_json", return_value=mock_model_info)

    # Call the method
    result = functional_model_manager._from_local_dir(name="test-model", models_dir="/path/to/models")

    # Assertions - _from_local_dir returns ModelInfo, not FocoosModel
    assert isinstance(result, ModelInfo)
    assert result == mock_model_info


def test_from_local_dir_success_without_models_dir(mocker: MockerFixture, functional_model_manager, mock_model_info):
    # Mock os.path functions
    mocker.patch("os.path.exists", return_value=True)

    # Mock ModelInfo.from_json
    mocker.patch("focoos.ports.ModelInfo.from_json", return_value=mock_model_info)

    # Call the method
    result = functional_model_manager._from_local_dir(name="test-model")

    # Assertions - _from_local_dir returns ModelInfo, not FocoosModel
    assert isinstance(result, ModelInfo)
    assert result == mock_model_info


def test_from_local_dir_with_model_final_pth(mocker: MockerFixture, functional_model_manager, mock_model_info):
    # Mock os.path functions
    mocker.patch("os.path.exists", return_value=True)

    # Set weights_uri to model_final.pth
    mock_model_info.weights_uri = "model_final.pth"

    # Mock ModelInfo.from_json
    mocker.patch("focoos.ports.ModelInfo.from_json", return_value=mock_model_info)

    # Call the method
    result = functional_model_manager._from_local_dir(name="test-model", models_dir="/path/to/models")

    # Check if weights_uri was updated
    expected_weights_path = os.path.join("/path/to/models", "test-model", "model_final.pth")
    assert result.weights_uri == expected_weights_path


def test_from_local_dir_dir_not_found(mocker: MockerFixture, functional_model_manager):
    # Mock os.path.exists to return False (directory not found)
    mocker.patch("os.path.exists", return_value=False)

    # Call the method and expect a ValueError
    with pytest.raises(ValueError, match="Run test-model not found in /path/to/models"):
        functional_model_manager._from_local_dir(name="test-model", models_dir="/path/to/models")


def test_from_local_dir_model_info_not_found(mocker: MockerFixture, functional_model_manager):
    # Mock os.path.exists to return True for the run_dir but False for the model_info_path
    def mock_exists(path):
        return not path.endswith("model_info.json")

    mocker.patch("os.path.exists", side_effect=mock_exists)

    # Call the method and expect a ValueError
    with pytest.raises(ValueError):
        functional_model_manager._from_local_dir(name="test-model", models_dir="/path/to/models")


def test_from_local_dir_with_config(
    mocker: MockerFixture, functional_model_manager, mock_model_info, mock_model_config
):
    # Mock os.path functions
    mocker.patch("os.path.exists", return_value=True)

    # Mock ModelInfo.from_json
    mocker.patch("focoos.ports.ModelInfo.from_json", return_value=mock_model_info)

    # Call the method with config
    result = functional_model_manager._from_local_dir(
        name="test-model", models_dir="/path/to/models", config=mock_model_config
    )

    # Assertions - _from_local_dir returns ModelInfo, not FocoosModel
    assert isinstance(result, ModelInfo)
    assert result == mock_model_info


def test_from_hub_success(mocker: MockerFixture, functional_model_manager, mock_model_info):
    """Test successful model loading from hub."""
    # Mock hub and dependencies
    mock_hub = MagicMock()
    mock_hub.download_model_pth.return_value = "/path/to/model.pth"
    mock_hub.get_model_info.return_value = MagicMock()
    mock_hub.get_model_info.return_value.model_dump.return_value = {}

    # Mock file operations
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("focoos.ports.ModelInfo.from_json", side_effect=[mock_model_info, mock_model_info])

    # Mock ConfigManager
    mock_config = MagicMock()
    mocker.patch("focoos.model_manager.ConfigManager.from_dict", return_value=mock_config)

    # Mock model_info dump_json method
    mock_model_info.dump_json = MagicMock()

    # Call the method
    model_info, config = functional_model_manager._from_hub(hub_uri="hub://test/model", hub=mock_hub)

    # Assertions
    assert model_info == mock_model_info
    assert config == mock_config
    mock_hub.download_model_pth.assert_called_once()
    mock_hub.get_model_info.assert_called_once()


def test_from_hub_invalid_uri(mocker: MockerFixture, functional_model_manager):
    """Test hub method with invalid URI."""
    # Mock FocoosHUB to avoid authentication issues
    mock_hub = MagicMock()
    mocker.patch("focoos.model_manager.FocoosHUB", return_value=mock_hub)

    with pytest.raises(ValueError, match="Model ref is required"):
        functional_model_manager._from_hub(hub_uri="hub://")


# BackboneManager Tests


@pytest.fixture
def mock_backbone_config():
    """Fixture to provide a mock BackboneConfig."""
    from focoos.nn.backbone.base import BackboneConfig

    return MagicMock(spec=BackboneConfig, model_type="resnet")


def test_backbone_manager_from_config(mocker: MockerFixture, mock_backbone_config):
    """Test that BackboneManager.from_config correctly loads a backbone."""
    from focoos.model_manager import BackboneManager
    from focoos.nn.backbone.base import BaseBackbone

    # Mock the get_model_class method
    mock_backbone_class = MagicMock(spec=BaseBackbone)
    mocker.patch.object(BackboneManager, "get_model_class", return_value=mock_backbone_class)

    # Call the method
    result = BackboneManager.from_config(mock_backbone_config)

    # Assertions
    BackboneManager.get_model_class.assert_called_once_with(mock_backbone_config.model_type)
    mock_backbone_class.assert_called_once_with(mock_backbone_config)
    assert result == mock_backbone_class.return_value


def test_backbone_manager_from_config_unsupported(mock_backbone_config):
    """Test that BackboneManager.from_config raises for unsupported backbone."""
    from focoos.model_manager import BackboneManager

    # Set an unsupported model type
    mock_backbone_config.model_type = "unsupported_backbone"

    # Call the method and expect ValueError
    with pytest.raises(ValueError, match=f"Backbone {mock_backbone_config.model_type} not supported"):
        BackboneManager.from_config(mock_backbone_config)


def test_backbone_manager_get_model_class(mocker: MockerFixture):
    """Test that BackboneManager.get_model_class correctly imports and returns a class."""
    from focoos.model_manager import BackboneManager

    # Mock importlib.import_module
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.ResNet = mock_class
    mocker.patch("importlib.import_module", return_value=mock_module)

    # Call the method
    result = BackboneManager.get_model_class("resnet")

    # Assertions
    importlib.import_module.assert_called_once_with(".resnet", package="focoos.nn.backbone")
    assert result == mock_class


# ConfigManager Tests


@pytest.fixture
def mock_config_class():
    """Fixture to provide a mock config class."""
    return MagicMock(spec=ModelConfig)


def test_config_manager_register_config():
    """Test that ConfigManager.register_config correctly registers a config loader."""
    from focoos.model_manager import ConfigManager

    # Save original mapping for restoration
    original_mapping = ConfigManager._MODEL_CFG_MAPPING.copy()

    try:
        # Clear the mapping
        ConfigManager._MODEL_CFG_MAPPING = {}

        # Create mock loader
        mock_loader = MagicMock(return_value=MagicMock(spec=ModelConfig))

        # Register the loader
        ConfigManager.register_config(ModelFamily.DETR, mock_loader)

        # Assertions
        assert ModelFamily.DETR.value in ConfigManager._MODEL_CFG_MAPPING
        assert ConfigManager._MODEL_CFG_MAPPING[ModelFamily.DETR.value] == mock_loader
    finally:
        # Restore original mapping
        ConfigManager._MODEL_CFG_MAPPING = original_mapping


def test_config_manager_from_dict_registered(mocker: MockerFixture, mock_config_class):
    """Test ConfigManager.from_dict with a registered config."""
    from focoos.model_manager import ConfigManager

    # Save original mapping for restoration
    original_mapping = ConfigManager._MODEL_CFG_MAPPING.copy()

    try:
        # Clear the mapping and add our mock
        ConfigManager._MODEL_CFG_MAPPING = {ModelFamily.DETR.value: MagicMock(return_value=mock_config_class)}

        # Test data
        config_dict = {"num_classes": 10}

        # Mock necessary components
        mocker.patch.object(ConfigBackboneManager, "from_dict", return_value=MagicMock())

        # Call the method
        result = ConfigManager.from_dict(ModelFamily.DETR, config_dict)

        # Assertions
        assert result == mock_config_class.return_value
        mock_config_class.assert_called_once_with(**config_dict)
    finally:
        # Restore original mapping
        ConfigManager._MODEL_CFG_MAPPING = original_mapping


def test_config_manager_from_dict_with_backbone_config(mocker: MockerFixture):
    """Test ConfigManager.from_dict with a backbone config."""
    from focoos.model_manager import ConfigManager

    # Save original mapping for restoration
    original_mapping = ConfigManager._MODEL_CFG_MAPPING.copy()

    try:

        @dataclass
        class MockConfig(ModelConfig):
            backbone_config: Any

        sub_mock_config_class = MagicMock(spec=MockConfig, __name__="MockConfig", wraps=MockConfig)
        sub_mock_config_class.return_value = MagicMock()

        # Clear the mapping and add our mock
        ConfigManager._MODEL_CFG_MAPPING = {ModelFamily.DETR.value: MagicMock(return_value=sub_mock_config_class)}

        # Test data with backbone_config
        config_dict = {"backbone_config": {"model_type": "resnet"}, "num_classes": 10}

        # Mock ConfigBackboneManager.from_dict
        mock_backbone_config = MagicMock()
        mocker.patch.object(ConfigBackboneManager, "from_dict", return_value=mock_backbone_config)

        # Call the method
        result = ConfigManager.from_dict(ModelFamily.DETR, config_dict)

        # Assertions
        ConfigBackboneManager.from_dict.assert_called_once_with({"model_type": "resnet"})
        assert result == sub_mock_config_class.return_value
        # Check that backbone_config was properly replaced
        sub_mock_config_class.assert_called_once_with(**config_dict)
    finally:
        # Restore original mapping
        ConfigManager._MODEL_CFG_MAPPING = original_mapping


def test_config_manager_from_dict_with_kwargs(mocker: MockerFixture, mock_config_class):
    """Test ConfigManager.from_dict with kwargs."""
    from focoos.model_manager import ConfigManager

    # Save original mapping for restoration
    original_mapping = ConfigManager._MODEL_CFG_MAPPING.copy()

    try:

        @dataclass
        class MockConfig(ModelConfig):
            param1: str
            param2: str

        sub_mock_config_class = MagicMock(spec=MockConfig, __name__="MockConfig", wraps=MockConfig)

        # Clear the mapping and add our mock
        ConfigManager._MODEL_CFG_MAPPING = {ModelFamily.DETR.value: MagicMock(return_value=sub_mock_config_class)}

        # Mock update method
        mock_config = MagicMock()
        sub_mock_config_class.return_value = mock_config

        # Test data
        config_dict = {"param1": "value1"}
        kwargs = {"param2": "value2"}

        # Call the method
        result = ConfigManager.from_dict(ModelFamily.DETR, config_dict, **kwargs)

        # Assertions
        assert result == mock_config
        mock_config.update.assert_called_once_with({"param2": "value2"})
    finally:
        # Restore original mapping
        ConfigManager._MODEL_CFG_MAPPING = original_mapping


def test_config_manager_from_dict_with_invalid_kwargs(mocker: MockerFixture, mock_config_class):
    """Test ConfigManager.from_dict with invalid kwargs."""
    from focoos.model_manager import ConfigManager

    # Save original mapping for restoration
    original_mapping = ConfigManager._MODEL_CFG_MAPPING.copy()

    try:

        class MockConfig(ModelConfig):
            param1: str
            param2: str

        sub_mock_config_class = MagicMock(spec=MockConfig)
        sub_mock_config_class.__name__ = "MockConfig"

        # Clear the mapping and add our mock
        ConfigManager._MODEL_CFG_MAPPING = {ModelFamily.DETR.value: MagicMock(return_value=sub_mock_config_class)}

        # Mock fields function to return what we need
        mock_field = MagicMock()
        mock_field.name = "param1"
        mocker.patch("dataclasses.fields", return_value=[mock_field])

        # Test data
        config_dict = {"param1": "value1"}
        kwargs = {"invalid_param": "value2"}

        # Call the method and expect ValueError
        with pytest.raises(ValueError, match="Invalid parameters"):
            ConfigManager.from_dict(ModelFamily.DETR, config_dict, **kwargs)
    finally:
        # Restore original mapping
        ConfigManager._MODEL_CFG_MAPPING = original_mapping


def test_config_manager_from_dict_unsupported_family(mocker: MockerFixture):
    """Test ConfigManager.from_dict with an unsupported family."""
    from focoos.model_manager import ConfigManager

    # Save original mapping for restoration
    original_mapping = ConfigManager._MODEL_CFG_MAPPING.copy()

    try:
        # Clear the mapping
        ConfigManager._MODEL_CFG_MAPPING = {}

        # Mock importlib.import_module to do nothing
        mock_module = MagicMock()
        # No _register function to be found
        mocker.patch("importlib.import_module", return_value=mock_module)

        # Test data
        config_dict = {"param1": "value1"}

        # Call the method and expect ValueError
        with pytest.raises(ValueError, match=f"Model {ModelFamily.DETR} not supported"):
            ConfigManager.from_dict(ModelFamily.DETR, config_dict)
    finally:
        # Restore original mapping
        ConfigManager._MODEL_CFG_MAPPING = original_mapping


# ConfigBackboneManager Tests


def test_config_backbone_manager_get_model_class(mocker: MockerFixture):
    """Test that ConfigBackboneManager.get_model_class correctly imports and returns a class."""
    from focoos.model_manager import ConfigBackboneManager

    # Mock importlib.import_module
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.ResnetConfig = mock_class
    mocker.patch("importlib.import_module", return_value=mock_module)

    # Call the method
    result = ConfigBackboneManager.get_model_class("resnet")

    # Assertions
    importlib.import_module.assert_called_once_with(".resnet", package="focoos.nn.backbone")
    assert result == mock_class


def test_config_backbone_manager_from_dict(mocker: MockerFixture):
    """Test that ConfigBackboneManager.from_dict correctly creates a config."""
    from focoos.model_manager import ConfigBackboneManager

    # Mock data
    config_dict = {"model_type": "resnet", "param1": "value1"}

    # Mock get_model_class
    mock_config_class = MagicMock()
    mock_config = MagicMock()
    mock_config_class.return_value = mock_config
    mocker.patch.object(ConfigBackboneManager, "get_model_class", return_value=mock_config_class)

    # Call the method
    result = ConfigBackboneManager.from_dict(config_dict)

    # Assertions
    ConfigBackboneManager.get_model_class.assert_called_once_with(config_dict["model_type"])
    mock_config_class.assert_called_once_with(**config_dict)
    assert result == mock_config


def test_config_backbone_manager_from_dict_unsupported(mocker: MockerFixture):
    """Test that ConfigBackboneManager.from_dict raises for unsupported backbone."""
    from focoos.model_manager import ConfigBackboneManager

    # Test data with unsupported backbone
    config_dict = {"model_type": "unsupported_backbone"}

    # Call the method and expect ValueError
    with pytest.raises(ValueError, match=f"Backbone {config_dict['model_type']} not supported"):
        ConfigBackboneManager.from_dict(config_dict)
