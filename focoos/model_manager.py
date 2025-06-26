import importlib
import os
from dataclasses import fields
from typing import Callable, Dict, Optional, Tuple, Type

from focoos.hub.focoos_hub import FocoosHUB
from focoos.model_registry.model_registry import ModelRegistry
from focoos.models.focoos_model import BaseModelNN, FocoosModel
from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.ports import MODELS_DIR, ArtifactName, ModelConfig, ModelFamily, ModelInfo
from focoos.utils.logger import get_logger

logger = get_logger("ModelManager")


class ModelManager:
    """Automatic model manager with lazy loading.

    The ModelManager provides a unified interface for loading models from various sources:
    - From ModelInfo objects
    - From the Focoos Hub (hub:// protocol)
    - From local directories
    - From the model registry

    It handles model registration, configuration management, and weights loading automatically.
    Models are loaded lazily when requested and can be accessed through the `get` method.

    Examples:
        Load a registered model:
        >>> model = ModelManager.get("model_name")

        Load a model from hub:
        >>> model = ModelManager.get("hub://username/model_ref")

        Load a model with custom config:
        >>> model = ModelManager.get("model_name", config=custom_config)
    """

    _models_family_map: Dict[str, Callable[[], Type[BaseModelNN]]] = {}  # {"fai-detr": load_fai_detr()}

    @classmethod
    def get(
        cls,
        name: str,
        model_info: Optional[ModelInfo] = None,
        config: Optional[ModelConfig] = None,
        hub: Optional[FocoosHUB] = None,
        cache: bool = True,
        **kwargs,
    ) -> FocoosModel:
        """
        Unified entrypoint to load a model by name or ModelInfo.

        This method provides a single interface for loading models from various sources:
        - From a ModelInfo object (when model_info is provided)
        - From the Focoos Hub (when name starts with "hub://")
        - From the ModelRegistry (for pretrained models)
        - From a local directory (when name is a local path)

        Args:
            name: Model name, path, or hub reference (e.g., "hub://username/model_ref")
            model_info: Optional ModelInfo object to load the model from directly
            config: Optional custom model configuration to override defaults
            hub: Optional FocoosHUB instance to use for hub:// references
            cache: Optional boolean to cache the model info and weights when loading from hub (defaults to True)
            **kwargs: Additional keyword arguments passed to the model configuration

        Returns:
            FocoosModel: The loaded model instance

        Raises:
            ValueError: If the model cannot be found or loaded
        """
        if model_info is not None:
            # Load model directly from provided ModelInfo
            return cls._from_model_info(model_info=model_info, config=config, **kwargs)

        # If name starts with "hub://", load from Focoos Hub
        if name.startswith("hub://"):
            model_info, hub_config = cls._from_hub(hub_uri=name, hub=hub, cache=cache, **kwargs)
            if config is None:
                config = hub_config  # Use hub config if no config is provided
        # If model exists in ModelRegistry, load as pretrained model
        elif ModelRegistry.exists(name):
            model_info = ModelRegistry.get_model_info(name)
        # Otherwise, attempt to load from a local directory
        else:
            model_info = cls._from_local_dir(name=name)
        # Load model from the resolved ModelInfo
        return cls._from_model_info(model_info=model_info, config=config, **kwargs)

    @classmethod
    def register_model(cls, model_family: ModelFamily, model_loader: Callable[[], Type[BaseModelNN]]):
        """
        Register a loader for a specific model family.

        This method associates a model family with a loader function that returns
        the model class when called. This enables lazy loading of model classes.

        Args:
            model_family: The ModelFamily enum value to register
            model_loader: A callable that returns the model class when invoked
        """
        cls._models_family_map[model_family.value] = model_loader

    @classmethod
    def _ensure_family_registered(cls, model_family: ModelFamily):
        """
        Ensure the model family is registered, importing if needed.

        This method checks if a model family is registered and if not, attempts to
        import and register it automatically by calling any registration functions
        in the family module.

        Args:
            model_family: The ModelFamily enum value to ensure is registered
        """
        if model_family.value in cls._models_family_map:
            return
        family_module = importlib.import_module(f"focoos.models.{model_family.value}")
        for attr_name in dir(family_module):
            if attr_name.startswith("_register"):
                register_func = getattr(family_module, attr_name)
                if callable(register_func):
                    register_func()

    @classmethod
    def _from_model_info(cls, model_info: ModelInfo, config: Optional[ModelConfig] = None, **kwargs) -> FocoosModel:
        """
        Load a model from ModelInfo, handling config and weights.

        This method instantiates a model based on the ModelInfo, applying the provided
        configuration (or using the one from ModelInfo) and loading weights if available.

        Args:
            model_info: ModelInfo object containing model metadata and references
            config: Optional model configuration to override the one in ModelInfo
            **kwargs: Additional keyword arguments passed to the model configuration

        Returns:
            FocoosModel: The instantiated model with weights loaded if available

        Raises:
            ValueError: If the model family is not supported
        """
        cls._ensure_family_registered(model_info.model_family)
        if model_info.model_family.value not in cls._models_family_map:
            raise ValueError(f"Model {model_info.model_family} not supported")
        model_class = cls._models_family_map[model_info.model_family.value]()
        config = config or ConfigManager.from_dict(model_info.model_family, model_info.config, **kwargs)
        model_info.config = config
        nn_model = model_class(model_info.config)
        model = FocoosModel(nn_model, model_info)
        return model

    @classmethod
    def _from_local_dir(cls, name: str) -> ModelInfo:
        """
        Load a model from a local experiment directory.

        This method loads a model from a local directory by reading its ModelInfo file
        and resolving paths to weights and other artifacts.

        Args:
            name: Name or path of the model directory relative to models_dir

        Returns:
            ModelInfo: The model information loaded from the local directory

        Raises:
            ValueError: If the model directory or ModelInfo file cannot be found
        """
        if os.path.exists(name):
            run_dir = name
        else:
            run_dir = os.path.join(MODELS_DIR, name)

        if not os.path.exists(run_dir):
            raise ValueError(f"Run {name} not found in {MODELS_DIR}")

        model_info_path = os.path.join(run_dir, ArtifactName.INFO)
        if not os.path.exists(model_info_path):
            raise ValueError(f"Model info not found in {run_dir}")
        model_info = ModelInfo.from_json(model_info_path)

        if model_info.weights_uri == ArtifactName.WEIGHTS:
            model_info.weights_uri = os.path.join(run_dir, model_info.weights_uri)

        return model_info

    @classmethod
    def _from_hub(
        cls, hub_uri: str, hub: Optional[FocoosHUB] = None, cache: bool = True, **kwargs
    ) -> Tuple[ModelInfo, ModelConfig]:
        """
        Load a model from the Focoos Hub.

        This method downloads a model from the Focoos Hub using the provided URI,
        which should be in the format "hub://username/model_ref".

        Args:
            hub_uri: Hub URI in the format "hub://username/model_ref"
            hub: Optional FocoosHUB instance to use (creates a new one if not provided)
            **kwargs: Additional keyword arguments passed to the model configuration

        Returns:
            Tuple[ModelInfo, ModelConfig]: The model information and configuration

        Raises:
            ValueError: If the model reference is invalid or the model cannot be downloaded
        """
        hub = hub or FocoosHUB()
        model_ref = hub_uri.split("hub://")[1]

        if not model_ref:
            raise ValueError("Model ref is required")

        model_pth_path = hub.download_model_pth(model_ref=model_ref, skip_if_exists=cache)

        model_info_path = os.path.join(MODELS_DIR, model_ref, ArtifactName.INFO)
        if not os.path.exists(model_info_path) or not cache:
            logger.info(f"📥 Downloading model info from hub for model: {model_ref}")
            remote_model_info = hub.get_model_info(model_ref=model_ref)
            model_info = ModelInfo.from_json(remote_model_info.model_dump(mode="json"))
            model_info.dump_json(model_info_path)
        else:
            logger.info(f"📥 Loading model info from cache: {model_info_path}")
            model_info = ModelInfo.from_json(model_info_path)

        if not model_info.weights_uri:
            model_info.weights_uri = model_pth_path

        model_config = ConfigManager.from_dict(model_info.model_family, model_info.config, **kwargs)

        return (model_info, model_config)


class BackboneManager:
    """
    Automatic backbone manager with lazy loading.

    The BackboneManager provides a unified interface for loading neural network backbones
    (feature extractors) from their configurations. It supports multiple backbone architectures
    like ResNet, STDC, Swin Transformer, MobileNetV2, and others.

    The manager maintains a mapping between backbone type names and their implementation paths,
    and handles the dynamic loading of the appropriate classes.
    """

    _BACKBONE_MAPPING: Dict[str, str] = {
        "resnet": "resnet.ResNet",
        "stdc": "stdc.STDC",
        "swin": "swin.Swin",
        "mobilenet_v2": "mobilenet_v2.MobileNetV2",
        "mit": "mit.MIT",
        "convnextv2": "convnextv2.ConvNeXtV2",
    }

    @classmethod
    def from_config(cls, config: BackboneConfig) -> BaseBackbone:
        """
        Load a backbone from a configuration.

        This method instantiates a backbone model based on the provided configuration,
        dynamically loading the appropriate backbone class based on the model_type.

        Args:
            config: The backbone configuration containing model_type and other parameters

        Returns:
            BaseBackbone: The instantiated backbone model

        Raises:
            ValueError: If the backbone type is not supported
        """
        if config.model_type not in cls._BACKBONE_MAPPING:
            raise ValueError(f"Backbone {config.model_type} not supported")
        backbone_class = cls.get_model_class(config.model_type)
        return backbone_class(config)

    @classmethod
    def get_model_class(cls, model_type: str):
        """
        Get the model class based on the model type.

        This method dynamically imports and returns the backbone class
        corresponding to the specified model type.

        Args:
            model_type: The type of backbone model to load (e.g., "resnet", "swin")

        Returns:
            Type[BaseBackbone]: The backbone class

        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class is not found in the module
        """
        import importlib

        module_path, class_name = cls._BACKBONE_MAPPING[model_type].split(".")
        module = importlib.import_module(f".{module_path}", package="focoos.nn.backbone")
        return getattr(module, class_name)


class ConfigManager:
    """
    Automatic model configuration management.

    The ConfigManager provides a centralized system for managing model configurations.
    It maintains a registry of configuration classes for different model families and
    handles the creation of appropriate configuration objects from dictionaries.

    The manager supports dynamic registration of configuration classes and automatic
    importing of model family modules as needed.
    """

    _MODEL_CFG_MAPPING: Dict[str, Callable[[], Type[ModelConfig]]] = {}

    @classmethod
    def register_config(cls, model_family: ModelFamily, model_config_loader: Callable[[], Type[ModelConfig]]):
        """
        Register a loader for a specific model configuration.

        This method associates a model family with a loader function that returns
        the configuration class when called. This enables lazy loading of configuration
        classes.

        Args:
            model_family: The ModelFamily enum value to register
            model_config_loader: A callable that returns the configuration class when invoked
        """
        cls._MODEL_CFG_MAPPING[model_family.value] = model_config_loader

    @classmethod
    def from_dict(cls, model_family: ModelFamily, config_dict: dict, **kwargs) -> ModelConfig:
        """
        Create a configuration from a dictionary.

        This method instantiates a model configuration object based on the model family
        and the provided configuration dictionary. It handles nested configurations
        like backbone_config and validates the parameters.

        Args:
            model_family: The model family enum value
            config_dict: Dictionary containing configuration parameters
            **kwargs: Additional keyword arguments to override configuration values

        Returns:
            ModelConfig: The instantiated configuration object

        Raises:
            ValueError: If the model family is not supported or if invalid parameters are provided
        """
        if model_family.value not in cls._MODEL_CFG_MAPPING:
            # Import the family module
            family_module = importlib.import_module(f"focoos.models.{model_family.value}")

            # Iteratively register all models in the family
            for attr_name in dir(family_module):
                if attr_name.startswith("_register"):
                    register_func = getattr(family_module, attr_name)
                    if callable(register_func):
                        register_func()

        if model_family.value not in cls._MODEL_CFG_MAPPING:
            raise ValueError(f"Model {model_family} not supported")

        config_class = cls._MODEL_CFG_MAPPING[model_family.value]()  # this return the config class

        # Convert the input dict to the actual config type
        if "backbone_config" in config_dict and config_dict["backbone_config"] is not None:
            config_dict["backbone_config"] = ConfigBackboneManager.from_dict(config_dict["backbone_config"])

            # Validate the parameters kwargs
        valid_fields = {f.name for f in fields(config_class)}
        invalid_kwargs = set(kwargs.keys()) - valid_fields
        if invalid_kwargs:
            raise ValueError(
                f"Invalid parameters for {config_class.__name__}: {invalid_kwargs}\nValid parameters: {valid_fields}"
            )

        config_dict = config_class(**config_dict)

        # Update the config with the kwargs
        if kwargs:
            config_dict.update(kwargs)

        return config_dict


class ConfigBackboneManager:
    """
    Automatic backbone configuration manager with lazy loading.

    The ConfigBackboneManager provides a specialized manager for handling backbone
    configurations. It maintains a mapping between backbone type names and their
    configuration classes, and handles the dynamic loading of these classes.

    This manager is used primarily by the ConfigManager when processing nested
    backbone configurations within model configurations.
    """

    _BACKBONE_MAPPING: Dict[str, str] = {
        "resnet": "resnet.ResnetConfig",
        "stdc": "stdc.STDCConfig",
        "swin": "swin.SwinConfig",
        "mobilenet_v2": "mobilenet_v2.MobileNetV2Config",
        "mit": "mit.MITConfig",
        "convnextv2": "convnextv2.ConvNeXtV2Config",
    }

    @classmethod
    def get_model_class(cls, model_type: str):
        """
        Get the configuration class based on the model type.

        This method dynamically imports and returns the backbone configuration class
        corresponding to the specified model type.

        Args:
            model_type: The type of backbone model (e.g., "resnet", "swin")

        Returns:
            Type[BackboneConfig]: The backbone configuration class

        Raises:
            ImportError: If the module cannot be imported
            AttributeError: If the class is not found in the module
        """
        import importlib

        module_path, class_name = cls._BACKBONE_MAPPING[model_type].split(".")
        module = importlib.import_module(f".{module_path}", package="focoos.nn.backbone")
        return getattr(module, class_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> BackboneConfig:
        """
        Create a backbone configuration from a dictionary.

        This method instantiates a backbone configuration object based on the
        model_type specified in the configuration dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters including model_type

        Returns:
            BackboneConfig: The instantiated backbone configuration object

        Raises:
            ValueError: If the backbone type is not supported
        """
        if config_dict["model_type"] not in cls._BACKBONE_MAPPING:
            raise ValueError(f"Backbone {config_dict['model_type']} not supported")

        config_class = cls.get_model_class(config_dict["model_type"])
        return_config = config_class(**config_dict)
        return return_config
