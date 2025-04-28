import importlib
import os
from dataclasses import fields
from typing import Callable, Dict, Optional, Type

from focoos.model_registry.model_registry import ModelRegistry
from focoos.models.fai_model import BaseModelNN, FocoosModel, ModelConfig
from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.ports import ModelFamily
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class AutoConfig:
    """Automatic model configuration management"""

    _MODEL_MAPPING: Dict[str, Callable[[], Type[ModelConfig]]] = {}
    _REGISTERED_MODELS: set = set()

    @classmethod
    def register_model(cls, model_family: ModelFamily, model_config_loader: Callable[[], Type[ModelConfig]]):
        """
        Register a loader for a specific model

        Args:
            model_family: Model family
            model_loader: Function that loads the model
        """
        cls._MODEL_MAPPING[model_family.value] = model_config_loader
        cls._REGISTERED_MODELS.add(model_family.value)

    @classmethod
    def from_dict(cls, model_family: ModelFamily, config_dict: dict, **kwargs) -> ModelConfig:
        """
        Create a configuration from a dictionary
        """
        if model_family not in cls._MODEL_MAPPING:
            raise ValueError(f"Model {model_family} not supported")
        config_class = cls._MODEL_MAPPING[model_family.value]()  # this return the config class

        # Convert the input dict to the actual config type
        if "backbone_config" in config_dict and config_dict["backbone_config"] is not None:
            config_dict["backbone_config"] = AutoConfigBackbone.from_dict(config_dict["backbone_config"])

            # Validate the parameters kwargs
        valid_fields = {f.name for f in fields(config_class)}
        invalid_kwargs = set(kwargs.keys()) - valid_fields
        if invalid_kwargs:
            raise ValueError(
                f"Invalid parameters for {config_class.__name__}: {invalid_kwargs}\nValid parameters: {valid_fields}"
            )

        config_dict = config_class(**config_dict)

        # Update the config with the kwargs
        config_dict.update(kwargs)

        return config_dict


class AutoModel:
    """Automatic model manager with lazy loading"""

    _MODEL_MAPPING: Dict[str, Callable[[], Type[BaseModelNN]]] = {}
    _REGISTERED_MODELS: set = set()

    @classmethod
    def register_model(cls, model_family: ModelFamily, model_loader: Callable[[], Type[BaseModelNN]]):
        """
        Register a loader for a specific model

        Args:
            model_family: Model family
            model_loader: Function that loads the model
        """
        cls._MODEL_MAPPING[model_family.value] = model_loader
        cls._REGISTERED_MODELS.add(model_family.value)

    @classmethod
    def _import_model_family(cls, model_family: ModelFamily):
        """Dynamically import a model family"""
        try:
            module_name = f"focoos.models.{model_family}"
            importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Unable to import model family {model_family}. Error: {str(e)}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, config: Optional[ModelConfig] = None, **kwargs) -> FocoosModel:
        """
        Load a pretrained model

        Args:
            pretrained_model_name: Name of the pretrained model
            config: Model configuration (optional)
            **kwargs: Configuration parameters to override

        Returns:
            FocoosModel: Loaded model

        Raises:
            ValueError: If the model doesn't exist or is not supported
            RuntimeError: If there are errors during loading
        """
        model_info = ModelRegistry.get_model_info(pretrained_model_name)
        if model_info is None:
            raise ValueError(f"Model {pretrained_model_name} not found")

        # Import the family module only if not already registered
        if model_info.model_family not in cls._REGISTERED_MODELS:
            # Import the family module
            family_module = importlib.import_module(f"focoos.models.{model_info.model_family.value}")

            # Iteratively register all models in the family
            for attr_name in dir(family_module):
                if attr_name.startswith("_register_"):
                    register_func = getattr(family_module, attr_name)
                    if callable(register_func):
                        register_func()

        if model_info.model_family not in cls._MODEL_MAPPING:
            raise ValueError(f"Model {pretrained_model_name} not supported")

        if config is None:
            config = AutoConfig.from_dict(model_info.model_family, model_info.config, **kwargs)

        model_info.config = config

        try:
            model_class = cls._MODEL_MAPPING[model_info.model_family.value]()
            model = model_class(config)
            focoos_model = FocoosModel(model, model_info)

            if model_info.weights_uri:
                weights = PretrainedWeightsManager.get_weights_dict(model_info.weights_uri)
                if weights:
                    focoos_model.load_weights(weights)
                    logger.info(f"✅ Weights loaded for model {pretrained_model_name}")
            else:
                logger.warning(f"⚠️ Model {pretrained_model_name} has no pretrained weights")

            return focoos_model

        except Exception as e:
            raise RuntimeError(f"Error loading model {pretrained_model_name}: {str(e)}")


class AutoConfigBackbone:
    """Automatic backbone configuration manager with lazy loading"""

    _BACKBONE_MAPPING: Dict[str, str] = {
        "resnet": "resnet.ResnetConfig",
        "stdc": "stdc.STDCConfig",
        "swin": "swin.SwinConfig",
        "timm": "timmbackbone.TimmBackboneConfig",
        "mobilenet_v2": "mobilenet_v2.MobileNetV2Config",
        "mit": "mit.MITConfig",
        "convnextv2": "convnextv2.ConvNeXtV2Config",
    }

    @classmethod
    def get_model_class(cls, model_type: str):
        """Get the model class based on the model type"""
        import importlib

        module_path, class_name = cls._BACKBONE_MAPPING[model_type].split(".")
        module = importlib.import_module(f".{module_path}", package="focoos.nn.backbone")
        return getattr(module, class_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> BackboneConfig:
        """Load a backbone from a configuration"""
        if config_dict["model_type"] not in cls._BACKBONE_MAPPING:
            raise ValueError(f"Backbone {config_dict['model_type']} not supported")

        config_class = cls.get_model_class(config_dict["model_type"])

        return config_class(**config_dict)


class AutoBackbone:
    """Automatic backbone manager with lazy loading"""

    _BACKBONE_MAPPING: Dict[str, str] = {
        "resnet": "resnet.ResNet",
        "stdc": "stdc.STDC",
        "swin": "swin.Swin",
        "timm": "timmbackbone.TimmBackbone",
        "mobilenet_v2": "mobilenet_v2.MobileNetV2",
        "mit": "mit.MIT",
        "convnextv2": "convnextv2.ConvNeXtV2",
    }

    @classmethod
    def from_config(cls, config: BackboneConfig) -> BaseBackbone:
        """Load a backbone from a configuration"""
        if config.model_type not in cls._BACKBONE_MAPPING:
            raise ValueError(f"Backbone {config.model_type} not supported")
        backbone_class = cls.get_model_class(config.model_type)
        return backbone_class(config)

    @classmethod
    def get_model_class(cls, model_type: str):
        """Get the model class based on the model type"""
        import importlib

        module_path, class_name = cls._BACKBONE_MAPPING[model_type].split(".")
        module = importlib.import_module(f".{module_path}", package="focoos.nn.backbone")
        return getattr(module, class_name)


class PretrainedWeightsManager:
    """Manager for pretrained model weights"""

    @staticmethod
    def get_weights_dict(weights_uri: str) -> Optional[dict]:
        """
        Load weights for a given model

        Args:
            weights_uri: Model name

        Returns:
            Optional[dict]: Dictionary of weights or None if not available

        Raises:
            ValueError: If the model doesn't exist
        """
        try:
            import torch

            if not os.path.exists(weights_uri):
                raise FileNotFoundError(f"Weights file not found: {weights_uri}")

            # Load weights from file
            state_dict = torch.load(weights_uri, map_location="cpu")

            # If the file contains a dictionary with a 'model' key, extract only that part
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            return state_dict

        except Exception as e:
            logger.error(f"Error loading weights for {weights_uri}: {str(e)}")
            return None

    @staticmethod
    def get_checkpoint_url(model_name: str) -> Optional[str]:
        """Get the checkpoint URL for a given model"""
        model_info = ModelRegistry.get_model_info(model_name)
        if model_info is None:
            logger.warning(f"⚠️ Model {model_name} not found")
            raise ValueError(f"Model {model_name} not found")
        if model_info.weights_uri is None:
            logger.warning(f"⚠️ Model {model_name} has no pretrained weights")
            return None
        return model_info.weights_uri
