import importlib
import os
from dataclasses import fields
from pathlib import Path
from typing import Callable, Dict, Optional, Type
from urllib.parse import urlparse

from focoos.hub.api_client import ApiClient
from focoos.model_registry.model_registry import ModelRegistry
from focoos.models.fai_model import BaseModelNN, FocoosModel, ModelConfig
from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.ports import MODELS_DIR, ModelFamily, ModelInfo
from focoos.utils.logger import get_logger

logger = get_logger("ModelManager")


class ModelManager:
    """Automatic model manager with lazy loading (refactored)"""

    _MODEL_MAPPING: Dict[str, Callable[[], Type[BaseModelNN]]] = {}
    _REGISTERED_MODELS: set = set()

    @classmethod
    def register_model(cls, model_family: ModelFamily, model_loader: Callable[[], Type[BaseModelNN]]):
        """
        Register a loader for a specific model
        """
        cls._MODEL_MAPPING[model_family.value] = model_loader
        cls._REGISTERED_MODELS.add(model_family.value)

    @classmethod
    def _ensure_family_registered(cls, model_family: ModelFamily):
        """Ensure the model family is registered, importing if needed."""
        if model_family not in cls._REGISTERED_MODELS:
            family_module = importlib.import_module(f"focoos.models.{model_family.value}")
            for attr_name in dir(family_module):
                if attr_name.startswith("_register"):
                    register_func = getattr(family_module, attr_name)
                    if callable(register_func):
                        register_func()

    @classmethod
    def _from_model_info(cls, model_info: ModelInfo, config: Optional[ModelConfig] = None, **kwargs) -> FocoosModel:
        """Load a model from ModelInfo, handling config and weights."""
        cls._ensure_family_registered(model_info.model_family)
        if model_info.model_family.value not in cls._MODEL_MAPPING:
            raise ValueError(f"Model {model_info.model_family} not supported")
        model_class = cls._MODEL_MAPPING[model_info.model_family.value]()
        if config is None:
            config = ConfigManager.from_dict(model_info.model_family, model_info.config, **kwargs)
        model_info.config = config
        nn_model = model_class(model_info.config)
        model = FocoosModel(nn_model, model_info)
        if model_info.weights_uri:
            weights = ArtifactsManager.get_weights_dict(model_info)
            if weights:
                model.load_weights(weights)
                logger.info(f"✅ Weights loaded for model {model_info.name}")
        else:
            logger.warning(f"⚠️ Model {model_info.name} has no pretrained weights")
        return model

    @classmethod
    def _from_local_dir(
        cls, name: str, models_dir: Optional[str] = None, config: Optional[ModelConfig] = None, **kwargs
    ) -> FocoosModel:
        """Load a model from a local experiment directory."""
        if models_dir is None:
            models_dir = MODELS_DIR
        run_dir = os.path.join(models_dir, name)
        if not os.path.exists(run_dir):
            raise ValueError(f"Run {name} not found in {models_dir}")
        model_info_path = os.path.join(run_dir, "model_info.json")
        if not os.path.exists(model_info_path):
            raise ValueError(f"Model info not found in {run_dir}")
        model_info = ModelInfo.from_json(model_info_path)
        if model_info.weights_uri == "model_final.pth":
            model_info.weights_uri = os.path.join(run_dir, model_info.weights_uri)
        return cls._from_model_info(model_info, config=config, **kwargs)

    @classmethod
    def _from_hub(cls, name: str, api_key: Optional[str] = None, **kwargs) -> FocoosModel:
        # TODO: implement hub loading logic
        raise NotImplementedError("Hub loading is not implemented yet.")

    @classmethod
    def get(
        cls,
        name: str,
        model_info: Optional[ModelInfo] = None,
        config: Optional[ModelConfig] = None,
        models_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> FocoosModel:
        """
        Unified entrypoint to load a model by name or ModelInfo.
        """
        if model_info is not None:
            return cls._from_model_info(model_info, config=config, **kwargs)
        if name.startswith("hub://"):
            return cls._from_hub(name, api_key=api_key, **kwargs)
        if ModelRegistry.exists(name):
            model_info = ModelRegistry.get_model_info(name)
            return cls._from_model_info(model_info, config=config, **kwargs)
        return cls._from_local_dir(name, models_dir=models_dir, config=config, **kwargs)


class BackboneManager:
    """Automatic backbone manager with lazy loading"""

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


class ConfigManager:
    """Automatic model configuration management"""

    _MODEL_MAPPING: Dict[str, Callable[[], Type[ModelConfig]]] = {}
    _REGISTERED_MODELS: set = set()

    @classmethod
    def register_config(cls, model_family: ModelFamily, model_config_loader: Callable[[], Type[ModelConfig]]):
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
            # Import the family module
            family_module = importlib.import_module(f"focoos.models.{model_family.value}")

            # Iteratively register all models in the family
            for attr_name in dir(family_module):
                if attr_name.startswith("_register"):
                    register_func = getattr(family_module, attr_name)
                    if callable(register_func):
                        register_func()

        if model_family not in cls._MODEL_MAPPING:
            raise ValueError(f"Model {model_family} not supported")

        config_class = cls._MODEL_MAPPING[model_family.value]()  # this return the config class

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
        config_dict.update(kwargs)

        return config_dict


class ConfigBackboneManager:
    """Automatic backbone configuration manager with lazy loading"""

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
        return_config = config_class(**config_dict)
        return return_config


class ArtifactsManager:
    @staticmethod
    def get_weights_dict(model_info: ModelInfo) -> Optional[dict]:
        """
        Load weights for a given model

        Args:
            model_info: ModelInfo

        Returns:
            Optional[dict]: Dictionary of weights or None if not available

        Raises:
            ValueError: If the model doesn't exist
        """
        if not model_info.weights_uri:
            logger.warning(f"⚠️ Model {model_info.name} has no pretrained weights")
            return None

        # Determine if weights are remote or local
        parsed_uri = urlparse(model_info.weights_uri)
        is_remote = bool(parsed_uri.scheme and parsed_uri.netloc)

        # Get weights path
        if is_remote:
            logger.info(f"Downloading weights from remote URL: {model_info.weights_uri}")
            model_dir = Path(MODELS_DIR) / model_info.name
            weights_path = ApiClient().download_ext_file(model_info.weights_uri, str(model_dir), skip_if_exists=True)
        else:
            logger.info(f"Using weights from local path: {model_info.weights_uri}")
            weights_path = model_info.weights_uri

        try:
            import torch

            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            # Load weights and extract model state if needed
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            return state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict

        except Exception as e:
            logger.error(f"Error loading weights for {model_info.name}: {str(e)}")
            return None
