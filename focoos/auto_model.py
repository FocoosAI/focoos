import importlib
import os
from dataclasses import fields
from typing import Callable, Dict, Optional, Type

from focoos.model_registry import ModelRegistry
from focoos.models.fai_model import BaseModelNN, ModelConfig
from focoos.nn.backbone.base import BackboneConfig, BaseBackbone
from focoos.ports import ModelFamily
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class AutoConfig:
    """Automatic model configuration management"""

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, **kwargs) -> ModelConfig:
        """
        Create a configuration for a pretrained model

        Args:
            pretrained_model_name: Name of the pretrained model
            **kwargs: Configuration parameters to override

        Returns:
            ModelConfig: Model configuration

        Raises:
            ValueError: If the model doesn't exist or the parameters are invalid
        """
        model_info = ModelRegistry.get_model_info(pretrained_model_name)
        if model_info is None:
            raise ValueError(
                f"Model {pretrained_model_name} not supported. Available models: {ModelRegistry.list_models()}"
            )

        base_config = model_info.config

        # Validazione dei parametri
        valid_fields = {f.name for f in fields(model_info.config_class)}
        invalid_kwargs = set(kwargs.keys()) - valid_fields
        if invalid_kwargs:
            raise ValueError(
                f"Invalid parameters for {model_info.config_class.__name__}: {invalid_kwargs}"
                f"\nValid parameters: {valid_fields}"
            )

        # Creazione configurazione
        config_dict = {field.name: getattr(base_config, field.name) for field in fields(base_config)}
        config_dict.update(kwargs)

        return model_info.config_class(**config_dict)


class AutoModel:
    """Automatic model manager with lazy loading"""

    _MODEL_MAPPING: Dict[str, Callable[[], Type[BaseModelNN]]] = {}
    _REGISTERED_MODELS: set = set()

    @classmethod
    def register_model(cls, model_name: str, model_loader: Callable[[], Type[BaseModelNN]]):
        """
        Register a loader for a specific model

        Args:
            model_name: Model name
            model_loader: Function that loads the model
        """
        cls._MODEL_MAPPING[model_name] = model_loader
        cls._REGISTERED_MODELS.add(model_name)

    @classmethod
    def _import_model_family(cls, model_family: ModelFamily):
        """Dynamically import a model family"""
        try:
            module_name = f"focoos.models.{model_family}"
            importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Unable to import model family {model_family}. Error: {str(e)}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name: str, config: Optional[ModelConfig] = None, **kwargs) -> BaseModelNN:
        """
        Load a pretrained model

        Args:
            pretrained_model_name: Name of the pretrained model
            config: Model configuration (optional)
            **kwargs: Configuration parameters to override

        Returns:
            BaseNNModel: Loaded model

        Raises:
            ValueError: If the model doesn't exist or is not supported
            RuntimeError: If there are errors during loading
        """
        model_info = ModelRegistry.get_model_info(pretrained_model_name)
        if model_info is None:
            raise ValueError(f"Model {pretrained_model_name} not found")

        # Import the family module only if not already registered
        if pretrained_model_name not in cls._REGISTERED_MODELS:
            # Import the family module
            family_module = importlib.import_module(f"focoos.models.{model_info.model_family.value}")

            # Iteratively register all models in the family
            for attr_name in dir(family_module):
                if attr_name.startswith("_register_"):
                    register_func = getattr(family_module, attr_name)
                    if callable(register_func):
                        register_func()

        if pretrained_model_name not in cls._MODEL_MAPPING:
            raise ValueError(f"Model {pretrained_model_name} not supported")

        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name, **kwargs)

        try:
            model_class = cls._MODEL_MAPPING[pretrained_model_name]()
            model = model_class(config, model_info)

            if pretrained_model_name:
                weights = PretrainedWeightsManager.get_weights_dict(pretrained_model_name)
                if weights:
                    model.load_weights(weights)
                    logger.info(f"✅ Weights loaded for model {pretrained_model_name}")
            else:
                logger.warning(f"⚠️ Model {pretrained_model_name} has no pretrained weights")

            return model

        except Exception as e:
            raise RuntimeError(f"Error loading model {pretrained_model_name}: {str(e)}")


class AutoBackbone:
    """Automatic backbone manager with lazy loading"""

    _BACKBONE_MAPPING: Dict[str, str] = {
        "resnet": "presnet.PResNet",
        "stdc": "stdc.STDC",
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
    def get_weights_dict(model_name: str) -> Optional[dict]:
        """
        Load weights for a given model

        Args:
            model_name: Model name

        Returns:
            Optional[dict]: Dictionary of weights or None if not available

        Raises:
            ValueError: If the model doesn't exist
        """
        model_info = ModelRegistry.get_model_info(model_name)
        if model_info is None:
            raise ValueError(f"Model {model_name} not found")

        try:
            import torch

            if model_info.weights_uri is None:
                logger.warning(f"⚠️ Model {model_name} has no pretrained weights")
                return None
            weights_path = model_info.weights_uri
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            # Load weights from file
            state_dict = torch.load(weights_path, map_location="cpu")

            # If the file contains a dictionary with a 'model' key, extract only that part
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            return state_dict

        except Exception as e:
            logger.error(f"Error loading weights for {model_name}: {str(e)}")
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
