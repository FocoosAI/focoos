import os
from pathlib import Path
from typing import Dict, List, Optional

from focoos.ports import ModelInfo
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Central registry of pretrained models.

    This class serves as a centralized registry for all pretrained models in the Focoos system.
    It provides methods to access model information, list available models, and check model existence.

    Attributes:
        _registry_path (Path): Path to the directory containing model JSON files.
        _pretrained_models (Dict[str, str]): Dictionary mapping model names to their JSON file paths.
    """

    _registry_path = Path(__file__).parent
    _pretrained_models: Optional[Dict[str, str]] = None

    @classmethod
    def _load_models_cfgs(cls) -> Dict[str, str]:
        """Load model configurations from JSON files.

        Returns:
            Dict[str, str]: Dictionary mapping model names to their JSON file paths.
        """
        if cls._pretrained_models is not None:
            return cls._pretrained_models

        models = {}
        try:
            json_files = [f for f in cls._registry_path.iterdir() if f.is_file() and f.suffix == ".json"]

            for json_file in json_files:
                model_name = json_file.stem  # Remove .json extension
                models[model_name] = str(json_file)
        except OSError as e:
            logger.error(f"Failed to load model configurations: {e}")
            models = {}

        cls._pretrained_models = models
        return models

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo:
        """Get the model information for a given model name.

        Args:
            model_name (str): The name of the model to retrieve information for.
                Can be either a pretrained model name or a path to a JSON file.

        Returns:
            ModelInfo: The model information object containing model details.

        Raises:
            ValueError: If the model is not found in the registry and the provided
                path does not exist.
        """
        models = cls._load_models_cfgs()

        if model_name in models:
            return ModelInfo.from_json(models[model_name])
        if not os.path.exists(model_name):
            logger.warning(f"⚠️ Model {model_name} not found")
            raise ValueError(f"⚠️ Model {model_name} not found")
        return ModelInfo.from_json(model_name)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all available pretrained models.

        Returns:
            List[str]: A list of all available pretrained model names.
        """
        models = cls._load_models_cfgs()
        return list(models.keys())

    @classmethod
    def exists(cls, model_name: str) -> bool:
        """Check if a model exists in the registry.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model exists in the pretrained models registry,
                False otherwise.
        """
        models = cls._load_models_cfgs()
        exists = model_name in models
        if not exists:
            logger.debug(f"Model '{model_name}' not found in registry")
        return exists


if __name__ == "__main__":
    print("Available models:", ModelRegistry.list_models())
    try:
        model_info = ModelRegistry.get_model_info("fai-detr-l-obj365")
        print(f"Model info: {model_info}")
    except ValueError as e:
        print(f"Error: {e}")
