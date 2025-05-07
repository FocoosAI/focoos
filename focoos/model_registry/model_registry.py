import os
from typing import Dict

from focoos.ports import MODELS_DIR, ModelInfo
from focoos.utils.logger import get_logger

logger = get_logger(__name__)

REGISTRY_PATH = os.path.dirname(__file__)


class ModelRegistry:
    """Central registry of pretrained models

    This class serves as a centralized registry for all pretrained models in the Focoos system.
    It provides methods to access model information, list available models, and display model details.

    Attributes:
        _pretrained_models (Dict[str, ModelInfo]): Dictionary of pretrained models with model name as key
        _user_models (Dict[str, ModelInfo]): Dictionary of user-defined models with model name as key

    Methods:
        get_model_info: Retrieves model information by name
        list_models: Lists all available models, optionally filtered by model family
        print_model_details: Displays detailed information about a specific model
    """

    _pretrained_models: Dict[str, str] = {
        "fai-detr-l-obj365": os.path.join(REGISTRY_PATH, "fai-detr-l-obj365.json"),
        "fai-detr-l-coco": os.path.join(REGISTRY_PATH, "fai-detr-l-coco.json"),
        "fai-detr-m-coco": os.path.join(REGISTRY_PATH, "fai-detr-m-coco.json"),
        "fai-detr-s-coco": os.path.join(REGISTRY_PATH, "fai-detr-s-coco.json"),
        "fai-detr-n-coco": os.path.join(REGISTRY_PATH, "fai-detr-n-coco.json"),
        "fai-mf-l-ade": os.path.join(REGISTRY_PATH, "fai-mf-l-ade.json"),
        "fai-mf-m-ade": os.path.join(REGISTRY_PATH, "fai-mf-m-ade.json"),
        "fai-mf-l-coco-ins": os.path.join(REGISTRY_PATH, "fai-mf-l-coco-ins.json"),
        "fai-mf-m-coco-ins": os.path.join(REGISTRY_PATH, "fai-mf-m-coco-ins.json"),
        "fai-mf-s-coco-ins": os.path.join(REGISTRY_PATH, "fai-mf-s-coco-ins.json"),
    }

    _user_models: Dict[str, ModelInfo] = {}

    def __init__(self):
        self._load_user_models()

    def _load_user_models(self):
        dir_list = [d for d in os.listdir(MODELS_DIR) if not d.startswith("fai-")]
        for model_ref in dir_list:
            info_keys = ["focoos_metadata.json", "model_info.json"]
            info_files = [os.path.join(MODELS_DIR, model_ref, info_key) for info_key in info_keys]
            for info_file in info_files:
                if os.path.exists(info_file):
                    self._user_models[model_ref] = ModelInfo.from_json(info_file)

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo:
        """Get the model information for a given model name"""
        if model_name in cls._pretrained_models:
            return ModelInfo.from_json(cls._pretrained_models[model_name])
        if not os.path.exists(model_name):
            logger.warning(f"⚠️ Model {model_name} not found")
            raise ValueError(f"⚠️ Model {model_name} not found")
        return ModelInfo.from_json(model_name)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available models"""
        return list(cls._pretrained_models.keys()) + list(cls._user_models.keys())
