from typing import Dict, Optional

from focoos.data.class_names import coco_classes, object365_classes
from focoos.models.fai_rtdetr.config_rtdetr_resnet import RTDetrResnetConfig
from focoos.models.fai_rtdetr.config_rtdetr_stdc import RTDetrStdCConfig
from focoos.ports import ModelFamily, ModelInfo, Task
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


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

    _pretrained_models: Dict[str, ModelInfo] = {
        "fai-rtdetr-l-obj365": ModelInfo(
            name="fai-rtdetr-l-obj365",
            description="RTDETR Large model (Object365)",
            model_family=ModelFamily.RTDETR,
            config_class=RTDetrResnetConfig,
            weights_uri="/home/ubuntu/anyma/pretrained_models/models/fai-rtdetr-m-obj365/model_final.pth",
            config=RTDetrResnetConfig(num_classes=365),
            task=Task.DETECTION,
            classes=object365_classes,
            val_dataset="object365",
            im_size=640,
        ),
        "fai-rtdetr-l-coco": ModelInfo(
            name="rtdetr-l-coco",
            description="RTDETR Large model (COCO)",
            model_family=ModelFamily.RTDETR,
            config_class=RTDetrResnetConfig,
            weights_uri="/home/ubuntu/anyma/pretrained_models/models/fai-rtdetr-l-coco/model_final.pth",
            config=RTDetrResnetConfig(num_classes=80),
            task=Task.DETECTION,
            classes=coco_classes,
            val_dataset="coco",
            im_size=640,
        ),
        "fai-rtdetr-m-coco": ModelInfo(
            name="rtdetr-m-coco",
            description="RTDETR Medium model (COCO)",
            model_family=ModelFamily.RTDETR,
            config_class=RTDetrStdCConfig,
            weights_uri="/home/ubuntu/anyma/pretrained_models/models/fai-rtdetr-m-coco/model_final.pth",
            config=RTDetrStdCConfig(num_classes=80),
            task=Task.DETECTION,
            classes=coco_classes,
            val_dataset="coco",
            im_size=640,
        ),
    }

    _user_models: Dict[str, ModelInfo] = {}

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelInfo]:
        """Ottiene le informazioni per un dato modello"""
        return cls._pretrained_models.get(model_name)

    @classmethod
    def list_models(cls, model_family: Optional[str] = None) -> list[str]:
        """Lista tutti i modelli disponibili, opzionalmente filtrati per famiglia"""
        if model_family is None:
            return list(cls._pretrained_models.keys())
        return [name for name, info in cls._pretrained_models.items() if info.model_family == model_family]

    @classmethod
    def print_model_details(cls, model_name: str):
        """Stampa i dettagli di un modello in formato leggibile"""
        info = cls.get_model_info(model_name)
        if info is None:
            logger.warning(f"⚠️ Model {model_name} not found")
            return

        logger.info(f"""
📋 Name: {info.name}
📝 Description: {info.description}
👪 Family: {info.model_family}
🎯 Task: {info.task}
🏷️ Classes: {info.classes}
🖼️ Im size: {info.im_size}
""")
