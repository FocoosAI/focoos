from typing import Dict, Optional

from focoos.ports import ModelInfo
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

    _pretrained_models: Dict[str, str] = {
        "fai-rtdetr-l-obj365": "../model_registry/fai-rtdetr-l-obj365.json",
        "fai-rtdetr-l-coco": "../model_registry/fai-rtdetr-l-coco.json",
        "fai-rtdetr-m-coco": "../model_registry/fai-rtdetr-m-coco.json",
        "fai-rtdetr-s-coco": "../model_registry/fai-rtdetr-s-coco.json",
    }
    # FIXME: the weights uri for the models is wrong and should be an s3 path or similar
    # _pretrained_models: Dict[str, ModelInfo] = {
    #     "fai-rtdetr-l-obj365": ModelInfo(
    #         name="fai-rtdetr-l-obj365",
    #         description="RTDETR Large model (Object365)",
    #         model_family=ModelFamily.RTDETR,
    #         weights_uri="/home/ubuntu/anyma/pretrained_models/models/fai-rtdetr-m-obj365/model_final.pth",
    #         config=RTDetrConfig(
    #             backbone_config=PResnetConfig(
    #                 depth=50,
    #                 variant="d",
    #                 freeze_at=-1,
    #                 num_stages=4,
    #                 freeze_norm=False,
    #             ),
    #             pixel_decoder_out_dim=256,
    #             pixel_decoder_feat_dim=256,
    #             pixel_decoder_num_encoder_layers=1,
    #             pixel_decoder_dim_feedforward=1024,
    #             transformer_predictor_out_dim=256,
    #             transformer_predictor_hidden_dim=256,
    #             transformer_predictor_dec_layers=6,
    #             transformer_predictor_dim_feedforward=1024,
    #             head_out_dim=256,
    #             num_queries=300,
    #             num_classes=365,
    #         ),
    #         task=Task.DETECTION,
    #         classes=object365_classes,
    #         val_dataset="object365",
    #         im_size=640,
    #     ),
    #     "fai-rtdetr-l-coco": ModelInfo(
    #         name="rtdetr-l-coco",
    #         description="RTDETR Large model (COCO)",
    #         model_family=ModelFamily.RTDETR,
    #         weights_uri="/home/ubuntu/anyma/pretrained_models/models/fai-rtdetr-l-coco/model_final.pth",
    #         config=RTDetrConfig(
    #             backbone_config=PResnetConfig(
    #                 depth=50,
    #                 variant="d",
    #                 freeze_at=-1,
    #                 num_stages=4,
    #                 freeze_norm=False,
    #             ),
    #             pixel_decoder_out_dim=256,
    #             pixel_decoder_feat_dim=256,
    #             pixel_decoder_num_encoder_layers=1,
    #             pixel_decoder_dim_feedforward=1024,
    #             transformer_predictor_out_dim=256,
    #             transformer_predictor_hidden_dim=256,
    #             transformer_predictor_dec_layers=6,
    #             transformer_predictor_dim_feedforward=1024,
    #             head_out_dim=256,
    #             num_queries=300,
    #             num_classes=80,
    #         ),
    #         task=Task.DETECTION,
    #         classes=coco_classes,
    #         val_dataset="coco",
    #         im_size=640,
    #     ),
    #     "fai-rtdetr-m-coco": ModelInfo(
    #         name="rtdetr-m-coco",
    #         description="RTDETR Medium model (COCO)",
    #         model_family=ModelFamily.RTDETR,
    #         weights_uri="/home/ubuntu/anyma/pretrained_models/models/fai-rtdetr-m-coco/model_final.pth",
    #         config=RTDetrConfig(
    #             backbone_config=STDCConfig(
    #                 base=64,
    #                 layers=[4, 5, 3],  # STDC-2
    #             ),
    #             pixel_decoder_out_dim=128,
    #             pixel_decoder_feat_dim=128,
    #             pixel_decoder_num_encoder_layers=0,
    #             pixel_decoder_dim_feedforward=1024,
    #             transformer_predictor_out_dim=128,
    #             transformer_predictor_hidden_dim=256,
    #             transformer_predictor_dec_layers=3,
    #             transformer_predictor_dim_feedforward=1024,
    #             head_out_dim=128,
    #             num_queries=300,
    #             num_classes=80,
    #         ),
    #         task=Task.DETECTION,
    #         classes=coco_classes,
    #         val_dataset="coco",
    #         im_size=640,
    #     ),
    #     "fai-rtdetr-s-coco": ModelInfo(
    #         name="rtdetr-s-coco",
    #         description="RTDETR Small model (COCO)",
    #         model_family=ModelFamily.RTDETR,
    #         weights_uri="/home/ubuntu/anyma/pretrained_models/models/fai-rtdetr-s-coco/model_final.pth",
    #         config=RTDetrConfig(
    #             backbone_config=STDCConfig(
    #                 base=64,
    #                 layers=[4, 5, 3],  # STDC-2
    #             ),
    #             pixel_decoder_out_dim=128,
    #             pixel_decoder_feat_dim=128,
    #             pixel_decoder_expansion=0.5,
    #             pixel_decoder_num_encoder_layers=0,
    #             pixel_decoder_dim_feedforward=512,
    #             transformer_predictor_out_dim=128,
    #             transformer_predictor_hidden_dim=128,
    #             transformer_predictor_dec_layers=3,
    #             transformer_predictor_dim_feedforward=512,
    #             head_out_dim=128,
    #             num_queries=300,
    #             num_classes=80,
    #         ),
    #         task=Task.DETECTION,
    #         classes=coco_classes,
    #         val_dataset="coco",
    #         im_size=640,
    #     ),
    # }

    # _user_models: Dict[str, ModelInfo] = {}

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelInfo]:
        """Get the model information for a given model name"""
        if model_name in cls._pretrained_models:
            return ModelInfo.from_json(cls._pretrained_models[model_name])
        return ModelInfo.from_json(model_name)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available models"""
        return list(cls._pretrained_models.keys())

    @classmethod
    def print_model_details(cls, model_name: str):
        """Print the details of a model in a readable format"""
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
