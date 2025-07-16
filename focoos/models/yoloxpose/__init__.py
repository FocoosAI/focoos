def _register():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily
    from focoos.processor import ProcessorManager

    def load_model():
        from focoos.models.yoloxpose.modelling import YOLOXPose

        return YOLOXPose

    def load_config():
        from focoos.models.yoloxpose.config import YOLOXPoseConfig

        return YOLOXPoseConfig

    def load_processor():
        from focoos.models.yoloxpose.processor import YOLOXPoseProcessor

        return YOLOXPoseProcessor

    ModelManager.register_model(ModelFamily.YOLOXPOSE, load_model)
    ConfigManager.register_config(ModelFamily.YOLOXPOSE, load_config)
    ProcessorManager.register_processor(ModelFamily.YOLOXPOSE, load_processor)
