def _register():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily
    from focoos.processor import ProcessorManager

    def load_model():
        from focoos.models.fai_detr.modelling import FAIDetr

        return FAIDetr

    def load_config():
        from focoos.models.fai_detr.config import DETRConfig

        return DETRConfig

    def load_processor():
        from focoos.models.fai_detr.processor import DETRProcessor

        return DETRProcessor

    ModelManager.register_model(ModelFamily.DETR, load_model)
    ConfigManager.register_config(ModelFamily.DETR, load_config)
    ProcessorManager.register_processor(ModelFamily.DETR, load_processor)
