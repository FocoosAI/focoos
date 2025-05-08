def _register():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily

    def load_model():
        from focoos.models.fai_detr.modelling import FAIDetr

        return FAIDetr

    def load_config():
        from focoos.models.fai_detr.config import DETRConfig

        return DETRConfig

    ModelManager.register_model(ModelFamily.DETR, load_model)
    ConfigManager.register_config(ModelFamily.DETR, load_config)
