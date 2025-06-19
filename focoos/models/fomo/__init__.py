def _register_cls():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily
    from focoos.processor import ProcessorManager

    def load_model():
        from focoos.models.fomo.modelling import FOMO

        return FOMO

    def load_config():
        from focoos.models.fomo.config import FOMOConfig

        return FOMOConfig

    def load_processor():
        from focoos.models.fomo.processor import FOMOProcessor

        return FOMOProcessor

    # Register the model and config loaders
    ModelManager.register_model(ModelFamily.FOMO, load_model)
    ConfigManager.register_config(ModelFamily.FOMO, load_config)
    ProcessorManager.register_processor(ModelFamily.FOMO, load_processor)