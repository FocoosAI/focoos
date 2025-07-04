def _register():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily
    from focoos.processor import ProcessorManager

    def load_model():
        from focoos.models.eomt.modelling import EoMT

        return EoMT

    def load_config():
        from focoos.models.eomt.config import EoMTConfig

        return EoMTConfig

    def load_processor():
        from focoos.models.eomt.processor import EoMTProcessor

        return EoMTProcessor

    ModelManager.register_model(ModelFamily.EOMT, load_model)
    ConfigManager.register_config(ModelFamily.EOMT, load_config)
    ProcessorManager.register_processor(ModelFamily.EOMT, load_processor)
