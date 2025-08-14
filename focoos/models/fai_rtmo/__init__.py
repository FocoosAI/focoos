def _register():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily
    from focoos.processor import ProcessorManager

    def load_model():
        from focoos.models.rtmo.modelling import RTMO

        return RTMO

    def load_config():
        from focoos.models.rtmo.config import RTMOConfig

        return RTMOConfig

    def load_processor():
        from focoos.models.rtmo.processor import RTMOProcessor

        return RTMOProcessor

    ModelManager.register_model(ModelFamily.RTMO, load_model)
    ConfigManager.register_config(ModelFamily.RTMO, load_config)
    ProcessorManager.register_processor(ModelFamily.RTMO, load_processor)
