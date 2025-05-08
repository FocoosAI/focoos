def _register():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily

    def load_model():
        # Questa importazione avviene SOLO quando load_rtdetr_model viene chiamata
        from focoos.models.fai_mf.modelling import FAIMaskFormer

        return FAIMaskFormer

    def load_config():
        from focoos.models.fai_mf.config import MaskFormerConfig

        return MaskFormerConfig

    # Qui registriamo solo la funzione load_rtdetr_model, NON viene eseguita
    ModelManager.register_model(ModelFamily.MASKFORMER, load_model)
    ConfigManager.register_config(ModelFamily.MASKFORMER, load_config)
