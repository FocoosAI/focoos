def _register():
    from focoos.auto_model import AutoConfig, AutoModel
    from focoos.ports import ModelFamily

    def load_model():
        # Questa importazione avviene SOLO quando load_rtdetr_model viene chiamata
        from focoos.models.fai_mf.modelling import FAIMaskFormer

        return FAIMaskFormer

    def load_config():
        from focoos.models.fai_mf.config import MaskFormerConfig

        return MaskFormerConfig

    # Qui registriamo solo la funzione load_rtdetr_model, NON viene eseguita
    AutoModel.register_model(ModelFamily.M2F, load_model)
    AutoConfig.register_model(ModelFamily.M2F, load_config)
