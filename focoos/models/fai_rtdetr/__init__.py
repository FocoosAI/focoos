def _register_rtdetr():
    from focoos.auto_model import AutoConfig, AutoModel
    from focoos.ports import ModelFamily

    def load_rtdetr_model():
        # Questa importazione avviene SOLO quando load_rtdetr_model viene chiamata
        from focoos.models.fai_rtdetr.modelling import FAIRTDetr

        return FAIRTDetr

    def load_rtdetr_config():
        from focoos.models.fai_rtdetr.config import RTDetrConfig

        return RTDetrConfig

    # Qui registriamo solo la funzione load_rtdetr_model, NON viene eseguita
    AutoModel.register_model(ModelFamily.RTDETR, load_rtdetr_model)
    AutoConfig.register_model(ModelFamily.RTDETR, load_rtdetr_config)
