def _register_rtdetr():
    from focoos.auto_model import AutoModel
    from focoos.ports import ModelFamily

    def load_rtdetr_model():
        # Questa importazione avviene SOLO quando load_rtdetr_model viene chiamata
        from focoos.models.fai_rtdetr.modelling import FAIRTDetr

        return FAIRTDetr

    # Qui registriamo solo la funzione load_rtdetr_model, NON viene eseguita
    AutoModel.register_model(ModelFamily.RTDETR, load_rtdetr_model)
