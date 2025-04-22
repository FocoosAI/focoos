def _register_rtdetr_resnet():
    from focoos.auto_model import AutoModel

    def load_rtdetr_resnet_model():
        # Questa importazione avviene SOLO quando load_rtdetr_model viene chiamata
        from focoos.models.fai_rtdetr.modelling import FAIRTDetr

        return FAIRTDetr

    # Qui registriamo solo la funzione load_rtdetr_model, NON viene eseguita
    AutoModel.register_model("fai-rtdetr-l-coco", load_rtdetr_resnet_model)
    AutoModel.register_model("fai-rtdetr-l-obj365", load_rtdetr_resnet_model)


def _register_rtdetr_stdc():
    from focoos.auto_model import AutoModel

    def load_rtdetr_stdc_model():
        from focoos.models.fai_rtdetr.modelling import FAIRTDetr

        return FAIRTDetr

    AutoModel.register_model("fai-rtdetr-m-coco", load_rtdetr_stdc_model)
