def _register_cls():
    from focoos.auto_model import AutoConfig, AutoModel
    from focoos.ports import ModelFamily

    def load_cls_model():
        # This import happens ONLY when load_cls_model is called
        from focoos.models.fai_cls.modelling import FAIClassification

        return FAIClassification

    def load_cls_config():
        from focoos.models.fai_cls.config import ClassificationConfig

        return ClassificationConfig

    # Register the model and config loaders
    AutoModel.register_model(ModelFamily.IMAGE_CLASSIFIER, load_cls_model)
    AutoConfig.register_model(ModelFamily.IMAGE_CLASSIFIER, load_cls_config)
