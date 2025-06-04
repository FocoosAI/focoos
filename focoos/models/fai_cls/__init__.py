def _register_cls():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily
    from focoos.processor import ProcessorManager

    def load_classifier_model():
        # This import happens ONLY when load_cls_model is called
        from focoos.models.fai_cls.modelling import FAIClassification

        return FAIClassification

    def load_classifier_config():
        from focoos.models.fai_cls.config import ClassificationConfig

        return ClassificationConfig

    def load_classifier_processor():
        from focoos.models.fai_cls.processor import ClassificationProcessor

        return ClassificationProcessor

    # Register the model and config loaders
    ModelManager.register_model(ModelFamily.IMAGE_CLASSIFIER, load_classifier_model)
    ConfigManager.register_config(ModelFamily.IMAGE_CLASSIFIER, load_classifier_config)
    ProcessorManager.register_processor(ModelFamily.IMAGE_CLASSIFIER, load_classifier_processor)
