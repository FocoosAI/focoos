def _register():
    from focoos.model_manager import ConfigManager, ModelManager
    from focoos.ports import ModelFamily
    from focoos.processor import ProcessorManager

    def load_model():
        # Questa importazione avviene SOLO quando load_rtdetr_model viene chiamata
        from focoos.models.bisenetformer.modelling import BisenetFormer

        return BisenetFormer

    def load_config():
        from focoos.models.bisenetformer.config import BisenetFormerConfig

        return BisenetFormerConfig

    def load_processor():
        from focoos.models.bisenetformer.processor import BisenetFormerProcessor

        return BisenetFormerProcessor

    # Qui registriamo solo la funzione load_rtdetr_model, NON viene eseguita
    ModelManager.register_model(ModelFamily.BISENETFORMER, load_model)
    ConfigManager.register_config(ModelFamily.BISENETFORMER, load_config)
    ProcessorManager.register_processor(ModelFamily.BISENETFORMER, load_processor)
