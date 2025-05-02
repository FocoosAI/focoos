def _register():
    from focoos.auto_model import AutoConfig, AutoModel
    from focoos.ports import ModelFamily

    def load_model():
        from focoos.models.fai_detr.modelling import FAIDetr

        return FAIDetr

    def load_config():
        from focoos.models.fai_detr.config import DETRConfig

        return DETRConfig

    AutoModel.register_model(ModelFamily.DETR, load_model)
    AutoConfig.register_model(ModelFamily.DETR, load_config)
