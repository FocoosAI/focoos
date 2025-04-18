from torch import nn

from focoos.ports import ModelConfig, ModelInfo
from focoos.structures import Instances
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModelNN(nn.Module):
    def __init__(self, config: ModelConfig, model_info: ModelInfo):
        super().__init__()
        self.model_info = model_info

    def forward(self, x):
        pass

    def load_weights(self, weights: dict):
        checkpoint_state_dict = weights
        model_state_dict = self.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        incompatible = self.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            logger.warning(f"Missing keys in checkpoint: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")
        logger.info("Loaded weights!")
        return len(incompatible.missing_keys) + len(incompatible.unexpected_keys)

    def post_process(self, outputs, batched_inputs) -> list[Instances]:
        raise NotImplementedError("Post-processing is not implemented for this model.")

    def predict(self, batched_inputs) -> list[Instances]:
        raise NotImplementedError("Prediction is not implemented for this model.")


class FocoosModel:
    def __init__(
        self,
        model: BaseModelNN,
    ):
        self.model = model

    def forward(self, x):
        pass

    def predict(self, x):
        pass

    def train(self, x):
        pass
