from typing import Any, Literal, Union

import numpy as np
import torch
from PIL import Image

from focoos.ports import DatasetEntry, FocoosDetections, ModelConfig, ModelOutput


class BaseProcessor:
    def __init__(self, config: ModelConfig):
        self.config = config

    def preprocess(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
        training: bool = False,
        device: Literal["cuda", "cpu"] = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, Any]:
        raise NotImplementedError("Pre-processing is not implemented for this model.")

    def postprocess(
        self,
        outputs: ModelOutput,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
        class_names: list[str] = [],
        **kwargs,
    ) -> list[FocoosDetections]:
        raise NotImplementedError("Post-processing is not implemented for this model.")

    def eval_post_process(self, outputs: ModelOutput, inputs: list[DatasetEntry]):
        raise NotImplementedError("Post-processing is not implemented for this model.")

    def get_image_sizes(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
    ):
        image_sizes = []

        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            # Single tensor/array input
            if isinstance(inputs, torch.Tensor):
                height, width = inputs.shape[-2:]
            else:  # numpy array
                height, width = inputs.shape[-3:-1] if inputs.ndim > 3 else inputs.shape[:2]
            image_sizes.append((height, width))
        elif isinstance(inputs, Image.Image):
            # Single PIL image
            width, height = inputs.size
            image_sizes.append((height, width))
        elif isinstance(inputs, list):
            # List of inputs
            for img in inputs:
                if isinstance(img, torch.Tensor):
                    height, width = img.shape[-2:]
                elif isinstance(img, np.ndarray):
                    height, width = img.shape[-3:-1] if img.ndim > 3 else img.shape[:2]
                elif isinstance(img, Image.Image):
                    width, height = img.size
                else:
                    raise ValueError(f"Unsupported input type in list: {type(img)}")
                image_sizes.append((height, width))
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        return image_sizes

    def get_tensors(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(inputs, (Image.Image, np.ndarray, torch.Tensor)):
            inputs_list = [inputs]
        else:
            inputs_list = inputs

        # Process each input based on its type
        processed_inputs = []
        for inp in inputs_list:
            # todo check for tensor of 4 dimesions.
            if isinstance(inp, Image.Image):
                inp = np.array(inp)
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp)

            # Ensure input has correct shape and type
            if inp.dim() == 3:  # Add batch dimension if missing
                inp = inp.unsqueeze(0)
            if inp.shape[1] != 3 and inp.shape[-1] == 3:  # Convert HWC to CHW if needed
                inp = inp.permute(0, 3, 1, 2)

            processed_inputs.append(inp)

        # Stack all inputs into a single batch tensor
        # use pixel mean to get dtype -> If fp16, pixel_mean is fp16, so inputs will be fp16
        # TODO: this will break with different image sizes
        images_torch = torch.cat(processed_inputs, dim=0)

        return images_torch

    def tensors_to_model_output(self, tensors: Union[list[np.ndarray], list[torch.Tensor]]) -> ModelOutput:
        raise NotImplementedError("Tensors to model output is not implemented for this model.")
