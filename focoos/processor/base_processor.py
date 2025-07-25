from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.ports import DatasetEntry, DynamicAxes, FocoosDetections, ModelConfig, ModelOutput


class Processor(ABC):
    """Abstract base class for model processors that handle preprocessing and postprocessing.

    This class defines the interface for processing inputs and outputs for different model types.
    Subclasses must implement the abstract methods to provide model-specific processing logic.

    Attributes:
        config (ModelConfig): Configuration object containing model-specific settings.
        training (bool): Flag indicating whether the processor is in training mode.
    """

    def __init__(self, config: ModelConfig):
        """Initialize the processor with the given configuration.

        Args:
            config (ModelConfig): Model configuration containing settings and parameters.
        """
        self.config = config
        self.training = False

    def eval(self):
        """Set the processor to evaluation mode.

        Returns:
            Processor: Self reference for method chaining.
        """
        self.training = False
        return self

    def train(self, training: bool = True):
        """Set the processor training mode.

        Args:
            training (bool, optional): Whether to set training mode. Defaults to True.

        Returns:
            Processor: Self reference for method chaining.
        """
        self.training = training
        return self

    @abstractmethod
    def preprocess(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
        device: Union[Literal["cuda", "cpu"], torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        image_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, Any]:
        """Preprocess input data for model inference.

        This method must be implemented by subclasses to handle model-specific preprocessing
        such as resizing, normalization, and tensor formatting.

        Args:
            inputs: Input data which can be single or multiple images in various formats.
            device: Target device for tensor placement. Defaults to "cuda".
            dtype: Target data type for tensors. Defaults to torch.float32.
            image_size: Optional target image size for resizing. Defaults to None.

        Returns:
            tuple[torch.Tensor, Any]: Preprocessed tensor and any additional metadata.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Pre-processing is not implemented for this model.")

    @abstractmethod
    def postprocess(
        self,
        outputs: ModelOutput,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
        class_names: list[str] = [],
        threshold: float = 0.5,
        **kwargs,
    ) -> list[FocoosDetections]:
        """Postprocess model outputs to generate final detection results.

        This method must be implemented by subclasses to convert raw model outputs
        into structured detection results.

        Args:
            outputs (ModelOutput): Raw outputs from the model.
            inputs: Original input data for reference during postprocessing.
            class_names (list[str], optional): List of class names for detection labels.
                Defaults to empty list.
            threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.
            **kwargs: Additional keyword arguments for model-specific postprocessing.

        Returns:
            list[FocoosDetections]: List of detection results for each input.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Post-processing is not implemented for this model.")

    @abstractmethod
    def export_postprocess(
        self,
        output: Union[list[torch.Tensor], list[np.ndarray]],
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        threshold: float = 0.5,
        **kwargs,
    ) -> list[FocoosDetections]:
        """Postprocess outputs from exported model for inference.

        This method handles postprocessing for models that have been exported
        (e.g., to ONNX format) and may have different output formats.

        Args:
            output: Raw outputs from exported model as tensors or numpy arrays.
            inputs: Original input data for reference during postprocessing.
            threshold: Optional confidence threshold for detections. Defaults to None.
            **kwargs: Additional keyword arguments for export-specific postprocessing.

        Returns:
            list[FocoosDetections]: List of detection results for each input.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Export post-processing is not implemented for this model.")

    @abstractmethod
    def get_dynamic_axes(self) -> DynamicAxes:
        """Get dynamic axes configuration for model export.

        This method defines which axes can vary in size during model export,
        typically used for ONNX export with dynamic batch sizes or image dimensions.

        Returns:
            DynamicAxes: Configuration specifying which axes are dynamic.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Export axes are not implemented for this model.")

    @abstractmethod
    def eval_postprocess(self, outputs: ModelOutput, inputs: list[DatasetEntry]):
        """Postprocess model outputs for evaluation purposes.

        This method handles postprocessing specifically for model evaluation,
        which may differ from inference postprocessing.

        Args:
            outputs (ModelOutput): Raw outputs from the model.
            inputs (list[DatasetEntry]): List of dataset entries used as inputs.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Post-processing is not implemented for this model.")

    def get_image_sizes(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
    ):
        """Extract image dimensions from various input formats.

        This utility method determines the height and width of images from different
        input types including tensors, numpy arrays, and PIL images.

        Args:
            inputs: Input data containing one or more images in various formats.

        Returns:
            list[tuple[int, int]]: List of (height, width) tuples for each image.

        Raises:
            ValueError: If input type is not supported.
        """
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
        """Convert various input formats to a batched PyTorch tensor.

        This utility method standardizes different input types (PIL Images, numpy arrays,
        PyTorch tensors) into a single batched tensor with consistent format (BCHW).

        Args:
            inputs: Input data containing one or more images in various formats.

        Returns:
            torch.Tensor: Batched tensor with shape (B, C, H, W) where:
                - B is batch size
                - C is number of channels (typically 3 for RGB)
                - H is height
                - W is width

        Note:
            This method may break with different image sizes as it uses torch.cat
            which requires consistent dimensions across inputs.
        """
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
