import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from focoos.data.datasets.map_dataset import MapDataset
from focoos.hub.api_client import ApiClient
from focoos.hub.focoos_hub import FocoosHUB
from focoos.infer.infer_model import InferModel
from focoos.models.base_model import BaseModelNN
from focoos.ports import (
    MODELS_DIR,
    ArtifactName,
    ExportFormat,
    FocoosDetections,
    LatencyMetrics,
    ModelInfo,
    ModelStatus,
    RuntimeType,
    TrainerArgs,
    TrainingInfo,
)
from focoos.processor.processor_manager import ProcessorManager
from focoos.utils.distributed.dist import launch
from focoos.utils.env import TORCH_VERSION
from focoos.utils.logger import get_logger
from focoos.utils.system import get_cpu_name, get_device_name, get_focoos_version, get_system_info

logger = get_logger("FocoosModel")


class ExportableModel(torch.nn.Module):
    """A wrapper class for making models exportable to different formats.

    This class wraps a BaseModelNN model to make it compatible with export formats
    like ONNX and TorchScript by handling the output formatting.

    Args:
        model: The base model to wrap for export.
        device: The device to move the model to. Defaults to "cuda".
    """

    def __init__(self, model: BaseModelNN, device="cuda"):
        """Initialize the ExportableModel.

        Args:
            model: The base model to wrap for export.
            device: The device to move the model to. Defaults to "cuda".
        """
        super().__init__()
        self.model = model.eval().to(device)

    def forward(self, x):
        """Forward pass through the wrapped model.

        Args:
            x: Input tensor to pass through the model.

        Returns:
            Model output converted to tuple format for export compatibility.
        """
        return self.model(x).to_tuple()


class FocoosModel:
    """Main model class for Focoos computer vision models.

    This class provides a high-level interface for training, testing, exporting,
    and running inference with Focoos models. It handles model configuration,
    weight loading, preprocessing, and postprocessing.

    Args:
        model: The underlying neural network model.
        model_info: Metadata and configuration information for the model.
    """

    def __init__(self, model: BaseModelNN, model_info: ModelInfo):
        """Initialize the FocoosModel.

        Args:
            model: The underlying neural network model.
            model_info: Metadata and configuration information for the model.
        """
        self.model = model
        self.model_info = model_info
        self.processor = ProcessorManager.get_processor(self.model_info.model_family, self.model_info.config)
        if self.model_info.weights_uri:
            self._load_weights()
        else:
            logger.warning(f"⚠️ Model {self.model_info.name} has no pretrained weights")

    def __str__(self):
        """Return string representation of the model.

        Returns:
            String containing model name and family.
        """
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def __repr__(self):
        """Return detailed string representation of the model.

        Returns:
            String containing model name and family.
        """
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def _setup_model_for_training(self, train_args: TrainerArgs, data_train: MapDataset, data_val: MapDataset):
        """Set up the model and metadata for training.

        This method configures the model information with training parameters,
        device information, dataset metadata, and initializes training status.

        Args:
            train_args: Training configuration arguments.
            data_train: Training dataset.
            data_val: Validation dataset.
        """
        device = get_cpu_name()
        system_info = get_system_info()
        if system_info.gpu_info and system_info.gpu_info.devices and len(system_info.gpu_info.devices) > 0:
            device = system_info.gpu_info.devices[0].gpu_name
        self.model_info.ref = None

        self.model_info.train_args = train_args  # type: ignore
        self.model_info.val_dataset = data_val.dataset.metadata.name
        self.model_info.val_metrics = None
        self.model_info.classes = data_val.dataset.metadata.classes
        self.model_info.focoos_version = get_focoos_version()
        self.model_info.status = ModelStatus.TRAINING_STARTING
        self.model_info.updated_at = datetime.now().isoformat()
        self.model_info.latency = []
        self.model_info.metrics = None
        self.model_info.training_info = TrainingInfo(
            instance_device=device,
            main_status=ModelStatus.TRAINING_STARTING,
            start_time=datetime.now().isoformat(),
            status_transitions=[
                dict(
                    status=ModelStatus.TRAINING_STARTING,
                    timestamp=datetime.now().isoformat(),
                )
            ],
        )

        self.model_info.classes = data_train.dataset.metadata.classes
        self.model_info.config["num_classes"] = len(data_train.dataset.metadata.classes)
        self._reload_model()
        self.model_info.name = train_args.run_name.strip()
        self.processor = ProcessorManager.get_processor(self.model_info.model_family, self.model_info.config)
        assert self.model_info.task == data_train.dataset.metadata.task, "Task mismatch between model and dataset."

    def train(self, args: TrainerArgs, data_train: MapDataset, data_val: MapDataset, hub: Optional[FocoosHUB] = None):
        """Train the model on the provided datasets.

        This method handles both single-GPU and multi-GPU distributed training.
        It sets up the model for training, optionally syncs with Focoos Hub,
        and manages the training process.

        Args:
            args: Training configuration and hyperparameters.
            data_train: Training dataset containing images and annotations.
            data_val: Validation dataset for model evaluation.
            hub: Optional Focoos Hub instance for model syncing.

        Raises:
            AssertionError: If task mismatch between model and dataset.
            AssertionError: If number of classes mismatch between model and dataset.
            AssertionError: If num_gpus is 0 (GPU training is required).
            FileNotFoundError: If training artifacts are not found after completion.
        """
        from focoos.trainer.trainer import run_train

        assert data_train.dataset.metadata.num_classes > 0, "Number of dataset classes must be greater than 0"
        self._setup_model_for_training(args, data_train, data_val)

        assert self.model_info.task == data_train.dataset.metadata.task, "Task mismatch between model and dataset."
        assert self.model_info.config["num_classes"] == data_train.dataset.metadata.num_classes, (
            "Number of classes mismatch between model and dataset."
        )
        assert args.num_gpus, "Training without GPUs is not supported. num_gpus must be greater than 0"
        if args.num_gpus > 1:
            launch(
                run_train,
                args.num_gpus,
                dist_url="auto",
                args=(args, data_train, data_val, self.model, self.processor, self.model_info, hub),
            )

            logger.info("Training done, resuming main process.")
            # here i should restore the best model and config since in DDP it is not updated
            final_folder = os.path.join(args.output_dir, args.run_name)
            model_path = os.path.join(final_folder, ArtifactName.WEIGHTS)
            metadata_path = os.path.join(final_folder, ArtifactName.INFO)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Training did not end correctly, model file not found at {model_path}")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Training did not end correctly, metadata file not found at {metadata_path}")
            self.model_info = ModelInfo.from_json(metadata_path)

            logger.info(f"Reloading weights from {self.model_info.weights_uri}")
            self._reload_model()
        else:
            run_train(args, data_train, data_val, self.model, self.processor, self.model_info, hub)

    def eval(self, args: TrainerArgs, data_test: MapDataset, save_json: bool = True):
        """evaluate the model on the provided test dataset.

        This method evaluates the model performance on a test dataset,
        supporting both single-GPU and multi-GPU testing.

        Args:
            args: Test configuration arguments.
            data_test: Test dataset for model evaluation.

        Raises:
            AssertionError: If task mismatch between model and dataset.
            AssertionError: If num_gpus is 0 (GPU testing is required).
        """
        from focoos.trainer.trainer import run_eval

        self.model_info.val_dataset = data_test.dataset.metadata.name
        self.model_info.val_metrics = None
        self.model_info.classes = data_test.dataset.metadata.classes
        self.model_info.config["num_classes"] = data_test.dataset.metadata.num_classes
        assert self.model_info.task == data_test.dataset.metadata.task, "Task mismatch between model and dataset."

        assert args.num_gpus, "Testing without GPUs is not supported. num_gpus must be greater than 0"
        if args.num_gpus > 1:
            launch(
                run_eval,
                args.num_gpus,
                dist_url="auto",
                args=(args, data_test, self.model, self.processor, self.model_info, save_json),
            )
            logger.info("Testing done, resuming main process.")
            # here i should restore the best model and config since in DDP it is not updated
            final_folder = os.path.join(args.output_dir, args.run_name)
            metadata_path = os.path.join(final_folder, ArtifactName.INFO)
            self.model_info = ModelInfo.from_json(metadata_path)
        else:
            run_eval(
                train_args=args,
                data_val=data_test,
                image_model=self.model,
                processor=self.processor,
                model_info=self.model_info,
                save_json=save_json,
            )

    @property
    def name(self):
        return self.model_info.name

    @property
    def device(self):
        """Get the device where the model is located.

        Returns:
            The device (CPU or CUDA) where the model is currently located.
        """
        return self.model.device

    @property
    def resolution(self):
        """Get the input resolution of the model.

        Returns:
            The input image resolution expected by the model.
        """
        return self.model_info.config["resolution"]

    @property
    def config(self) -> dict:
        """Get the model configuration.

        Returns:
            Dictionary containing the model configuration parameters.
        """
        return self.model_info.config

    @property
    def classes(self):
        """Get the class names the model can predict.

        Returns:
            List of class names that the model was trained to recognize.
        """
        return self.model_info.classes

    @property
    def task(self):
        """Get the computer vision task type.

        Returns:
            The type of computer vision task (e.g., detection, classification).
        """
        return self.model_info.task

    def export(
        self,
        runtime_type: RuntimeType = RuntimeType.TORCHSCRIPT_32,
        onnx_opset: int = 17,
        out_dir: Optional[str] = None,
        device: Literal["cuda", "cpu"] = "cuda",
        overwrite: bool = False,
        image_size: Optional[int] = None,
    ) -> InferModel:
        """Export the model to different runtime formats.

        This method exports the model to formats like ONNX or TorchScript
        for deployment and inference optimization.

        Args:
            runtime_type: Target runtime format for export.
            onnx_opset: ONNX opset version to use for ONNX export.
            out_dir: Output directory for exported model. If None, uses default location.
            device: Device to use for export ("cuda" or "cpu").
            overwrite: Whether to overwrite existing exported model files.
            image_size: Custom image size for export. If None, uses model's default size.

        Returns:
            InferModel instance for the exported model.

        Raises:
            ValueError: If unsupported PyTorch version or export format.
        """
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA is not available. Using CPU for export.")
        if out_dir is None:
            out_dir = os.path.join(MODELS_DIR, self.model_info.ref or self.model_info.name)

        format = runtime_type.to_export_format()
        exportable_model = ExportableModel(self.model, device=device)
        os.makedirs(out_dir, exist_ok=True)
        if image_size is None:
            data = 128 * torch.randn(1, 3, self.model_info.im_size, self.model_info.im_size).to(device)
        else:
            data = 128 * torch.randn(1, 3, image_size, image_size).to(device)
            self.model_info.im_size = image_size

        export_model_name = ArtifactName.ONNX if format == ExportFormat.ONNX else ArtifactName.PT
        _out_file = os.path.join(out_dir, export_model_name)

        dynamic_axes = self.processor.get_dynamic_axes()

        # Hack to warm up the model and record the spacial shapes if needed
        self.model(data)

        if not overwrite and os.path.exists(_out_file):
            logger.info(f"Model file {_out_file} already exists. Set overwrite to True to overwrite.")
            return InferModel(model_dir=out_dir, runtime_type=runtime_type)

        if format == "onnx":
            with torch.no_grad():
                logger.info("🚀 Exporting ONNX model..")
                if TORCH_VERSION >= (2, 5):
                    exp_program = torch.onnx.export(
                        exportable_model,
                        (data,),
                        f=_out_file,
                        opset_version=onnx_opset,
                        verbose=False,
                        verify=True,
                        dynamo=False,
                        external_data=False,  # model weights external to model
                        input_names=dynamic_axes.input_names,
                        output_names=dynamic_axes.output_names,
                        dynamic_axes=dynamic_axes.dynamic_axes,
                        do_constant_folding=True,
                        export_params=True,
                        # dynamic_shapes={
                        #    "x": {
                        #        0: torch.export.Dim("batch", min=1, max=64),
                        #        #2: torch.export.Dim("height", min=18, max=4096),
                        #        #3: torch.export.Dim("width", min=18, max=4096),
                        #    }
                        # },
                    )
                elif TORCH_VERSION >= (2, 0):
                    torch.onnx.export(
                        exportable_model,
                        (data,),
                        f=_out_file,
                        opset_version=onnx_opset,
                        verbose=False,
                        input_names=dynamic_axes.input_names,
                        output_names=dynamic_axes.output_names,
                        dynamic_axes=dynamic_axes.dynamic_axes,
                        do_constant_folding=True,
                        export_params=True,
                    )
                else:
                    raise ValueError(f"Unsupported Torch version: {TORCH_VERSION}. Install torch 2.x")
                # if exp_program is not None:
                #    exp_program.optimize()
                #    exp_program.save(_out_file)
                logger.info(f"✅ Exported {format} model to {_out_file}")

        elif format == "torchscript":
            with torch.no_grad():
                logger.info("🚀 Exporting TorchScript model..")
                exp_program = torch.jit.trace(exportable_model, data)
                if exp_program is not None:
                    _out_file = os.path.join(out_dir, ArtifactName.PT)
                    torch.jit.save(exp_program, _out_file)
                    logger.info(f"✅ Exported {format} model to {_out_file} ")
                else:
                    raise ValueError(f"Failed to export {format} model")

        # Fixme: this may override the model_info with the one from the exportable model
        self.model_info.dump_json(os.path.join(out_dir, ArtifactName.INFO))
        return InferModel(model_dir=out_dir, runtime_type=runtime_type)

    def __call__(
        self,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        **kwargs,
    ) -> FocoosDetections:
        """Run inference on input images.

        This method performs end-to-end inference including preprocessing,
        model forward pass, and postprocessing to return detections.

        Args:
            inputs: Input images in various formats (PIL, numpy, torch tensor, or lists).
            **kwargs: Additional arguments passed to postprocessing.

        Returns:
            FocoosDetections containing the detection results.
        """
        model = self.model.eval()
        processor = self.processor.eval()
        try:
            model = model.cuda()
        except Exception:
            logger.warning("Unable to use CUDA")
        images, _ = processor.preprocess(
            inputs,
            device=model.device,
            dtype=model.dtype,
            image_size=self.model_info.im_size,
        )  # second output is targets that we're not using
        with torch.no_grad():
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model.forward(images)
            except Exception:
                output = model.forward(images)
        class_names = self.model_info.classes
        output_fdet = processor.postprocess(output, inputs, class_names=class_names, **kwargs)

        # FIXME: we don't support batching yet
        return output_fdet[0]

    def _reload_model(self):
        """Reload the model with updated configuration.

        This method recreates the model instance with the current configuration
        and reloads the weights. Used when configuration changes during training.
        """
        from focoos.model_manager import ConfigManager  # here to avoid circular import

        torch.cuda.empty_cache()
        model_class = self.model.__class__
        # without the next line, the inner config may be not a ModelConfig but a dict
        config = ConfigManager.from_dict(self.model_info.model_family, self.model_info.config)
        self.model_info.config = config
        model = model_class(config)
        self.model = model
        self._load_weights()

    def _load_weights(self) -> int:
        """Load model weights from the specified URI.

        This method loads the model weights from either a local path or a remote URL,
        depending on the value of `self.model_info.weights_uri`. If the weights are remote,
        they are downloaded to a local directory. The method then loads the weights into
        the model, allowing for missing or unexpected keys (non-strict loading).

        Returns:
            The total number of missing or unexpected keys encountered during loading.
            Returns 0 if no weights are loaded or an error occurs.

        Raises:
            FileNotFoundError: If the weights file cannot be found at the specified path.
        """
        if not self.model_info.weights_uri:
            logger.warning(f"⚠️ Model {self.model_info.name} has no pretrained weights")
            return 0

        # Determine if weights are remote or local
        parsed_uri = urlparse(self.model_info.weights_uri)
        is_remote = bool(parsed_uri.scheme and parsed_uri.netloc)

        # Get weights path
        if is_remote:
            logger.info(f"Downloading weights from remote URL: {self.model_info.weights_uri}")
            model_dir = Path(MODELS_DIR) / self.model_info.name
            weights_path = ApiClient().download_ext_file(
                self.model_info.weights_uri, str(model_dir), skip_if_exists=True
            )
        else:
            logger.info(f"Using weights from local path: {self.model_info.weights_uri}")
            weights_path = self.model_info.weights_uri

        try:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            # Load weights and extract model state if needed
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            weights_dict = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict

        except Exception as e:
            logger.error(f"Error loading weights for {self.model_info.name}: {str(e)}")
            return 0

        incompatible = self.model.load_state_dict(weights_dict, strict=False)
        return len(incompatible.missing_keys) + len(incompatible.unexpected_keys)

    def benchmark(
        self,
        iterations: int = 50,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> LatencyMetrics:
        """Benchmark the model's inference performance.

        This method measures the raw model inference latency without
        preprocessing and postprocessing overhead.

        Args:
            iterations: Number of iterations to run for benchmarking.
            size: Input image size. If None, uses model's default size.
            device: Device to run benchmarking on ("cuda" or "cpu").

        Returns:
            LatencyMetrics containing performance statistics.
        """
        self.model.eval()

        if size is None:
            size = self.model_info.im_size
        if isinstance(size, int):
            size = (size, size)
        model = self.model.to(device)
        metrics = model.benchmark(size=size, iterations=iterations)
        return metrics

    def end2end_benchmark(self, iterations: int = 50, size: Optional[int] = None) -> LatencyMetrics:
        """Benchmark the complete end-to-end inference pipeline.

        This method measures the full inference latency including preprocessing,
        model forward pass, and postprocessing steps.

        Args:
            iterations: Number of iterations to run for benchmarking.
            size: Input image size. If None, uses model's default size.
            device: Device to run benchmarking on ("cuda" or "cpu").

        Returns:
            LatencyMetrics containing end-to-end performance statistics.
        """
        if size is None:
            size = self.model_info.im_size
        if self.model.device.type == "cpu":
            device_name = get_cpu_name()
        else:
            device_name = get_device_name()
        try:
            model = self.model.cuda()
        except Exception:
            logger.warning("Unable to use CUDA")
        logger.info(f"⏱️ Benchmarking End-to-End latency on {device_name} ({self.model.device}), size: {size}x{size}..")
        # warmup
        data = 128 * torch.randn(1, 3, size, size).to(model.device)

        durations = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream=torch.cuda.Stream())
            _ = self(data)
            end.record(stream=torch.cuda.Stream())
            torch.cuda.synchronize()
            durations.append(start.elapsed_time(end))

        durations = np.array(durations)
        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine=f"torch.{self.model.device}",
            mean=round(durations.mean().astype(float), 3),
            max=round(durations.max().astype(float), 3),
            min=round(durations.min().astype(float), 3),
            std=round(durations.std().astype(float), 3),
            im_size=size,
            device=str(self.model.device),
        )
        logger.info(f"🔥 FPS: {metrics.fps} Mean latency: {metrics.mean} ms ")
        return metrics
