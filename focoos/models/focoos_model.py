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
from focoos.utils.system import get_cpu_name, get_focoos_version, get_system_info

logger = get_logger("FocoosModel")


class ExportableModel(torch.nn.Module):
    def __init__(self, model: BaseModelNN, device="cuda"):
        super().__init__()
        self.model = model.eval().to(device)

    def forward(self, x):
        return self.model(x).to_tuple()


class FocoosModel:
    def __init__(self, model: BaseModelNN, model_info: ModelInfo):
        self.model = model
        self.model_info = model_info
        self.processor = ProcessorManager.get_processor(self.model_info.model_family, self.model_info.config)
        if self.model_info.weights_uri:
            self._load_weights()
        else:
            logger.warning(f"⚠️ Model {self.model_info.name} has no pretrained weights")

    def __str__(self):
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def __repr__(self):
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def _setup_model_for_training(self, train_args: TrainerArgs, data_train: MapDataset, data_val: MapDataset):
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
        assert self.model_info.task == data_train.dataset.metadata.task, "Task mismatch between model and dataset."

    def train(self, args: TrainerArgs, data_train: MapDataset, data_val: MapDataset, hub: Optional[FocoosHUB] = None):
        from focoos.trainer.trainer import run_train

        """Train the model.

        Args:
            train_args: Training arguments
            data_train: Training dataset
            data_val: Validation dataset
        """

        self._setup_model_for_training(args, data_train, data_val)
        assert self.model_info.task == data_train.dataset.metadata.task, "Task mismatch between model and dataset."
        assert self.model_info.config["num_classes"] == data_train.dataset.metadata.num_classes, (
            "Number of classes mismatch between model and dataset."
        )
        remote_model = None
        if args.sync_to_hub:
            hub = hub or FocoosHUB()
            remote_model = hub.new_model(self.model_info)

            self.model_info.ref = remote_model.ref
            logger.info(f"Model {self.model_info.name} created in hub with ref {self.model_info.ref}")

        assert args.num_gpus, "Training without GPUs is not supported. num_gpus must be greater than 0"
        if args.num_gpus > 1:
            launch(
                run_train,
                args.num_gpus,
                dist_url="auto",
                args=(args, data_train, data_val, self.model, self.processor, self.model_info, remote_model),
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
            run_train(args, data_train, data_val, self.model, self.processor, self.model_info, remote_model)

    def test(self, args: TrainerArgs, data_test: MapDataset):
        from focoos.trainer.trainer import run_test

        """Test the model.

        Args:
            args: Test arguments
            data_test: Test dataset
        """
        self.model_info.val_dataset = data_test.dataset.metadata.name
        self.model_info.val_metrics = None
        self.model_info.classes = data_test.dataset.metadata.classes
        self.model_info.config["num_classes"] = data_test.dataset.metadata.num_classes
        assert self.model_info.task == data_test.dataset.metadata.task, "Task mismatch between model and dataset."

        assert args.num_gpus, "Testing without GPUs is not supported. num_gpus must be greater than 0"
        if args.num_gpus > 1:
            launch(
                run_test,
                args.num_gpus,
                dist_url="auto",
                args=(args, data_test, self.model, self.processor, self.model_info),
            )
            logger.info("Testing done, resuming main process.")
            # here i should restore the best model and config since in DDP it is not updated
            final_folder = os.path.join(args.output_dir, args.run_name)
            metadata_path = os.path.join(final_folder, ArtifactName.INFO)
            self.model_info = ModelInfo.from_json(metadata_path)
        else:
            run_test(args, data_test, self.model, self.processor, self.model_info)

    @property
    def device(self):
        return self.model.device

    @property
    def resolution(self):
        return self.model_info.config["resolution"]

    @property
    def config(self) -> dict:
        return self.model_info.config

    @property
    def classes(self):
        return self.model_info.classes

    @property
    def task(self):
        return self.model_info.task

    def export(
        self,
        runtime_type: RuntimeType = RuntimeType.ONNX_CUDA32,
        onnx_opset: int = 17,
        out_dir: Optional[str] = None,
        device: Literal["cuda", "cpu"] = "cuda",
        overwrite: bool = False,
        image_size: Optional[int] = None,
    ) -> InferModel:
        if device is None:
            device = self.model.device

        if out_dir is None:
            out_dir = os.path.join(MODELS_DIR, self.model_info.name)

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
            return InferModel(model_dir=out_dir, model_info=self.model_info, runtime_type=runtime_type)

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
        return InferModel(model_dir=out_dir, model_info=self.model_info, runtime_type=runtime_type)

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
        torch.cuda.empty_cache()
        model_class = self.model.__class__
        model = model_class(self.model_info.config)  # type: ignore
        self.model = model
        self._load_weights()

    def _load_weights(self) -> int:
        """
        Load model weights from the specified URI.

        This method loads the model weights from either a local path or a remote URL,
        depending on the value of `self.model_info.weights_uri`. If the weights are remote,
        they are downloaded to a local directory. The method then loads the weights into
        the model, allowing for missing or unexpected keys (non-strict loading).

        Returns:
            int: The total number of missing or unexpected keys encountered during loading.
                 Returns 0 if no weights are loaded or an error occurs.

        Raises:
            FileNotFoundError: If the weights file cannot be found at the specified path.
            Exception: If any other error occurs during loading, it is logged and 0 is returned.
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
        """
        Benchmark the model's inference performance over multiple iterations.
        """
        self.model.eval()

        if size is None:
            size = self.model_info.im_size
        if isinstance(size, int):
            size = (size, size)
        model = self.model.to(device)
        metrics = model.benchmark(size=size, iterations=iterations)
        return metrics

    def end2end_benchmark(
        self, iterations: int = 50, size: Optional[int] = None, device: Literal["cuda", "cpu"] = "cuda"
    ) -> LatencyMetrics:
        """
        Benchmark the model's inference performance over multiple iterations.
        """
        if size is None:
            size = self.model_info.im_size

        try:
            model = self.model.cuda()
        except Exception:
            logger.warning("Unable to use CUDA")
        logger.info(f"⏱️ Benchmarking latency on {model.device}, size: {size}x{size}..")
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
