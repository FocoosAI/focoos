import os
from typing import Literal, Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from focoos.data.datasets.map_dataset import MapDataset
from focoos.infer.infer_model import InferModel
from focoos.ports import DatasetEntry, ExportCfg, FocoosDetections, ModelConfig, ModelInfo, ModelOutput, TrainerArgs
from focoos.structures import Instances
from focoos.trainer.export.onnx import onnx_export
from focoos.utils.distributed.dist import launch
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModelNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

    def forward(
        self,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
            list[DatasetEntry],
        ],
    ) -> ModelOutput:
        raise NotImplementedError("Forward is not implemented for this model.")

    def eval_post_process(self, outputs: ModelOutput, inputs: list[DatasetEntry]) -> list[dict[str, Instances]]:
        raise NotImplementedError("Post-processing is not implemented for this model.")

    def post_process(
        self,
        outputs: ModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        class_names: list[str] = [],
        **kwargs,
    ) -> list[FocoosDetections]:
        raise NotImplementedError("Post-processing is not implemented for this model.")


class BaseProcessor:
    def __init__(self, config: ModelConfig):
        self.config = config

    def preprocess(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
    ) -> Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]]:
        raise NotImplementedError("Pre-processing is not implemented for this model.")

    def post_process(
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
    ):
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


class FocoosModel:
    def __init__(self, model: BaseModelNN, model_info: ModelInfo):
        self.model = model
        self.model_info = model_info

    def __str__(self):
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def __repr__(self):
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def train(self, args: TrainerArgs, data_train: MapDataset, data_val: MapDataset):
        from focoos.trainer.trainer import run_train

        """Train the model.

        Args:
            train_args: Training arguments
            data_train: Training dataset
            data_val: Validation dataset
        """
        if self.model_info.config["num_classes"] != data_val.dataset.metadata.num_classes:
            logger.error(
                f"Number of classes in the model ({self.model_info.config['num_classes']}) does not match the number of classes in the dataset ({data_val.dataset.metadata.num_classes})."
            )
            # self.model_info.config["num_classes"] = data_val.dataset.metadata.num_classes
            return

        self.model_info.train_args = args  # type: ignore
        self.model_info.val_dataset = data_val.dataset.metadata.name
        self.model_info.val_metrics = None
        self.model_info.classes = data_val.dataset.metadata.classes
        assert self.model_info.task == data_val.dataset.metadata.task, "Task mismatch between model and dataset."

        assert args.num_gpus, "Training without GPUs is not supported. num_gpus must be greater than 0"
        if args.num_gpus > 1:
            launch(
                run_train,
                args.num_gpus,
                dist_url="auto",
                args=(args, data_train, data_val, self.model, self.model_info),
            )
            logger.info("Training done, resuming main process.")
            # here i should restore the best model and config since in DDP it is not updated
            final_folder = os.path.join(args.output_dir, args.run_name)
            model_path = os.path.join(final_folder, "model_final.pth")
            metadata_path = os.path.join(final_folder, "model_info.json")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Training did not end correctly, model file not found at {model_path}")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Training did not end correctly, metadata file not found at {metadata_path}")
            logger.info(f"Reloading weights from {model_path}")
            weights = torch.load(model_path)
            self.load_weights(weights)
            self.model_info = ModelInfo.from_json(metadata_path)
        else:
            run_train(args, data_train, data_val, self.model, self.model_info)

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
                args=(args, data_test, self.model, self.model_info),
            )
            logger.info("Testing done, resuming main process.")
            # here i should restore the best model and config since in DDP it is not updated
            final_folder = os.path.join(args.output_dir, args.run_name)
            metadata_path = os.path.join(final_folder, "focoos_metadata.json")
            self.model_info = ModelInfo.from_json(metadata_path)
        else:
            run_test(args, data_test, self.model, self.model_info)

    @property
    def device(self):
        return self.model.device

    @property
    def im_size(self):
        return self.model_info.config["im_size"]

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
        export_cfg: ExportCfg,
        # quantization_cfg: Optional[QuantizationCfg] = None,
        benchmark: bool = False,
        benchmark_iters: int = 50,
        device: Literal["cuda", "cpu"] = "cuda",
        store_metadata: bool = True,
    ) -> InferModel:
        if export_cfg.device is None:
            export_cfg.device = self.model.device
        if export_cfg.format == "onnx":
            model_path = os.path.join(export_cfg.out_dir, "model.onnx")
            onnx_export(
                model=self.model,
                size=(self.model_info.im_size, self.model_info.im_size),
                device="cuda",
                opset=export_cfg.onnx_opset,
                dynamic=export_cfg.onnx_dynamic,
                simplify=export_cfg.onnx_simplify,
                model_name=model_path,
            )
        pass

    #     model_to_export = self.model.exportable_model(
    #         fuse_layers=export_cfg.model_fuse, task=FocoosTasks(model_cfg.task)
    #     )

    #     exporter = ModelExporter(
    #         export_cfg=export_cfg,
    #         model=model_to_export,
    #         model_cfg=model_cfg,
    #     )

    #     self.logger.info(f"Exporting model {model_cfg.name} to {export_cfg.format}")

    #     model_path = exporter.export()

    #     if quantization_cfg:
    #         if quantization_cfg.size != model_cfg.im_size:
    #             self.logger.warning(
    #                 f"Quantization size {quantization_cfg.size} does not match model size {model_cfg.im_size}. Forcing quantization size to {model_cfg.im_size}."
    #             )
    #             quantization_cfg.size = model_cfg.im_size

    #         if not export_cfg.format == ExportFormat.ONNX.value:
    #             self.logger.warning("Only ONNX supports quantization")
    #         else:
    #             self.logger.info(f"Quantizing {model_path}.")
    #             quantizer = OnnxQuantizer(quantization_cfg)
    #             quantizer.quantize(
    #                 input_model_path=model_path,
    #                 output_model_path=model_path,
    #             )

    #     metrics = None
    #     if benchmark:
    #         self.logger.info(f"⏱️ Benchmarking {model_path}.")
    #         metrics = get_runtime(
    #             runtime_type=runtime_type,
    #             model_path=model_path,
    #         ).benchmark(iterations=benchmark_iters, size=model_cfg.im_size)
    #         self.logger.info(
    #             f"⏱️ Benchmarking done. Latency: {metrics.mean} ms, FPS: {metrics.fps} im_size: {metrics.im_size} engine: {metrics.engine} device: {metrics.device}"
    #         )

    #         if not model_cfg.latency:
    #             model_cfg.latency = []
    #         model_cfg.latency.append(metrics)

    #     if store_metadata:
    #         self.logger.info(f"Storing metadata in {export_cfg.out_dir}")
    #         self.store_metadata(export_cfg.out_dir)

    #     return model_path, metrics

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
    ) -> list[FocoosDetections]:
        model = self.model.eval()
        try:
            model = model.cuda()
        except Exception:
            logger.warning("Unable to use CUDA")
        output = model(inputs)
        class_names = self.model_info.classes
        output_fdet = model.post_process(output, inputs, class_names=class_names, **kwargs)
        return output_fdet

    def load_weights(self, weights: dict):
        checkpoint_state_dict = weights
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)

        if incompatible.missing_keys:
            logger.warning(f"Missing keys in checkpoint: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")
        logger.info("Loaded weights!")
        return len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
