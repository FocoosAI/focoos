import os
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from focoos.data.datasets.map_dataset import MapDataset
from focoos.ports import ModelConfig, ModelInfo, ModelOutput, TrainerArgs
from focoos.structures import Instances
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
        ],
    ):
        raise NotImplementedError("Forward is not implemented for this model.")

    def post_process(self, outputs, batched_inputs) -> list[dict[str, Instances]]:
        raise NotImplementedError("Post-processing is not implemented for this model.")


def run_train(
    train_args: TrainerArgs,
    data_train: MapDataset,
    data_val: MapDataset,
    image_model: BaseModelNN,
    model_info: ModelInfo,  # type: ignore  # noqa: F821
):
    """Run model training.

    Args:
        train_args: Training configuration
        data_train: Training dataset
        data_val: Validation dataset
        image_model: Model to train
        metadata: Model metadata/configuration

    Returns:
        tuple: (trained model, updated metadata)
    """
    from focoos.trainer.trainer import FocoosTrainer

    trainer = FocoosTrainer(
        args=train_args,
        model=image_model,
        model_info=model_info,
        data_train=data_train,
        data_val=data_val,
    )
    trainer.train()

    return image_model, model_info


def run_test(
    train_args: TrainerArgs,
    data_val: MapDataset,
    image_model: BaseModelNN,
    model_info: ModelInfo,
):
    from focoos.trainer.trainer import FocoosTrainer

    trainer = FocoosTrainer(
        args=train_args,
        model=image_model,
        model_info=model_info,
        data_val=data_val,
    )
    trainer.test()

    return image_model, model_info


class FocoosModel:
    def __init__(self, model: BaseModelNN, model_info: ModelInfo):
        self.model = model
        self.model_info = model_info

    def __str__(self):
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def __repr__(self):
        return f"{self.model_info.name} ({self.model_info.model_family.value})"

    def train(self, args: TrainerArgs, data_train: MapDataset, data_val: MapDataset):
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

    # def export(
    #     self,
    #     export_cfg: ExportCfg,
    #     quantization_cfg: Optional[QuantizationCfg] = None,
    #     benchmark: bool = False,
    #     benchmark_iters: int = 50,
    #     runtime_type: RuntimeTypes = RuntimeTypes.CUDA,
    #     store_metadata: bool = True,
    # ) -> Tuple[str, Optional[LatencyMetrics]]:
    #     """Export model to different formats.

    #     Args:
    #         export_cfg: Export configuration
    #         quantization_cfg: Optional quantization config
    #         benchmark: Whether to run benchmarks
    #         benchmark_iters: Number of benchmark iterations
    #         runtime_type: Runtime type for benchmarking
    #         store_metadata: Whether to store model metadata

    #     Returns:
    #         Tuple of (model path, optional latency metrics)
    #     """
    #     model_cfg = self.config
    #     if export_cfg.device is None:
    #         export_cfg.device = self.model.device

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
    ) -> ModelOutput:
        model = self.model.eval()
        try:
            model = model.cuda()
        except Exception:
            logger.warning("Unable to use CUDA")
        return model(inputs, **kwargs)

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
