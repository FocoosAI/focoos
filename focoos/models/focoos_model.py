import os
from typing import Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.data.datasets.map_dataset import MapDataset
from focoos.infer.infer_model import InferModel
from focoos.models.base_model import BaseModelNN
from focoos.ports import (
    MODELS_DIR,
    ExportFormat,
    FocoosDetections,
    ModelInfo,
    RuntimeType,
    TrainerArgs,
)
from focoos.processor.processor_manager import ProcessorManager
from focoos.utils.distributed.dist import launch
from focoos.utils.logger import get_logger

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
        onnx_dynamic: bool = True,
        model_fuse: bool = True,
        fp16: bool = False,
        out_dir: Optional[str] = None,
        device: Literal["cuda", "cpu"] = "cuda",
        overwrite: bool = False,
    ) -> InferModel:
        if device is None:
            device = self.model.device

        if out_dir is None:
            out_dir = os.path.join(MODELS_DIR, self.model_info.name)

        format = runtime_type.to_export_format()
        exportable_model = ExportableModel(self.model, device=device)
        os.makedirs(out_dir, exist_ok=True)
        data = 128 * torch.randn(1, 3, self.model_info.im_size, self.model_info.im_size).to(device)

        export_model_name = "model.onnx" if format == ExportFormat.ONNX else "model.pt"
        _out_file = os.path.join(out_dir, export_model_name)

        dynamic_axes = self.processor.get_dynamic_axes()

        # spec = InputSpec(tensors=[TensorSpec([Dim("batch", min=1, max=64), 3, 224, 224])])
        if not overwrite and os.path.exists(_out_file):
            logger.info(f"Model file {_out_file} already exists. Set overwrite to True to overwrite.")
            return InferModel(model_dir=out_dir, model_info=self.model_info, runtime_type=runtime_type)

        if format == "onnx":
            with torch.no_grad():
                logger.info("🚀 Exporting ONNX model..")
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
                # if exp_program is not None:
                #    exp_program.optimize()
                #    exp_program.save(_out_file)
                logger.info(f"✅ Exported {format} model to {_out_file}")

        elif format == "torchscript":
            with torch.no_grad():
                logger.info("🚀 Exporting TorchScript model..")
                exp_program = torch.jit.trace(exportable_model, data)
                if exp_program is not None:
                    _out_file = os.path.join(out_dir, "model.pt")
                    exp_program.save(_out_file)
                    logger.info(f"✅ Exported {format} model to {_out_file} ")
                else:
                    raise ValueError(f"Failed to export {format} model")

        self.model_info.dump_json(os.path.join(out_dir, "model_info.json"))
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
    ) -> list[FocoosDetections]:
        model = self.model.eval()
        try:
            model = model.cuda()
        except Exception:
            logger.warning("Unable to use CUDA")
        output = model.forward(inputs)
        class_names = self.model_info.classes
        output_fdet = self.processor.postprocess(output, inputs, class_names=class_names, **kwargs)
        return output_fdet

    def load_weights(self, weights: dict):
        # Merge with load weights of checkpointer
        incompatible = self.model.load_state_dict(weights, strict=False)
        return len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
