import os
import time
from typing import Optional

import onnx
import onnxslim
import torch

from focoos.utils.logger import get_logger

logger = get_logger(__name__)


def onnx_export(
    model_name: str,
    model: torch.nn.Module,
    size: tuple,
    device: str = "cuda",
    opset: int = 17,
    dynamic: bool = True,
    fp16: bool = False,
    dtype: torch.dtype = torch.float32,
    simplify: bool = False,
    weights: Optional[str] = None,
):
    logger.info(f"Exporting model to {model_name} on {device} with opset {opset}")
    model.to(device)
    model.eval()

    if weights is not None:
        ckpt = torch.load(weights, map_location=torch.device("cpu"))["model"]
        model.load_state_dict(ckpt, strict=True)
        logger.info(f"Weights loaded from {weights}")

    if not (isinstance(size, tuple) or isinstance(size, list)):
        size = [size, size]
    # , "width": 2048, 'height': 1024}, ]
    data = 128 * torch.ones(1, 3, size[0], size[1], dtype=dtype).to(device)

    input_names = ["input"]
    with torch.no_grad():
        res = model(data)
    output_names = [f"output_{i}" for i in range(len(res))]

    if dynamic:
        # shape(1,3,640,640)}
        dynamic_axes = {"input": {0: "batch", 2: "height", 3: "width"}}
        dynamic_axes["output"] = {0: "batch", 2: "height", 3: "width"}

    with torch.no_grad():
        model = model.eval()
        if fp16:
            with torch.autocast(device_type="cuda"):
                torch.onnx.export(
                    model,
                    args=(data,),
                    f=model_name,
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names,
                    export_params=True,
                    opset_version=opset,
                    do_constant_folding=True,
                    dynamic_axes=dynamic_axes if dynamic else None,
                )
        else:
            logger.info("Starting export...")
            torch.onnx.export(
                model,
                args=(data,),
                f=model_name,
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes if dynamic else None,
            )
        logger.info(f"Correctly export model at {model_name}.")

    model_onnx = onnx.load(model_name)
    onnx.checker.check_model(model_onnx)
    logger.info("Correctly checked model.")

    if simplify:
        t0 = time.time()
        try:
            logger.info(f"Slimming with onnxslim {onnxslim.__version__}...")
            simplified_onnx = onnxslim.slim(model_onnx)
            os.remove(model_name)
            if isinstance(simplified_onnx, onnx.ModelProto):
                onnx.save(simplified_onnx, model_name)
            else:
                logger.error("Failed to slim model.")

            logger.info("Correctly slimmed model.")
        except Exception as e:
            logger.error(f"Error slimming model: {e}.")

        logger.info(f"Simplify took: {time.time() - t0:.2f} seconds")
    return model_onnx
