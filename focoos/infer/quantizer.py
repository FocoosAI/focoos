import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import onnxruntime
import onnxruntime.quantization.qdq_loss_debug as debug
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process
from PIL import Image

from focoos.infer.infer_model import InferModel
from focoos.ports import LatencyMetrics, RuntimeType
from focoos.utils.logger import get_logger


@dataclass
class QuantizationCfg:
    """
    Configuration for model quantization.

    Attributes:
        calibration_images_folder (str): Path to the folder containing calibration images.
        data_reader_limit (int): Maximum number of images to use for calibration. Default is 100.
        normalize_images (bool): Whether to normalize images during preprocessing. Default is True.
        w_SNR (Optional[float]): Signal-to-noise ratio threshold for weights. If None, no threshold is applied.
        a_SNR (Optional[float]): Signal-to-noise ratio threshold for activations. If None, no threshold is applied.
        size (int): Target size for input images (height and width). Default is 512.
        format (Literal["QDQ", "QO"]): Quantization format, either "QDQ" or "QO". Default is "QDQ".
        per_channel (bool): Whether to use per-channel quantization. Default is True.
    """

    calibration_images_folder: str
    data_reader_limit: int = 100
    normalize_images: bool = True
    w_SNR: Optional[float] = None
    a_SNR: Optional[float] = None
    size: int = 512
    format: Literal["QDQ", "QO"] = "QDQ"
    per_channel: bool = True


class DataReader(CalibrationDataReader):
    def __init__(
        self,
        calibration_image_folder: str,
        limit=1,
        size: Union[int, Tuple[int]] = 512,
        model_path: Optional[str] = None,
        normalize_images: bool = True,
    ):
        self.enum_data = None
        self.calibration_image_folder = calibration_image_folder

        # Use inference session to get input shape.
        if model_path:
            session = onnxruntime.InferenceSession(model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            if not isinstance(height, int) or not isinstance(width, int):
                raise ValueError("Input shape are dynamic, please export the model without dynamic axes")
            print(f"Input shape: {height}, {width}")
            self.input_name = session.get_inputs()[0].name
        else:
            height, width = size if not isinstance(size, int) else (size, size)
            self.input_name = "images"

        # Convert image to input data
        self.nhwc_data_list = self._preprocess_images(
            calibration_image_folder, height, width, size_limit=limit, normalize_images=normalize_images
        )

        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

    def _preprocess_images(
        self, images_folder: str, height: int, width: int, size_limit=100, normalize_images: bool = True
    ):
        """
        Loads a batch of images and preprocess them
        parameter images_folder: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        """
        image_names = os.listdir(images_folder)
        pixel_mean: np.ndarray = np.array([123.675, 116.28, 103.53]).astype(np.float32)
        pixel_std: np.ndarray = np.array([58.395, 57.12, 57.375]).astype(np.float32)
        image_names = [img for img in image_names if any(img.lower().endswith(fmt) for fmt in ["jpg", "png", "jpeg"])]
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names
        unconcatenated_batch_data = []

        for image_name in batch_filenames:
            image_filepath = images_folder + "/" + image_name
            pillow_img = Image.new("RGB", (width, height))
            pillow_img.paste(Image.open(image_filepath).resize((width, height)))
            input_data = np.float32(pillow_img)

            if normalize_images:
                input_data = input_data - pixel_mean
                input_data = input_data / pixel_std
            nhwc_data = np.expand_dims(input_data, axis=0)
            nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
            unconcatenated_batch_data.append(nchw_data)
        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
        return batch_data


class OnnxQuantizer:
    """
    Handles ONNX model quantization, benchmarking, and error analysis.

    This class provides a high-level interface for quantizing ONNX models using calibration images,
    benchmarking both the original and quantized models, and computing signal-to-noise ratios (SNR)
    for weights and activations to assess quantization quality.

    Args:
        cfg (QuantizationCfg): Configuration object containing quantization parameters, calibration image folder,
            quantization format, benchmarking and error thresholds, and other options.
        input_model_path (Union[str, Path]): Path to the input ONNX model to be quantized.

    Attributes:
        cfg (QuantizationCfg): Quantization configuration.
        logger (logging.Logger): Logger for quantization process.
        input_model_path (Union[str, Path]): Path to the input ONNX model.
        data_reader (DataReader): Data reader for calibration images.

    Methods:
        quantize(): Quantizes the ONNX model, benchmarks it if requested, and computes SNR for weights and activations
            if thresholds are provided. Returns the path to the quantized model.
        benchmark(model_path, size=640, runs=20): Benchmarks the inference time of a given ONNX model.
        singal_noise_ratio(x, y): Computes the signal-to-noise ratio between two tensors.
    """

    def __init__(self, cfg: QuantizationCfg, input_model_path: Union[str, Path]):
        assert str(input_model_path).endswith(".onnx"), "Input model must be an ONNX model"
        self.cfg = cfg
        self.logger = get_logger(name="OnnxQuantizer")
        self.logger.info(f"Setting up data reader with calibration images: {cfg.calibration_images_folder}")
        self.input_model_path = input_model_path
        self.data_reader = DataReader(
            calibration_image_folder=cfg.calibration_images_folder,
            limit=cfg.data_reader_limit,
            size=self.cfg.size,
            model_path=str(input_model_path) if input_model_path else None,
        )
        # TBI: pass also the other quantization options using a struct or as init arguments

    def quantize(
        self,
        benchmark: bool = True,
    ) -> Path:
        """
        Quantizes the ONNX model using the provided configuration and calibration images.

        This method performs the following steps:
        - Preprocesses the ONNX model for quantization.
        - Runs static quantization using the specified quantization format and types.
        - Optionally benchmarks both the original and quantized models.
        - Optionally computes and logs SNR for weights and activations if thresholds are set.

        Returns:
            Path: Path to the quantized ONNX model.
        """
        w_SNR = self.cfg.w_SNR
        a_SNR = self.cfg.a_SNR

        # Generate output path by replacing the .onnx extension with _quant.onnx
        output_model_path = str(self.input_model_path)[:-5] + "_int8.onnx"

        # preprocess onnx model
        quant_pre_process(
            input_model_path=self.input_model_path,
            output_model_path=output_model_path,
            auto_merge=True,
            skip_symbolic_shape=False,
            verbose=False,
        )
        self.logger.info(f"🔧 Quantizing model from {self.input_model_path} to {output_model_path}")
        activation_quant_type = QuantType.QInt8 if self.cfg.format == "QDQ" else QuantType.QUInt8
        weight_quant_type = QuantType.QInt8
        quantize_static(
            output_model_path,
            output_model_path,
            self.data_reader,
            quant_format=QuantFormat.QDQ if self.cfg.format == "QDQ" else QuantFormat.QOperator,
            per_channel=self.cfg.per_channel,
            weight_type=weight_quant_type,
            activation_type=activation_quant_type,
            calibrate_method=CalibrationMethod.MinMax,
        )

        self.logger.info(f"✅ Quantized model saved successfully to {output_model_path}")

        if benchmark:
            self.logger.info("================== BENCHMARKING FP32 MODEL ==================")
            self.benchmark(self.input_model_path, device="cpu", iterations=100)
            self.logger.info("================================================================")
            self.logger.info("================== BENCHMARKING INT8 MODEL ==================")
            self.benchmark(output_model_path, device="cpu", iterations=100)
            self.logger.info("================================================================")

        if w_SNR is not None:
            self.logger.info("Computing weight error...")
            w_SNR_thres = w_SNR
            matching = self._match_weights(self.input_model_path, output_model_path)
            dict_matching = dict(matching)
            for x in matching:
                try:
                    if len(matching[x]["float"]) == 0:
                        self.logger.info(f"should never enter: {x}, {len(matching[x]['float'])}")
                except KeyError:
                    self.logger.error(f"not a list: {x}, {matching[x]['float']}")
                    del dict_matching[x]
                    continue

            # pprint(debug.compute_weight_error(new_matching, lambda x, y: np.abs(x-y).max()))
            weight_error = debug.compute_weight_error(dict_matching)
            for k in weight_error:
                if weight_error[k] < w_SNR_thres:
                    self.logger.info(f"{k}, {weight_error[k]}")

        if a_SNR is not None:
            self.logger.info("Computing activation error...")
            a_SNR_thres = a_SNR

            activations_float, activations_quant = self._compute_activations(self.input_model_path, output_model_path)
            matching = self._match_activations(activations_float, activations_quant)
            error = {}
            for k in sorted(matching):
                # try:
                #     qdb_diff = singal_noise_ratio(matching[k]['pre_qdq'], matching[k]['post_qdq'][0]) # np.abs(matching[k]['pre_qdq'][0] - matching[k]['post_qdq'][0]).max()
                # except:
                #     self._print("ERROR Q", k, matching[k]['pre_qdq'], matching[k]['post_qdq'][0])
                try:
                    float_diff = self.singal_noise_ratio(
                        matching[k]["float"], matching[k]["post_qdq"][0]
                    )  # np.abs(matching[k]['pre_qdq'][0] - matching[k]['float'][0]).max()
                except KeyError:
                    self.logger.error(f"ERROR F, {k}, {matching[k]['float']}, {matching[k]['post_qdq'][0]}")
                error[k] = float_diff
                # if "backbone" not in k and "pixel_decoder" not in k and "predictor/Add" in k:
                if error[k] < a_SNR_thres:
                    self.logger.info(f"{k}, {float_diff, matching[k]['post_qdq'][0].shape}")

        return Path(output_model_path)

    def _match_weights(self, input_model_path, output_model_path):
        """
        Matches weights between the original and quantized models for error analysis.

        Args:
            input_model_path (str or Path): Path to the original model.
            output_model_path (str or Path): Path to the quantized model.

        Returns:
            dict: Mapping of matched weights between models.
        """
        return debug.create_weight_matching(input_model_path, output_model_path)

    def _compute_activations(self, input_model_path, output_model_path):
        """
        Collects activations from both the original and quantized models using calibration data.

        Args:
            input_model_path (str or Path): Path to the original model.
            output_model_path (str or Path): Path to the quantized model.

        Returns:
            tuple: (activations_float, activations_quant) from both models.
        """
        calibration_dataset_path = self.data_reader.calibration_image_folder
        dr = DataReader(calibration_dataset_path, limit=1, model_path=input_model_path)
        debug.modify_model_output_intermediate_tensors(input_model_path, input_model_path + ".tmp")
        activations_float = debug.collect_activations(input_model_path + ".tmp", dr)

        dr = DataReader(calibration_dataset_path, limit=1, model_path=output_model_path)
        debug.modify_model_output_intermediate_tensors(output_model_path, output_model_path + ".tmp")
        activations_quant = debug.collect_activations(output_model_path + ".tmp", dr)
        return activations_float, activations_quant

    def _match_activations(self, af, aq):
        """
        Matches activations between the original and quantized models for error analysis.

        Args:
            af: Activations from the original model.
            aq: Activations from the quantized model.

        Returns:
            dict: Mapping of matched activations between models.
        """
        return debug.create_activation_matching(af, aq)

    def singal_noise_ratio(self, x, y):
        """
        Computes the signal-to-noise ratio (SNR) in decibels between two tensors.

        Args:
            x (np.ndarray): Reference tensor (signal).
            y (np.ndarray): Comparison tensor (signal + noise).

        Returns:
            float: SNR value in decibels.
        """
        left = np.array(x).flatten()
        right = np.array(y).flatten()

        epsilon = 1e-5
        tensor_norm = max(np.linalg.norm(left), epsilon)
        diff_norm = max(np.linalg.norm(left - right), epsilon)
        assert tensor_norm > 0, f"Tensor norm < 0 {tensor_norm}"
        assert diff_norm > 0, f"Diff norm < 0 {diff_norm}"
        res = tensor_norm / diff_norm
        try:
            return 20 * math.log10(res)
        except:
            self.logger.error(f"{res}")
            raise

    def benchmark(self, model_path, device: Literal["cuda", "cpu"] = "cpu", iterations: int = 100) -> LatencyMetrics:
        """
        Benchmarks the inference time of an ONNX model.

        Args:
            model_path (str or Path): Path to the ONNX model to benchmark.
            device (Literal["cuda", "cpu"]): Device to run benchmarking on ("cuda" or "cpu"). Default is "cpu".
            iterations (int): Number of inference runs for averaging. Default is 100.

        Logs:
            Average inference time in milliseconds.
        """
        runtime_type = RuntimeType.ONNX_CPU if device == "cpu" else RuntimeType.ONNX_CUDA32
        infer_model = InferModel(model_path, runtime_type=runtime_type)
        metrics = infer_model.benchmark(iterations=iterations)
        return metrics
