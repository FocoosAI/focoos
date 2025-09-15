import logging
import math
import os
import time
from dataclasses import dataclass
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

from focoos.utils.logger import get_logger


@dataclass
class QuantizationCfg:
    calibration_images: Optional[str] = None
    benchmark: bool = False
    data_reader_limit: int = 1
    w_SNR: Optional[float] = None
    a_SNR: Optional[float] = None
    file_log: bool = False
    size: int = 512
    format: Literal["QDQ", "QO"] = "QDQ"


class DataReader(CalibrationDataReader):
    def __init__(
        self,
        calibration_image_folder: str,
        limit=1,
        size: Union[int, Tuple[int]] = 512,
        model_path: Optional[str] = None,
    ):
        self.enum_data = None
        self.calibration_image_folder = calibration_image_folder

        # Use inference session to get input shape.
        if model_path:
            session = onnxruntime.InferenceSession(model_path, None)
            (_, _, height, width) = session.get_inputs()[0].shape
            print(f"Input shape: {height}, {width}")
            self.input_name = session.get_inputs()[0].name
        else:
            height, width = size if not isinstance(size, int) else (size, size)
            self.input_name = "images"

        # Convert image to input data
        self.nhwc_data_list = self._preprocess_images(calibration_image_folder, height, width, size_limit=limit)

        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

    def _preprocess_images(self, images_folder: str, height: int, width: int, size_limit=100):
        """
        Loads a batch of images and preprocess them
        parameter images_folder: path to folder storing images
        parameter height: image height in pixels
        parameter width: image width in pixels
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        """
        image_names = os.listdir(images_folder)
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
            nhwc_data = np.expand_dims(input_data, axis=0)
            nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
            unconcatenated_batch_data.append(nchw_data)
        batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
        return batch_data


class OnnxQuantizer:
    def __init__(self, cfg: QuantizationCfg, input_model_path: Optional[str] = None):
        self.cfg = cfg
        self.logger = get_logger(name="quantizer")

        self.logger.info(f"Setting up data reader with calibration images: {cfg.calibration_images}")
        self.dr = DataReader(
            calibration_image_folder=cfg.calibration_images,
            limit=cfg.data_reader_limit,
            size=self.cfg.size,
            model_path=input_model_path if input_model_path else None,
        )

        self.file_log = cfg.file_log
        if self.file_log:
            self._print = self._file_print
        else:
            self._print = print
        # TBI: pass also the other quantization options using a struct or as init arguments

    def quantize(self, input_model_path: str, output_model_path: str):
        benchmark = self.cfg.benchmark
        w_SNR = self.cfg.w_SNR
        a_SNR = self.cfg.a_SNR

        #!fixme that's bad python, anyway, let's do it
        self.log_file = output_model_path[:-5] + ".log"
        if self.file_log:
            # Get the root logger
            root_logger = logging.getLogger()

            # Remove all handlers if they exist (to reset logging)
            if root_logger.hasHandlers():
                root_logger.handlers.clear()

            log_path = self.log_file

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
                handlers=[
                    logging.FileHandler(log_path, mode="w"),
                    # logging.StreamHandler()  # Optional: to also log to console
                ],
            )
            self.logger = logging.getLogger()

        # preprocess onnx model
        quant_pre_process(input_model_path, output_model_path, auto_merge=True)
        self.logger.info(f"Quantizing model from {input_model_path} to {output_model_path}")
        quantize_static(
            output_model_path,
            output_model_path,
            self.dr,
            quant_format=QuantFormat.QDQ if self.cfg.format == "QDQ" else QuantFormat.QOperator,
            per_channel=True,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )

        if self.file_log:
            logging.shutdown()

        self._print("Calibrated and quantized model saved.")

        if benchmark:
            imgsize = self.cfg.size
            self._print("benchmarking fp32 model...")
            self.benchmark(input_model_path, imgsize)

            self._print("benchmarking int8 model...")
            self.benchmark(output_model_path, imgsize)

        if w_SNR is not None:
            self._print("Computing weight error...")
            w_SNR_thres = w_SNR
            matching = self._match_weights(input_model_path, output_model_path)
            dict_matching = dict(matching)
            for x in matching:
                try:
                    if len(matching[x]["float"]) == 0:
                        self._print(f"should never enter: {x}, {len(matching[x]['float'])}")
                except KeyError:
                    self._print(f"not a list: {x}, {matching[x]['float']}")
                    del dict_matching[x]
                    continue

            # pprint(debug.compute_weight_error(new_matching, lambda x, y: np.abs(x-y).max()))
            weight_error = debug.compute_weight_error(dict_matching)
            for k in weight_error:
                if weight_error[k] < w_SNR_thres:
                    self._print(f"{k}, {weight_error[k]}")

        if a_SNR is not None:
            self._print("Computing activation error...")
            a_SNR_thres = a_SNR

            activations_float, activations_quant = self._compute_activations(input_model_path, output_model_path)
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
                    self._print(f"ERROR F, {k}, {matching[k]['float']}, {matching[k]['post_qdq'][0]}")
                error[k] = float_diff
                # if "backbone" not in k and "pixel_decoder" not in k and "predictor/Add" in k:
                if error[k] < a_SNR_thres:
                    self._print(f"{k}, {float_diff, matching[k]['post_qdq'][0].shape}")

        return 1

    def _match_weights(self, input_model_path, output_model_path):
        return debug.create_weight_matching(input_model_path, output_model_path)

    def _compute_activations(self, input_model_path, output_model_path):
        calibration_dataset_path = self.dr.calibration_image_folder
        dr = DataReader(calibration_dataset_path, limit=1, model_path=input_model_path)
        debug.modify_model_output_intermediate_tensors(input_model_path, input_model_path + ".tmp")
        activations_float = debug.collect_activations(input_model_path + ".tmp", dr)

        dr = DataReader(calibration_dataset_path, limit=1, model_path=output_model_path)
        debug.modify_model_output_intermediate_tensors(output_model_path, output_model_path + ".tmp")
        activations_quant = debug.collect_activations(output_model_path + ".tmp", dr)
        return activations_float, activations_quant

    def _match_activations(self, af, aq):
        return debug.create_activation_matching(af, aq)

    def singal_noise_ratio(self, x, y):
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
            self._print(f"{res}")
            raise

    def benchmark(self, model_path, size=640, runs=20):
        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        total = 0.0
        input_data = np.zeros((1, 3, size, size), np.float32)
        # Warming up
        _ = session.run([], {input_name: input_data})
        for i in range(runs):
            start = time.perf_counter()
            _ = session.run([], {input_name: input_data})
            end = (time.perf_counter() - start) * 1000
            total += end
            # self._print(f"{end:.2f}ms")
        total /= runs
        self._print(f"Avg: {total:.2f}ms")

    def _file_print(self, str):
        with open(self.log_file, "a") as f:
            f.write(str + "\n")
