import io
from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from focoos.ports import OnnxEngineOpts
from focoos.utils.logger import get_logger

GPU_ID = 0


def preprocess_image(bytes, dtype=np.float32) -> Tuple[np.ndarray, Image.Image]:
    pil_img = Image.open(io.BytesIO(bytes))
    img_numpy = np.ascontiguousarray(
        np.array(pil_img).transpose(2, 0, 1)[np.newaxis, :]  # HWC->CHW
    ).astype(dtype)
    return img_numpy, pil_img


def postprocess_image(
    cmapped_image: np.ndarray, input_image: Image.Image
) -> Image.Image:
    out = Image.fromarray(cmapped_image)
    return Image.blend(input_image, out, 0.6)


def image_to_byte_array(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


class ONNXRuntime:
    def __init__(self, model: str, opts: OnnxEngineOpts):
        self.logger = get_logger()
        self.logger.info(f"[Onnxruntime device] {ort.get_device()}")
        self.logger.info(
            f"[Onnxruntime available providers] {ort.get_available_providers()}"
        )
        self.name = Path(model).stem
        self.opts = opts

        options = ort.SessionOptions()
        if opts.verbose:
            options.log_severity_level = 0
        options.enable_profiling = opts.verbose
        # options.intra_op_num_threads = 1
        available_providers = ort.get_available_providers()
        # Set providers
        providers = []
        dtype = np.float32
        binding = None
        if opts.trt and "TensorrtExecutionProvider" in available_providers:
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        # 'trt_max_workspace_size': 1073741824,  # 1 GB
                        "trt_fp16_enable": opts.fp16,
                        "trt_force_sequential_engine_build": False,
                    },
                )
            )
            dtype = np.float32
        elif opts.vino and "OpenVINOExecutionProvider" in available_providers:
            providers.append(
                (
                    "OpenVINOExecutionProvider",
                    {
                        "device_type": "MYRIAD_FP16",
                        "enable_vpu_fast_compile": True,
                        "num_of_threads": 1,
                    },
                    # 'use_compiled_network': False}
                )
            )
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            dtype = np.float32
            binding = None
        elif opts.cuda and "CUDAExecutionProvider" in available_providers:
            binding = "cuda"
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": GPU_ID,
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": 16 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        elif opts.coreml and "CoreMLExecutionProvider" in available_providers:
            # options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers.append("CoreMLExecutionProvider")
        else:
            binding = None
        providers.append("CPUExecutionProvider")
        self.dtype = dtype
        self.binding = binding
        self.ort_sess = ort.InferenceSession(model, options, providers=providers)
        self.logger.info(f"[OnnxRuntime] Providers:{self.ort_sess.get_providers()}")
        if self.ort_sess.get_inputs()[0].type == "tensor(uint8)":
            self.dtype = np.uint8
        else:
            self.dtype = np.float32
        self.logger.info(f"[OnnxRuntime] dtype {self.dtype}")
        if self.opts.warmup_iter > 0:
            for i in range(0, self.opts.warmup_iter):
                np_image = np.random.rand(1, 3, 640, 640).astype(self.dtype)
                input_name = self.ort_sess.get_inputs()[0].name
                out_name = [output.name for output in self.ort_sess.get_outputs()]
                t0 = perf_counter()
                if self.binding is not None:
                    io_binding = self.ort_sess.io_binding()
                    io_binding.bind_input(
                        input_name,
                        self.binding,
                        device_id=GPU_ID,
                        element_type=self.dtype,
                        shape=np_image.shape,
                        buffer_ptr=np_image.ctypes.data,
                    )
                    io_binding.bind_cpu_input(input_name, np_image)
                    io_binding.bind_output(out_name[0], self.binding)
                    t0 = perf_counter()
                    self.ort_sess.run_with_iobinding(io_binding)
                    t1 = perf_counter()
                    io_binding.copy_outputs_to_cpu()
                    self.logger.info(f"Warmup {i} time: {t1 - t0:.3f} ms")
                else:
                    self.ort_sess.run(out_name, {input_name: np_image})

            self.logger.info(f"[ONNX] {self.name} WARMUP DONE")

    def __call__(self, img_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
        np_image, pil_img = preprocess_image(img_bytes, dtype=self.dtype)
        out_name = None
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]
        t_all_0 = perf_counter()
        if self.binding is not None:
            io_binding = self.ort_sess.io_binding()

            io_binding.bind_input(
                input_name,
                self.binding,
                device_id=GPU_ID,
                element_type=self.dtype,
                shape=np_image.shape,
                buffer_ptr=np_image.ctypes.data,
            )

            io_binding.bind_cpu_input(input_name, np_image)
            io_binding.bind_output(out_name[0], self.binding)
            t0 = perf_counter()
            self.ort_sess.run_with_iobinding(io_binding)
            t1 = perf_counter()
            out = io_binding.copy_outputs_to_cpu()
        else:
            t0 = perf_counter()
            out = self.ort_sess.run(out_name, {input_name: np_image})
            t1 = perf_counter()
        t_all_1 = perf_counter()
        print(out[0][0].shape)
        t_post0 = perf_counter()
        if len(out[0].shape) == 3:
            postprocess_out = np.argmax(out, axis=1)[0]
        else:
            postprocess_out = out[0]
        t_post1 = perf_counter()

        output_colored = self.cmap[postprocess_out]
        out_img = postprocess_image(cmapped_image=output_colored, input_image=pil_img)

        self.logger.debug(f"[{self.name}][Inference Time] {t1-t0:.3f} ms")
        self.logger.debug(f"[{self.name}][Argmax Time] {t_post1-t_post0:.3f} ms")
        self.logger.debug(
            f"[{self.name}][Inference TotalTime] {t_post1-t_all_0:.3f} ms"
        )
        self.logger.debug(
            f"[{self.name}][Inference with Data Time] {t_all_1-t_all_0:.3f} ms"
        )
        return postprocess_out, out_img

    def benchmark(
        self, batch_size: int = 1, input_shape: tuple = (3, 640, 640), iters: int = 100
    ):
        pass
        # !TODO add benchmark
