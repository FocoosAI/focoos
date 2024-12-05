import io
from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from supervision import Detections

from focoos.ports import (
    FocoosTask,
    LatencyMetrics,
    ModelMetadata,
    OnnxEngineOpts,
    RuntimeTypes,
)
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


def det_postprocess(
    out: np.ndarray, im0_shape: Tuple[int, int], conf_threshold: float
) -> Detections:
    cls_ids, boxes, confs = out
    boxes[:, 0::2] *= im0_shape[1]
    boxes[:, 1::2] *= im0_shape[0]
    high_conf_indices = np.where(confs > conf_threshold)

    return Detections(
        xyxy=boxes[high_conf_indices].astype(int),
        class_id=cls_ids[high_conf_indices].astype(int),
        confidence=confs[high_conf_indices].astype(float),
    )


def semseg_postprocess(
    out: np.ndarray, im0_shape: Tuple[int, int], conf_threshold: float
) -> Detections:
    cls_ids, mask, confs = out[0][0], out[1][0], out[2][0]
    masks = np.zeros((len(cls_ids), *mask.shape), dtype=bool)
    for i, cls_id in enumerate(cls_ids):
        masks[i, mask == i] = True
    high_conf_indices = np.where(confs > conf_threshold)[0]
    masks = masks[high_conf_indices].astype(bool)
    cls_ids = cls_ids[high_conf_indices].astype(int)
    confs = confs[high_conf_indices].astype(float)
    return Detections(
        mask=masks,
        # xyxy is required from supervisio
        xyxy=np.zeros(shape=(len(high_conf_indices), 4), dtype=np.uint8),
        class_id=cls_ids,
        confidence=confs,
    )


class ONNXRuntime:
    def __init__(
        self, model_path: str, opts: OnnxEngineOpts, model_metadata: ModelMetadata
    ):
        self.logger = get_logger()
        self.logger.debug(f"[onnxruntime device] {ort.get_device()}")
        self.logger.debug(
            f"[onnxruntime available providers] {ort.get_available_providers()}"
        )
        self.name = Path(model_path).stem
        self.opts = opts
        self.model_metadata = model_metadata
        self.postprocess_fn = (
            det_postprocess
            if model_metadata.task == FocoosTask.DETECTION
            else semseg_postprocess
        )
        options = ort.SessionOptions()
        if opts.verbose:
            options.log_severity_level = 0
        options.enable_profiling = opts.verbose
        # options.intra_op_num_threads = 1
        available_providers = ort.get_available_providers()
        if opts.cuda and "CUDAExecutionProvider" not in available_providers:
            self.logger.warning("CUDA ExecutionProvider not found.")
        if opts.trt and "TensorrtExecutionProvider" not in available_providers:
            self.logger.warning("Tensorrt ExecutionProvider not found.")
        if opts.vino and "OpenVINOExecutionProvider" not in available_providers:
            self.logger.warning("OpenVINO ExecutionProvider not found.")
        if opts.coreml and "CoreMLExecutionProvider" not in available_providers:
            self.logger.warning("CoreML ExecutionProvider not found.")
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
            #     # options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers.append("CoreMLExecutionProvider")
        else:
            binding = None

        binding = None  # TODO: remove this
        providers.append("CPUExecutionProvider")
        self.dtype = dtype
        self.binding = binding
        self.ort_sess = ort.InferenceSession(model_path, options, providers=providers)
        self.logger.info(
            f"[onnxruntime] Active providers:{self.ort_sess.get_providers()}"
        )
        if self.ort_sess.get_inputs()[0].type == "tensor(uint8)":
            self.dtype = np.uint8
        else:
            self.dtype = np.float32
        if self.opts.warmup_iter > 0:
            self.logger.info(f"⏱️ [onnxruntime] Warming up model ..")
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
                else:
                    self.ort_sess.run(out_name, {input_name: np_image})

            self.logger.info(f"⏱️ [onnxruntime] {self.name} WARMUP DONE")

    def __call__(self, im: np.ndarray, conf_threshold: float) -> Detections:
        out_name = None
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = [output.name for output in self.ort_sess.get_outputs()]
        if self.binding is not None:
            print(f"binding {self.binding}")
            io_binding = self.ort_sess.io_binding()

            io_binding.bind_input(
                input_name,
                self.binding,
                device_id=GPU_ID,
                element_type=self.dtype,
                shape=im.shape,
                buffer_ptr=im.ctypes.data,
            )

            io_binding.bind_cpu_input(input_name, im)
            io_binding.bind_output(out_name[0], self.binding)
            self.ort_sess.run_with_iobinding(io_binding)
            out = io_binding.copy_outputs_to_cpu()
        else:
            out = self.ort_sess.run(out_name, {input_name: im})

        detections = self.postprocess_fn(
            out, (im.shape[2], im.shape[3]), conf_threshold
        )
        return detections

    def benchmark(self, iterations=20, size=640) -> LatencyMetrics:
        self.logger.info(f"⏱️ [onnxruntime] Benchmarking latency..")
        size = size if isinstance(size, (tuple, list)) else (size, size)

        durations = []
        np_input = (255 * np.random.random((1, 3, size[0], size[1]))).astype(self.dtype)
        input_name = self.ort_sess.get_inputs()[0].name
        out_name = self.ort_sess.get_outputs()[0].name
        if self.binding:
            io_binding = self.ort_sess.io_binding()

            io_binding.bind_input(
                input_name,
                "cuda",
                device_id=0,
                element_type=self.dtype,
                shape=np_input.shape,
                buffer_ptr=np_input.ctypes.data,
            )

            io_binding.bind_cpu_input(input_name, np_input)
            io_binding.bind_output(out_name, "cuda")
        else:
            out_name = [output.name for output in self.ort_sess.get_outputs()]

        for step in range(iterations + 5):
            if self.binding:
                start = perf_counter()
                self.ort_sess.run_with_iobinding(io_binding)
                # print(len(outputs))
                end = perf_counter()
                # out = io_binding.copy_outputs_to_cpu()
            else:
                start = perf_counter()
                out = self.ort_sess.run(out_name, {input_name: np_input})
                end = perf_counter()

            if step >= 5:
                durations.append((end - start) * 1000)
        durations = np.array(durations)
        # time.sleep(0.1)
        metrics = LatencyMetrics(
            fps=int(1000 / durations.mean()),
            engine="onnx",
            mean=round(durations.mean(), 3),
            max=round(durations.max(), 3),
            min=round(durations.min(), 3),
            std=round(durations.std(), 3),
            im_size=size[0],
            device="",
        )
        self.logger.info(f"🔥 FPS: {metrics.fps}")
        return metrics


def get_runtime(
    runtime_type: RuntimeTypes,
    model_path: str,
    model_metadata: ModelMetadata,
    warmup_iter: int = 0,
) -> ONNXRuntime:
    if runtime_type == RuntimeTypes.ONNX_CUDA32:
        opts = OnnxEngineOpts(
            cuda=True, verbose=False, fp16=False, warmup_iter=warmup_iter
        )
    elif runtime_type == RuntimeTypes.ONNX_TRT32:
        opts = OnnxEngineOpts(
            cuda=False, verbose=False, trt=True, fp16=False, warmup_iter=warmup_iter
        )
    elif runtime_type == RuntimeTypes.ONNX_TRT16:
        opts = OnnxEngineOpts(
            cuda=False, verbose=False, trt=True, fp16=True, warmup_iter=warmup_iter
        )
    elif runtime_type == RuntimeTypes.ONNX_CPU:
        opts = OnnxEngineOpts(cuda=False, verbose=False, warmup_iter=warmup_iter)
    elif runtime_type == RuntimeTypes.ONNX_COREML:
        opts = OnnxEngineOpts(
            cuda=False, verbose=False, coreml=True, warmup_iter=warmup_iter
        )
    return ONNXRuntime(model_path, opts, model_metadata)
