"""Benchmark command implementation.

This module implements the benchmarking command for the Focoos CLI. It provides
comprehensive functionality to measure and analyze performance metrics of computer
vision models including inference latency, throughput (FPS), and statistical analysis
across multiple benchmark runs.

**Key Features:**
- **Latency Analysis**: Measures mean, min, max, and standard deviation of inference times
- **Throughput Metrics**: Calculates frames per second (FPS) performance
- **Device Support**: Benchmarks on both CUDA GPU and CPU devices
- **Statistical Reporting**: Provides comprehensive performance statistics
- **Flexible Configuration**: Supports custom iterations and image sizes

**Use Cases:**
- Performance optimization and profiling
- Model comparison and selection
- Hardware capability assessment
- Production deployment planning
- Performance regression testing

**Metrics Reported:**
- **FPS**: Frames per second throughput
- **Mean Latency**: Average inference time per frame
- **Min/Max Latency**: Best and worst case performance
- **Standard Deviation**: Performance consistency measure
- **Runtime Information**: Engine and device details

Examples:
    Basic benchmarking:
    ```bash
    focoos benchmark --model fai-detr-m-coco
    ```

    Custom configuration:
    ```bash
    focoos benchmark --model fai-detr-m-coco --iterations 100 --device cuda --im-size 1024
    ```

    Programmatic usage:
    ```python
    from focoos.cli.commands import benchmark_command

    benchmark_command(model_name="fai-detr-m-coco", iterations=50, device="cuda")
    ```

See Also:
    - [`focoos.model_manager.ModelManager`][focoos.model_manager.ModelManager]: Model loading and management
    - [`focoos.models.base_model.BaseModel.benchmark`][focoos.models.base_model.BaseModel.benchmark]: Core benchmarking functionality
"""

from typing import Literal, Optional, Tuple, Union

from focoos.model_manager import ModelManager
from focoos.utils.logger import get_logger

logger = get_logger("benchmark")


def benchmark_command(
    model_name: str,
    iterations: Optional[int] = None,
    device: Literal["cuda", "cpu"] = "cuda",
    im_size: Optional[Union[int, Tuple[int, int]]] = None,
    models_dir: Optional[str] = None,
):
    """Benchmark a model's inference performance with comprehensive metrics.

    Loads a specified model and runs performance benchmarks to measure detailed
    latency and throughput metrics. The benchmark performs multiple inference
    iterations with synthetic data to provide statistical analysis of model
    performance characteristics.

    **Performance Metrics:**
    - **FPS (Frames Per Second)**: Overall throughput measure
    - **Mean Latency**: Average inference time across all iterations
    - **Min/Max Latency**: Best and worst case performance times
    - **Standard Deviation**: Measure of performance consistency
    - **Runtime Engine**: Backend engine used for inference

    **Benchmark Process:**
    1. Model loading and initialization
    2. Warmup iterations to stabilize performance
    3. Timed inference iterations with synthetic input
    4. Statistical analysis and metric calculation
    5. Comprehensive results reporting

    Args:
        model_name (str): Name or identifier of the model to benchmark.
            Can be a pretrained model name (e.g., "fai-detr-m-coco") or
            path to a local model file.
        iterations (Optional[int], optional): Number of benchmark iterations to run
            for statistical analysis. More iterations provide more accurate statistics
            but take longer to complete. Defaults to 50 if None.
        device (Literal["cuda", "cpu"], optional): Target device for benchmarking.
            - "cuda": Use GPU acceleration (requires CUDA-compatible GPU)
            - "cpu": Use CPU-only inference
            Defaults to "cuda".
        im_size (Optional[Union[int, Tuple[int, int]]], optional): Input image size
            for benchmark synthetic data. Can be:
            - Single integer: Square image (e.g., 640 → 640x640)
            - Tuple: Specific dimensions (width, height)
            - None: Use model's default input size
            Defaults to None.
        models_dir (Optional[str], optional): Custom directory to search for local
            model files. If None, uses the default models directory.
            Defaults to None.

    Raises:
        Exception: If model loading fails, device is unavailable, or benchmark
            execution encounters an error.
        FileNotFoundError: If specified model name/path cannot be found.
        RuntimeError: If CUDA device is specified but not available.

    Examples:
        Basic GPU benchmarking:
        ```python
        benchmark_command("fai-detr-m-coco")
        ```

        Custom configuration:
        ```python
        benchmark_command(model_name="fai-detr-m-coco", iterations=100, device="cuda", im_size=1024)
        ```

        CPU benchmarking:
        ```python
        benchmark_command(model_name="my-custom-model", iterations=20, device="cpu", im_size=(800, 600))
        ```

    Note:
        - GPU benchmarking requires a CUDA-compatible device and drivers
        - Benchmark results may vary based on system load and thermal conditions
        - First-run results may include model loading overhead
        - Consider running multiple benchmark sessions for production planning

    See Also:
        - [`focoos.model_manager.ModelManager.get`][focoos.model_manager.ModelManager.get]: Model loading
        - [`focoos.models.base_model.BaseModel.benchmark`][focoos.models.base_model.BaseModel.benchmark]: Core benchmark method
    """
    logger.info(f"🔄 Loading model: {model_name}")
    model = ModelManager.get(model_name, models_dir=models_dir)

    logger.info(f"🚀 Starting benchmark with {iterations or 50} iterations on {device}")
    latency_metrics = model.benchmark(
        iterations=iterations or 50,
        size=im_size,
        device=device,
    )

    logger.info("📊 Benchmark Results:")
    logger.info("=" * 50)
    logger.info(f"  🔧 Runtime: {latency_metrics.engine}")
    logger.info(f"  📐 Image size: {latency_metrics.im_size}")
    logger.info(f"  💻 Device: {latency_metrics.device}")
    logger.info(f"  ⚡ FPS: {latency_metrics.fps:.2f}")
    logger.info(f"  📈 Mean latency: {latency_metrics.mean:.2f} ms")
    logger.info(f"  🟢 Min latency: {latency_metrics.min:.2f} ms")
    logger.info(f"  🔴 Max latency: {latency_metrics.max:.2f} ms")
    logger.info(f"  📊 Std deviation: {latency_metrics.std:.2f} ms")
    logger.info("=" * 50)

    logger.info("✅ Benchmark completed successfully!")
