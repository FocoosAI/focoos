"""Benchmark command implementation.

This module implements the benchmarking command for the Focoos CLI. It provides
functionality to measure and report performance metrics of computer vision models
including latency, throughput (FPS), and statistical analysis across multiple runs.
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
    """Benchmark a model's inference performance.

    Loads a model and runs performance benchmarks to measure latency metrics
    including FPS, mean/min/max latency, and standard deviation. Results are
    logged to the console.

    Args:
        model_name (str): Name of the model to benchmark.
        iterations (Optional[int], optional): Number of benchmark iterations to run.
            Defaults to 50 if None.
        device (Literal["cuda", "cpu"], optional): Device to run the benchmark on.
            Defaults to "cuda".
        im_size (Optional[Union[int, Tuple[int, int]]], optional): Input image size
            for the benchmark. Can be a single integer or tuple of (width, height).
            Defaults to None.

    Raises:
        Exception: If model loading fails or benchmark execution encounters an error.
    """
    logger.info(
        f"⚡ Starting benchmark - Model: {model_name}, Iterations: {iterations}, Size: {im_size}, Device: {device}"
    )

    try:
        logger.info(f"Loading model: {model_name}")
        model = ModelManager.get(model_name, models_dir=models_dir)

        latency_metrics = model.benchmark(
            iterations=iterations or 50,
            size=im_size,
            device=device,
        )

        logger.info("📊 Benchmark Results:")
        logger.info(f"  Runtime: {latency_metrics.engine}")
        logger.info(f"  Image size: {latency_metrics.im_size}")
        logger.info(f"  Device: {latency_metrics.device}")
        logger.info(f"  FPS: {latency_metrics.fps}")
        logger.info(f"  Mean latency: {latency_metrics.mean:.2f} ms")
        logger.info(f"  Min latency: {latency_metrics.min:.2f} ms")
        logger.info(f"  Max latency: {latency_metrics.max:.2f} ms")
        logger.info(f"  Std deviation: {latency_metrics.std:.2f} ms")

        logger.info("✅ Benchmark completed!")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise
