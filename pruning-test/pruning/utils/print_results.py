import json
import os

from focoos import Task


def calculate_model_size_mb(model_path):
    """Calculate model size in MB from file path.

    Args:
        model_path (str): Path to the model file (.pth)

    Returns:
        float: Model size in MB, or 0.0 if file doesn't exist
    """
    if not os.path.exists(model_path):
        return 0.0

    file_size_bytes = os.path.getsize(model_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    return file_size_mb


def load_eval_metrics_from_model_info(model_info_path, task_type=Task.DETECTION):
    """Load evaluation metrics from model_info.json file.

    Args:
        model_info_path (str): Path to model_info.json file
        task_type (Task, optional): Task type to determine which metrics to extract

    Returns:
        dict: Dictionary containing relevant metrics for the task or None if not found
    """
    if not os.path.exists(model_info_path):
        return None

    try:
        with open(model_info_path, "r") as f:
            model_info = json.load(f)

        # Extract metrics from val_metrics
        val_metrics = model_info.get("val_metrics", {})
        if not val_metrics:
            return None

        # Define task-specific metric prefixes and relevant metrics
        task_metric_configs = {
            "classification": {"prefix": "classification/", "metrics": ["F1", "Precision", "Recall"]},
            "detection": {"prefix": "bbox/", "metrics": ["AP", "AP50", "AP75"]},
            "semseg": {"prefix": "sem_seg/", "metrics": ["mIoU", "Pixel_Accuracy"]},
            "instance_segmentation": {"prefix": "segm/", "metrics": ["AP", "AP50", "AP75"]},
            "keypoint": {"prefix": "keypoints/", "metrics": ["AP", "AP50", "AP75"]},
        }

        # If task_type is provided, use specific config, otherwise try all
        if task_type and task_type.value in task_metric_configs:
            configs_to_try = [task_metric_configs[task_type.value]]
        else:
            configs_to_try = task_metric_configs.values()

        # Look for metrics based on task configuration
        extracted_metrics = {}
        for config in configs_to_try:
            prefix = config["prefix"]
            target_metrics = config["metrics"]

            for key, value in val_metrics.items():
                if key.startswith(prefix):
                    metric_name = key.replace(prefix, "")
                    if metric_name in target_metrics:
                        extracted_metrics[metric_name] = value

            # If we found metrics for this task, return them
            if extracted_metrics:
                return extracted_metrics

        return None

    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not load evaluation metrics from {model_info_path}: {e}")
        return None


def format_row(label, model_name, metrics, folder_path, eval_metrics=None, metric_columns=None, model_size_mb=None):
    if metrics is None:
        fps_str = "N/A"
        mean_str = "N/A"
        std_str = "N/A"
        device_str = "N/A"
    else:
        fps_str = f"{metrics.fps:>5}"
        mean_str = f"{metrics.mean:<6}"
        std_str = f"{metrics.std:<6}"
        device_str = f"{metrics.device:<12}"

    # Format model size
    if model_size_mb is not None:
        size_str = f"{model_size_mb:<8.2f}"
    else:
        size_str = f"{'N/A':<8}"

    # Format evaluation metrics dynamically based on available metrics
    eval_strs = []
    if eval_metrics is not None and metric_columns is not None:
        for col in metric_columns:
            value = eval_metrics.get(col, "N/A")
            if value != "N/A":
                # Format with 3 decimal places, fixed width 8
                eval_strs.append(f"{float(value):<9.3f}")
            else:
                eval_strs.append(f"{'N/A':<9}")
    else:
        # Fallback to default columns if no metrics available
        eval_strs = [f"{'N/A':<9}" for _ in range(3)]

    eval_metrics_str = " | ".join(eval_strs)

    return (
        f"{label:<16} | {model_name:<16} | {fps_str} | "
        f"{mean_str} | {std_str} | {device_str} | {size_str} | "
        f"{eval_metrics_str} | {folder_path:<40}"
    )


def print_results(
    result_original_model,
    result_pruned_model,
    MODEL_NAME,
    OUTPUT_DIRECTORY,
    original_eval_metrics=None,
    pruned_eval_metrics=None,
    task_type=Task.DETECTION,
    original_model_size_mb=None,
    pruned_model_size_mb=None,
):
    """Print benchmark results in a formatted table and save to summary.txt."""
    print("\nBenchmark Results:")

    ORIGINAL_FOLDER = "Focoos AI ModelRegistry"  # Assuming OUTPUT_DIRECTORY is for original model

    # Determine metric columns based on available metrics
    metric_columns = []
    if original_eval_metrics:
        metric_columns = list(original_eval_metrics.keys())
    elif pruned_eval_metrics:
        metric_columns = list(pruned_eval_metrics.keys())
    else:
        # Fallback to default classification metrics
        metric_columns = ["F1", "Precision", "Recall"]

    # Create header dynamically, fixed width 6 for each metric column
    metric_header = " | ".join([f"{col:<9}" for col in metric_columns])
    header = (
        f"{'Model':<16} | {'MODEL_NAME':<16} | {'FPS':>5} | "
        f"{'Mean':<6} | {'Std':<6} | {'Device':<12} | {'Size (MB)':<8} | "
        f"{metric_header} | {'Folder path':<40}"
    )
    separator = "-" * len(header)
    row_original = format_row(
        "Original Model",
        MODEL_NAME,
        result_original_model,
        ORIGINAL_FOLDER,
        original_eval_metrics,
        metric_columns,
        original_model_size_mb,
    )
    row_pruned = format_row(
        "Pruned Model",
        MODEL_NAME,
        result_pruned_model,
        OUTPUT_DIRECTORY,
        pruned_eval_metrics,
        metric_columns,
        pruned_model_size_mb,
    )

    print(header)
    print(separator)
    print(row_original)
    print(row_pruned)

    # Save to summary.txt in OUTPUT_DIRECTORY
    summary_path = os.path.join(OUTPUT_DIRECTORY, "summary_eval-benchmark_results.txt")
    try:
        with open(summary_path, "w") as f:
            f.write(header + "\n")
            f.write(separator + "\n")
            f.write(row_original + "\n")
            f.write(row_pruned + "\n")
    except Exception as e:
        print(f"Warning: Could not write summary to {summary_path}: {e}")
