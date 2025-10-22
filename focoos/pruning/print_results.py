import json
import os

from focoos import Task

# Define column widths for alignment - shared across functions
COL_WIDTHS = {
    "label": 17,  # 16 + 1 for space
    "fps": 7,  # 5 + 2 for space
    "mean": 11,  # 9 + 2 for space
    "std": 10,  # 8 + 2 for space
    "size": 11,  # 8 + 3 for space
    "metric": 11,  # 10 + 1 for space
    "folder": 40,
}


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


def format_row(label, metrics, folder_path, eval_metrics=None, metric_columns=None, model_size_mb=None):
    if metrics is None:
        fps_str = f"{'N/A':>{COL_WIDTHS['fps'] - 2}}"
        mean_str = f"{'N/A':<{COL_WIDTHS['mean'] - 2}}"
        std_str = f"{'N/A':<{COL_WIDTHS['std'] - 2}}"
    else:
        fps_str = f"{metrics.fps:>{COL_WIDTHS['fps'] - 2}}"
        mean_str = f"{metrics.mean:<{COL_WIDTHS['mean'] - 2}}"
        std_str = f"{metrics.std:<{COL_WIDTHS['std'] - 2}}"

    # Format model size
    if model_size_mb is not None:
        size_str = f"{model_size_mb:<{COL_WIDTHS['size'] - 2}.2f}"
    else:
        size_str = f"{'N/A':<{COL_WIDTHS['size'] - 2}}"

    # Format evaluation metrics dynamically based on available metrics
    eval_strs = []
    if eval_metrics is not None and metric_columns is not None:
        for col in metric_columns:
            value = eval_metrics.get(col, "N/A")
            if value != "N/A":
                eval_strs.append(f"{float(value):<{COL_WIDTHS['metric'] - 1}.3f}")
            else:
                eval_strs.append(f"{'N/A':<{COL_WIDTHS['metric'] - 1}}")
    else:
        # Fallback to default columns if no metrics available
        eval_strs = [f"{'N/A':<{COL_WIDTHS['metric'] - 1}}" for _ in range(3)]

    eval_metrics_str = " | ".join(eval_strs)

    return (
        f"{label:<{COL_WIDTHS['label']}}|"
        f" {fps_str} |"
        f" {mean_str} |"
        f" {std_str} |"
        f" {size_str} | "
        f"{eval_metrics_str}| "
        f"{folder_path:<{COL_WIDTHS['folder']}}"
    )


def show_results(
    result_original_model,
    result_pruned_model,
    original_model_directory,
    MODEL_NAME,
    OUTPUT_DIRECTORY,
    original_eval_metrics=None,
    pruned_eval_metrics=None,
    task_type=Task.DETECTION,
    original_model_size_mb=None,
    pruned_model_size_mb=None,
    resolution=None,
    prune_ratio=None,
):
    """Print benchmark results in a formatted table, save to summary.txt, and return summary as dict."""

    # Try to get device from the result objects, fallback to "N/A"
    device = None
    if result_original_model and hasattr(result_original_model, "device"):
        device = result_original_model.device
    elif result_pruned_model and hasattr(result_pruned_model, "device"):
        device = result_pruned_model.device
    else:
        device = "N/A"

    print("\nBenchmark Results:\n")
    print(f"- Model: {MODEL_NAME}")
    print(f"- Device: {device}")
    print(f"- Resolution: {resolution}")
    print(f"- Prune ratio: {prune_ratio}")
    print()

    if original_model_directory:
        original_folder = original_model_directory
    else:
        original_folder = "Focoos AI ModelRegistry"

    # Determine metric columns based on available metrics
    metric_columns = []
    if original_eval_metrics:
        metric_columns = list(original_eval_metrics.keys())
    elif pruned_eval_metrics:
        metric_columns = list(pruned_eval_metrics.keys())
    else:
        # Fallback to default classification metrics
        metric_columns = ["F1", "Precision", "Recall"]

    # Create header dynamically, fixed width for each metric column
    metric_header = " | ".join([f"{col:<{COL_WIDTHS['metric'] - 1}}" for col in metric_columns])

    # Format header strings to match data row spacing
    fps_header = f"{'FPS':>{COL_WIDTHS['fps'] - 2}}"
    mean_header = f"{'Mean (ms)':<{COL_WIDTHS['mean'] - 2}}"
    std_header = f"{'Std (ms)':<{COL_WIDTHS['std'] - 2}}"
    size_header = f"{'Size (MB)':<{COL_WIDTHS['size'] - 2}}"

    header = (
        f"{'':<{COL_WIDTHS['label']}}|"
        f" {fps_header} |"
        f" {mean_header} |"
        f" {std_header} |"
        f" {size_header} | "
        f"{metric_header}| "
        f"{'Folder path':<{COL_WIDTHS['folder']}}"
    )
    separator = "-" * len(header)

    row_original = format_row(
        "Original Model",
        result_original_model,
        original_folder,
        original_eval_metrics,
        metric_columns,
        original_model_size_mb,
    )
    row_pruned = format_row(
        "Pruned Model",
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
    print(separator)
    print()

    # Save to summary.txt in OUTPUT_DIRECTORY
    summary_path = os.path.join(OUTPUT_DIRECTORY, "summary_eval-benchmark_results.txt")
    try:
        with open(summary_path, "w") as f:
            f.write("Benchmark Results:\n")
            f.write(f"- Model: {MODEL_NAME}\n")
            f.write(f"- Device: {device}\n")
            f.write(f"- Resolution: {resolution}\n")
            f.write(f"- Prune ratio: {prune_ratio}\n")
            f.write("\n")
            f.write(header + "\n")
            f.write(separator + "\n")
            f.write(row_original + "\n")
            f.write(row_pruned + "\n")
            f.write(separator + "\n")
    except Exception as e:
        print(f"Warning: Could not write summary to {summary_path}: {e}")

    # Helper to collect all relevant metrics into a dict
    def _extract_result(label, metrics, folder_path, eval_metrics, metric_columns, model_size_mb):
        result = {
            "label": label,
            "fps": None,
            "mean": None,
            "std": None,
            "size_mb": model_size_mb if model_size_mb is not None else "N/A",
            "folder_path": folder_path,
            "metrics": {},
        }
        if metrics is not None:
            result["fps"] = metrics.fps if hasattr(metrics, "fps") else "N/A"
            result["mean"] = float(metrics.mean) if hasattr(metrics, "mean") else "N/A"
            result["std"] = float(metrics.std) if hasattr(metrics, "std") else "N/A"
        else:
            result["fps"] = "N/A"
            result["mean"] = "N/A"
            result["std"] = "N/A"

        # Fill in each eval metric from metric_columns
        if eval_metrics is not None and metric_columns is not None:
            for col in metric_columns:
                value = eval_metrics.get(col, "N/A")
                try:
                    # Try to cast to float if not "N/A"
                    if value != "N/A":
                        value = float(value)
                except Exception:
                    pass
                result["metrics"][col] = value
        else:
            for col in metric_columns:
                result["metrics"][col] = "N/A"
        return result

    dict_results = {
        "model": MODEL_NAME,
        "device": device,
        "resolution": resolution,
        "prune_ratio": prune_ratio,
        "results": {
            "original_model": _extract_result(
                "Original Model",
                result_original_model,
                original_folder,
                original_eval_metrics,
                metric_columns,
                original_model_size_mb,
            ),
            "pruned_model": _extract_result(
                "Pruned Model",
                result_pruned_model,
                OUTPUT_DIRECTORY,
                pruned_eval_metrics,
                metric_columns,
                pruned_model_size_mb,
            ),
        },
    }

    with open(os.path.join(OUTPUT_DIRECTORY, "summary_results.json"), "w") as f:
        json.dump(dict_results, f, indent=4)

    return dict_results
