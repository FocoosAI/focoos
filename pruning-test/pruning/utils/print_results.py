def format_row(label, model_name, metrics, folder_path):
    if metrics is None:
        return (
            f"{label:<16} | {model_name:<16} | {'N/A':>5} | {'N/A':<6} | {'N/A':<6} | {'N/A':<12} | {folder_path:<40}"
        )
    return (
        f"{label:<16} | {model_name:<16} | {metrics.fps:>5} | "
        f"{metrics.mean:<6} | "
        f"{metrics.std:<6} | {metrics.device:<12} | {folder_path:<40}"
    )


def print_results(result_original_model, result_pruned_model, MODEL_NAME, OUTPUT_DIRECTORY):
    """Print benchmark results in a formatted table."""
    print("\nBenchmark Results:")

    ORIGINAL_FOLDER = "Focoos AI ModelRegistry"  # Assuming OUTPUT_DIRECTORY is for original model

    # Print formatted table
    header = (
        f"{'Model':<16} | {'MODEL_NAME':<16} | {'FPS':>5} | "
        f"{'Mean':<6} | {'Std':<6} | {'Device':<12} | {'Folder path':<40}"
    )
    print(header)
    print("-" * len(header))
    print(format_row("Original Model", MODEL_NAME, result_original_model, ORIGINAL_FOLDER))
    print(format_row("Pruned Model", MODEL_NAME, result_pruned_model, OUTPUT_DIRECTORY))
