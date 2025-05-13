from datetime import datetime

import orjson
from colorama import Fore, Style

from focoos.ports import Metrics

OBJECT_DETECTIONS_VALIDATION_METRICS = [
    "bbox/AP",
    "bbox/AP50",
    "bbox/AP75",
    "bbox/APs",
    "bbox/APm",
    "bbox/APl",
]
SEGMENTATION_VALIDATION_METRICS = [
    "sem_seg/mIoU",
    "sem_seg/fwIoU",
    "sem_seg/pACC",
    "sem_seg/mAcc",
]
INSTANCE_SEGMENTATION_VALIDATION_METRICS = [
    "segm/AP",
    "segm/AP50",
    "segm/AP75",
    "segm/APs",
    "segm/APm",
    "segm/APl",
]
PANOPTIC_SEGMENTATION_VALIDATION_METRICS = ["panoptic_seg/PQ"]


class MetricsVisualizer:
    def __init__(self, metrics: Metrics):
        self.metrics = metrics
        self.green_up = f"{Fore.GREEN}(↑){Style.RESET_ALL}"
        self.red_down = f"{Fore.RED}(↓){Style.RESET_ALL}"

    def log_metrics(self):
        def format_time_metrics(iteration, metric, is_valid=False, is_best=False):
            color = Fore.GREEN if is_valid else Fore.BLUE
            star = f"{Fore.YELLOW}*{Style.RESET_ALL}" if is_best else ""
            text = f"{color}[iter {iteration} {'valid' if is_valid else ''}]{star}{Style.RESET_ALL}: "
            metric_text = []
            for key, value in metric.items():
                if key not in ["device", "engine", "is_valid"]:
                    metric_text.append(f"{key}: {value}")
            return text + " ".join(metric_text)

        def format_infer_metrics(metric):
            color = Fore.BLUE
            text = f"{color}[device={metric.get('device')}, engine={metric.get('engine')}]{Style.RESET_ALL}: "
            for key, value in metric.items():
                if key != "device" and key != "engine":
                    text += f"{key}: {value} "
            return text

        if self.metrics.train_metrics:
            print(f"{Fore.YELLOW}[Training metrics]{Style.RESET_ALL}")
            for item in self.metrics.valid_metrics:
                item["is_valid"] = True
            ordered_metrics = sorted(
                self.metrics.train_metrics + self.metrics.valid_metrics, key=lambda x: int(x.get("iteration", -1))
            )
            previous_valid_metric = None
            for metric in ordered_metrics:
                iter = metric["iteration"]
                is_valid = metric.get("is_valid", False)
                is_best = self.metrics.best_valid_metric and self.metrics.best_valid_metric["iteration"] == iter
                text = format_time_metrics(iter, metric, is_valid, is_best)  # type: ignore

                if is_valid and previous_valid_metric:
                    for key, value in metric.items():
                        if key in previous_valid_metric and key not in ["device", "engine", "is_valid", "iteration"]:
                            prev_value = previous_valid_metric[key]
                            if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
                                if value > prev_value:
                                    text = text.replace(f"{key}: {value}", f"{key}: {value} {self.green_up}")
                                elif value < prev_value:
                                    text = text.replace(f"{key}: {value}", f"{key}: {value} {self.red_down}")
                    previous_valid_metric = metric
                elif is_valid:
                    previous_valid_metric = metric
                print(text)

        if self.metrics.infer_metrics:
            print(f"{Fore.YELLOW}[Inference metrics]{Style.RESET_ALL}")
            for metric in self.metrics.infer_metrics:
                text = format_infer_metrics(metric)
                print(text)

    def notebook_plot_training_metrics(self):
        """
        Plots training and validation metrics on a grid.
        Each key containing 'loss' is plotted as a separate line, excluding 'total_loss'.
        'total_loss' and valid_metrics are plotted on separate graphs.
        """
        import matplotlib.pyplot as plt

        def plot_on_axis(ax, x_values, y_values_dict, xlabel, ylabel, title):
            for label, y_values in y_values_dict.items():
                ax.plot(x_values, y_values, linestyle="-", label=label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)

        if not self.metrics.train_metrics:
            print("No training metrics available to plot.")
            return

        iterations = [metric["iteration"] for metric in self.metrics.train_metrics]
        loss_keys = [key for key in self.metrics.train_metrics[0].keys() if "loss" in key and key != "total_loss"]
        losses_dict = {
            loss_key: [metric[loss_key] for metric in self.metrics.train_metrics if loss_key in metric]
            for loss_key in loss_keys
        }

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 12))

        plot_on_axis(ax1, iterations, losses_dict, "Iterations", "Loss", "Training Metrics: Losses vs Iterations")

        if "total_loss" in self.metrics.train_metrics[0]:
            total_losses = [metric["total_loss"] for metric in self.metrics.train_metrics if "total_loss" in metric]
            plot_on_axis(
                ax2,
                iterations,
                {"total_loss": total_losses},
                "Iterations",
                "Total Loss",
                "Training Metrics: Total Loss vs Iterations",
            )

        if self.metrics.valid_metrics:
            val_keys_start = ["bbox/", "sem_seg/", "segm/", "panoptic_seg/PQ"]
            valid_iterations = [metric["iteration"] for metric in self.metrics.valid_metrics]
            valid_keys = [
                key
                for key in self.metrics.valid_metrics[0].keys()
                if any(key.startswith(prefix) for prefix in val_keys_start)
            ]
            valid_values_dict = {
                valid_key: [metric[valid_key] for metric in self.metrics.valid_metrics if valid_key in metric]
                for valid_key in valid_keys
            }
            plot_on_axis(
                ax3,
                valid_iterations,
                valid_values_dict,
                "Iterations",
                "Validation Metrics",
                "Validation Metrics vs Iterations",
            )
        if self.metrics.infer_metrics:
            infer_fps = [metric["fps"] for metric in self.metrics.infer_metrics if "fps" in metric]
            if infer_fps:
                ax4.bar(range(len(infer_fps)), infer_fps, color="skyblue")
                ax4.set_xlabel("Inference Sample")
                ax4.set_ylabel("FPS")
                ax4.set_title("Inference Performance (FPS)")
                ax4.grid(True, axis="y")

        plt.tight_layout()
        plt.show()


def parse_metrics(metrics_text: str) -> Metrics:
    """
    Parse the given metrics text and extract validation, training, and inference metrics.

    Args:
        metrics_text (str): A string containing JSON lines of metrics data.

    Returns:
        FocoosMetrics: An instance of FocoosMetrics containing parsed metrics categorized into
                       'valid_metrics', 'train_metrics', 'infer_metrics', along with 'iterations'
                       and 'best_valid_metric'.
    """
    # Initialize the FocoosMetrics object with empty lists for metrics and None for iterations
    res = Metrics(
        valid_metrics=[],
        train_metrics=[],
        infer_metrics=[],
        iterations=None,
        best_valid_metric=None,
    )

    # Parse the input metrics text into a list of dictionaries, ignoring empty lines
    content = [orjson.loads(line.replace("NaN", "null")) for line in metrics_text.split("\n") if line.strip()]

    # Create a set of all possible validation metrics for easy lookup
    #! TODO: this is a hack to get the best metric to use delta, not stable for overlapping metrics
    valid_metrics_set = set(
        INSTANCE_SEGMENTATION_VALIDATION_METRICS
        + OBJECT_DETECTIONS_VALIDATION_METRICS
        + SEGMENTATION_VALIDATION_METRICS
        + PANOPTIC_SEGMENTATION_VALIDATION_METRICS
    )

    # Iterate over each metric dictionary in the content
    for metric in content:
        # Filter out keys that end with a _digit, as they are not needed (loss_100, etc)
        metric = {
            k: round(v, 4) if isinstance(v, (int, float)) else v
            for k, v in metric.items()
            if not (k[-2] == "_" and k[-1].isdigit())
        }
        # Determine if the metric belongs to validation or training based on the presence of validation metrics
        if valid_metrics_set.intersection(metric):
            res.valid_metrics.append(metric)
        else:
            res.train_metrics.append(metric)

    # If there are training metrics, set the total iterations to the last iteration value
    if len(res.train_metrics) > 0:
        res.iterations = res.train_metrics[-1].get("iteration", -1)

    # If there are validation metrics, update total iterations and find the best iteration
    if len(res.valid_metrics) > 0:
        # Update total iterations if the last validation iteration is greater
        if res.valid_metrics[-1].get("iteration", -1) > (res.iterations or 0):
            res.iterations = res.valid_metrics[-1].get("iteration")

        delta_keys = [
            INSTANCE_SEGMENTATION_VALIDATION_METRICS[0],
            OBJECT_DETECTIONS_VALIDATION_METRICS[0],
            SEGMENTATION_VALIDATION_METRICS[0],
            PANOPTIC_SEGMENTATION_VALIDATION_METRICS[0],
        ]

        # Determine the best metric to use for finding the best iteration
        best_metric = None
        for k in delta_keys:
            if k in res.valid_metrics[0]:
                best_metric = k
                break
        print(f"best metric to use delta: {best_metric}")
        # If a best metric is found, determine the best iteration based on it
        if best_metric:
            res.best_valid_metric = max(res.valid_metrics, key=lambda x: x.get(best_metric, 0))
    res.updated_at = datetime.now()
    return res
