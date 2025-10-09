import os

import torch
from layers_to_prune import layers_prunable_fai_cls_n_coco
from pruning.utils.print_results import calculate_model_size_mb, load_eval_metrics_from_model_info, print_results
from pruning.utils.utils import PrunedBaseModel, PruningCompatibleModel, prune_model_with_torch_pruning

from focoos import DATASETS_DIR, MODELS_DIR, DatasetLayout, DatasetSplitType, ModelManager, Task, TrainerArgs
from focoos.data import AutoDataset, get_default_by_task
from focoos.ports import get_gpus_count
from focoos.utils.logger import get_logger

logger = get_logger("pruning_and_benchmark")


class FocoosPruning:
    def __init__(
        self,
        task,
        dataset_name,
        dataset_layout,
        device,
        verbose,
        do_eval,
        root_dir,
        model_name,
        resolution,
        prune_ratio,
        benchmark_iterations,
        layers_to_prune,
    ):
        self.logger = get_logger("focoos_pruning")

        # Configuration parameters
        self.task = task
        self.dataset_name = dataset_name
        self.dataset_layout = dataset_layout
        self.device = device
        self.verbose = verbose
        self.do_eval = do_eval
        self.root_dir = root_dir
        self.model_name = model_name
        self.resolution = resolution
        self.prune_ratio = prune_ratio
        self.benchmark_iterations = benchmark_iterations
        self.layers_to_prune = layers_to_prune

        # Pipeline state
        self.focoos_model = None
        self.original_model = None
        self.original_model_size_mb = None
        self.auto_dataset = None
        self.valid_dataset = None
        self.trainer_args = None
        self.original_eval_metrics = None
        self.result_original_model = None
        self.model = None
        self.output_directory = None
        self.output_path = None
        self.model_pruned = None
        self.model_pruned_wrapper = None
        self.result_pruned_model = None
        self.pruned_eval_metrics = None
        self.pruned_model_size_mb = None

    def run(self):
        """Main pipeline execution"""
        self.logger.info("Starting pruning and benchmark pipeline")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Resolution: {self.resolution}")
        self.logger.info(f"Prune ratio: {self.prune_ratio}")
        self.logger.info(f"Layers to prune: {self.layers_to_prune}")
        self.logger.info(f"Benchmark iterations: {self.benchmark_iterations}")

        self.load_model()
        self.evaluate_model()
        self.benchmark_model()
        self.prune_model()
        self.save_model_info()
        self.prepare_pruned_model()
        self.evaluate_pruned_model()
        self.benchmark_pruned_model()
        self.calculate_pruned_model_size()
        self.print_results()

        self.logger.info("Pruning pipeline completed successfully")
        self.logger.info(f"Output directory: {self.output_directory}")

    def load_model(self):
        """Load the original model and calculate its size"""
        self.logger.info(f"1/12 - Loading model: {self.model_name}")
        self.focoos_model = ModelManager.get(self.model_name)
        self.original_model = self.focoos_model.model

        # Calculate original model size
        self.logger.info("1.1/12 - Calculating original model size")
        original_model_path = os.path.expanduser(os.path.join(MODELS_DIR, self.model_name, "model_final.pth"))
        if os.path.exists(original_model_path):
            self.original_model_size_mb = calculate_model_size_mb(original_model_path)
        else:
            self.original_model_size_mb = None

    def evaluate_model(self):
        """Evaluate the original model"""
        self.logger.info("2/12 - Evaluating original model")
        self.auto_dataset = AutoDataset(
            dataset_name=self.dataset_name,
            task=self.task,
            layout=self.dataset_layout,
            datasets_dir=DATASETS_DIR,
        )
        train_augs, val_augs = get_default_by_task(self.task, resolution=self.resolution)
        self.valid_dataset = self.auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

        self.trainer_args = TrainerArgs(
            run_name=f"{self.focoos_model.name}_{self.valid_dataset.name}",
            batch_size=16,
            max_iters=50,
            eval_period=50,
            learning_rate=0.0008,
            sync_to_hub=False,
            device=self.device,
            num_gpus=get_gpus_count() if self.device == "cuda" else -1,
        )

        # Evaluate original model
        if self.do_eval:
            self.focoos_model.eval(self.trainer_args, self.valid_dataset)
        original_eval_dir = os.path.join(self.trainer_args.output_dir, f"{self.trainer_args.run_name.strip()}_eval")
        original_model_info_path = os.path.join(original_eval_dir, "model_info.json")
        self.original_eval_metrics = load_eval_metrics_from_model_info(original_model_info_path, task_type=self.task)

    def benchmark_model(self):
        """Benchmark the original model"""
        self.logger.info("3/12 - Benchmarking original model")
        self.result_original_model = self.focoos_model.benchmark(
            iterations=self.benchmark_iterations,
            size=(self.resolution, self.resolution),
            device=self.device,
        )

    def prune_model(self):
        """Wrap model for pruning compatibility and run pruning"""
        self.logger.info("4/12 - Wrapping model for pruning compatibility")
        self.model = PruningCompatibleModel(self.original_model)

        # Create output directory
        self.logger.info("5/12 - Creating output directory")
        NAME = f"{self.model_name}-pruned"
        FOLDER_NAME = f"{NAME}_RATIO={self.prune_ratio}_NUM_LAYERS={len(self.layers_to_prune)}"
        self.output_directory = f"{self.root_dir}/models/{FOLDER_NAME}"
        os.makedirs(self.output_directory, exist_ok=True)

        # Run pruning
        self.logger.info(f"6/12 - Running pruning with ratio {self.prune_ratio} on {len(self.layers_to_prune)} layers")
        dummy_input = torch.randn(1, 3, self.resolution, self.resolution)
        self.output_path = os.path.join(self.output_directory, "model_pruned.pth")

        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        prune_model_with_torch_pruning(
            self.model,
            dummy_input,
            self.layers_to_prune,
            prune_ratio=self.prune_ratio,
            norm_type=2,
            output_path=self.output_path,
            verbose=self.verbose,
        )
        self.logger.info(f"Pruned model saved to {self.output_path}")

    def save_model_info(self):
        """Save model structure and state dict info"""
        self.logger.info("7/12 - Saving model structure and state dict info")
        NAME = f"{self.model_name}-pruned"
        FOLDER_NAME = f"{NAME}_RATIO={self.prune_ratio}_NUM_LAYERS={len(self.layers_to_prune)}"

        with open(os.path.join(self.output_directory, f"layers_{FOLDER_NAME}.txt"), "w") as f:
            f.write(str(self.model))

        state_dict_path = os.path.join(self.output_directory, f"state_dict_{FOLDER_NAME}.txt")
        state_dict_path_shape = os.path.join(self.output_directory, f"state_dict_shape_{FOLDER_NAME}.txt")
        if os.path.exists(state_dict_path):
            os.remove(state_dict_path)
        if os.path.exists(state_dict_path_shape):
            os.remove(state_dict_path_shape)
        with open(state_dict_path, "a") as f:
            for k, v in self.model.state_dict().items():
                print(f"{k}: {v}", file=f)
        with open(state_dict_path_shape, "a") as f:
            for k, v in self.model.state_dict().items():
                print(f"{k}: {v.shape}", file=f)

    def prepare_pruned_model(self):
        """Load and prepare pruned model for export"""
        self.logger.info("8/12 - Loading pruned model and preparing for export")
        self.model_pruned = torch.load(self.output_path, map_location="cpu", weights_only=False)
        state_dict = self.model_pruned.state_dict()

        # Remove ".model" prefix from state_dict
        PREFIX = "model."
        keys_to_update = [k for k in state_dict.keys() if k.startswith(PREFIX)]
        for k in keys_to_update:
            state_dict[k.replace(PREFIX, "")] = state_dict.pop(k)

        self.model_pruned.state_dict = state_dict

        # Verify state_dict is correct
        for i, k in enumerate(state_dict.keys()):
            if k.startswith(PREFIX):
                print(f"Error: {k} starts with {PREFIX} at index {i}")
                exit()

        self.logger.info("State_dict is correct")

        # Create PrunedBaseModel wrapper
        self.logger.info("9/12 - Creating PrunedBaseModel wrapper")
        input_tensor = torch.randn(1, 3, self.resolution, self.resolution).to(self.device)
        self.model_pruned_wrapper = PrunedBaseModel(
            self.model_pruned, config=self.focoos_model.model.config, device=self.device
        )
        self.model_pruned_wrapper = self.model_pruned_wrapper.to(self.device)
        self.model_pruned_wrapper.eval()

        # Warm up the model
        self.logger.info("10/12 - Warming up the model")
        for i in range(50):
            self.model_pruned_wrapper(input_tensor)

    def evaluate_pruned_model(self):
        """Evaluate the pruned model"""
        self.logger.info("11/12 - Evaluating pruned model")
        self.focoos_model.model = self.model_pruned_wrapper

        # Evaluate pruned model (reusing the same trainer_args)
        if self.do_eval:
            self.focoos_model.eval(self.trainer_args, self.valid_dataset)
        pruned_eval_dir = os.path.join(self.trainer_args.output_dir, f"{self.trainer_args.run_name.strip()}_eval")
        pruned_model_info_path = os.path.join(pruned_eval_dir, "model_info.json")
        self.pruned_eval_metrics = load_eval_metrics_from_model_info(pruned_model_info_path, task_type=self.task)

    def benchmark_pruned_model(self):
        """Benchmark the pruned model"""
        self.logger.info("12/12 - Benchmarking pruned model")
        self.result_pruned_model = self.focoos_model.benchmark(
            iterations=self.benchmark_iterations,
            size=(self.resolution, self.resolution),
            device=self.device,
        )

    def calculate_pruned_model_size(self):
        """Calculate pruned model size"""
        self.logger.info("13/13 - Calculating pruned model size")
        pruned_model_path = os.path.join(self.output_directory, "model_pruned.pth")
        self.pruned_model_size_mb = calculate_model_size_mb(pruned_model_path)
        self.logger.info(f"Pruned model size: {self.pruned_model_size_mb:.2f} MB")

    def print_results(self):
        """Print final results"""
        print_results(
            self.result_original_model,
            self.result_pruned_model,
            self.model_name,
            self.output_directory,
            self.original_eval_metrics,
            self.pruned_eval_metrics,
            task_type=self.task,
            original_model_size_mb=self.original_model_size_mb,
            pruned_model_size_mb=self.pruned_model_size_mb,
        )


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Configuration
    TASK = Task.CLASSIFICATION
    DATASET_NAME = "coco_2017_cls"
    DATASET_LAYOUT = DatasetLayout.CATALOG
    DEVICE = "cpu"
    VERBOSE = False
    DO_EVAL = True  # Do not compute eval metrics

    MODEL_NAME = "fai-cls-n-coco"
    RESOLUTION = 224
    PRUNE_RATIO = 0.99
    BENCHMARK_ITERATIONS = 10

    # Get layers to prune from layers_to_prune.py
    LAYERS_TO_PRUNE = layers_prunable_fai_cls_n_coco

    pipeline = FocoosPruning(
        task=TASK,
        dataset_name=DATASET_NAME,
        dataset_layout=DATASET_LAYOUT,
        device=DEVICE,
        verbose=VERBOSE,
        do_eval=DO_EVAL,
        root_dir=root_dir,
        model_name=MODEL_NAME,
        resolution=RESOLUTION,
        prune_ratio=PRUNE_RATIO,
        benchmark_iterations=BENCHMARK_ITERATIONS,
        layers_to_prune=LAYERS_TO_PRUNE,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
