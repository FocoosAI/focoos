import os
from typing import List

import torch
from layers_to_prune import layers_prunable_fai_cls_n_coco
from pruning.utils.print_results import calculate_model_size_mb, load_eval_metrics_from_model_info, print_results
from pruning.utils.utils import PrunedBaseModel, PruningCompatibleModel, prune_model_with_torch_pruning

from focoos import DATASETS_DIR, MODELS_DIR, DatasetLayout, DatasetSplitType, ModelManager, Task, TrainerArgs
from focoos.data import AutoDataset, get_default_by_task
from focoos.ports import get_gpus_count
from focoos.utils.logger import get_logger

logger = get_logger("pruning_and_benchmark")

# Configuration
TASK = Task.CLASSIFICATION
DATASET_NAME = "coco_2017_cls"
DATASET_LAYOUT = DatasetLayout.CATALOG
DEVICE = "cpu"
VERBOSE = False
DO_EVAL = True  # Do not compute eval metrics

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "fai-cls-n-coco"
RESOLUTION = 224
PRUNE_RATIO = 0.99
BENCHMARK_ITERATIONS = 10

# Get layers to prune from layers_to_prune.py
LAYERS_TO_PRUNE: List[str] = layers_prunable_fai_cls_n_coco


class FocoosPruning:
    def __init__(self):
        self.logger = get_logger("focoos_pruning")
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
        self.logger.info(f"Device: {DEVICE}")
        self.logger.info(f"Model: {MODEL_NAME}")
        self.logger.info(f"Resolution: {RESOLUTION}")
        self.logger.info(f"Prune ratio: {PRUNE_RATIO}")
        self.logger.info(f"Layers to prune: {LAYERS_TO_PRUNE}")
        self.logger.info(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")

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
        self.logger.info(f"1/12 - Loading model: {MODEL_NAME}")
        self.focoos_model = ModelManager.get(MODEL_NAME)
        self.original_model = self.focoos_model.model

        # Calculate original model size
        self.logger.info("1.1/12 - Calculating original model size")
        original_model_path = os.path.expanduser(os.path.join(MODELS_DIR, MODEL_NAME, "model_final.pth"))
        if os.path.exists(original_model_path):
            self.original_model_size_mb = calculate_model_size_mb(original_model_path)
        else:
            self.original_model_size_mb = None

    def evaluate_model(self):
        """Evaluate the original model"""
        self.logger.info("2/12 - Evaluating original model")
        self.auto_dataset = AutoDataset(
            dataset_name=DATASET_NAME,
            task=TASK,
            layout=DATASET_LAYOUT,
            datasets_dir=DATASETS_DIR,
        )
        train_augs, val_augs = get_default_by_task(TASK, resolution=RESOLUTION)
        self.valid_dataset = self.auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

        self.trainer_args = TrainerArgs(
            run_name=f"{self.focoos_model.name}_{self.valid_dataset.name}",
            batch_size=16,
            max_iters=50,
            eval_period=50,
            learning_rate=0.0008,
            sync_to_hub=False,
            device=DEVICE,
            num_gpus=get_gpus_count() if DEVICE == "cuda" else -1,
        )

        # Evaluate original model
        if DO_EVAL:
            self.focoos_model.eval(self.trainer_args, self.valid_dataset)
        original_eval_dir = os.path.join(self.trainer_args.output_dir, f"{self.trainer_args.run_name.strip()}_eval")
        original_model_info_path = os.path.join(original_eval_dir, "model_info.json")
        self.original_eval_metrics = load_eval_metrics_from_model_info(original_model_info_path, task_type=TASK)

    def benchmark_model(self):
        """Benchmark the original model"""
        self.logger.info("3/12 - Benchmarking original model")
        self.result_original_model = self.focoos_model.benchmark(
            iterations=BENCHMARK_ITERATIONS,
            size=(RESOLUTION, RESOLUTION),
            device=DEVICE,
        )

    def prune_model(self):
        """Wrap model for pruning compatibility and run pruning"""
        self.logger.info("4/12 - Wrapping model for pruning compatibility")
        self.model = PruningCompatibleModel(self.original_model)

        # Create output directory
        self.logger.info("5/12 - Creating output directory")
        NAME = f"{MODEL_NAME}-pruned"
        FOLDER_NAME = f"{NAME}_RATIO={PRUNE_RATIO}_NUM_LAYERS={len(LAYERS_TO_PRUNE)}"
        self.output_directory = f"{ROOT_DIR}/models/{FOLDER_NAME}"
        os.makedirs(self.output_directory, exist_ok=True)

        # Run pruning
        self.logger.info(f"6/12 - Running pruning with ratio {PRUNE_RATIO} on {len(LAYERS_TO_PRUNE)} layers")
        dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)
        self.output_path = os.path.join(self.output_directory, "model_pruned.pth")

        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        prune_model_with_torch_pruning(
            self.model,
            dummy_input,
            LAYERS_TO_PRUNE,
            prune_ratio=PRUNE_RATIO,
            norm_type=2,
            output_path=self.output_path,
            verbose=VERBOSE,
        )
        self.logger.info(f"Pruned model saved to {self.output_path}")

    def save_model_info(self):
        """Save model structure and state dict info"""
        self.logger.info("7/12 - Saving model structure and state dict info")
        NAME = f"{MODEL_NAME}-pruned"
        FOLDER_NAME = f"{NAME}_RATIO={PRUNE_RATIO}_NUM_LAYERS={len(LAYERS_TO_PRUNE)}"

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
        input_tensor = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
        self.model_pruned_wrapper = PrunedBaseModel(
            self.model_pruned, config=self.focoos_model.model.config, device=DEVICE
        )
        self.model_pruned_wrapper = self.model_pruned_wrapper.to(DEVICE)
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
        if DO_EVAL:
            self.focoos_model.eval(self.trainer_args, self.valid_dataset)
        pruned_eval_dir = os.path.join(self.trainer_args.output_dir, f"{self.trainer_args.run_name.strip()}_eval")
        pruned_model_info_path = os.path.join(pruned_eval_dir, "model_info.json")
        self.pruned_eval_metrics = load_eval_metrics_from_model_info(pruned_model_info_path, task_type=TASK)

    def benchmark_pruned_model(self):
        """Benchmark the pruned model"""
        self.logger.info("12/12 - Benchmarking pruned model")
        self.result_pruned_model = self.focoos_model.benchmark(
            iterations=BENCHMARK_ITERATIONS,
            size=(RESOLUTION, RESOLUTION),
            device=DEVICE,
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
            MODEL_NAME,
            self.output_directory,
            self.original_eval_metrics,
            self.pruned_eval_metrics,
            task_type=TASK,
            original_model_size_mb=self.original_model_size_mb,
            pruned_model_size_mb=self.pruned_model_size_mb,
        )


def main():
    """Main function to run the pruning pipeline"""
    pipeline = FocoosPruning()
    pipeline.run()


if __name__ == "__main__":
    main()
