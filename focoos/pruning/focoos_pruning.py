import os

import torch

from focoos import (
    DATASETS_DIR,
    MODELS_DIR,
    DatasetSplitType,
    ModelManager,
    RuntimeType,
    TrainerArgs,
)
from focoos.data import AutoDataset, get_default_by_task
from focoos.ports import get_gpus_count
from focoos.pruning.print_results import calculate_model_size_mb, load_eval_metrics_from_model_info, show_results
from focoos.pruning.utils import PrunedBaseModel, PruningCompatibleModel, prune_model_with_torch_pruning
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
        self.runtime_type = RuntimeType.TORCHSCRIPT_32  # do not change, other runtimes are not supported now

        # Pipeline state
        self.original_model = None
        self.original_model_size_mb = None
        self.auto_dataset = None
        self.valid_dataset = None
        self.trainer_args = None
        self.original_eval_metrics = None
        self.result_original_model = None
        self.model = None
        self.original_model_directory = None
        self.output_path = None
        self.model_pruned = None
        self.model_pruned_wrapper = None
        self.result_pruned_model = None
        self.pruned_eval_metrics = None
        self.pruned_model_size_mb = None

    def run(self):
        """Main pipeline execution"""
        self.logger.info("Starting pruning pipeline")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Resolution: {self.resolution}")
        self.logger.info(f"Prune ratio: {self.prune_ratio}")
        self.logger.info(f"Layers to prune: {self.layers_to_prune}")
        self.logger.info(f"Benchmark iterations: {self.benchmark_iterations}")

        # Create output directory
        self.logger.info("Creating output directory")
        if os.path.isdir(self.model_name):
            model_name = os.path.basename(self.model_name)
            self.original_model_directory = self.model_name
        else:
            model_name = self.model_name
        NAME = f"{model_name}-pruned"
        FOLDER_NAME = f"{NAME}_RATIO={self.prune_ratio}_NUM_LAYERS={len(self.layers_to_prune)}"
        self.output_directory = f"{self.root_dir}/models/{FOLDER_NAME}"
        os.makedirs(self.output_directory, exist_ok=True)

        self._load_model()
        self._evaluate_model()
        self._benchmark_model()
        self._prune_model()
        self._save_model_info()
        self._prepare_pruned_model()
        self._evaluate_pruned_model()
        self._benchmark_pruned_model()
        self._calculate_pruned_model_size()
        results = self._print_results()

        self.logger.info("Pruning pipeline completed successfully")
        self.logger.info(f"Output directory: {self.output_directory}")

        return results

    def _load_model(self):
        """Load the original model and calculate its size"""
        folder_name = self.model_name
        if self.original_model_directory:
            folder_name = os.path.basename(self.original_model_directory)
        self.logger.info(f"Loading model: {self.model_name}")
        self.focoos_model = ModelManager.get(self.model_name)
        self.original_model = self.focoos_model.model

        # Calculate original model size
        self.logger.info("Calculating original model size")
        if self.original_model_directory:
            original_model_path = os.path.join(self.original_model_directory, "model_final.pth")
        else:
            original_model_path = os.path.expanduser(os.path.join(MODELS_DIR, folder_name, "model_final.pth"))

        if os.path.exists(original_model_path):
            self.original_model_size_mb = calculate_model_size_mb(original_model_path)
        else:
            self.original_model_size_mb = None

    def _evaluate_model(self):
        """Evaluate the original model"""
        self.logger.info("Evaluating original model")
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
            if self.original_model_directory and os.path.isdir(self.original_model_directory):
                self.trainer_args.output_dir = os.path.join(self.original_model_directory, "eval")
            else:
                folder_name = self.model_name
                self.trainer_args.output_dir = os.path.join(MODELS_DIR, folder_name, "eval")
            self.focoos_model.eval(self.trainer_args, self.valid_dataset)
            original_eval_dir = os.path.join(self.trainer_args.output_dir, f"{self.trainer_args.run_name.strip()}_eval")
            original_model_info_path = os.path.join(original_eval_dir, "model_info.json")
            self.original_eval_metrics = load_eval_metrics_from_model_info(
                original_model_info_path, task_type=self.task
            )

    def _benchmark_model(self):
        """Benchmark the original model"""
        self.logger.info("Benchmarking original model")

        optimized_model = self.focoos_model.export(runtime_type=self.runtime_type)
        self.result_original_model = optimized_model.benchmark(
            iterations=self.benchmark_iterations, size=(self.resolution, self.resolution)
        )

    def _prune_model(self):
        """Wrap model for pruning compatibility and run pruning"""
        self.logger.info("Wrapping model for pruning compatibility")
        self.model = PruningCompatibleModel(self.original_model, task=self.task, is_eval=False)

        # Run pruning
        self.logger.info(f"Running pruning with ratio {self.prune_ratio} on {len(self.layers_to_prune)} layers")
        dummy_input = torch.randn(1, 3, self.resolution, self.resolution)
        self.output_path = os.path.join(self.output_directory, "model_pruned.pth")

        model_info = self.focoos_model.model_info
        model_info.weights_uri = os.path.abspath(self.output_path)
        model_info.dump_json(os.path.join(self.output_directory, "model_info.json"))

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

    def _save_model_info(self):
        """Save model structure and state dict info"""
        self.logger.info("Saving model structure and state dict info")
        if os.path.isdir(self.model_name):
            self.model_name = os.path.basename(self.model_name)

        with open(os.path.join(self.output_directory, "layers.txt"), "w") as f:
            f.write(str(self.model))

        with open(os.path.join(self.output_directory, "layers_pruned.txt"), "w") as f:
            for layer in self.layers_to_prune:
                print(layer, file=f)

        state_dict_path = os.path.join(self.output_directory, "state_dict.txt")
        state_dict_path_shape = os.path.join(self.output_directory, "state_dict_shape.txt")
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

    def _prepare_pruned_model(self):
        """Load and prepare pruned model for export"""
        self.logger.info("Loading pruned model")
        state_dict = torch.load(self.output_path, map_location="cpu", weights_only=False)
        self.model_pruned = PruningCompatibleModel(self.original_model, task=self.task, is_eval=True)

        # Remove ".model" prefix from state_dict if present
        PREFIX = "model."
        keys_to_update = [k for k in state_dict.keys() if k.startswith(PREFIX)]
        for k in keys_to_update:
            state_dict[k.replace(PREFIX, "")] = state_dict.pop(k)

        # Load the cleaned state_dict into the model
        self.model_pruned.load_state_dict(state_dict)

        # Verify state_dict is correct
        for i, k in enumerate(state_dict.keys()):
            if k.startswith(PREFIX):
                print(f"Error: {k} starts with {PREFIX} at index {i}")
                exit()

        self.logger.info("State_dict is correct")

        # Create PrunedBaseModel wrapper
        self.logger.info("Creating PrunedBaseModel wrapper")
        input_tensor = torch.randn(1, 3, self.resolution, self.resolution).to(self.device)
        self.model_pruned_wrapper = PrunedBaseModel(
            self.model_pruned, config=self.focoos_model.model.config, device=self.device, task=self.task
        )
        self.model_pruned_wrapper = self.model_pruned_wrapper.to(self.device)
        self.model_pruned_wrapper.eval()
        self.model_pruned_wrapper.model.is_eval = True  # self.model_pruned.is_eval = True

        # Warm up the model
        self.logger.info("Warming up the model")
        for i in range(50):
            self.model_pruned_wrapper(input_tensor)

    def _evaluate_pruned_model(self):
        """Evaluate the pruned model"""
        self.logger.info("Evaluating pruned model")
        self.focoos_model.model = self.model_pruned_wrapper

        # Evaluate pruned model (reusing the same trainer_args)
        if self.do_eval:
            self.focoos_model.eval(self.trainer_args, self.valid_dataset)
            pruned_eval_dir = os.path.join(self.trainer_args.output_dir, f"{self.trainer_args.run_name.strip()}_eval")
            pruned_model_info_path = os.path.join(pruned_eval_dir, "model_info.json")
            self.pruned_eval_metrics = load_eval_metrics_from_model_info(pruned_model_info_path, task_type=self.task)

    def _benchmark_pruned_model(self):
        """Benchmark the pruned model"""
        self.logger.info("Benchmarking pruned model")
        optimized_model = self.focoos_model.export(runtime_type=self.runtime_type)
        self.result_pruned_model = optimized_model.benchmark(
            iterations=self.benchmark_iterations, size=(self.resolution, self.resolution)
        )

    def _calculate_pruned_model_size(self):
        """Calculate pruned model size"""
        self.logger.info("Calculating pruned model size")
        pruned_model_path = os.path.join(self.output_directory, "model_pruned.pth")
        self.pruned_model_size_mb = calculate_model_size_mb(pruned_model_path)
        self.logger.info(f"Pruned model size: {self.pruned_model_size_mb:.2f} MB")

    def _print_results(self):
        """Print final results"""
        results = show_results(
            self.result_original_model,
            self.result_pruned_model,
            self.original_model_directory,
            self.model_name,
            self.output_directory,
            self.original_eval_metrics,
            self.pruned_eval_metrics,
            task_type=self.task,
            original_model_size_mb=self.original_model_size_mb,
            pruned_model_size_mb=self.pruned_model_size_mb,
            resolution=self.resolution,
            prune_ratio=self.prune_ratio,
        )
        return results
