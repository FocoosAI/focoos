import os

import torch
from pruning.utils.print_results import load_eval_metrics_from_model_info, print_results
from pruning.utils.utils import PrunedBaseModel, PruningCompatibleModel, prune_model_with_torch_pruning

from focoos import DatasetLayout, DatasetSplitType, ModelManager, Task, TrainerArgs
from focoos.data import AutoDataset, get_default_by_task
from focoos.ports import get_gpus_count
from focoos.utils.logger import get_logger

logger = get_logger("pruning_and_benchmark")

# Configuration
TASK = Task.CLASSIFICATION
DATASETS_DIR = "/Users/andreapellegrino_focoosai/FocoosAI/datasets"
DATASET_NAME = "coco_2017_cls"
DATASET_LAYOUT = DatasetLayout.CATALOG
DEVICE = "cpu"
VERBOSE = False

ROOT_DIR = "/Users/andreapellegrino_focoosai/Work/focoos-1/pruning-test"
MODEL_NAME = "fai-cls-n-coco"
RESOLUTION = 224
PRUNE_RATIO = 0.99
BENCHMARK_ITERATIONS = 10_000
LAYERS_TO_PRUNE = [
    "model.backbone.features.2.conv_list.0.conv",
    "model.backbone.features.2.conv_list.1.conv",
    "model.backbone.features.2.conv_list.2.conv",
    "model.backbone.features.2.conv_list.3.conv",
    "model.backbone.features.3.conv_list.0.conv",
    "model.backbone.features.3.conv_list.1.conv",
    "model.backbone.features.3.conv_list.2.conv",
    "model.backbone.features.3.conv_list.3.conv",
]


def main():
    logger.info("Starting pruning and benchmark pipeline")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Resolution: {RESOLUTION}")
    logger.info(f"Prune ratio: {PRUNE_RATIO}")
    logger.info(f"Layers to prune: {LAYERS_TO_PRUNE}")
    logger.info(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")

    # Step 1: Load the original model
    logger.info(f"1/12 - Loading model: {MODEL_NAME}")
    focoos_model = ModelManager.get(MODEL_NAME)
    original_model = focoos_model.model

    # Step 2: Evaluate original model
    logger.info("2/12 - Evaluating original model")
    auto_dataset = AutoDataset(
        dataset_name=DATASET_NAME,
        task=TASK,
        layout=DATASET_LAYOUT,
        datasets_dir=DATASETS_DIR,
    )
    train_augs, val_augs = get_default_by_task(TASK, resolution=RESOLUTION)
    valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

    args_original = TrainerArgs(
        run_name=f"{focoos_model.name}_{valid_dataset.name}_original",
        batch_size=16,
        max_iters=50,
        eval_period=50,
        learning_rate=0.0008,
        sync_to_hub=False,
        device=DEVICE,
        num_gpus=get_gpus_count() if DEVICE == "cuda" else -1,
    )

    # Evaluate original model
    focoos_model.eval(args_original, valid_dataset)
    original_eval_dir = os.path.join(args_original.output_dir, f"{args_original.run_name.strip()}_eval")
    original_model_info_path = os.path.join(original_eval_dir, "model_info.json")
    original_eval_metrics = load_eval_metrics_from_model_info(original_model_info_path, task_type=TASK)

    # Step 3: Benchmark original model
    logger.info("3/12 - Benchmarking original model")
    result_original_model = focoos_model.benchmark(
        iterations=BENCHMARK_ITERATIONS,
        size=(RESOLUTION, RESOLUTION),
        device=DEVICE,
    )

    # Step 4: Wrap the model for pruning compatibility
    logger.info("4/12 - Wrapping model for pruning compatibility")
    model = PruningCompatibleModel(original_model)

    # Step 5: Create output directory
    logger.info("5/12 - Creating output directory")
    NAME = f"{MODEL_NAME}-pruned"
    FOLDER_NAME = f"{NAME}_RATIO={PRUNE_RATIO}_LAYERS={len(LAYERS_TO_PRUNE)}"
    OUTPUT_DIRECTORY = f"{ROOT_DIR}/models/{FOLDER_NAME}"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Step 6: Run pruning
    logger.info(f"6/12 - Running pruning with ratio {PRUNE_RATIO} on {len(LAYERS_TO_PRUNE)} layers")
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)
    output_path = os.path.join(OUTPUT_DIRECTORY, "model_pruned.pth")

    if os.path.exists(output_path):
        os.remove(output_path)

    prune_model_with_torch_pruning(
        model,
        dummy_input,
        LAYERS_TO_PRUNE,
        prune_ratio=PRUNE_RATIO,
        norm_type=2,
        output_path=output_path,
        verbose=VERBOSE,
    )
    logger.info(f"Pruned model saved to {output_path}")

    # Step 7: Save model structure and state dict info
    logger.info("7/12 - Saving model structure and state dict info")
    with open(os.path.join(OUTPUT_DIRECTORY, f"layers_{FOLDER_NAME}.txt"), "w") as f:
        f.write(str(model))

    state_dict_path = os.path.join(OUTPUT_DIRECTORY, f"state_dict_{FOLDER_NAME}.txt")
    state_dict_path_shape = os.path.join(OUTPUT_DIRECTORY, f"state_dict_shape_{FOLDER_NAME}.txt")
    if os.path.exists(state_dict_path):
        os.remove(state_dict_path)
    if os.path.exists(state_dict_path_shape):
        os.remove(state_dict_path_shape)
    with open(state_dict_path, "a") as f:
        for k, v in model.state_dict().items():
            print(f"{k}: {v}", file=f)
    with open(state_dict_path_shape, "a") as f:
        for k, v in model.state_dict().items():
            print(f"{k}: {v.shape}", file=f)

    # Step 8: Load and prepare pruned model for export
    logger.info("8/12 - Loading pruned model and preparing for export")
    model_pruned = torch.load(output_path, map_location="cpu", weights_only=False)
    state_dict = model_pruned.state_dict()

    # Remove ".model" prefix from state_dict
    PREFIX = "model."
    keys_to_update = [k for k in state_dict.keys() if k.startswith(PREFIX)]
    for k in keys_to_update:
        state_dict[k.replace(PREFIX, "")] = state_dict.pop(k)

    model_pruned.state_dict = state_dict

    # Verify state_dict is correct
    for i, k in enumerate(state_dict.keys()):
        if k.startswith(PREFIX):
            print(f"Error: {k} starts with {PREFIX} at index {i}")
            exit()

    logger.info("State_dict is correct")

    # Step 9: Create PrunedBaseModel wrapper
    logger.info("9/12 - Creating PrunedBaseModel wrapper")
    input_tensor = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
    model_pruned_wrapper = PrunedBaseModel(model_pruned, config=focoos_model.model.config, device=DEVICE)
    model_pruned_wrapper = model_pruned_wrapper.to(DEVICE)
    model_pruned_wrapper.eval()

    # Step 10: Warm up the model
    logger.info("10/12 - Warming up the model")
    for i in range(50):
        model_pruned_wrapper(input_tensor)

    # Step 11: Evaluate pruned model
    logger.info("11/12 - Evaluating pruned model")
    focoos_model.model = model_pruned_wrapper

    # Step 12: Benchmark pruned model
    logger.info("12/12 - Benchmarking pruned model")
    result_pruned_model = focoos_model.benchmark(
        iterations=BENCHMARK_ITERATIONS,
        size=(RESOLUTION, RESOLUTION),
        device=DEVICE,
    )

    args_pruned = TrainerArgs(
        run_name=f"{focoos_model.name}_{valid_dataset.name}_pruned",
        batch_size=16,
        max_iters=50,
        eval_period=50,
        learning_rate=0.0008,
        sync_to_hub=False,
        device=DEVICE,
        num_gpus=get_gpus_count() if DEVICE == "cuda" else -1,
    )

    # Evaluate pruned model
    focoos_model.eval(args_pruned, valid_dataset)
    pruned_eval_dir = os.path.join(args_pruned.output_dir, f"{args_pruned.run_name.strip()}_eval")
    pruned_model_info_path = os.path.join(pruned_eval_dir, "model_info.json")
    pruned_eval_metrics = load_eval_metrics_from_model_info(pruned_model_info_path, task_type=TASK)

    # Step 13: Print results
    print_results(
        result_original_model,
        result_pruned_model,
        MODEL_NAME,
        OUTPUT_DIRECTORY,
        original_eval_metrics,
        pruned_eval_metrics,
        task_type=TASK,
    )

    logger.info("Pruning pipeline completed successfully")
    logger.info(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
