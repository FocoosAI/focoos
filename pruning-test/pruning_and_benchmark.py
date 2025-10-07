import os

import torch
import torch_pruning as tp
from pruning.utils.print_results import print_results

from focoos import DatasetLayout, DatasetSplitType, ModelManager, Task, TrainerArgs
from focoos.data import AutoDataset, get_default_by_task
from focoos.models.base_model import BaseModelNN
from focoos.models.fai_cls.ports import ClassificationModelOutput

# Configuration
DATASETS_DIR = "/home/andre/FocoosAI/datasets"
DEVICE = "cuda:0"
ROOT_DIR = "/home/andre/focoos-1/pruning-test"

MODEL_NAME = "fai-cls-n-coco"
RESOLUTION = 224
PRUNE_RATIO = 0.99
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


class PruningCompatibleModel(torch.nn.Module):
    """Wrapper to make the model compatible with torch-pruning by ensuring clean tensor outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Call the original model and return only the logits tensor
        output = self.model(x)
        # Ensure we return a clean tensor that torch-pruning can handle
        return output.logits


def prune_model_with_torch_pruning(
    model, dummy_input, layers_to_prune, prune_ratio=0.2, norm_type=2, output_path="pruned_model.pth"
):
    """
    Prunes specified layers of a model using torch-pruning DependencyGraph and saves the pruned model.

    Args:
        model (torch.nn.Module): Input model to prune.
        dummy_input (torch.Tensor): Example input tensor to trace model.
        layers_to_prune (list of str): List of layer names to prune (e.g., ['conv1', 'layer1.0.conv1']).
        prune_ratio (float): Fraction of output channels to prune in each specified layer.
        norm_type (int): Norm type (1 for L1 norm, 2 for L2 norm) used to select channels to prune.
        output_path (str): Path to save the pruned model.
    """

    # Ensure model and input are on the same device
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    model = model.to(device)
    model.eval()

    print(f"Building dependency graph for model on device: {device}")
    print(f"Model type: {type(model)}")

    # Build dependency graph for the model
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=dummy_input)

    if DG is None:
        raise RuntimeError(
            "Failed to build dependency graph. This might be due to unsupported model architecture or torch-pruning compatibility issues."
        )

    for layer_name in layers_to_prune:
        # Access the layer by attribute chain (support nested modules like layer1.0.conv1)
        layer = model
        for attr in layer_name.split("."):
            layer = getattr(layer, attr)

        # Check if layer has weight attribute (conv layers)
        if not hasattr(layer, "weight"):
            print(f"Layer {layer_name} does not have weight attribute, skipping.")
            continue

        # Compute norms over output channels
        if norm_type == 1:
            norms = layer.weight.data.abs().sum(dim=(1, 2, 3)).to(device)
        else:
            norms = torch.norm(layer.weight.data.view(layer.weight.size(0), -1), p=norm_type, dim=1).to(device)

        num_channels = norms.size(0)
        num_prune = int(num_channels * prune_ratio)
        # num_prune = 64 # OVERWRITE PRUNE RATIO

        if num_prune == 0:
            print(f"Skipping pruning for {layer_name}, no channels to prune.")
            continue

        # Get indices of channels with smallest norms to prune
        prune_indices = torch.argsort(norms)[:num_prune].tolist()

        # Print the name of the layer that will be pruned
        print(f"Preparing to prune layer: {layer_name}")
        print(f"Layer: {layer}")
        print(f"Prune indices: {prune_indices}")
        print(f"Prune indices ratio: {len(prune_indices) / num_channels}")
        print(f"len(Prune indices): {len(prune_indices)}")
        print("--------------------------------")

        # Get pruning group from dependency graph
        prune_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=prune_indices)
        print(f"Prune group: {prune_group}")

        if DG.check_pruning_group(prune_group):
            prune_group.prune()
            print(f"Pruned {num_prune} channels from {layer_name}")
        else:
            print(f"Pruning group for {layer_name} invalid, skipping.")

    torch.save(model, output_path)
    print(f"Pruned model state dict saved to {output_path}")


class PrunedBaseModel(BaseModelNN):
    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.config = config

    @property
    def device(self):
        return torch.device(DEVICE)

    @property
    def dtype(self):
        return torch.float

    def forward(self, x):
        return ClassificationModelOutput(logits=self.model(x), loss=None)


def main():
    print("=" * 60)
    print("STARTING PRUNING AND BENCHMARK PIPELINE")
    print("=" * 60)

    # Step 1: Load the original model
    print(f"\n1. Loading model: {MODEL_NAME}")
    focoos_model = ModelManager.get(MODEL_NAME)
    original_model = focoos_model.model
    # Step 2: Benchmark original model
    print("1.5. Benchmarking original model")
    result_original_model = focoos_model.benchmark(
        iterations=200,
        size=(RESOLUTION, RESOLUTION),
    )

    # Step 2: Wrap the model for pruning compatibility
    print("2. Wrapping model for pruning compatibility")
    model = PruningCompatibleModel(original_model)

    # Step 3: Create output directory
    NAME = f"{MODEL_NAME}-pruned"
    FOLDER_NAME = f"{NAME}_RATIO={PRUNE_RATIO}_LAYERS={len(LAYERS_TO_PRUNE)}"
    OUTPUT_DIRECTORY = f"{ROOT_DIR}/models/{FOLDER_NAME}"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Step 4: Run pruning
    print(f"3. Running pruning with ratio {PRUNE_RATIO} on {len(LAYERS_TO_PRUNE)} layers")
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION)
    output_path = os.path.join(OUTPUT_DIRECTORY, "model_pruned.pth")

    if os.path.exists(output_path):
        os.remove(output_path)

    prune_model_with_torch_pruning(
        model, dummy_input, LAYERS_TO_PRUNE, prune_ratio=PRUNE_RATIO, norm_type=2, output_path=output_path
    )

    # Step 5: Save model structure and state dict info
    print("4. Saving model structure and state dict info")
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

    # Step 6: Load and prepare pruned model for export
    print("5. Loading pruned model and preparing for export")
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
            return

    print("State_dict is correct")

    # Step 7: Create PrunedBaseModel wrapper
    print("6. Creating PrunedBaseModel wrapper")
    input_tensor = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
    model_pruned_wrapper = PrunedBaseModel(model_pruned, config=focoos_model.model.config)
    model_pruned_wrapper = model_pruned_wrapper.to(DEVICE)
    model_pruned_wrapper.eval()

    # Step 8: Warm up the model
    print("7. Warming up the model")
    for i in range(50):
        model_pruned_wrapper(input_tensor)

    # Step 10: Export pruned model
    print("9. Exporting pruned model")
    focoos_model.model = model_pruned_wrapper
    # focoos_model.export(runtime_type=RuntimeType.TORCHSCRIPT_32, out_dir=OUTPUT_DIRECTORY, overwrite=True)

    # Step 11: Benchmark pruned model
    print("10. Benchmarking pruned model")
    result_pruned_model = focoos_model.benchmark(
        iterations=200,
        size=(RESOLUTION, RESOLUTION),
    )

    auto_dataset = AutoDataset(
        dataset_name="coco_2017_cls",
        task=Task.CLASSIFICATION,
        layout=DatasetLayout.CATALOG,
        datasets_dir=DATASETS_DIR,
    )

    train_augs, val_augs = get_default_by_task(Task.CLASSIFICATION, resolution=RESOLUTION)

    # train_dataset = auto_dataset.get_split(augs=train_augs.get_augmentations(), split=DatasetSplitType.TRAIN)
    valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

    args = TrainerArgs(
        run_name=f"{focoos_model.name}_{valid_dataset.name}",
        batch_size=16,
        max_iters=50,
        eval_period=50,
        learning_rate=0.0008,
        sync_to_hub=False,  # use this to sync model info, weights and metrics on the hub
        device=DEVICE,
    )

    focoos_model.eval(args, valid_dataset)

    # Step 12: Print results
    print("11. Printing results")
    print_results(result_original_model, result_pruned_model, MODEL_NAME, OUTPUT_DIRECTORY)

    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Output directory: {OUTPUT_DIRECTORY}")
    print("=" * 60)


if __name__ == "__main__":
    main()
