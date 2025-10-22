import os
import re
from typing import Optional

import torch
import torch_pruning as tp

from focoos import ModelManager, Task
from focoos.models.base_model import BaseModelNN
from focoos.models.fai_cls.ports import ClassificationModelOutput
from focoos.models.fai_detr.ports import DETRModelOutput


class PruningCompatibleModel(torch.nn.Module):
    """Wrapper to make the model compatible with torch-pruning by ensuring clean tensor outputs."""

    def __init__(self, model, task, is_eval=False):
        super().__init__()
        self.model = model
        self.task = task
        self.is_eval = is_eval

    def forward(self, x):
        # Call the original model and return only the logits tensor
        output = self.model(x)
        # Ensure we return a clean tensor that torch-pruning can handle
        if self.is_eval:
            return output
        else:
            if self.task == Task.CLASSIFICATION:
                return output.logits
            elif self.task == Task.DETECTION:
                return output.boxes
            else:
                raise ValueError(f"Task {self.task} not supported")

    def __getattr__(self, name):
        """Delegate attribute access to the underlying model to support nested attribute access."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to load into the underlying model."""
        return self.model.load_state_dict(state_dict, strict=strict)


class PrunedBaseModel(BaseModelNN):
    def __init__(self, model, config, device, task):
        super().__init__(config)
        self.model = model
        self.config = config
        self._device = device
        self.task = task

    @property
    def device(self):
        return torch.device(self._device)

    @property
    def dtype(self):
        return torch.float

    def forward(self, x):
        output = self.model(x)
        if self.task == Task.DETECTION:
            return DETRModelOutput(boxes=output.boxes, logits=output.logits, loss=None)  # output.shape == (B, N_DET, 4)
        elif self.task == Task.CLASSIFICATION:
            return ClassificationModelOutput(logits=output.logits, loss=None)  # output.shape == (B, N_DET, num_logits)
        else:  # add more here
            raise ValueError(f"Task {self.task} not supported")


def prune_model_with_torch_pruning(
    model, dummy_input, layers_to_prune, prune_ratio=0.2, norm_type=2, output_path="pruned_model.pth", verbose=False
) -> torch.nn.Module:
    """
    Prunes specified layers of a model using torch-pruning DependencyGraph and saves the pruned model.

    Args:
        model (torch.nn.Module): Input model to prune.
        dummy_input (torch.Tensor): Example input tensor to trace model.
        layers_to_prune (list of str): List of layer names to prune (e.g., ['conv1', 'layer1.0.conv1']).
        prune_ratio (float): Fraction of output channels to prune in each specified layer.
        norm_type (int): Norm type (1 for L1 norm, 2 for L2 norm) used to select channels to prune.
        output_path (str): Path to save the pruned model.
        verbose (bool): Whether to print verbose output.
    """

    # Ensure model and input are on the same device
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    model = model.to(device)
    model.eval()

    if verbose:
        print(f"Building dependency graph for model on device: {device}")
        print(f"Model type: {type(model)}")

    # Build dependency graph for the model
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=dummy_input, verbose=verbose)

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
        if verbose:
            print(f"Preparing to prune layer: {layer_name}")
            print(f"Layer: {layer}")
            print(f"Prune indices: {prune_indices}")
            print(f"Prune indices ratio: {len(prune_indices) / num_channels}")
            print(f"len(Prune indices): {len(prune_indices)}")
            print("--------------------------------")

        # Get pruning group from dependency graph
        prune_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=prune_indices)
        if verbose:
            print(f"Prune group: {prune_group}")

        if DG.check_pruning_group(prune_group):
            prune_group.prune()
            print(f"Pruned {num_prune} channels from {layer_name}")
        else:
            print(f"Pruning group for {layer_name} invalid, skipping.")

    torch.save(model.model.state_dict(), output_path)
    if verbose:
        print(f"Pruned model state dict saved to {output_path}")
    return model


def get_model_layers(model_name, output_folder_path: Optional[str] = ""):
    """Get the layers of the model"""
    focoos_model = ModelManager.get(model_name)
    state_dict = focoos_model.model.state_dict()

    layers = []
    if output_folder_path:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        with open(os.path.join(output_folder_path, f"state_dict_shape_{model_name}.txt"), "a") as f:
            for k, v in state_dict.items():
                str_layer = str(f"{k}: {v.shape}")
                print(str_layer, file=f)
                print(str_layer)
                layers.append(str_layer)
    return layers


def get_layers_to_prune(regex_pattern: str, layers_file_path: str) -> list[str]:
    """
    Returns a list of layer names from the given file that match the provided regex pattern.

    Args:
        regex_pattern (str): Regular expression pattern to match layer names.
        layers_file_path (str): Path to the file containing layer names (one per line, as keys before ':').

    Returns:
        list[str]: List of matching layer names.
    """
    pattern = re.compile(regex_pattern)
    matching_layers = []
    with open(layers_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            layer_name = line.split(":", 1)[0].strip()
            if pattern.fullmatch(layer_name) or pattern.search(layer_name):
                matching_layers.append(layer_name)

    # remove suffix ".weight"
    suffix = ".weight"
    matching_layers = [layer.replace(suffix, "") for layer in matching_layers]
    return matching_layers


def load_layers_from_file(layers_file_path: str) -> list[str]:
    with open(layers_file_path, "r") as f:
        return [line.strip() for line in f.readlines()]
