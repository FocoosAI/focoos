import torch
import torch_pruning as tp

from focoos.models.base_model import BaseModelNN
from focoos.models.fai_cls.ports import ClassificationModelOutput


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
    def __init__(self, model, config, device):
        super().__init__(config)
        self.model = model
        self.config = config
        self._device = device

    @property
    def device(self):
        return torch.device(self._device)

    @property
    def dtype(self):
        return torch.float

    def forward(self, x):
        return ClassificationModelOutput(logits=self.model(x), loss=None)
