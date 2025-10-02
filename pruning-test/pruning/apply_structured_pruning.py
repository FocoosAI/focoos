import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from focoos import ModelManager


def ln_structured_prune_and_save(model, norm_type=2, amount=0.2, output_directory="output", filename="pruned_model.pt"):
    """
    Prunes channels in all Conv2d layers using L-norm criterion and saves the model.

    Args:
        model (nn.Module): PyTorch model.
        norm_type (int): 1 for L1 norm, 2 for L2 norm.
        amount (float): Fraction of channels to prune.
        output_directory (str): Directory to save the pruned model.
        filename (str): Filename for saving.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=norm_type, dim=1)
            prune.remove(module, "weight")

    os.makedirs(output_directory, exist_ok=True)
    save_path = os.path.join(output_directory, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Pruned model saved to {save_path}")


model = ModelManager.get("fai-cls-n-coco")
model = model.model

FOLDER_NAME = "test-1"
OUTPUT_DIRECTORY = f"/home/ubuntu/focoos-2/pruning-test/models/{FOLDER_NAME}"

ln_structured_prune_and_save(
    model, norm_type=2, amount=0.5, output_directory=OUTPUT_DIRECTORY, filename="model_final.pt"
)

with open(os.path.join(OUTPUT_DIRECTORY, f"layers_{FOLDER_NAME}.txt"), "w") as f:
    f.write(str(model))

state_dict_path = os.path.join(OUTPUT_DIRECTORY, f"state_dict_{FOLDER_NAME}.txt")
if os.path.exists(state_dict_path):
    os.remove(state_dict_path)
with open(state_dict_path, "a") as f:
    for k, v in model.state_dict().items():
        print(f"{k}: {v}", file=f)
