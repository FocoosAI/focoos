import torch

from focoos import DatasetLayout, DatasetSplitType, ModelManager, RuntimeType, Task, TrainerArgs
from focoos.data import AutoDataset, get_default_by_task
from focoos.models.base_model import BaseModelNN
from focoos.models.fai_cls.ports import ClassificationModelOutput

MODEL_NAME = "fai-cls-m-coco"
MODEL = f"/home/ubuntu/focoos-2/pruning-test/models/{MODEL_NAME}-pruned_RATIO=0.99_LAYERS=5"
RESOLUTION = 224


class PrunedBaseModel(BaseModelNN):
    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.config = config

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def dtype(self):
        return torch.float

    def forward(self, x):
        return ClassificationModelOutput(logits=self.model(x), loss=None)


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


model = ModelManager.get(MODEL_NAME)
model.benchmark(
    iterations=200,
    size=(RESOLUTION, RESOLUTION),
)

auto_dataset = AutoDataset(
    dataset_name="coco_2017_cls",
    task=Task.CLASSIFICATION,
    layout=DatasetLayout.CATALOG,
    datasets_dir="/home/ubuntu/anyma/datasets",
)

train_augs, val_augs = get_default_by_task(Task.CLASSIFICATION, resolution=RESOLUTION)

# train_dataset = auto_dataset.get_split(augs=train_augs.get_augmentations(), split=DatasetSplitType.TRAIN)
valid_dataset = auto_dataset.get_split(augs=val_augs.get_augmentations(), split=DatasetSplitType.VAL)

args = TrainerArgs(
    run_name=f"{model.name}_{valid_dataset.name}",
    batch_size=16,
    max_iters=50,
    eval_period=50,
    learning_rate=0.0008,
    sync_to_hub=False,  # use this to sync model info, weights and metrics on the hub
)

# Remove ".model" prefix from state_dict
PREFIX = "model."
model_pruned = torch.load(f"{MODEL}/model_pruned.pth", map_location="cpu", weights_only=False)
state_dict = model_pruned.state_dict()
keys_to_update = [k for k in state_dict.keys() if k.startswith(PREFIX)]
for k in keys_to_update:
    state_dict[k.replace(PREFIX, "")] = state_dict.pop(k)

model_pruned.state_dict = state_dict
# torch.save(state_dict, f"{MODEL}/model_pruned.pth")

# check if the state_dict is correct
for i, k in enumerate(state_dict.keys()):
    if k.startswith(PREFIX):
        print(f"Error: {k} starts with {PREFIX} at index {i}")

print("State_dict is correct")


# model_pruned = ModelManager.get(MODEL)

# state_dict = torch.load(f"{MODEL}/model_pruned.pth", map_location="cpu", weights_only=False)
input_tensor = torch.randn(1, 3, RESOLUTION, RESOLUTION).to("cuda")
model_pruned = PrunedBaseModel(model_pruned, config=model.model.config)
model_pruned = model_pruned.to("cuda")
model_pruned.eval()

for i in range(50):
    model_pruned(input_tensor)
# print(state_dict.keys())

# model.export(
#     format=ExportFormat.TORCHSCRIPT,
#     output_dir=MODEL
#     )

model.model = model_pruned
model.eval(args, valid_dataset)
infer_model = model.export(runtime_type=RuntimeType.TORCHSCRIPT_32, out_dir=MODEL, overwrite=True)
model.benchmark(
    iterations=200,
    size=(RESOLUTION, RESOLUTION),
)
