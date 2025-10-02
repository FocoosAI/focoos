import torch

from focoos import DatasetLayout, DatasetSplitType, ModelManager, Task, TrainerArgs
from focoos.data import AutoDataset, get_default_by_task

MODEL = "/home/ubuntu/focoos-2/pruning-test/models/fai-cls-m-coco-pruned_RATIO=0.99_LAYERS=8"
RESOLUTION = 224

# Remove ".model" prefix from state_dict
PREFIX = "model."
state_dict = torch.load(f"{MODEL}/model_pruned.pth", map_location="cpu")
keys_to_update = [k for k in state_dict.keys() if k.startswith(PREFIX)]
for k in keys_to_update:
    state_dict[k.replace(PREFIX, "")] = state_dict.pop(k)
torch.save(state_dict, f"{MODEL}/model_pruned.pth")

# check if the state_dict is correct
for i, k in enumerate(state_dict.keys()):
    if k.startswith(PREFIX):
        print(f"Error: {k} starts with {PREFIX} at index {i}")

print("State_dict is correct")


model = ModelManager.get(MODEL)
# exit() # DEBUG
# with open(f"{MODEL}/layers.txt", "w") as f:
#     f.write(str(model.model))

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

model.eval(args, valid_dataset)
model.benchmark(iterations=50, size=(RESOLUTION, RESOLUTION))
