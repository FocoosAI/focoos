DATASETS = {
    "bottles": {
        "name": "bottles",
        "path": "../data/bottles",
        "task": "detection",
        "workspace": "roboflow-100",
        "project": "soda-bottles",
        "version": 4,
    },
    "blister": {
        "name": "blister",
        "path": "../data/blister",
        "task": "instance_segmentation",
        "workspace": "blisterdetection",
        "project": "blister-pills-segmentation",
        "version": 1,
    },
    "boxes": {
        "name": "boxes",
        "path": "../data/boxes",
        "task": "detection",
        "workspace": "moyed-chowdhury",
        "project": "mv_train_data",
        "version": 2,
    },
    "cable": {
        "name": "cable",
        "path": "../data/cable",
        "task": "detection",
        "workspace": "roboflow-100",
        "project": "cable-damage",
        "version": 2,
    },
    "concrete": {
        "name": "concrete",
        "path": "../data/concrete",
        "task": "instance_segmentation",
        "workspace": "focoosai",
        "project": "concrete-merge-d91ow",
        "version": 1,
    },
    "lettuce": {
        "name": "lettuce",
        "path": "../data/lettuce",
        "task": "detection",
        "workspace": "object-detection",
        "project": "lettuce-pallets",
        "version": 1,
    },
    "peanuts": {
        "name": "Peanuts",
        "path": "../data/peanuts",
        "task": "detection",
        "workspace": "roboflow-100",
        "project": "peanuts-sd4kf",
        "version": 1,
    },
    "safety": {
        "name": "Safety",
        "path": "../data/safety",
        "task": "detection",
        "workspace": "roboflow-100",
        "project": "construction-safety-gsnvb",
        "version": 1,
    },
    "strawberry": {
        "name": "Strawberries",
        "path": "../data/strawberries",
        "task": "instance_segmentation",
        "workspace": "marstrawberry",
        "project": "strawberry-disease-uudgf",
        "version": 1,
    },
}


def get_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Dataset {name} not found")
    return DATASETS[name]
