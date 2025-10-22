import json
import os

from focoos import PACKAGE_DIR, DatasetLayout, Task
from focoos.pruning.focoos_pruning import FocoosPruning

# Configuration
TASK = Task.DETECTION
DATASET_NAME = "coco_2017_det"
DATASET_LAYOUT = DatasetLayout.CATALOG
DEVICE = "cpu"
VERBOSE = False
DO_EVAL = True  # compute or not eval metrics

MODEL_NAME = "fai-detr-m-coco"
OUTPUT_FOLDER = "pruning_outputs"

RESOLUTION = 640
PRUNE_RATIO = 0.75
BENCHMARK_ITERATIONS = 200

# Get layers to prune from prunable_layers.json
prunable_layers = json.load(open(os.path.join(PACKAGE_DIR, "pruning", "prunable_layers.json")))

if MODEL_NAME not in prunable_layers:
    raise ValueError(f"Model {MODEL_NAME} not found in prunable_layers.json")
LAYERS_TO_PRUNE = prunable_layers[MODEL_NAME]


def main():
    pipeline = FocoosPruning(
        task=TASK,
        dataset_name=DATASET_NAME,
        dataset_layout=DATASET_LAYOUT,
        device=DEVICE,
        verbose=VERBOSE,
        do_eval=DO_EVAL,
        root_dir=OUTPUT_FOLDER,
        model_name=MODEL_NAME,
        resolution=RESOLUTION,
        prune_ratio=PRUNE_RATIO,
        benchmark_iterations=BENCHMARK_ITERATIONS,
        layers_to_prune=LAYERS_TO_PRUNE,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
