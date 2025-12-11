<a href="https://www.focoos.ai" target="_blank">
  <img src="https://public.focoos.ai/library/focoos_banner.png" alt="FocoosAI" style="max-width:100%;">
</a>

![Tests](https://github.com/FocoosAI/focoos/actions/workflows/test.yml/badge.svg??event=push&branch=main)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/tutorials/training.ipynb)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://focoosai.github.io/focoos/)

# Welcome to Focoos AI

Focoos AI provides an advanced development platform designed to empower developers and businesses with efficient, customizable computer vision solutions. Whether you're working with data from cloud infrastructures or deploying on edge devices, Focoos AI enables you to select, fine-tune, and deploy state-of-the-art models optimized for your unique needs.

## Overview

<!-- Unlock the full potential of Focoos AI with the Focoos library! 🚀  -->
The Focoos library is your gateway to easily access cutting-edge computer vision models and development tools. With just a few lines of code, you can **fine tune** pre-trained models tailored to your specific needs.

Whether you're working in the cloud or on edge devices, the Focoos library seamlessly integrates into your workflow, accelerating development and simplifying the implementation of computer vision solutions.

### Key Features 🔑

1. **Frugal Pretrained Models** 🌿
   Get started quickly by selecting one of our efficient, [pre-trained models](https://focoosai.github.io/focoos/models/) that best suits your data and application needs.
   Focoos Model Registry give access to 11 pretrained models of different size from different families: RTDetr, Maskformer, BisenetFormer

2. **Fine Tune Your Model** ✨ Adapt the model to your specific use case by customize its config and training it on your own dataset.

4. **Optimized Inference** 🖥️ Export Models and run inference efficiently, Leverage hardware acceleration through Torchscript, TensorRT and ONNX for maximum performance.

5. **FocoosHub Integration** 🔄 Seamlessly integrate with Focoos Cloud to access your models and data, you can also run cloud inference on managed models.

# Quickstart 🚀
Ready to dive in? Get started with the setup in just a few simple steps!

## Installation
**Install** the Focoos library (for more options, see [setup](https://focoosai.github.io/focoos/setup))

```bash linenums="0"
uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
```

## Inference

```python
from focoos import ModelManager


im = "https://public.focoos.ai/samples/motogp.jpg" # can be local/remote path, np.array, PIL image

model = ModelManager.get("fai-detr-l-obj365") # any models from ModelRegistry, FocoosHub or local folder

detections = model.infer(im,annotate=True)

```

## Training

```python
from focoos import ModelManager,DatasetSplitType, DatasetLayout, RuntimeType, TrainerArgs, Task
from focoos.data import get_default_by_task
from focoos.data import AutoDataset



ds_name = "my_dataset.zip"
task = Task.DETECTION
layout = DatasetLayout.ROBOFLOW_COCO

auto_dataset = AutoDataset(dataset_name=ds_name, task=task, layout=layout)

train_augs, val_augs = get_default_by_task(task, 640, advanced=False)
train_dataset = auto_dataset.get_split(augs=train_augs, split=DatasetSplitType.TRAIN)
valid_dataset = auto_dataset.get_split(augs=val_augs, split=DatasetSplitType.VAL)


model = ModelManager.get("fai-detr-l-obj365")

args = TrainerArgs(
    run_name=f"{model.name}_{train_dataset.name}",
    batch_size=16,
    max_iters=50,
    eval_period=50,
    learning_rate=0.0008,
    sync_to_hub=False,  # use this to sync model info, weights and metrics on the hub
)


model.train(args, train_dataset, valid_dataset)

# if you want to do only evaluation

model.eval(args, valid_dataset)

```


## Export, Benchmark and optimized Inference

```python
from focoos import RuntimeType, ModelManager

model = ModelManager.get("fai-detr-l-obj365")
infer_model = model.export(runtime_type=RuntimeType.TORCHSCRIPT_32)
infer_model.benchmark()

```

# Our Models 🧠
Focoos AI offers the best models in object detection, semantic and instance segmentation, and more is coming soon.

Using Focoos AI helps you save both time and money while delivering high-performance AI models 💪:

- **10x Faster** ⏳: Our models are able to process images up to ten times faster than traditional methods.
- **4x Cheaper** 💰: Our models require up to 4x less computational power, letting you save on hardware or cloud bill while ensuring high-quality results.
- **Tons of CO2 saved annually per model** 🌱: Our models are energy-efficient, helping you reduce your carbon footprint by using less powerful hardware with respect to mainstream models.

See the list of our models in the [models](https://focoosai.github.io/focoos/models/) section.

---
### Start now!
By choosing Focoos AI, you can save time, reduce costs, and achieve superior model performance, all while ensuring the privacy and efficiency of your deployments.
[Reach out to us](mailto:info@focoos.ai) to ask for your API key for free and power your computer vision projects. 🚀


## ❤️ Contributors
[![Contributors](https://contrib.rocks/image?repo=FocoosAI/focoos)](https://github.com/FocoosAI/focoos/graphs/contributors)
