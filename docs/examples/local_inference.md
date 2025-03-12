# Select and Inference with Focoos Models

This section covers how to perform inference using the [Focoos Models](../models.md) on the cloud or locally using the `focoos` library.

As a reference, the following example demonstrates how to perform inference using the [`fai-rtdetr-m-obj365`](../models/fai-rtdetr-m-obj365.md) model, but you can use any of the models listed in the [models](../models.md) section.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/inference.ipynb)

## 🤖 Local Inference
!!! Note
    To perform local inference, you need to install the package with one of the extra modules (`[cpu]`, `[torch]`, `[cuda]`, `[tensorrt]`). See the [installation](../setup.md) page for more details.

You can perform inference locally by selecting the model you want to use and calling the [`infer` method](/focoos/api/local_model/#focoos.local_model.LocalModel.infer) on your image. If it's the first time you run the model locally, the model will be downloaded from the cloud and saved on your machine. Additionally, if you use CUDA or TensorRT, the model will be optimized for your GPU before running the inference (it can take few seconds, especially for TensorRT).

## Pretrained FAI Models

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

model = focoos.get_local_model("fai-rtdetr-m-obj365")

image_path = "<PATH-TO-YOUR-IMAGE>"
result, preview = model.infer(image_path, threshold=0.5, annotate=True)

```
## User Models

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

model = focoos.get_local_model("<YOUR-MODEL-REF>")

image_path = "<PATH-TO-YOUR-IMAGE>"
result, preview = model.infer(image_path, threshold=0.5, annotate=True)

```
`result` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](/focoos/api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference.

As for remote inference, you can pass the `annotate` parameter to return a preview of the prediction.
