# Select and Inference with Focoos Models

This section covers how to perform inference using the [Focoos Models](../models.md) on the cloud or locally using the `focoos` library.

As a reference, the following example demonstrates how to perform inference using the [`fai-rtdetr-m-obj365`](../models/fai-rtdetr-m-obj365.md) model, but you can use any of the models listed in the [models](../models.md) section.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/inference.ipynb)

## ☁️ Cloud Inference
Making inference on the cloud is straightforward, you just need to select the model you want to use and call the [`infer` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.infer) on your image. The image will be uploaded on the FocoosAI cloud, where the model will perform the inference and return the results.

## Pretrained FAI Models
```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

image_path = "<PATH-TO-YOUR-IMAGE>"
model = focoos.get_remote_model("fai-rtdetr-m-obj365")
result, preview = model.infer(image_path, threshold=0.5, annotate=True)

```

## User Models
```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

image_path = "<PATH-TO-YOUR-IMAGE>"
model = focoos.get_remote_model("<YOUR-MODEL-REF>")
result, preview = model.infer(image_path, threshold=0.5, annotate=True)

```


`result` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](/focoos/api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference.

The `threshold` parameter is optional and defines the minimum confidence score for a detection to be considered valid (predictions with a confidence score lower than the threshold are discarded).

The `preview` parameter is optional and return annotated image with the detections.
