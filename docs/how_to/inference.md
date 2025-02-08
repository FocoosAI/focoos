# Select and Inference with Focoos Models

This section covers how to perform inference using the [Focoos Models](./models.md) on the cloud or locally using the `focoos` library.

As a reference, the following example demonstrates how to perform inference using the [`fai-rtdetr-m-obj365`](./models/fai-rtdetr-l-obj365.md) model, but you can use any of the models listed in the [models](./models.md) section.

## 📈 See Focoos Models metrics
You can see the metrics of the Focoos Models by calling the [`metrics` method](../../api/remote_model/#focoos.remote_model.RemoteModel.metrics) on the model.

```python
model = focoos.get_remote_model("fai-rtdetr-m-obj365")
metrics = model.metrics()
print(f"Best validation metrics:")
for k, v in metrics.best_valid_metric.items():
    print(f"  {k}: {v}")
```
This code snippet will print the best validation metrics of the model, both considering average and per-class metrics.

## ☁️ Cloud Inference
Making inference on the cloud is straightforward, you just need to select the model you want to use and call the [`infer` method](../../api/remote_model/#focoos.remote_model.RemoteModel.infer) on your image. The image will be uploaded on the FocoosAI cloud, where the model will perform the inference and return the results.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

image_path = "<PATH-TO-YOUR-IMAGE>"
model = focoos.get_remote_model("fai-rtdetr-m-obj365")
result, _ = model.infer(image_path, threshold=0.5, annotate=False)

for det in results.detections:
    print(f"Found {det.label} with confidence {det.conf:.2f}")
    print(f"Bounding box: {det.bbox}")
    if det.mask:
        print("Instance segmentation mask included")

```
`result` is a [FocoosDetections](../../api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](../../api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference.

The `threshold` parameter is optional and defines the minimum confidence score for a detection to be considered valid (predictions with a confidence score lower than the threshold are discarded).

Optionally, you can preview the results by passing the `annotate` parameter to the `infer` method.
```python
from PIL import Image

output, preview = model.infer(image_path, threshold=0.5, annotate=True)
preview = Image.fromarray(preview[:,:,[2,1,0]]) # invert to make it RGB
```

## 🤖 Local Inference
!!! Note
    To perform local inference, you need to install the package with one of the extra modules (`[cpu]`, `[torch]`, `[cuda]`, `[tensorrt]`). See the [installation](./setup.md) page for more details.

You can perform inference locally by selecting the model you want to use and calling the [`infer` method](../../api/local_model/#focoos.local_model.LocalModel.infer) on your image. If it's the first time you run the model locally, the model will be downloaded from the cloud and saved on your machine. Additionally, if you use CUDA or TensorRT, the model will be optimized for your GPU before running the inference (it can take few seconds, especially for TensorRT).

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

model = focoos.get_local_model("fai-rtdetr-m-obj365")

image_path = "<PATH-TO-YOUR-IMAGE>"
result, _ = model.infer(image_path, threshold=0.5, annotate=False)

for det in results.detections:
    print(f"Found {det.label} with confidence {det.conf:.2f}")
    print(f"Bounding box: {det.bbox}")
    if det.mask:
        print("Instance segmentation mask included")

```
`result` is a [FocoosDetections](../../api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](../../api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference.

As for remote inference, you can pass the `annotate` parameter to return a preview of the prediction.

## 🖼️ Cloud Inference with Gradio
You can further use Gradio to create a web interface for your model.

First, install the `gradio` extra dependency.

```bash linenums="0"
pip install '.[gradio]'
```

To use it, use an environment variable with your Focoos API key and run the app (you will select the model from the UI).
```bash linenums="0"
export FOCOOS_API_KEY_GRADIO=<YOUR-API-KEY>; python gradio/app.py
```
