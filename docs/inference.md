# How to Use a Computer Vision Model with Focoos
Focoos provides a powerful inference framework that makes it easy to deploy and use state-of-the-art computer vision models in production. Whether you're working on object detection, image classification, or other vision tasks, Focoos offers flexible deployment options that adapt to your specific needs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/tutorials/inference.ipynb)

Key features of the Focoos inference framework include:

- **Multiple Deployment Options**: Choose between cloud-based inference, local PyTorch deployment, or optimized runtime deployment
- **Easy Model Loading**: Seamlessly load models from the Focoos Hub or your local environment
- **Production-Ready Features**:
    - Optimized inference performance
    - Hardware acceleration compatibility
    - Memory-efficient execution
- **Simple Integration**: Easy-to-use APIs that work seamlessly with your existing applications

In the following sections, we'll guide you through the different ways to use Focoos models for inference, from cloud deployment to local optimization.

## 🎨 There are three ways to use a model:

1. [🌍 Remote Inference](#1-remote-inference)
2. [🔥 Pytorch Inference](#2-pytorch-inference)
3. [🔨 Optimized Inference](#3-optimized-inference)

## 0. \[Optional\] Connect to the Focoos Hub

Focoos can be used without having an accont on the [Focoos Hub](http://app.focoos.ai). With it, you will unlock additional functionalities, as we will see below. If you have it, just connect to the HUB.
```python
from focoos.hub import FocoosHUB

FOCOOS_API_KEY = os.getenv("FOCOOS_API_KEY")  # write here your API key os set env variable FOCOOS_API_KEY, will be used as default
hub = FocoosHUB(api_key=FOCOOS_API_KEY)
```

You can see the models available for you on the platform with an intuitive user interface.
However, you can also list them using the Hub functionalities.

```python
models = hub.list_remote_models()

```


## 1. 🌍 Remote Inference

In this section, you'll run a model on the Focoos' servers instead of on your machine. The image will be packed and sent on the network to the servers, where it is processed and the results is retured to your machine, all in few milliseconds. If you want an example model, you can try `fai-detr-l-obj365`.



```python
model_ref = "<YOUR-MODEL-REF>"
dataset = hub.get_remote_model(model_ref)
```

Using the model is as simple as it could! Just call it with an image.

```python
from PIL import Image
image = Image.open("<PATH-TO-IMAGE>")
detections = model(image)
```

`detections` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](/focoos/api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference. The `FocoosDet` object contains the following attributes:

- `bbox`: Bounding box coordinates in x1y1x2y2 absolute format.
- `conf`: Confidence score (from 0 to 1).
- `cls_id`: Class ID (0-indexed).
- `label`: Label (name of the class).
- `mask`: Mask (base64 encoded string having origin in the top left corner of bbox and the same width and height of the bbox).

If you want to visualize the result on the image, there's a utily for you.

```python
from focoos.utils.vision import annotate_image

annotate_image(image, detections, task=model.model_info.task, classes=model.model_info.classes).save("predictions.png")
```

## 2. 🔥 PyTorch Inference

This section demonstrates how to perform local inference using a plain Pytorch model.
We will load a model and then run inference on a sample image.

First, let's get a model. We need to use the `ModelManager` that will take care of instaciating the right model starting from a model reference (for example, the `fai-detr-l-obj365`). If you want to use a model from the Hub, please remember to add `hub://` as prefix to the model reference.

=== "Pretrained Model"

    ```python
    from focoos.model_manager import ModelManager
    model_name = "fai-detr-l-obj365"

    model = ModelManager.get(model_name)
    ```

=== "HUB Model"

```python
from focoos.model_manager import ModelManager

model_ref = "<YOUR-MODEL-REF>"


model = ModelManager.get(f"hub://{model_ref}")
```

=== "Local Model "

```python
from focoos.model_manager import ModelManager

model_path = "/path/to/model"


model = ModelManager.get(model_path)
```

Now, again, you can now run the model by simply passing it an image and visualize the results.

```python
from focoos.utils.vision import annotate_image

detections = model(image)

annotate_image(image, detections, task=model.model_info.task, classes=model.model_info.classes).save("predictions.png")
```

`detections` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object.

How fast is this model locally? We can compute it's speed by using the benchmark utility.

```python
model.benchmark(iterations=10, size=640)
```

## 3. 🔨 Optimized Inference

As you can see, using the torch model is great, but we can achieve better performance by exporting and running it with a optimized runtime, such as Torchscript, TensorRT, CoreML or the ones available on ONNXRuntime.

In the following cells, we will export the previous model for one of these and run it.

### Torchscript

We already provide multiple inference runtime, that you can see on the [`RuntimeTypes`](focoos/api/ports/#focoos.ports.RuntimeType) enum. Let's select Torchscript as an example.

```python
from focoos.ports import RuntimeType

runtime = RuntimeType.TORCHSCRIPT_32
```

It's time to export the model. We can use the export method of the models.

```python
optimized_model = model.export(runtime_type=runtime, image_size=512)
```

Let's visualize the output. As you will see, there are not differences from the model in pure torch.

```python
from focoos.utils.vision import annotate_image

detections = optimized_model(image)
annotate_image(image, detections, task=model.model_info.task, classes=model.model_info.classes).save("prediction.png")
```
`detections` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object.


But, let's see its latency, that should be substantially lower than the pure pytorch model.
```python
optimized_model.benchmark(iterations=10, size=512)
```

You can use different runtimes that may fit better your device, such as TensorRT. See the list of available Runtimes at [`RuntimeTypes`](focoos/api/ports/#focoos.ports.RuntimeType). Please note that you need to install the relative packages for onnx and tensorRT for using them.

### ONNX with TensorRT
```python
from focoos.ports import RuntimeType
from focoos.utils.vision import annotate_image

runtime = RuntimeType.ONNX_TRT16
optimized_model = model.export(runtime_type=runtime)

detections = optimized_model(image)
display(annotate_image(image, detections, task=model.model_info.task, classes=model.model_info.classes))

optimized_model.benchmark(iterations=10, size=640)
```
