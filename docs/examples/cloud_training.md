# Create and train custom models

This section covers the steps to create a model and train it in the cloud using the `focoos` library. The following example demonstrates how to interact with the Focoos API to manage models, datasets, and training jobs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/training.ipynb)

In this guide, we will perform the following steps:

1. [📦 Load or select a dataset](#1-load-or-select-a-dataset)
2. [🎯 Create a model](#2-create-a-model)
3. [🏃‍♂️ Train the model](#3-train-the-model)
4. [📊 Visualize training metrics](#4-visualize-training-metrics)
5. [🧪 Test your model](#5-test-your-model)

---

## 1. Load or select a dataset

- You can list the public shared datasets using the following code:
    ```python
    from focoos import Focoos

    focoos = Focoos(api_key="<YOUR-API-KEY>")

    datasets = focoos.list_shared_datasets()
    print(datasets)
    ```
- You can also create your own dataset and upload it to the platform [here](datasets.md)
    ```python
    from focoos import Focoos

    focoos = Focoos(api_key="<YOUR-API-KEY>")

    dataset = focoos.create_dataset(name="<YOUR-DATASET-NAME>", description="<YOUR-DATASET-DESCRIPTION>")
    ```

Find the dataset reference you want to use and store it in a variable:
```python
dataset_ref = "<YOUR-DATASET-REFERENCE>"
```

## 2. Create a model
The first step to personalize your model is to create a model.
You can create a model by calling the [`new_model` method](/focoos/api/focoos/#focoos.focoos.Focoos.new_model) on the `Focoos` object. You can choose the model you want to personalize from the list of [Focoos Models](../models.md) available on the platform. Make sure to select the correct model for your task.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

model = focoos.new_model(
    name="<YOUR-MODEL-NAME>",
    description="<YOUR-MODEL-DESCRIPTION>",
    focoos_model="<FOCOOS-MODEL-NAME>",
)
```
An example of how to create a model is the following:
```python
model = focoos.new_model(
    name="my-model",
    description="my-model-description",
    focoos_model="fai-rtdetr-m-obj365",
)
```
This function will return a new [`RemoteModel`](/focoos/api/remote_model/#focoos.remote_model.RemoteModel) object that you can use to train the model and to perform remote inference.

## 3. Train the model
Once the model is created, you can start the training process by calling the [`train` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.train) on the model object.

```python
from focoos.ports import Hyperparameters

res = model.train(
    dataset_ref=dataset_ref,
    hyperparameters=Hyperparameters(
        learning_rate=0.0001, # custom learning rate
        batch_size=16, # custom batch size
        max_iters=1500, # custom max iterations
    ),
)
```
For selecting the `dataset_ref` see the [step 1](#1-load-or-select-a-dataset).
You can further customize the training process by passing additional parameters to the [`train` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.train) (such as the instance type, the volume size, the maximum runtime, etc.) or use additional hyperparameters (see the list [available hyperparameters](/focoos/api/ports/#focoos.ports.Hyperparameters)).

Futhermore, you can monitor the training progress by polling the training status. Use the [`notebook_monitor_train`](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.notebook_monitor_train) method on a jupyter notebook:
```python
model.notebook_monitor_train(interval=30, plot_metrics=True)
```

You can also get the training logs by calling the [`train_logs` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.train_logs):
```python
logs = model.train_logs()
pprint(logs)
```

Finally, if for some reason you need to cancel the training, you can do so by calling the [`stop_training` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.stop_training):
```python
model.stop_training()
```

## 4. Visualize training metrics
You can visualize the training metrics by calling the [`metrics` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.metrics):
```python
metrics = model.metrics()
visualizer = MetricsVisualizer(metrics)
visualizer.log_metrics()
```
The function will return an object of type [`Metrics`](/focoos/api/ports/#focoos.ports.Metrics) that you can use to visualize the training metrics using a `MetricsVisualizer` object.

On notebooks, you can also plot the metrics by calling the [`notebook_plot_training_metrics`](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.notebook_plot_training_metrics) method:
```python
visualizer.notebook_plot_training_metrics()
```

## 5. Test your model

### Remote Inference
Once the training is over, you can test your model using remote inference by calling the [`infer` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.infer) on the model object.

```python
image_path = "<PATH-TO-YOUR-IMAGE>"
result, _ = model.infer(image_path, threshold=0.5, annotate=False)

for det in result.detections:
    print(f"Found {det.label} with confidence {det.conf:.2f}")
    print(f"Bounding box: {det.bbox}")
    if det.mask:
        print("Instance segmentation mask included")
```
`result` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](/focoos/api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference.

The `threshold` parameter is optional and defines the minimum confidence score for a detection to be considered valid (predictions with a confidence score lower than the threshold are discarded).

Optionally, you can preview the results by passing the `annotate` parameter to the `infer` method.
```python
from PIL import Image

output, preview = model.infer(image_path, threshold=0.5, annotate=True)
preview = Image.fromarray(preview[:,:,[2,1,0]]) # invert to make it RGB
```

### Local Inference
!!! Note
    To perform local inference, you need to install the package with one of the extra modules (`[cpu]`, `[torch]`, `[cuda]`, `[tensorrt]`). See the [installation](../setup.md) page for more details.

You can perform inference locally by getting the [`LocalModel`](/focoos/api/local_model) you already trained and calling the [`infer` method](/focoos/api/local_model/#focoos.local_model.LocalModel.infer) on your image. If it's the first time you run the model locally, the model will be downloaded from the cloud and saved on your machine. Additionally, if you use CUDA or TensorRT, the model will be optimized for your GPU before running the inference (it can take few seconds, especially for TensorRT).

```python
model = focoos.get_local_model(model.model_ref) # get the local model

image_path = "<PATH-TO-YOUR-IMAGE>"
result, _ = model.infer(image_path, threshold=0.5, annotate=False)

for det in result.detections:
    print(f"Found {det.label} with confidence {det.conf:.2f}")
    print(f"Bounding box: {det.bbox}")
    if det.mask:
        print("Instance segmentation mask included")
```
As for remote inference, you can pass the `annotate` parameter to return a preview of the prediction and play with the `threshold` parameter to change the minimum confidence score for a detection to be considered valid.
