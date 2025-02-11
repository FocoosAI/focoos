# Model Management

This section covers the steps to monitor the status of your models on the FocoosAI platform.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/model_management.ipynb)

## How to list the Focoos Models
To list all the models available on the FocoosAI platform, you can use the following code:
```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

models = focoos.list_focoos_models()
for model in models:
    print(f"Name: {model.name}")
    print(f"Reference: {model.ref}")
    print(f"Status: {model.status}")
    print(f"Task: {model.task}")
    print(f"Description: {model.description}")
    print("-" * 50)


```
`models` is a list of [`ModelPreview`](/focoos/api/ports/#focoos.ports.ModelPreview) objects that contains the following information:

- `name`: The name of the model.
- `ref`: The reference of the model.
- `status`: The status of the model.
- `task`: The task of the model.
- `description`: The description of the model.
- `status`: The status of the model, which indicates its current state (e.g. CREATED, TRAINING_RUNNING, TRAINING_COMPLETED - see [`ModelStatus`](/focoos/api/ports/#focoos.ports.ModelStatus)).

## How to list all your models
To list all your models, the library provides a `list_models` function. This function will return a list of `Model` objects.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

models = focoos.list_models()
for model in models:
    print(f"Name: {model.name}")
    print(f"Reference: {model.ref}")
    print(f"Status: {model.status}")
    print(f"Task: {model.task}")
    print(f"Description: {model.description}")
    print(f"Focoos Model: {model.focoos_model}")
    print("-" * 50)

```
`models` is a list of [`ModelPreview`](../../api/ports/#focoos.ports.ModelPreview) objects that contains the following information:

- `name`: The name of the model.
- `ref`: The reference of the model.
- `status`: The status of the model.
- `task`: The task of the model.
- `description`: The description of the model.
- `focoos_model`: The starting Focoos Model used for training.
- `status`: The status of the model, which indicates its current state (e.g. CREATED, TRAINING_RUNNING, TRAINING_COMPLETED - see [`ModelStatus`](../../api/ports/#focoos.ports.ModelStatus)).

## See the metrics for a model
To see the validation metrics of a model, you can use the [`metrics` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.metrics) on the model object.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

model = focoos.get_remote_model("my-model")
metrics = model.metrics()

if metrics.best_valid_metric:
    print(f"Best validation metrics:")
    for k, v in metrics.best_valid_metric.items():
        print(f"  {k}: {v}")

if metrics.valid_metrics:
    print(f"Last iteration validation metrics:")
    for k, v in metrics.valid_metrics[-1].items():
        print(f"  {k}: {v}")

if metrics.train_metrics:
    print(f"Last iteration training metrics:")
    for k, v in metrics.train_metrics[-1].items():
        print(f"  {k}: {v}")

```
`metrics` is a [`Metrics`](/focoos/api/ports/#focoos.ports.Metrics) object that contains the validation metrics of the model.

## Delete a model
To delete a model, you can use the [`delete_model` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.delete_model) on the model object.

!!! warning
    This action is irreversible and the model will be deleted forever from the platform.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

model = focoos.get_remote_model("my-model")
model.delete_model()
```


<!-- ## How to list all the shared datasets
To list all the shared datasets, the library provides a `list_shared_datasets` function. This function will return a list of `Dataset` objects.

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

datasets = focoos.list_shared_datasets()
for dataset in datasets:
    print(f"Name: {dataset.name}")
    print(f"Reference: {dataset.ref}")
    print(f"Task: {dataset.task}")
    print(f"Description: {dataset.description}")
    print("-" * 50)

```
`datasets` is a list of [`DatasetMetadata`](/focoos/api/ports/#focoos.ports.DatasetMetadata) objects that contains the following information:

- `name`: The name of the dataset.
- `ref`: The reference of the dataset.
- `task`: The task of the dataset.
- `description`: The description of the dataset.

     -->
