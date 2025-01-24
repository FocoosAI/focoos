# Cloud Training

This section covers the steps to train a model in the cloud using the `focoos` library. The following example demonstrates how to interact with the Focoos API to manage models, datasets, and training jobs.

---

## Listing Available Datasets

Before training a model, you can list all available shared datasets:

```python
from pprint import pprint
import os
from focoos import Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

datasets = focoos.list_shared_datasets()
pprint(datasets)
```

##  Initiating a Cloud Training Job
To start training, configure the model, dataset, and training parameters as shown below:

```python
from focoos.ports import Hyperparameters, TrainInstance

model = focoos.get_remote_model("<YOUR-MODEL-ID>")

res = model.train(
    dataset_ref="<YOUR-DATASET-ID>",
    instance_type=TrainInstance.ML_G4DN_XLARGE,
    volume_size=50,
    max_runtime_in_seconds=36000,
    hyperparameters=Hyperparameters(
        learning_rate=0.0001,
        batch_size=16,
        max_iters=1500,
        eval_period=100,
        resolution=640,
    ),  # type: ignore
)
```

##  Monitoring Training Progress on jupyter notebook

Once the training job is initiated, monitor its progress by polling the training status. Use the following code:

```python
from focoos import  Focoos

focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

model = focoos.get_remote_model("<YOUR-MODEL-ID>")
model.notebook_monitor_train(interval=30, plot_metrics=True)
```

##  Retrieving Training Logs

After the training process is complete, retrieve the logs for detailed insights:

```python
logs = model.train_logs()
pprint(logs)
```

## Retrieve and Visualize Training Metrics

```python
from focoos.utils.metrics import MetricsVisualizer

metrics = model.metrics()
visualizer = MetricsVisualizer(metrics)
visualizer.log_metrics()
```
