# Datasets

With the Focoos SDK, you can leverage a diverse collection of datasets shared by FocoosAI team or create and upload your own datasets to the platform.


## Create and upload your own dataset

You can create and upload your own dataset to the platform by using the SDK or the WebAPP.
actually Focoos support 3 differents layout for dataset:

- Object Detection
    - Roboflow COCO
- Instance Segmentation
    - Roboflow SEG
- Semantic Segmentation
    - Roboflow COCO
    - Supervisely

**Dataset Structure for each layout is defined in [DatasetLayout](https://focoosai.github.io/focoos/api/ports/#focoos.ports.DatasetLayout)**

1. create a dataset
```python
from focoos import DatasetLayout, Focoos, FocoosTask

focoos = Focoos()

ds = focoos.add_remote_dataset(
    name="my-dataset", description="my-dataset", layout=DatasetLayout.ROBOFLOW_COCO, task=FocoosTask.DETECTION
)
```

2. upload dataset zip from local folder
```python
ds_spec = ds.upload_data("./datasets/my_dataset.zip")
```


After uploading the dataset, you can see the dataset specification in the application or with Focoos SDK:

```python
ds_spec = ds.get_info()
```
or list all datasets (both user and shared datasets):

```python
from focoos import DatasetLayout, Focoos, FocoosTask

focoos = Focoos()
focoos.list_datasets()
```
