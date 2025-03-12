# Datasets

With the Focoos SDK, you can leverage a diverse collection of datasets shared by FocoosAI team or create and upload your own datasets to the platform.


## Create and upload your own dataset

You can create and upload your own dataset to the platform by using the SDK or the WebAPP.
actually Focoos support 3 differents [Dataset Layout](/focoos/api/ports/#datasetlayout) for dataset:

- **ROBOFLOW_COCO** (Detection,Instance Segmentation):
```python
root/
    train/
        - _annotations.coco.json
        - img_1.jpg
        - img_2.jpg
    valid/
        - _annotations.coco.json
        - img_3.jpg
        - img_4.jpg
```
- **ROBOFLOW_SEG** (Semantic Segmentation):
```python
root/
    train/
        - _classes.csv (comma separated csv)
        - img_1.jpg
        - img_2.jpg
    valid/
        - _classes.csv (comma separated csv)
        - img_3.png
        - img_4.png
```
- **SUPERVISELY** (Semantic Segmentation):
```python
root/
    train/
        meta.json
        img/
        ann/
        mask/
    valid/
        meta.json
        img/
        ann/
        mask/
```




## Create Remote Dataset
```python
from focoos import DatasetLayout, Focoos, FocoosTask

focoos = Focoos()

ds = focoos.add_remote_dataset(
    name="my-dataset", description="my-dataset", layout=DatasetLayout.ROBOFLOW_COCO, task=FocoosTask.DETECTION
)
```

## Upload dataset zip archive from local folder
```python
ds_spec = ds.upload_data("./datasets/my_dataset.zip")
```


After uploading the dataset, you can see the dataset [specs](/focoos/api/ports/#datasetpreview) in the application or with Focoos SDK:

```python
ds_spec = ds.get_info()
```
or list all datasets (both user and shared datasets):

```python
from focoos import DatasetLayout, Focoos, FocoosTask

focoos = Focoos()
focoos.list_datasets()
```
