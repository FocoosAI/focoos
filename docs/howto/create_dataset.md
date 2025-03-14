# Dataset Management

This section covers the steps to create, upload, and manage datasets in Focoos using the SDK.
The `focoos` library supports multiple dataset formats, making it flexible for various machine learning tasks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/dataset.ipynb)


In this guide, we will show the following steps:

1. [🧬 Dataset format](#1-dataset-format)
2. [📸 Create dataset](#2-create-dataset)
3. [📤 Upload data](#3-upload-data)
4. [📥 Download your own dataset from Focoos](#4-download-your-own-dataset-from-focoos-platform)
5. [🌍 Download dataset from external sources](#5-download-dataset-from-external-sources)
6. [🗑️ Delete data](#6-delete-data)
7. [🚮 Delete dataset](#7-delete-dataset)


## 1. Dataset format
The `focoos` library currently supports three distinct dataset layouts, providing seamless compatibility with various machine learning workflows. Below are the supported formats along with their respective folder structures:

- **ROBOFLOW_COCO** (Detection, Instance Segmentation):
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
        - img_3_mask.png
        - img_4_mask.png
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

!!! Note
    More dataset formats will be added soon. If you need support for a specific format, feel free to reach out via email at [support@focoos.ai](mailto:support@focoos.ai)


## 2. Create dataset
The `focoos` library enables you to create datasets tailored for specific deep learning tasks, such as object detection and semantic segmentation. The available computer vision tasks are defined in the [FocoosTask function](../api/ports.md/#focoos.ports.FocoosTask). Each dataset must follow a specific structure to ensure compatibility with the Focoos platform. You can select the appropriate dataset format from the supported options detailed in [Dataset Format](#1-dataset-format).

Use the following code to create a new dataset:

```python
from focoos import DatasetLayout, Focoos, FocoosTask

focoos = Focoos(api_key="<YOUR-API-KEY>")

# Create a new remote dataset
dataset = focoos.add_remote_dataset(
    name="my-dataset",
    description="My custom dataset for object detection",
    layout=DatasetLayout.ROBOFLOW_COCO,  # Choose dataset format
    task=FocoosTask.DETECTION  # Specify the task type
)
```


## 3. Upload data
Once you've created a dataset, you can upload your data as a ZIP archive from your local folder:

```python
dataset.upload_data("./datasets/my_dataset.zip")
```

After the upload, you can check dataset [preview](../api/ports.md/#focoos.ports.DatasetPreview) using:

```python
dataset_info = dataset.get_info()
print(dataset_info)
```

Alternatively, you can list all available datasets (both personal and shared):

```python
datasets = focoos.list_datasets()
for dataset in datasets:
    print(f"Name: {dataset.name}")
    print(f"Reference: {dataset.ref}")
    print(f"Task: {dataset.task}")
    print(f"Description: {dataset.description}")
    print(f"spec: {dataset.spec}")
    print("-" * 50)
```


## 4. Download your own dataset from Focoos platform
If you have previously uploaded a dataset to Focoos platform, you can retrieve it by following these steps.
First, list all your datasets to identify the dataset reference:


```python
datasets = focoos.list_datasets()

for dataset in datasets:
    print(f"Name: {dataset.name}")
    print(f"Reference: {dataset.ref}")
```

Once you have the dataset reference, use the following code to download the associated data to a predefined local folder:

```python
dataset_ref = "<YOUR-DATASET-REFERENCE>"
dataset = focoos.get_remote_dataset(dataset_ref)

dataset.download_data("./<YOUR-DATA-FOLDER>/")
```



## 5. Download dataset from external sources
You can also download datasets from external sources like Dataset-Ninja (Supervisely) and Roboflow Universe, then upload them to the Focoos platform for use in your projects.

=== "pip"
    ```bash linenums="0"
    pip install dataset-tools roboflow
    pip install setuptools
    ```

- **Dataset Ninja**:
```python
import dataset_tools as dtools

dtools.download(dataset="dacl10k", dst_dir="./datasets/dataset-ninja/")
```

- **Roboflow**:
```python
import os

from roboflow import Roboflow

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(14)
dataset = version.download("coco")
```


## 6. Delete data
If you need to remove specific files from an existing dataset without deleting the entire dataset, you can do so by specifying the filename. This is useful when updating or refining your dataset.

Use the following command:
```python
dataset_ref = "<YOUR-DATASET-REFERENCE>"
dataset = focoos.get_remote_dataset(dataset_ref)
dataset.delete_data()
```
!!! warning
    This will permanently remove the specified file from your dataset in Focoos platform. Be sure to double-check the filename before executing the command, as deleted data cannot be recovered.



## 7. Delete dataset
If you want to remove an entire dataset from the Focoos platform, use the following command:

```python
dataset_ref = "<YOUR-DATASET-REFERENCE>"
dataset = focoos.get_remote_dataset(dataset_ref)
dataset.delete()
```
!!! warning
    Deleting a dataset is irreversible. Once deleted, all data associated with the dataset is permanently lost and cannot be recovered.

##
