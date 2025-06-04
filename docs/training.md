# How to Train a Computer Vision Model with Focoos

Focoos provides a comprehensive training framework that makes it easy to train state-of-the-art computer vision models on your own datasets. Whether you're working on object detection, image classification, or other vision tasks, Focoos offers an intuitive training pipeline that handles everything from data preparation to model optimization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/tutorials/training.ipynb)

Key features of the Focoos training framework include:

- **Easy Dataset Integration**: Seamlessly import and prepare your datasets using Focoos' data loading utilities
- **Flexible Model Architecture**: Choose from a variety of pre-built model architectures or customize your own
- **Advanced Training Features**:
    - Mixed precision training for faster training and reduced memory usage
    - Automatic learning rate scheduling
    - Early stopping and model checkpointing
    - Distributed training support
- **Experiment Tracking**: Monitor training progress, visualize metrics, and compare experiments through the Focoos Hub

In the following sections, we'll guide you through the process of training a model with Focoos, from setting up your environment to deploying your trained model.

## 🎨 Fine-tune a model in 3 steps

In this guide, we will perform the following steps:

1. [📦 Select dataset](#1-select-dataset)
2. [🏃‍♂️ Train model](#2-train-model)
3. [🧪 Test model](#3-test-model)

## 0. \[Optional\] Connect to the Focoos Hub

Focoos can be used without having an accont on the [Focoos Hub](app.focoos.ai). With it, you will unlock additional functionalities, as we will see below. If you have it, just connect to the HUB.
```
from focoos.hub import FocoosHUB

FOCOOS_API_KEY = None  # write here your API key
hub = FocoosHUB(api_key=FOCOOS_API_KEY)
```

## 1. Select dataset

Before starting the training, we need to get a dataset. You can either use a local dataset or you can download one from the hub.

### \[Optional\] Download the data from the Hub
If you want to download a dataset from the hub, you can use it to directly store it in your local environment.
Check the reference of your dataset on the platform and use it in the following cell. If you want to try an example dataset, you can use [Football Player Detection](https://app.focoos.ai/datasets/3a7cec8afb6b4780) with reference `3a7cec8afb6b4780`.

```python
dataset_ref = "<YOUR-DATASET-REFERENCE>"
dataset = hub.get_remote_dataset(dataset_ref)
print(dataset)

dataset_path = dataset.download_data()
```

### Get the training dataset

Now that we downloaded the dataset, we can magically 🪄 instanciate the dataset using the [`AutoDataset`]() as will be used in the training. You can optionally specify aumgentations for the training using the [`DatasetAugmentation`]() dataclass.

```python
from focoos.data.auto_dataset import AutoDataset
from focoos.data.default_aug import DatasetAugmentations
from focoos.ports import DatasetSplitType

task = dataset.task  # see ports.Task for more information
layout = dataset.layout  # see ports.DatasetLayout for more information
auto_dataset = AutoDataset(dataset_name=dataset_path, task=task, layout=layout)

augs = DatasetAugmentations(resolution=512).get_augmentations()

train_dataset = auto_dataset.get_split(augs=augs, split=DatasetSplitType.TRAIN)
valid_dataset = auto_dataset.get_split(augs=augs, split=DatasetSplitType.VAL)
```

## 2. Train the Model

### Instanciate a model
The first step to personalize your model is to instance a model. You can get a model using the ModelManager by specifying a model name. Optionally, you can also get one of your trained models on the hub. If you want to follow the example, just use `fai-detr-m-coco` as the model reference.

```python
from focoos.model_manager import ModelManager

model_ref = "<YOUR-MODEL-REF>"
model = ModelManager.get("hub://" + model_ref, hub=hub)
```

### Select the hyper-parameters
The next step is to create a [`TrainerArgs`]() with the hyper-parameters such as the learning rate, the number of iterations and so on.
Optionally, if you are using the hub, you can specify `sync_to_hub=True` to track the experiment on the Focoos Hub.

```python
from focoos.ports import TrainerArgs

args = TrainerArgs(
    run_name="football-tutorial",  # the name of the experiment
    output_dir="./experiments",  # the folder where the model is saved
    batch_size=16,  # how many images in each iteration
    max_iters=500,  # how many iterations lasts the training
    eval_period=100,  # period after we eval the model on the validation (in iterations)
    learning_rate=0.0001,  # learning rate
    weight_decay=0.0001,  # regularization strenght (set it properly to avoid under/over fitting)
    sync_to_hub=True,  # Use this to see the model under training on the platform
)
```

### Train the model
Now we are set up. We can directly call the train function of the model.

```python
model.train(args, train_dataset, valid_dataset, hub=hub)
```


## 3. Test the Model
Now that the model is ready, let's see how it behaves.

```python
import random
from PIL import Image
from focoos.utils.vision import annotate_image

index = random.randint(0, len(valid_dataset))

ground_truth = valid_dataset.preview(index, use_augmentations=False).save("ground_truth.jpg")

image = Image.open(valid_dataset[index]["file_name"])
outputs = model(image)

prediction = annotate_image(image, outputs, task=task, classes=model.model_info.classes).save("prediction.jpg")
```
