![Tests](https://github.com/FocoosAI/focoos/actions/workflows/test.yml/badge.svg??event=push&branch=main)

# Welcome to Focoos AI

Focoos AI provides an advanced development platform designed to empower developers and businesses with efficient, customizable computer vision solutions. Whether you're working with data from cloud infrastructures or deploying on edge devices, Focoos AI enables you to select, fine-tune, and deploy state-of-the-art models optimized for your unique needs.

## SDK Overview

<!-- Unlock the full potential of Focoos AI with the Focoos Python SDK! 🚀  -->
This powerful SDK gives you seamless access to our cutting-edge computer vision models and tools, allowing you to effortlessly interact with the Focoos API. With just a few lines of code, you can easily **select, customize, test, and deploy** pre-trained models tailored to your specific needs.

Whether you're deploying in the cloud or on edge devices, the Focoos Python SDK integrates smoothly into your workflow, speeding up your development process.

### Key Features 🔑

1. **Select Ready-to-use Models** 🧩
   Get started quickly by selecting one of our efficient, [pre-trained models](https://focoosai.github.io/focoos/models/) that best suits your data and application needs.

2. **Personalize Your Model** ✨
   Customize the selected model for higher accuracy through [fine-tuning](https://focoosai.github.io/focoos/how_to/cloud_training/). Adapt the model to your specific use case by training it on your own dataset.

3. **Test and Validate** 🧪
   Upload your data sample to [test the model](https://focoosai.github.io/focoos/how_to/inference/)'s accuracy and efficiency. Iterate the process to ensure the model performs to your expectations.

4. **Remote and Local Inference** 🖥️
   Deploy the model on your devices or use it on our servers. Download the model to run it locally, without sending any data over the network, ensuring full privacy.


## Quickstart 🚀
Ready to dive in? Get started with the setup in just a few simple steps!

### Installation
**Install** the Focoos Python SDK (for more options, see [setup](https://focoosai.github.io/focoos/setup))

**uv**
```bash linenums="0"
uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
```

**pip**, **conda**
```bash linenums="0"
pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
```

### Inference
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/inference.ipynb)

🚀 [Directly use](https://focoosai.github.io/focoos/how_to/inference/) our **Efficient Models**, optimized for different data, applications, and hardware.



```python
from focoos import Focoos
from PIL import Image

# Initialize the Focoos client with your API key
focoos = Focoos(api_key="<YOUR-API-KEY>")

# Get the remote model (fai-rtdetr-m-obj365) from Focoos API
model = focoos.get_remote_model("fai-rtdetr-m-obj365")

# Run inference on an image
detections, preview = model.infer(image_path, annotate=True)

# Output the detections
Image.fromarray(preview)
```

### Training
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/training.ipynb)

⚙️ **Customize** the models to your specific needs by [fine-tuning](https://focoosai.github.io/focoos/how_to/cloud_training/) on your own dataset.



```python
from focoos import Focoos
from focoos.ports import Hyperparameters

focoos = Focoos(api_key="<YOUR-API-KEY>")
model = focoos.new_model(name="awesome",
                         focoos_model="fai-rtdetr-m-obj365",
                         description="An awesome model")

res = model.train(
    dataset_ref="<YOUR-DATASET-ID>",
    hyperparameters=Hyperparameters(
        learning_rate=0.0001,
        batch_size=16,
        max_iters=1500,
    )
)
```

See more examples in the [how to](https://focoosai.github.io/focoos/how_to) section.

## Our Models 🧠
Focoos AI offers the best models in object detection, semantic and instance segmentation, and more is coming soon.

Using Focoos AI helps you save both time and money while delivering high-performance AI models 💪:

- **10x Faster** ⏳: Our models are able to process images up to ten times faster than traditional methods.
- **4x Cheaper** 💰: Our models require up to 4x less computational power, letting you save on hardware or cloud bill while ensuring high-quality results.
- **Tons of CO2 saved annually per model** 🌱: Our models are energy-efficient, helping you reduce your carbon footprint by using less powerful hardware with respect to mainstream models.

These are not empty promises, but the result of years of research and development by our team 🔬:
<div style="space-between; margin: 20px 0;">
    <div style="flex: 1; margin-right: 10px;">
        <img src="https://raw.githubusercontent.com/FocoosAI/focoos/refs/heads/main/docs/models/fai-ade.png" alt="ADE-20k Semantic Segmentation" style="width: 100%;">
        <figcaption style="text-align: center;">ADE-20k <a href="https://focoosai.github.io/focoos/models/#semantic-segmentation">Semantic Segmentation</a> Results</figcaption>
    </div>
    <div style="flex: 1; margin-left: 10px;">
        <img src="https://raw.githubusercontent.com/FocoosAI/focoos/refs/heads/main/docs/models/fai-coco.png" alt="COCO Object Detection" style="width: 100%;">
        <figcaption style="text-align: center;">COCO <a href="https://focoosai.github.io/focoos/models/#object-detection">Object Detection</a> Results</figcaption>
    </div>
</div>

See the list of our models in the [models](https://focoosai.github.io/focoos/models/) section.

---
### Start now!
By choosing Focoos AI, you can save time, reduce costs, and achieve superior model performance, all while ensuring the privacy and efficiency of your deployments.
[Reach out to us](mailto:info@focoos.ai) to ask for your API key for free and power your computer vision projects. 🚀
