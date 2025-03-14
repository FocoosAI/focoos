# Welcome to the Focoos AI SDK

Focoos AI provides a powerful development platform designed to help developers and businesses deploy high-performance, cost-efficient computer vision models. Whether you're processing images in the cloud, running inference on edge devices, or training custom models on your dataset, the Focoos AI SDK makes it seamless.


### Why choose Focoos AI?

- 🔹 Blazing Fast Inference. Up to 10x faster than traditional methods.
- 💰 Optimized Cost. Requires 4x less computation, reducing cloud and hardware expenses.
- ⚡ Quick Deployment. Deploy and fine-tune models with minimal effort.


## SDK Overview
The Focoos Python SDK provides seamless access to our state-of-the-art computer vision models. With just a few lines of code, you can easily **select, customize, test, and deploy** pre-trained models tailored to your specific needs.
Whether you're deploying in the cloud or on edge devices, the Focoos Python SDK integrates smoothly into your workflow, speeding up your development process.


### Quickstart 🚀
Ready to dive in? Get started with the setup in just a few simple steps!

**Install** the Focoos Python SDK (for more options, see [setup](setup.md))
=== "uv"
    ```bash linenums="0"
    uv pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
    ```

=== "pip"
    ```bash linenums="0"
    pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
    ```

=== "conda"
    ```bash linenums="0"
    pip install 'focoos @ git+https://github.com/FocoosAI/focoos.git'
    ```

⚙️ **Customize** the models to your specific needs by [fine-tuning](howto/personalize_model.md) on your own dataset.
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

🚀 **Use** the model with just few lines of [code](howto/use_model).

```python
from focoos import Focoos
from PIL import Image
# Initialize the Focoos client with your API key
focoos = Focoos(api_key="<YOUR-API-KEY>")

# Get the remote model from Focoos API
model = focoos.get_remote_model("<MODEL-REF>")

# Run inference on an image
detections, preview = model.infer(image_path, threshold=0.5, annotate=True)

# Output the detections
Image.fromarray(preview)
```


### Our Models 🧠
Focoos AI offers the best models in object detection, semantic and instance segmentation, and more is coming soon.

Using Focoos AI helps you save both time and money while delivering high-performance AI models 💪:

- **10x Faster** 🚀: our models are able to process images up to ten times faster than traditional methods.
- **4x Cheaper** 💰: our models require up to 4x less computational power, letting you save on hardware or cloud bill while ensuring high-quality results.
- **90% Time Saved** ⏱️: our platform accelerates computer vision model development and deployment, enabling faster model training, seamless integration, and optimized performance with minimal effort.

These are not empty promises, but the result of years of research and development by our team 🔬:
<div style="space-between; margin: 20px 0;">
    <div style="flex: 1; margin-right: 10px;">
        <img src="./models/fai-ade.png" alt="ADE-20k Semantic Segmentation" style="width: 100%;">
        <figcaption style="text-align: center;">ADE-20k <a href="models/#semantic-segmentation">Semantic Segmentation</a> Results</figcaption>
    </div>
    <div style="flex: 1; margin-left: 10px;">
        <img src="./models/fai-coco.png" alt="COCO Object Detection" style="width: 100%;">
        <figcaption style="text-align: center;">COCO <a href="models/#object-detection">Object Detection</a> Results</figcaption>
    </div>
</div>

See the list of our models in the [models](models.md) section.

---
## Start now!
By choosing Focoos AI, you can save time, reduce costs, and achieve superior model performance, all while ensuring the privacy and efficiency of your deployments.

[Reach out to us](mailto:support@focoos.ai) to ask for your API key and power your computer vision projects.

Otherwise [Book A Demo](https://www.focoos.ai/book-a-demo) now to access the platform and test by yourself Focoos AI in action.
