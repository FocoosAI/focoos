# Quickstart 🚀

Getting started with Focoos AI has never been easier! In just a few steps, you can quickly set up remote inference using our built-in models. Here's a simple example of how to perform object detection with the **focoos_object365** model:

## Step 1: Install the SDK

First, make sure you've installed the Focoos Python SDK by following the [installation guide](../installation).

## Step 2: Set Up Remote Inference

With the SDK installed, you can start using the Focoos API to run inference remotely. Here's a basic code snippet to detect objects in an image using a pre-trained model:

```python
from focoos import Focoos
import os

# Initialize the Focoos client with your API key
focoos = Focoos(api_key=os.getenv("FOCOOS_API_KEY"))

# Get the remote model (focoos_object365) from Focoos API
model = focoos.get_remote_model("focoos_object365")

# Deploy the model
model.deploy()

# Run inference on an image
detections = model.infer("./image.jpg", threshold=0.4)

# Output the detections
print(detections)
```
