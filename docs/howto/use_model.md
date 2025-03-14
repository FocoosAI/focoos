# Select and Inference with Focoos Models

This section covers how to perform inference using the [Focoos Models](../models.md) on the cloud or locally using the `focoos` library.

As a reference, the following example demonstrates how to perform inference using the [`fai-rtdetr-m-obj365`](../models/fai-rtdetr-m-obj365.md) model, but you can use any of the models listed in the [models](../models.md) section.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FocoosAI/focoos/blob/main/notebooks/inference.ipynb)

In this guide, we will cover the following topics:

1. [☁️ Cloud Inference](#cloud-inference-for-image-processing)
!<!-- 2. [☁️ Cloud Inference for Video Processing](#cloud-inference-for-video-processing)-->
2. [☁️ Cloud Inference with Gradio](#cloud-inference-with-gradio)
3. [🏠 Local Inference](#local-inference)


## Cloud inference for image processing
Running inference on the cloud is simple and efficient. Select the model you want to use and call the [`infer` method](/focoos/api/remote_model/#focoos.remote_model.RemoteModel.infer) on your image. The image will be securely uploaded to the Focoos AI platform, where the selected model processes it and returns the results.

To get the model reference you can refere to the [Model Management section](manage_models.md). Here the code to handle a single image inference:

```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

image_path = "<PATH-TO-YOUR-IMAGE>"
model = focoos.get_remote_model("<MODEL-REF>")
result, _ = model.infer(image_path, threshold=0.5, annotate=True)

for det in result.detections:
    print(f"Found {det.label} with confidence {det.conf:.2f}")
    print(f"Bounding box: {det.bbox}")
    if det.mask:
        print("Instance segmentation mask included")
        print(f"Mask shape: {det.mask.shape}")
```

`result` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](/focoos/api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference. The `FocoosDet` object contains the following attributes:

- `bbox`: Bounding box coordinates in x1y1x2y2 absolute format.
- `conf`: Confidence score (from 0 to 1).
- `cls_id`: Class ID (0-indexed).
- `label`: Label (name of the class).
- `mask`: Mask (base64 encoded string having origin in the top left corner of bbox and the same width and height of the bbox).


Optional parameters are:

- `threshold` (default: 0.5) – Sets the minimum confidence score required for prediction to be considered valid. Predictions below this threshold are discarded.
- `annotate` (default: False) – If set to True, the method returns preview, an annotated image with the detected objects.


### Image Preview

You can preview the results by passing the `annotate` parameter to the `infer` method:

```python
from PIL import Image

output, preview = model.infer(image_path, threshold=0.5, annotate=True)
preview = Image.fromarray(preview)

```

<!--
## Cloud inference for video processing
This guide explains how to process a video using Focoos AI's cloud inference. The script extracts frames from a video, sends them to Focoos AI for processing, and saves an annotated video with predictins.

Other than the `focoos` library, ensure you have the required dependencies installed:

```bash linenums="0"
pip install opencv-python numpy
```

The following Python script:

- Reads frames from a video file.
- Sends each frame to Focoos AI cloud for inference.
- Receives the processed frame and saves it in a new video file.


```python
import cv2

model = focoos.get_remote_model("3c8442ffb4f54a7d")

input_video_path = "/Users/antoniotavera/Desktop/package.mp4"
output_video_path = "/Users/antoniotavera/Desktop/package_analysis.mp4"

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)  # Default to 30 FPS if unavailable
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing {frame_count} frames...")
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    frame_index += 1
    print(f"Processing frame {frame_index}/{frame_count}")

    temp_frame_path = f"frame_{frame_index}.jpg"
    cv2.imwrite(temp_frame_path, frame)

    result, preview = model.infer(temp_frame_path, threshold=0.7, annotate=True)

    out.write(preview)


# Release resources
cap.release()
out.release()

print(f"Annotated video saved as: {output_video_path}")

```
-->

## Cloud inference with Gradio
You can easily create a web interface for your model using Gradio.

First, install the required `dev` dependencies:

```bash linenums="0"
pip install '.[dev]'
```

Set your Focoos API key as an environment variable and start the application. The model selection will be available from the UI:

```bash linenums="0"
export FOCOOS_API_KEY_GRADIO=<YOUR-API-KEY> && python gradio/app.py
```
Now, your model is accessible through a user-friendly web interface! 🚀


## Local inference
!!! Note
    To perform local inference, you need to install the package with one of the extra modules (`[cpu]`, `[torch]`, `[cuda]`, `[tensorrt]`). See the [installation](../setup.md) page for more details.

You can run inference locally by selecting a model and calling the [`infer` method](/focoos/api/local_model/#focoos.local_model.LocalModel.infer) on your image.
If this is the first time you are running the model locally, it will be downloaded from the cloud and stored on your machine.
If you are using CUDA or TensorRT, the model will be optimized for your GPU before inference. This process may take a few seconds, especially for TensorRT.

Example code:
```python
from focoos import Focoos

focoos = Focoos(api_key="<YOUR-API-KEY>")

model = focoos.get_local_model("<MODEL-REF>")

image_path = "<PATH-TO-YOUR-IMAGE>"
result, _ = model.infer(image_path, threshold=0.5, annotate=True)

for det in result.detections:
    print(f"Found {det.label} with confidence {det.conf:.2f}")
    print(f"Bounding box: {det.bbox}")
    if det.mask:
        print("Instance segmentation mask included")
        print(f"Mask shape: {det.mask.shape}")

```

`result` is a [FocoosDetections](/focoos/api/ports/#focoos.ports.FocoosDetections) object, containing a list of [FocoosDet](/focoos/api/ports/#focoos.ports.FocoosDet) objects and optionally a dict of information about the latency of the inference. The `FocoosDet` object contains the following attributes:

- `bbox`: Bounding box coordinates in x1y1x2y2 absolute format.
- `conf`: Confidence score (from 0 to 1).
- `cls_id`: Class ID (0-indexed).
- `label`: Label (name of the class).
- `mask`: Mask (base64 encoded string having origin in the top left corner of bbox and the same width and height of the bbox).



As for remote inference, you can use the `annotate` parameter to return a preview of the prediction.
Local inference provides faster results and reduces cloud dependency, making it ideal for real-time applications and edge deployments.
