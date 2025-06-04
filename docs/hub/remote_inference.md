# 🌍 Remote Inference

Remote inference allows you to run computer vision models in the cloud without needing local GPU resources. This is perfect for production deployments, edge devices, or when you want to avoid the overhead of managing local model inference.

## What is Remote Inference?

Remote inference uses the Focoos cloud infrastructure to run your models. Instead of loading models locally, you send images to the cloud API and receive inference results. This provides several advantages:

- **No Local GPU Required**: Run inference on any device, including CPU-only machines
- **Scalability**: Handle varying inference loads without managing infrastructure
- **Always Updated**: Use the latest version of your models automatically
- **Cost Efficient**: Pay per inference without maintaining dedicated hardware
- **Low Latency**: Optimized cloud infrastructure for fast inference

## Getting Started

### Basic Remote Inference

Here's how to perform remote inference with a model:

```python
from focoos import FocoosHUB

# Initialize the HUB client
hub = FocoosHUB()

# Get a remote model instance
model_ref = "fai-detr-l-obj365"  # Use any available model
remote_model = hub.get_remote_model(model_ref)

# Perform inference
results = remote_model.infer("path/to/image.jpg", threshold=0.5)

# Process results
for detection in results.detections:
    print(f"Class ID: {detection.cls_id}")
    print(f"Confidence: {detection.conf:.3f}")
    print(f"Bounding Box: {detection.bbox}")
```

### Using the Callable Interface

Remote models can also be called directly like functions:

```python
# This is equivalent to calling remote_model.infer()
results = remote_model("path/to/image.jpg", threshold=0.5)
```

## Supported Input Types

Remote inference accepts various input types:

### File Paths
```python
results = remote_model.infer("./images/photo.jpg")
```

### NumPy Arrays
```python
import cv2
import numpy as np

# Load image as numpy array
image = cv2.imread("photo.jpg")
results = remote_model.infer(image, threshold=0.3)
```

### PIL Images
```python
from PIL import Image

# Load with PIL
pil_image = Image.open("photo.jpg")
results = remote_model.infer(pil_image)
```

### Raw Bytes
```python
# Image as bytes
with open("photo.jpg", "rb") as f:
    image_bytes = f.read()

results = remote_model.infer(image_bytes)
```

## Inference Parameters

### Threshold Control

Control detection sensitivity with the threshold parameter:

```python
# High threshold - only very confident detections
results = remote_model.infer("image.jpg", threshold=0.8)

# Low threshold - more detections, potentially less accurate
results = remote_model.infer("image.jpg", threshold=0.2)

# Default threshold (usually 0.5)
results = remote_model.infer("image.jpg")
```

## Working with Results

### Detection Results

For object detection models, results contain bounding boxes and classifications:

```python
results = remote_model.infer("image.jpg")

print(f"Found {len(results.detections)} objects")

for i, detection in enumerate(results.detections):
    print(f"Detection {i+1}:")
    print(f"  Class ID: {detection.cls_id}")
    print(f"  Confidence: {detection.conf:.3f}")
    print(f"  Bounding Box: {detection.bbox}")

    # Box coordinates
    if detection.bbox:
        x1, y1, x2, y2 = detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]
    print(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
```

### Visualization

Visualize results using the built-in utilities:

```python
from focoos.utils.vision import annotate_image

results = model.infer(image=image, threshold=0.5)

annotated_image = annotate_image(
    im=image, detections=results, task=model.model_info.task, classes=model.model_info.classes
)
```

## Model Management for Remote Inference

### Checking Model Status

Before using a model for inference, check its status:

```python
model_info = remote_model.get_info()

if model_info.status == ModelStatus.TRAINING_COMPLETED:
    print("Model is ready for inference")
    results = remote_model.infer("image.jpg")
elif model_info.status == ModelStatus.TRAINING_RUNNING:
    print("Model is still training")
elif model_info.status == ModelStatus.TRAINING_ERROR:
    print("Model has an error")
```

### Model Information

Get detailed information about the remote model:

```python
model_info = remote_model.get_info()

print(f"Model: {model_info.name}")
print(f"Task: {model_info.task}")
print(f"Classes: {model_info.classes}")
print(f"Image Size: {model_info.im_size}")
print(f"Status: {model_info.status}")
```

## Comparison: Remote vs Local Inference

| Aspect | Remote Inference | Local Inference |
|--------|-----------------|-----------------|
| **Hardware** | No GPU required | GPU recommended |
| **Setup** | Instant | Model download required |
| **Scalability** | Automatic | Manual scaling |
| **Cost** | Pay per use | Infrastructure costs |
| **Latency** | Network dependent | Very low |
| **Privacy** | Data sent to cloud | Data stays local |
| **Offline** | Requires internet | Works offline |

## Best Practices

1. **Optimize Images**: Resize large images to reduce upload time and costs
2. **Handle Errors**: Implement retry logic for network issues
3. **Batch Smartly**: Group related inferences to minimize overhead
4. **Monitor Usage**: Track inference costs and quotas
5. **Cache Results**: Store results for identical inputs when appropriate
6. **Use Appropriate Thresholds**: Tune detection thresholds for your use case

## See Also

- [HUB](hub.md) - Complete HUB documentation
- [Overview](overview.md) - HUB architecture overview
- [API Reference](../api/hub.md) - Detailed API documentation
