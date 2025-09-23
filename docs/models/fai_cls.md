# FAI-CLS (FocoosAI Classification)

## Overview

Fai-cls is a versatile image classification model developed by FocoosAI that can utilize any backbone architecture for feature extraction. This model is designed for both single-label and multi-label image classification tasks, offering flexibility in architecture choices and training configurations.

The model employs a simple yet effective approach: a configurable backbone extracts features from input images, followed by a classification head that produces class predictions. This design enables easy adaptation to different domains and datasets while maintaining high performance and computational efficiency.

## Available Models

Currently, you can find 3 fai-cls models on the Focoos Hub, all trained on COCO dataset for image classification.

| Model Name | Architecture | Domain (Classes) | Dataset | Metric | FPS Nvidia-T4 |
|------------|--------------|------------------|----------|---------|--------------|
| fai-cls-n-coco | Classification (STDC-Small) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | F1: 48.66<br>Precision: 58.48<br>Recall: 41.66 | - |
| fai-cls-s-coco | Classification (STDC-Small) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | F1: 61.92<br>Precision: 68.69<br>Recall: 56.37 | - |
| fai-cls-m-coco | Classification (STDC-Large) | Common Objects (80) | [COCO](https://cocodataset.org/#home) | F1: 66.98<br>Precision: 73.00<br>Recall: 61.88 | - |

## Supported dataset
- [ROBOFLOW_COCO](/focoos/api/ports/#focoos.ports.DatasetLayout) (multi-class)

- [CLASSIFICATION_FOLDER](/focoos/api/ports/#focoos.ports.DatasetLayout)

## Neural Network Architecture

The FAI-CLS architecture consists of two main components:

### Backbone
- **Purpose**: Feature extraction from input images
- **Design**: Configurable backbone network (ResNet, EfficientNet, STDC, etc.)
- **Output**: High-level feature representations
- **Feature Selection**: Uses specified feature level (default: "res5" for highest-level features)
- **Flexibility**: Supports any backbone that provides the required output shape

### Classification Head
- **Architecture**: Multi-layer perceptron (MLP) with configurable depth
- **Components**:

  - Global Average Pooling (AdaptiveAvgPool2d) for spatial dimension reduction
  - Flatten layer to convert 2D features to 1D
  - Linear layers with ReLU activation
  - Dropout for regularization
  - Final linear layer for class predictions
- **Configurations**:

  - **Single Layer**: Direct mapping from features to classes
  - **Two Layer**: Hidden layer with ReLU and dropout for better feature transformation

## Configuration Parameters

### Core Model Parameters
- `num_classes` (int): Number of classification classes
- `backbone_config` (BackboneConfig): Backbone network configuration

### Architecture Configuration
- `hidden_dim` (int, default=512): Hidden layer dimension for two-layer classifier
- `dropout_rate` (float, default=0.2): Dropout probability for regularization
- `features` (str, default="res5"): Feature level to extract from backbone
- `num_layers` (int, default=2): Number of classification layers (1 or 2)

### Loss Configuration
- `use_focal_loss` (bool, default=False): Use focal loss instead of cross-entropy
- `focal_alpha` (float, default=0.75): Alpha parameter for focal loss
- `focal_gamma` (float, default=2.0): Gamma parameter for focal loss
- `label_smoothing` (float, default=0.0): Label smoothing factor
- `multi_label` (bool, default=False): Enable multi-label classification

## Supported Tasks

### Single-Label Classification
- **Output**: Single class prediction per image
- **Use Cases**:

    - Image categorization (animals, objects, scenes)
    - Medical image diagnosis
    - Quality control in manufacturing
    - Content moderation
    - Agricultural crop classification

- **Loss**: Cross-entropy or focal loss
- **Configuration**: Set `multi_label=False`

### Multi-Label Classification
- **Output**: Multiple class predictions per image
- **Use Cases**:

    - Multi-object recognition
    - Image tagging and annotation
    - Scene attribute recognition
    - Medical condition classification
    - Content-based image retrieval

- **Loss**: Binary cross-entropy with logits
- **Configuration**: Set `multi_label=True`

## Model Outputs

### Training Output (`ClassificationModelOutput`)
- `logits` (torch.Tensor): Shape [B, num_classes] - Raw class predictions
- `loss` (Optional[dict]): Training loss including:
    - `loss_cls`: Classification loss (cross-entropy, focal, or BCE)

### Inference Output
For each detected object:

- `conf` (float): Confidence score
- `cls_id` (int): Class identifier
- `label` (Optional[str]): Human-readable class name


## Losses

The model supports multiple loss function configurations:

### Cross-Entropy Loss (Default)
- **Use Case**: Standard single-label classification
- **Features**: Optional label smoothing for better generalization
- **Activation**: Softmax for probability distribution


### Binary Cross-Entropy Loss
- **Use Case**: Multi-label classification tasks
- **Features**: Independent probability for each class
- **Activation**: Sigmoid for per-class probabilities

## Architecture Variants

### Single-Layer Classifier
```
AdaptiveAvgPool2d(1) → Flatten → Dropout → Linear(features → num_classes)
```
- **Benefits**: Faster inference, fewer parameters
- **Use Case**: Simple datasets or when computational efficiency is critical

### Two-Layer Classifier
```
AdaptiveAvgPool2d(1) → Flatten → Linear(features → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → num_classes)
```
- **Benefits**: Better feature transformation, improved accuracy
- **Use Case**: Complex datasets requiring more sophisticated feature processing

## Training Strategies

### Standard Training
- Use cross-entropy loss with appropriate learning rate scheduling
- Apply data augmentation for better generalization
- Monitor validation accuracy for early stopping

### Imbalanced Data
- Enable focal loss with appropriate α and γ parameters
- Consider class weighting strategies
- Use stratified sampling for validation

### Multi-Label Scenarios
- Set `multi_label=True` in configuration
- Use appropriate evaluation metrics (F1-score, mAP)
- Consider threshold optimization for final predictions

This flexible architecture makes FAI-CLS suitable for a wide range of image classification applications, from simple binary classification to complex multi-label scenarios, while maintaining computational efficiency and ease of use.


### Quick Start with Pre-trained Model

```python
from focoos import ASSETS_DIR, ModelManager
from PIL import Image

# Load a pre-trained model
model = ModelManager.get("fai-cls-m-coco")

image = ASSETS_DIR / "federer.jpg"
result = model.infer(image,threshold=0.5, annotate=True)

# Process results
for detection in result.detections:
    print(f"Class: {detection.label}, Confidence: {detection.conf:.3f}")

# Visualize image
Image.fromarray(result.image)

```
For the training process, please refer to the specific section of the documentation.


## Custom Model Configuration

### Single-Label Classification Setup

```python
from focoos.models.fai_cls.config import ClassificationConfig
from focoos.models.fai_cls.modelling import FAIClassification
from focoos.nn.backbone.resnet import ResnetConfig

# Configure with ResNet backbone
backbone_config = ResnetConfig(
    model_type="resnet",
    depth=50,
    pretrained=True,
)

config = ClassificationConfig(
    backbone_config=backbone_config,
    num_classes=1000,  # ImageNet classes
    resolution=224,
    num_layers=2,
    hidden_dim=512,
    dropout_rate=0.2,
)

# Create model
model = FAIClassification(config)
```

### Multi-Label Classification Setup

```python
from focoos.models.fai_cls.config import ClassificationConfig
from focoos.models.fai_cls.modelling import FAIClassification
from focoos.nn.backbone.resnet import ResnetConfig

# Configure with ResNet backbone
backbone_config = ResnetConfig(
    model_type="resnet",
    depth=50,
    pretrained=True,
)

config = ClassificationConfig(
    backbone_config=backbone_config,
    num_classes=1000,  # ImageNet classes
    resolution=224,
    num_layers=2,
    hidden_dim=512,
    dropout_rate=0.2,
    multi_label=True,
)

# Create model
model = FAIClassification(config)
```
