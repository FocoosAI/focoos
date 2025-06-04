# FAI-CLS (FocoosAI Classification)

## Overview

FAI-CLS is a versatile image classification model developed by FocoosAI that can utilize any backbone architecture for feature extraction. This model is designed for both single-label and multi-label image classification tasks, offering flexibility in architecture choices and training configurations.

The model employs a simple yet effective approach: a configurable backbone extracts features from input images, followed by a classification head that produces class predictions. This design enables easy adaptation to different domains and datasets while maintaining high performance and computational efficiency.

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
- `resolution` (int, default=224): Input image resolution

### Image Preprocessing
- `pixel_mean` (List[float]): RGB normalization means [123.675, 116.28, 103.53]
- `pixel_std` (List[float]): RGB normalization standard deviations [58.395, 57.12, 57.375]

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
- **Single-Label**: Class probabilities after softmax activation
- **Multi-Label**: Class probabilities after sigmoid activation
- **Post-Processing**: Can be processed into `FocoosDetections` format with confidence scores

## Key Features

### Flexibility
- **Backbone Agnostic**: Compatible with any feature extraction backbone
- **Configurable Depth**: Choose between 1 or 2-layer classification heads
- **Multi-Task Ready**: Supports both single-label and multi-label scenarios
- **Resolution Adaptive**: Configurable input resolution for different use cases

### Training Features
- **Advanced Loss Functions**: Focal loss for handling class imbalance
- **Label Smoothing**: Reduces overfitting and improves generalization
- **Dropout Regularization**: Prevents overfitting in the classification head
- **Multi-Label Support**: Binary cross-entropy for multi-label scenarios

### Performance Optimizations
- **Efficient Head Design**: Lightweight classification layers
- **Global Average Pooling**: Reduces spatial dimensions efficiently
- **Proper Initialization**: Truncated normal initialization for better training

## Loss Functions

The model supports multiple loss function configurations:

### Cross-Entropy Loss (Default)
- **Use Case**: Standard single-label classification
- **Features**: Optional label smoothing for better generalization
- **Activation**: Softmax for probability distribution

### Focal Loss
- **Use Case**: Imbalanced datasets with hard-to-classify examples
- **Parameters**:
  - Alpha (α): Controls importance of rare class
  - Gamma (γ): Focuses learning on hard examples
- **Benefits**: Improved performance on imbalanced datasets

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
