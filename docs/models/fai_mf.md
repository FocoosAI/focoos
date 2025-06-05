# FAI-MF (FocoosAI MaskFormer)

## Overview

The FAI-MF model is a [Mask2Former](https://github.com/facebookresearch/Mask2Former) implementation optimized by [FocoosAI](https://focoos.ai) for semantic and instance segmentation tasks.
Unlike traditional segmentation models such as [DeepLab](https://arxiv.org/abs/1802.02611), Mask2Former employs a mask-classification approach where predictions consist of segmentation masks paired with class probabilities.

## Neural Network Architecture

The FAI-MF model is built on the [Mask2Former](https://arxiv.org/abs/2112.01527) architecture, featuring a transformer-based encoder-decoder design with three main components:

![Mask2Former Architecture](./mask2former.png)

### Backbone
 - **Network**: Any backbone that can extract multi-scale features from an image
 - **Output**: Multi-scale features from stages 2-5 at resolutions 1/4, 1/8, 1/16, and 1/32

### Pixel Decoder
 - **Architecture**: Feature Pyramid Network (FPN)
 - **Input**: Features from backbone stages 2-5
 - **Modifications**: Deformable attention modules removed for improved portability and inference speed
 - **Output**: Upscaled multi-scale features for mask generation

### Transformer Decoder
 - **Design**: Lightweight version of the original Mask2Former decoder
 - **Layers**: N decoder layer (depending on the speed/accuracy trade-off)
 - **Queries**: Q learnable object queries (usually 100)
 - **Components**:
   - Self-attention layers
   - Masked cross-attention layers
   - Feed-forward networks (FFN)

## Configuration Parameters

### Core Model Parameters
- `num_classes` (int): Number of segmentation classes
- `num_queries` (int, default=100): Number of learnable object queries
- `resolution` (int, default=640): Input image resolution

### Backbone Configuration
- `backbone_config` (BackboneConfig): Backbone network configuration

### Architecture Dimensions
- `pixel_decoder_out_dim` (int, default=256): Pixel decoder output channels
- `pixel_decoder_feat_dim` (int, default=256): Pixel decoder feature channels
- `transformer_predictor_hidden_dim` (int, default=256): Transformer hidden dimension
- `transformer_predictor_dec_layers` (int, default=6): Number of decoder layers
- `head_out_dim` (int, default=256): Prediction head output dimension

### Inference Configuration
- `postprocessing_type` (str): Either "semantic" or "instance" segmentation
- `mask_threshold` (float, default=0.5): Binary mask threshold
- `threshold` (float, default=0.5): Confidence threshold for the classification scores
- `predict_all_pixels` (bool, default=False): Predict class for every pixel, this is usually better for semantic segmentation

## Supported Tasks

### Semantic Segmentation
- **Output**: Dense pixel-wise class predictions
- **Use case**: Scene understanding, medical imaging, autonomous driving
- **Configuration**: Set `postprocessing_type="semantic"`

### Instance Segmentation
- **Output**: Individual object instances with masks and bounding boxes
- **Use case**: Object detection and counting, robotics, surveillance
- **Configuration**: Set `postprocessing_type="instance"`

## Model Outputs

### Inner Model Output (`MaskFormerModelOutput`)
- `masks` (torch.Tensor): Shape [B, num_queries, H, W] - Query mask predictions
- `logits` (torch.Tensor): Shape [B, num_queries, num_classes] - Class predictions
- `loss` (Optional[dict]): Training losses including:
    - `loss_ce`: Cross-entropy classification loss
    - `loss_mask`: Binary cross-entropy mask loss
    - `loss_dice`: Dice coefficient loss

### Inference Output (`FocoosDetections`)
For each detected object:

- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2]
- `conf` (float): Confidence score
- `cls_id` (int): Class identifier
- `mask` (str): Base64-encoded binary mask
- `label` (Optional[str]): Human-readable class name

## Losses

The model employs three complementary loss functions as described in the [original paper](https://arxiv.org/abs/2112.01527):

1. **Cross-entropy Loss (`loss_ce`)**: Classification of object classes
2. **Dice Loss (`loss_dice`)**: Shape-aware segmentation loss
3. **Mask Loss (`loss_mask`)**: Binary cross-entropy on predicted masks

## Available Models
Currently, you can find 5 fai-mf models on the Focoos Hub, 2 for semantic segmentation and 3 for instance-segmentation.

### Semantic Segmentation Models

| Model Name | Architecture | Dataset | Metric | FPS Nvidia-T4 |
|------------|--------------|----------|---------|--------------|
| fai-mf-l-ade | Mask2Former (Resnet-101) | ADE20K | mIoU: 48.27<br>mAcc: 62.15 | 73 |
| fai-mf-m-ade | Mask2Former (STDC-2) | ADE20K | mIoU: 45.32<br>mACC: 57.75 | 127 |

### Instance Segmentation Models

| Model Name | Architecture | Dataset | Metric | FPS Nvidia-T4 |
|------------|--------------|----------|---------|--------------|
| fai-mf-s-coco-ins | Mask2Former (Resnet-50) | COCO | segm/AP: 41.45<br>segm/AP50: 64.12 | 86 |
| fai-mf-m-coco-ins | Mask2Former (Resnet-101) | COCO | segm/AP: 43.09<br>segm/AP50: 65.87 | 70 |
| fai-mf-l-coco-ins | Mask2Former (Resnet-101) | COCO | segm/AP: 44.23<br>segm/AP50: 67.53 | 55 |

## Example Usage

### Quick Start with Pre-trained Model

```python
from focoos.model_manager import ModelManager

# Load a pre-trained BisenetFormer model
model = ModelManager.get("fai-mf-l-ade")

# Run inference on an image
image = Image.open("path/to/image.jpg")
result = model(image)

# Process results
for detection in result.detections:
    print(f"Class: {detection.label}, Confidence: {detection.conf:.3f}")
```

### Custom Model Configuration

```python
from focoos.models.fai_mf.config import MaskFormerConfig
from focoos.models.fai_mf.modelling import FAIMaskFormer
from focoos.nn.backbone.stdc import STDCConfig

# Configure the backbone
backbone_config = STDCConfig(
    model_type="stdc",
    use_pretrained=True,
)

# Configure the model
config = MaskFormerConfig(
    backbone_config=backbone_config,
    num_classes=80,
    num_queries=300,
    transformer_predictor_dec_layers=3,
    threshold=0.5,
)

# Create the model
model = FAIMaskFormer(config)
```
