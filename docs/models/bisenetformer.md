# BisenetFormer (Bilateral Segmentation Network with Transformer)

## Overview

BisenetFormer is an advanced semantic segmentation model that combines the efficiency of BiSeNet (Bilateral Segmentation Network) with the power of transformer architectures. Developed by FocoosAI, this model is designed for real-time semantic segmentation tasks requiring both high accuracy and computational efficiency.

The model employs a dual-path architecture where spatial details are preserved through one path while semantic information is processed through another, then fused with transformer-based attention mechanisms for superior segmentation performance.

## Neural Network Architecture

The BisenetFormer architecture consists of four main components working in concert:

### Backbone
- **Purpose**: Feature extraction from input images
- **Design**: Configurable backbone network (e.g., ResNet, STDC)
- **Output**: Multi-scale features at different resolutions (1/4, 1/8, 1/16, 1/32)

### Context Path
- **Component**: Global context extraction path
- **Features**:
  - Attention Refinement Module (ARM) for feature enhancement
  - Global Average Pooling for context aggregation
  - Multi-scale feature fusion with upsampling
- **Purpose**: Captures high-level semantic information

### Spatial Path (Detail Branch)
- **Component**: Spatial detail preservation path
- **Features**:
  - Bilateral structure maintaining spatial resolution
  - ConvBNReLU blocks for efficient processing
  - Feature Fusion Module (FFM) for combining paths
- **Purpose**: Preserves fine-grained spatial details

### Transformer Decoder
- **Design**: Lightweight transformer decoder with attention mechanisms
- **Components**:
  - Self-attention layers for feature refinement
  - Cross-attention layers for multi-scale feature integration
  - Feed-forward networks (FFN) for feature transformation
  - 100 learnable object queries
- **Layers**: Configurable number of decoder layers (default: 6)

## Configuration Parameters

### Core Model Parameters
- `num_classes` (int): Number of segmentation classes
- `num_queries` (int, default=100): Number of learnable object queries
- `backbone_config` (BackboneConfig): Backbone network configuration

### Image Preprocessing
- `pixel_mean` (List[float]): RGB normalization means [123.675, 116.28, 103.53]
- `pixel_std` (List[float]): RGB normalization standard deviations [58.395, 57.12, 57.375]
- `size_divisibility` (int, default=0): Input size divisibility constraint

### Architecture Dimensions
- `pixel_decoder_out_dim` (int, default=256): Pixel decoder output channels
- `pixel_decoder_feat_dim` (int, default=256): Pixel decoder feature channels
- `transformer_predictor_hidden_dim` (int, default=256): Transformer hidden dimension
- `transformer_predictor_dec_layers` (int, default=6): Number of decoder layers
- `transformer_predictor_dim_feedforward` (int, default=1024): FFN dimension
- `head_out_dim` (int, default=256): Prediction head output dimension

### Inference Configuration
- `postprocessing_type` (str): Either "semantic" or "instance" segmentation
- `mask_threshold` (float, default=0.5): Binary mask threshold
- `threshold` (float, default=0.5): Confidence threshold for detections
- `top_k` (int, default=300): Maximum number of detections to return
- `use_mask_score` (bool, default=False): Whether to use mask quality scores
- `predict_all_pixels` (bool, default=False): Predict class for every pixel
- `cls_sigmoid` (bool, default=False): Use sigmoid activation for classification

### Loss Configuration
- `criterion_deep_supervision` (bool, default=True): Enable deep supervision
- `criterion_eos_coef` (float, default=0.1): End-of-sequence coefficient
- `criterion_num_points` (int, default=12544): Number of sampling points
- `weight_dict_loss_ce` (int, default=2): Cross-entropy loss weight
- `weight_dict_loss_mask` (int, default=5): Mask loss weight
- `weight_dict_loss_dice` (int, default=5): Dice loss weight

### Hungarian Matcher Configuration
- `matcher_cost_class` (int, default=2): Classification cost for matching
- `matcher_cost_mask` (int, default=5): Mask cost for matching
- `matcher_cost_dice` (int, default=5): Dice cost for matching

## Supported Tasks

### Semantic Segmentation
- **Output**: Dense pixel-wise class predictions
- **Use Cases**: Scene understanding, autonomous driving, medical imaging
- **Configuration**: Set `postprocessing_type="semantic"`

### Instance Segmentation
- **Output**: Individual object instances with masks and bounding boxes
- **Use Cases**: Object detection and counting, robotics applications
- **Configuration**: Set `postprocessing_type="instance"`

## Model Outputs

### Training Output (`BisenetFormerOutput`)
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

## Key Features

### Efficiency Optimizations
- **Bilateral Architecture**: Separate paths for spatial details and semantic context
- **Attention Refinement**: ARM modules enhance feature quality without computational overhead
- **Lightweight Transformer**: Reduced decoder complexity for faster inference

### Performance Advantages
- **Real-time Capable**: Optimized for efficient inference
- **Multi-scale Processing**: Leverages features at multiple resolutions
- **Context Preservation**: Global context path maintains semantic understanding
- **Detail Retention**: Spatial path preserves fine-grained details

### Training Features
- **Deep Supervision**: Auxiliary losses at multiple decoder layers
- **Hungarian Matching**: Optimal assignment between predictions and ground truth
- **Flexible Loss Functions**: Combines classification, mask, and shape-aware losses

## Architecture Innovations

The BisenetFormer introduces several key innovations:

1. **Hybrid Architecture**: Combines the efficiency of bilateral networks with transformer attention
2. **Feature Fusion Module**: Intelligent fusion of spatial and context paths
3. **Attention Refinement**: ARM modules refine features at multiple scales
4. **Query-based Segmentation**: Transformer queries enable instance-aware segmentation

This architecture achieves an optimal balance between accuracy and efficiency, making it suitable for both research and production deployments requiring real-time semantic segmentation capabilities.

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
| fai-m2f-s-coco-ins | Mask2Former (Resnet-50) | COCO | segm/AP: 41.45<br>segm/AP50: 64.12 | 86 |
| fai-m2f-m-coco-ins | Mask2Former (Resnet-101) | COCO | segm/AP: 43.09<br>segm/AP50: 65.87 | 70 |
| fai-m2f-l-coco-ins | Mask2Former (Resnet-101) | COCO | segm/AP: 44.23<br>segm/AP50: 67.53 | 55 |
