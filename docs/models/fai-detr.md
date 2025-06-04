# FAI-DETR (FocoosAI Detection Transformer)

## Overview

FAI-DETR is an advanced object detection model based on the DETR (Detection Transformer) architecture, optimized by FocoosAI for efficient and accurate object detection tasks. This model eliminates the need for hand-crafted components like non-maximum suppression (NMS) and anchor generation by using a transformer-based approach with learnable object queries.

The model employs a set-based global loss through bipartite matching and a transformer encoder-decoder architecture that directly predicts bounding boxes and class labels. This end-to-end approach simplifies the detection pipeline while achieving competitive performance.

## Neural Network Architecture

The FAI-DETR architecture consists of four main components:

### Backbone
- **Purpose**: Feature extraction from input images
- **Design**: Configurable backbone network (ResNet, STDC, etc.)
- **Output**: Multi-scale features at different resolutions
- **Integration**: Features are processed through the encoder for global context

### Encoder
- **Architecture**: Transformer encoder with multi-scale deformable attention
- **Components**:
  - Multi-scale deformable self-attention layers
  - Position embeddings (sine-based)
  - Feed-forward networks (FFN)
  - CSPRep layers for efficient feature processing
- **Features**: Processes multi-scale features to capture global context
- **Layers**: Configurable number of encoder layers (default: 1)

### Decoder
- **Architecture**: Multi-scale deformable transformer decoder
- **Components**:
  - Self-attention layers for query interaction
  - Cross-attention layers with deformable attention
  - Feed-forward networks
  - Reference point refinement
- **Queries**: 300 learnable object queries (configurable)
- **Layers**: Configurable number of decoder layers (default: 6)

### Detection Head
- **Classification Head**: Predicts class probabilities for each query
- **Regression Head**: Predicts bounding box coordinates (center, width, height)
- **Output Format**: Direct box predictions without anchors or post-processing

## Configuration Parameters

### Core Model Parameters
- `num_classes` (int): Number of object detection classes
- `num_queries` (int, default=300): Number of learnable object queries
- `resolution` (int, default=640): Input image resolution
- `backbone_config` (BackboneConfig): Backbone network configuration

### Image Preprocessing
- `pixel_mean` (List[float]): RGB normalization means [123.675, 116.28, 103.53]
- `pixel_std` (List[float]): RGB normalization standard deviations [58.395, 57.12, 57.375]
- `size_divisibility` (int, default=0): Input size divisibility constraint

### Encoder Configuration
- `pixel_decoder_out_dim` (int, default=256): Encoder output dimension
- `pixel_decoder_feat_dim` (int, default=256): Encoder feature dimension
- `pixel_decoder_num_encoder_layers` (int, default=1): Number of encoder layers
- `pixel_decoder_expansion` (float, default=1.0): Channel expansion ratio
- `pixel_decoder_dim_feedforward` (int, default=1024): FFN dimension
- `pixel_decoder_dropout` (float, default=0.0): Dropout rate
- `pixel_decoder_nhead` (int, default=8): Number of attention heads

### Decoder Configuration
- `transformer_predictor_hidden_dim` (int, default=256): Decoder hidden dimension
- `transformer_predictor_dec_layers` (int, default=6): Number of decoder layers
- `transformer_predictor_dim_feedforward` (int, default=1024): FFN dimension
- `transformer_predictor_nhead` (int, default=8): Number of attention heads
- `transformer_predictor_out_dim` (int, default=256): Decoder output dimension
- `head_out_dim` (int, default=256): Detection head output dimension

### Inference Configuration
- `threshold` (float, default=0.5): Confidence threshold for detections
- `top_k` (int, default=300): Maximum number of detections to return

### Loss Configuration
- `criterion_deep_supervision` (bool, default=True): Enable deep supervision
- `criterion_eos_coef` (float, default=0.1): End-of-sequence coefficient
- `criterion_losses` (List[str]): Loss types ["vfl", "boxes"]
- `criterion_focal_alpha` (float, default=0.75): Focal loss alpha parameter
- `criterion_focal_gamma` (float, default=2.0): Focal loss gamma parameter
- `weight_dict_loss_vfl` (int, default=1): Varifocal loss weight
- `weight_dict_loss_bbox` (int, default=5): Bounding box loss weight
- `weight_dict_loss_giou` (int, default=2): GIoU loss weight

### Hungarian Matcher Configuration
- `matcher_cost_class` (int, default=2): Classification cost for matching
- `matcher_cost_bbox` (int, default=5): Bounding box cost for matching
- `matcher_cost_giou` (int, default=2): GIoU cost for matching
- `matcher_use_focal_loss` (bool, default=True): Use focal loss in matcher
- `matcher_alpha` (float, default=0.25): Matcher focal loss alpha
- `matcher_gamma` (float, default=2.0): Matcher focal loss gamma

## Supported Tasks

### Object Detection
- **Output**: Bounding boxes with class labels and confidence scores
- **Use Cases**:
  - General object detection in natural images
  - Autonomous driving (vehicle, pedestrian detection)
  - Surveillance and security applications
  - Industrial quality control
  - Medical image analysis
- **Performance**: End-to-end detection without NMS post-processing

## Model Outputs

### Training Output (`DETRModelOutput`)
- `boxes` (torch.Tensor): Shape [B, num_queries, 4] - Bounding boxes in XYXY format normalized to [0, 1]
- `logits` (torch.Tensor): Shape [B, num_queries, num_classes] - Class predictions
- `loss` (Optional[dict]): Training losses including:
  - `loss_vfl`: Varifocal loss for classification
  - `loss_bbox`: L1 loss for bounding box regression
  - `loss_giou`: Generalized IoU loss for box alignment

### Inference Output (`FocoosDetections`)
For each detected object:
- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2]
- `conf` (float): Confidence score
- `cls_id` (int): Class identifier
- `label` (Optional[str]): Human-readable class name

## Key Features

### Architecture Advantages
- **End-to-End Training**: Direct optimization of detection metrics
- **No Hand-Crafted Components**: Eliminates anchors, NMS, and heuristic post-processing
- **Set-Based Global Loss**: Bipartite matching enables global optimization
- **Parallel Prediction**: All objects predicted simultaneously

### Performance Optimizations
- **Deformable Attention**: Efficient multi-scale feature processing
- **CSP Layers**: Channel Split Pooling for computational efficiency
- **RepVGG Blocks**: Efficient convolution blocks for feature extraction
- **Focal Loss Variants**: Improved handling of class imbalance

### Training Features
- **Hungarian Matching**: Optimal assignment between predictions and ground truth
- **Deep Supervision**: Auxiliary losses at each decoder layer
- **Multi-Scale Training**: Robust to different object sizes
- **Flexible Loss Combination**: Balances classification and localization objectives

## Loss Functions

The model employs three main loss components:

1. **Varifocal Loss (`loss_vfl`)**:
   - Advanced focal loss variant for classification
   - Handles foreground-background imbalance
   - Joint optimization of classification and localization quality

2. **Bounding Box Loss (`loss_bbox`)**:
   - L1 loss for direct coordinate regression
   - Normalized coordinates for scale invariance

3. **Generalized IoU Loss (`loss_giou`)**:
   - Shape-aware bounding box loss
   - Better gradient flow for overlapping boxes
   - Improved localization accuracy

## Architecture Innovations

The FAI-DETR introduces several key innovations:

1. **Efficient Encoder Design**: Lightweight encoder with deformable attention
2. **Multi-Scale Processing**: Handles objects at different scales effectively
3. **Reference Point Refinement**: Iterative improvement of object localization
4. **CSP Integration**: Efficient feature processing with Channel Split Pooling
5. **Optimized Matching**: Hungarian algorithm for optimal query-target assignment

This architecture achieves an optimal balance between accuracy and efficiency, making it suitable for both research applications and production deployments requiring fast and accurate object detection capabilities.
