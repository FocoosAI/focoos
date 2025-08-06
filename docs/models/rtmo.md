# RTMO

## Overview

[RTMO](https://arxiv.org/abs/2312.07526) (Real-Time Multi-person One-stage) is a high-performance, one-stage framework for real-time multi-person pose estimation. It integrates a novel coordinate classification method into YOLOXPose architecture, achieving an excellent balance between speed and accuracy.

The model avoids the typical slowdown of two-stage (top-down) methods in crowded scenes by directly predicting keypoints for all individuals in a single pass. Its core innovation is the **Dynamic Coordinate Classifier (DCC)**, which represents keypoints using dual 1-D heatmaps, enabling precise localization without the computational overhead of high-resolution feature maps.

## Neural Network Architecture

![RTMO Architecture](./rtmo.png)
The RTMO architecture consists of three main components:

### Backbone
- **Purpose**: To extract multi-scale feature maps from the input image.
- **Design**: Configurable backbone network (e.g., ResNet, STDC), default is **CSPDarknet**.

### Neck
- **Purpose**: To fuse and refine the features from the backbone.
- **Design**: A **YOLO neck** processes the last three feature maps from the backbone (with downsampling rates of 8, 16 and 32) to generate enhanced features for the head.

### Head
- **Purpose**: To predict the final outputs for each grid cell on the feature map.
- **Components**:
  - **Prediction Layers**: Dual convolution blocks generate a classification score and a high-dimensional pose feature for each grid cell.
  - **Pose & BBox Prediction**: The pose feature is used to directly regress bounding boxes and keypoint visibility scores.
  - **Dynamic Coordinate Classifier (DCC)**: This is the core component for keypoint localization. It takes the pose feature and translates it into K pairs of 1-D heatmaps (one for the horizontal axis, one for the vertical). It consists of:
    - **Dynamic Bin Allocation**: Bins are not static across the whole image but are dynamically allocated within a region scaled to each instance's predicted bounding box. This optimizes bin utilization and accuracy.
    - **Dynamic Bin Encoding**: The coordinate of each dynamic bin is encoded using sine positional encodings to form a unique representation, allowing the model to calculate a precise probability for each keypoint at each bin location.

## Configuration Parameters

### Core Model parameters
- `num_classes` (int): Number of object classes (default=1 for "person" in COCO dataset).
- `num_keypoints` (int): Number of keypoints to predict (default=17 for person body).

### Backbone Configuration
- `backbone_config` (BackboneConfig): Backbone network configuration (default: `DarkNetConfig`)

### Neck
- `neck_feat_dim` (int, default=256): Input feature dimension for the neck.
- `neck_out_dim` (int, default=256): Output feature dimension from the neck.
- `c2f_depth` (int, default=2): Depth of the C2f blocks within the neck. <ins>TODO if YOLO neck is changed<ins>

### Head
- `in_channels` (int, default=256): Number of input channels to the head from the neck.
- `feat_channels` (int, default=256): Number of channels in the intermediate convolutional layers of the head.
- `pose_vec_channels` (int, default=256): Dimension of the output pose feature vector for each grid, which is later fed into the DCC.
- `stacked_convs` (int, default=2): Number of stacked convolution layers in the classification and regression branches of the head.
- `activation` (str, default="relu"): The activation function to use in the head's layers.
- `norm` (NormType, default="BN"): The normalization layer type to use (e.g., "BN" for BatchNorm).

<ins>TODO: maybe to remove stuff below<ins>
#### Dynamic Coordinate Classifier (DCC)
- `feat_channels_dcc` (int, default=128): The feature dimension used within the DCC for keypoint feature representation.
- `num_bins` (Tuple[int, int], default=(192, 256)): The number of bins for the horizontal (x) and vertical (y) heatmaps, respectively.
- `spe_channels` (int, default=128): The channel dimension for the Sine Positional Encoding (SPE) used to encode bin coordinates.
#### GAU
- `gau_s` (int, default=128): The dimension `s` in the Gated Attention Unit (GAU).
- `gau_expansion_factor` (int, default=2): The expansion factor for the hidden dimension in the GAU.
- `gau_dropout_rate` (float, default=0.0): Dropout rate applied within the GAU.

## Losses
1.  **Classification Loss (`loss_cls`)**: A **VariFocal Loss** is used to supervise the classification score of each grid.
2.  **Bounding Box Loss (`loss_bbox`)**: An **IoU (Intersection over Union) Loss** is applied to the decoded bounding boxes for positive grids.
3.  **MLE Loss (`loss_mle`)**: The core loss for keypoint localization. It is applied to the 1-D heatmaps from the DCC.
4.  **Keypoint Visibility Loss (`loss_vis`)**: A **Binary Cross-Entropy (BCE) Loss** is used to supervise the visibility prediction for each keypoint.
5. **OKS Loss (`loss_oks`)**: Auxiliary loss supervising keypoint regression with **OKS (Object Keypoint Similarity) Loss**. It uses DCC predictions as targets for further refinement.

The total loss is a weighted sum of these components.

## Supported Tasks

### Multi-Person Pose Estimation
- **Output**: Detects all individuals in an image, providing instance-level predictions. For each person, the model outputs a bounding box, a detection score, and the coordinates and visibility status for all keypoints.
- **Use Cases**: Action recognition, sports analytics, augmented reality, human-computer interaction.

## Model Outputs

### Internal Output (`RTMOModelOutput`)
- `outputs` (KeypointOutput):
    - `scores`: (torch.Tensor) Shape [B, num_detections] - Detection scores for each instance
    - `labels`: (torch.Tensor) Shape [B, num_detections] - Class labels for each instance
    - `pred_bboxes`: (torch.Tensor) Shape [B, num_detections, 4] - Predicted bounding boxes [x1, y1, x2, y2]
    - `bbox_scores`: (torch.Tensor) Shape [B, num_detections] - Scores for each bounding box
    - `pred_keypoints`: (torch.Tensor) Shape [B, num_detections, num_keypoints, 2] - Predicted keypoint coordinates
    - `keypoint_scores`: (torch.Tensor) Shape [B, num_detections, num_keypoints] - Scores for each keypoint
    - `keypoints_visible`: (torch.Tensor) Shape [B, num_detections, num_keypoints] - Visibility for keypoints
- `loss` (RTMOLoss): Training losses including:
    - `loss_bbox`: IoU loss for bounding box regression
    - `loss_vis`: Binary cross-entropy visibility loss
    - `loss_mle`: Maximum Likelihood Estimation (MLE) loss applied to 1-D keypoint heatmaps for keypoint localization
    - `loss_oks`: OKS (Object Keypoint Similarity) loss for better keypoint localization
    - `loss_cls`: VariFocal classification loss

### Inference Output (`FocoosDetections`)
For each detected object:

- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2]
- `conf` (float): Confidence score
- `cls_id` (int): Class identifier
- `label` (Optional[str]): Human-readable class name
- `keypoints` (List[List[int]]): Keypoints [x, y, visibility]


## Available Models

Currently, you can find 1 model on the Focoos Hub for multi-person Pose Estimation.

| Model Name | Architecture | Dataset | Metric | FPS Nvidia-T4 |
|------------|--------------|----------|---------|--------------|
| rtmo-s-coco | RTMO (STDC-2) | COCO | AP: 59.9<br>AP50: 83.3 | - |


## Example Usage

### Quick Start with Pre-trained Model

```python
from focoos.model_manager import ModelManager

# Load a pre-trained RTMO model
model = ModelManager.get("rtmo-s-coco")

# Run inference on an image
image = Image.open("path/to/image.jpg")
result = model.infer(image)

```

### Custom Model Configuration
```python
from focoos.models.rtmo.config import RTMOConfig
from focoos.models.rtmo.modelling import RTMO
from focoos.nn.backbone.stdc import STDCConfig

# Configure the backbone
backbone_config = STDCConfig(
    model_type="stdc",
    use_pretrained=True,
)

# Configure the RTMO model
config = RTMOConfig(
    backbone_config=backbone_config,
    num_classes=1,  # COCO Person classes
    num_keypoints=17 # Number of keypoints for COCO Person body pose
)

# Create the model
model = RTMO(config)
```
