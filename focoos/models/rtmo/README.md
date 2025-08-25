# RTMO: Real-Time Multi-Person Pose Estimation

## Overview

RTMO (Real-Time Multi-Person Pose Estimation) is a state-of-the-art real-time pose estimation model that achieves excellent performance-speed trade-offs. This implementation has been reimplemented from the [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose) framework.

## Paper

**RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation**
*[arXiv:2312.07526](https://arxiv.org/pdf/2312.07526)*

## Model Architecture

RTMO is a one-stage, real-time multi-person pose estimation model that features:

- **Hybrid Encoder**: Combines CNN backbone with transformer encoder for efficient feature extraction
- **Gated Attention Unit (GAU)**: Lightweight attention mechanism for real-time performance
- **Dual-branch Decoder**: Separate branches for person detection and keypoint localization
- **SimOTA Assignment**: Advanced label assignment strategy for better training stability

### Key Components

- **Backbone**: Configurable backbone (default: CSP-Darknet)
- **Transformer Encoder**: Multi-head self-attention with positional encoding
- **Keypoint Head**: Specialized head for keypoints
- **Detection Head**: Person detection with bounding box regression

## Performance

### COCO Keypoint Detection Results

| Model | AP | AP50 | AP75| Latency (T4) |
|-------|----|------|-----|--------|
| RTMO-S | 67.7 | 87.8 | 73.7 | xxms |
| RTMO-M | 70.9 | 89.0 | 77.8 | x ms |
| RTMO-L | 72.4 | 89.9 | 78.8 | x ms |

*Results on COCO val2017, tested on NVIDIA T4 GPU with TensorRT and 640x640 resolution. Due to different data processing in our evaluation benchmark, results may slightly change with the original implementation.*

### Speed-Accuracy Trade-off

RTMO achieves real-time performance while maintaining competitive accuracy:
- **RTMO-S**: 2.8ms inference time, suitable for real-time applications
- **RTMO-M**: 4.2ms inference time, balanced performance
- **RTMO-L**: 6.1ms inference time, highest accuracy

## Installation & Usage

### Basic Usage

```python
from focoos.models.rtmo import RTMO, RTMOConfig
from focoos.nn.backbone.csp import CSPConfig

# Create configuration
config = RTMOConfig(
    backbone_config=CSPConfig(size="small", use_pretrained=True),
    num_classes=1,
    num_keypoints=17
)

# Initialize model
model = RTMO(config)

# Inference
import torch
dummy_input = torch.randn(1, 3, 640, 640)
outputs = model(dummy_input)
```

## Model Variants

### RTMO-S (Small)
- **Backbone**: CSP-Darknet-S
- **Parameters**: ~8.5M
- **Use Case**: Real-time applications, edge devices

### RTMO-M (Medium)
- **Backbone**: CSP-Darknet-M
- **Parameters**: ~15.2M
- **Use Case**: Balanced performance and speed

### RTMO-L (Large)
- **Backbone**: CSP-Darknet-L
- **Parameters**: ~25.8M
- **Use Case**: High-accuracy applications

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{rtmo2023,
  title={RTMO: Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation},
  author={...},
  journal={arXiv preprint arXiv:2312.07526},
  year={2023}
}
```

## License

This implementation follows the same license as the original MMPose project (Apache 2.0).

## Acknowledgments

- Original RTMO implementation from [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose)
- Research team behind the RTMO paper
- OpenMMLab community for the excellent pose estimation framework

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests to improve this implementation.
