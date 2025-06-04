
## Overview
The models is a [Mask2Former](https://github.com/facebookresearch/Mask2Former) model otimized by [FocoosAI](https://focoos.ai) for the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/). It is a semantic segmentation model able to segment 150 classes, comprising both stuff (sky, road, etc.) and thing (dog, cat, car, etc.).

## Model Details
The model is based on the [Mask2Former](https://github.com/facebookresearch/Mask2Former) architecture. It is a segmentation model that uses a transformer-based encoder-decoder architecture.
Differently from traditional segmentation models (such as [DeepLab](https://arxiv.org/abs/1802.02611)), Mask2Former uses a mask-classification approach, where the prediction is made by a set of segmentation mask with associated class probabilities.

### Neural Network Architecture
The [Mask2Former](https://arxiv.org/abs/2112.01527) FocoosAI implementation optimize the original neural network architecture for improving the model's efficiency and performance. The original model is fully described in this [paper](https://arxiv.org/abs/2112.01527).

Mask2Former is a hybrid model that uses three main components: a *backbone* for extracting features, a *pixel decoder* for upscaling the features, and a *transformer-based decoder* for generating the segmentation output.

![alt text](./mask2former.png)

In this implementation:

 - the backbone is [STDC-1](https://github.com/MichaelFan01/STDC-Seg) that shows a trade-off tending to be more efficient.
 - the pixel decoder is a [FPN](https://arxiv.org/abs/1612.03144) getting the features from the stage 2 (1/4 resolution), 3 (1/8 resolution), 4 (1/16 resolution) and 5 (1/32 resolution) of the backbone. Differently from the original paper, for the sake of portability, we removed the deformable attention modules in the pixel decoder, speeding up the inference while only marginally affecting the accuracy.
 - the transformer decoder is a extremely light version of the original, having only 1 decoder layer (instead of 9) and 100 learnable queries.

### Losses
We use the same losses as the original paper:

- loss_ce: Cross-entropy loss for the classification of the classes
- loss_dice: Dice loss for the segmentation of the classes
- loss_mask: A binary cross-entropy loss applied to the predicted segmentation masks

Please refer to the [Mask2Former paper](https://arxiv.org/abs/2112.01527) for more details.

### Output Format
The pre-processed output of the model is set of masks with associated class probabilities. In particular, the output is composed by three tensors:

- class_ids: a tensor of 100 elements containing the class id associated with each mask (such as 1 for wall, 2 for building, etc.)
- scores: a tensor of 100 elements containing the corresponding probability of the class_id
- masks: a tensor of shape (100, H, W) where H and W are the height and width of the input image and the values represent the index of the class_id associated with the pixel

The model does not need NMS (non-maximum suppression) because the output is already a set of masks with associated class probabilities and has been trained to avoid overlapping masks.

After the post-processing, the output is a [Focoos Detections](https://github.com/FocoosAI/focoos/blob/4a317a269cb7758ea71b255faeba654d21182083/focoos/ports.py#L179) object containing the predicted masks with confidence greather than a specific threshold (0.5 by default).
