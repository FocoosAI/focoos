
## Overview
The models is the reimplementation of the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) model by [FocoosAI](https://focoos.ai) for the [COCO dataset](https://cocodataset.org/#home). It is a object detection model able to detect 80 thing (dog, cat, car, etc.) classes.


## Model Details
The model is based on the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) architecture. It is a object detection model that uses a transformer-based encoder-decoder architecture.

### Neural Network Architecture
This implementation is a reimplementation of the [RT-DETR](https://github.com/lyuwenyu/RT-DETR) model by [FocoosAI](https://focoos.ai). The original model is fully described in this [paper](https://arxiv.org/abs/2304.08069).

RT-DETR is a hybrid model that uses three main components: a *backbone* for extracting features, an *encoder* for upscaling the features, and a *transformer-based decoder* for generating the detection output.

![alt text](./rt-detr.png)

In this implementation:

- the backbone is a [Resnet-50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py),that guarantees a good performance while having good efficiency.
- the encoder is the Hybrid Encoder, as proposed by the paper, and it is a bi-FPN (bilinear feature pyramid network) that includes a transformer encoder on the smaller feature resolution for improving efficiency.
- The query selection mechanism select the features of the pixels (aka queries) with the highest probability of containing an object and pass them to a transformer decoder head that will generate the final detection output. In this implementation, we select 300 queries and use 6 transformer decoder layers.

### Losses
We use the same losses as the original paper:

- loss_vfl: a variant of the binary cross entropy loss for the classification of the classes that is weighted by the correctness of the predicted bounding boxes IoU.
- loss_bbox: an L1 loss computing the distance between the predicted bounding boxes and the ground truth bounding boxes.
- loss_giou: a loss minimizing the IoU the predicted bounding boxes and the ground truth bounding boxes. For more details look at [GIoU](https://giou.stanford.edu/).

These losses are applied to each output of the transformer decoder, meaning that we apply it on the output and on each auxiliary output of the transformer decoder layers.
Please refer to the [RT-DETR paper](https://arxiv.org/abs/2304.08069) for more details.

### Output Format
The pre-processed output of the model is set of bounding boxes with associated class probabilities. In particular, the output is composed by three tensors:

- class_ids: a tensor of 300 elements containing the class id associated with each bounding box (such as 1 for wall, 2 for building, etc.)
- scores: a tensor of 300 elements containing the corresponding probability of the class_id
- boxes: a tensor of shape (300, 4) where the values represent the coordinates of the bounding boxes in the format [x1, y1, x2, y2]

The model does not need NMS (non-maximum suppression) because the output is already a set of bounding boxes with associated class probabilities and has been trained to avoid overlaps.

After the post-processing, the output is a the output is a [Focoos Detections](https://github.com/FocoosAI/focoos/blob/4a317a269cb7758ea71b255faeba654d21182083/focoos/ports.py#L179) object containing the predicted bounding boxes with confidence greather than a specific threshold (0.5 by default).
