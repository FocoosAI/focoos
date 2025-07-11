# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from focoos.utils.box import bbox_overlaps


def bias_init_with_prob(prior_prob: Union[float, int]) -> float:
    """Initialize conv/fc bias value according to a given probability value.

    This function calculates the initial bias value for convolutional or fully
    connected layers based on a prior probability. It uses the inverse of the
    sigmoid function (logit) to compute the appropriate bias initialization.

    Args:
        prior_prob (Union[float, int]): Prior probability for the bias
            initialization. Should be a value between 0 and 1.

    Returns:
        float: Initialized bias value calculated as -log((1 - p)/p) where p
            is the prior probability.

    Example:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Initialized bias: {bias}")
        Initialized bias: -4.59511985013459
    """
    return float(-np.log((1 - prior_prob) / prior_prob))


def nms_torch(
    bboxes: Tensor, scores: Tensor, threshold: float = 0.65, iou_calculator=bbox_overlaps, return_group: bool = False
):
    """Perform Non-Maximum Suppression (NMS) on a set of bounding boxes using
    their corresponding scores.

    Args:

        bboxes (Tensor): list of bounding boxes (each containing 4 elements
            for x1, y1, x2, y2).
        scores (Tensor): scores associated with each bounding box.
        threshold (float): IoU threshold to determine overlap.
        iou_calculator (function): method to calculate IoU.
        return_group (bool): if True, returns groups of overlapping bounding
            boxes, otherwise returns the main bounding boxes.
    """

    _, indices = scores.sort(descending=True)
    groups = []
    while len(indices):
        idx, indices = indices[0], indices[1:]
        bbox = bboxes[idx]
        ious = iou_calculator(bbox, bboxes[indices])
        close_indices = torch.where(ious > threshold)[1]
        keep_indices = torch.ones_like(indices, dtype=torch.bool)
        keep_indices[close_indices] = 0
        groups.append(torch.cat((idx[None], indices[close_indices])))
        indices = indices[keep_indices]

    if return_group:
        return groups
    else:
        return torch.cat([g[:1] for g in groups])


def filter_scores_and_topk(
    scores: Tensor, score_thr: float, topk: int, results: Optional[dict or list or torch.Tensor] = None
):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f"Only supports dict or list or Tensor, but get {type(results)}.")
    return scores, labels, keep_idxs, filtered_results


def reduce_mean(tensor):
    """ "Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    scale = (bbox[..., 2:] - bbox[..., :2]) * padding
    center = (bbox[..., 2:] + bbox[..., :2]) * 0.5

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
