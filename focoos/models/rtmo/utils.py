# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from focoos.utils.env import TORCH_VERSION


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


def filter_scores_and_topk(
    scores: Tensor, score_thr: float, topk: int, results: Optional[Union[dict, list, torch.Tensor]] = None
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


def bbox_xyxy2cs(bbox: Tensor, padding: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - Tensor[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - Tensor[float32]: Scale (w, h) of the bbox in shape (2,) or
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

    return torch.Tensor(center), torch.Tensor(scale)


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


def flatten_predictions(preds: List[Tensor]) -> Optional[Tensor]:
    """Flattens the predictions from a list of tensors to a single
    tensor."""
    if len(preds) == 0:
        return None

    preds = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in preds]
    return torch.cat(preds, dim=1)


def decode_bbox(pred_bboxes: torch.Tensor, priors: torch.Tensor, stride: Union[torch.Tensor, int]) -> torch.Tensor:
    """Decode regression results (delta_x, delta_y, log_w, log_h) to
    bounding boxes (tl_x, tl_y, br_x, br_y).

    Note:
        - batch size: B
        - token number: N

    Args:
        pred_bboxes (torch.Tensor): Encoded boxes with shape (B, N, 4),
            representing (delta_x, delta_y, log_w, log_h) for each box.
        priors (torch.Tensor): Anchors coordinates, with shape (N, 2).
        stride (torch.Tensor | int): Strides of the bboxes. It can be a
            single value if the same stride applies to all boxes, or it
            can be a tensor of shape (N, ) if different strides are used
            for each box.

    Returns:
        torch.Tensor: Decoded bounding boxes with shape (N, 4),
            representing (tl_x, tl_y, br_x, br_y) for each box.
    """
    if isinstance(stride, int):
        stride = torch.tensor([stride], device=pred_bboxes.device, dtype=pred_bboxes.dtype)

    stride = stride.view(1, stride.size(0), 1)
    priors = priors.view(1, priors.size(0), 2)

    xys = (pred_bboxes[..., :2] * stride) + priors  # xys.shape = [batch_size, 8400, 2], cxcy ABS
    whs = pred_bboxes[..., 2:].exp() * stride

    # Calculate bounding box corners
    tl_x = xys[..., 0] - whs[..., 0] / 2
    tl_y = xys[..., 1] - whs[..., 1] / 2
    br_x = xys[..., 0] + whs[..., 0] / 2
    br_y = xys[..., 1] + whs[..., 1] / 2

    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes


def decode_kpt_reg(
    pred_kpt_offsets: torch.Tensor, priors: torch.Tensor, stride: torch.Tensor, num_keypoints: int
) -> torch.Tensor:
    """Decode regression results (delta_x, delta_y) to keypoints
    coordinates (x, y).

    Args:
        pred_kpt_offsets (torch.Tensor): Encoded keypoints offsets with
            shape (batch_size, num_anchors, num_keypoints, 2).
        priors (torch.Tensor): Anchors coordinates with shape
            (num_anchors, 2).
        stride (torch.Tensor): Strides of the anchors.

    Returns:
        torch.Tensor: Decoded keypoints coordinates with shape
            (batch_size, num_boxes, num_keypoints, 2).
    """
    stride = stride.view(1, stride.size(0), 1, 1)
    priors = priors.view(1, priors.size(0), 1, 2)
    pred_kpt_offsets = pred_kpt_offsets.reshape(
        *pred_kpt_offsets.shape[:-1], num_keypoints, 2
    )  # pred_kpt_offsets.shape = [batch_size, 8400, 17, 2]

    decoded_kpts = pred_kpt_offsets * stride + priors
    return decoded_kpts


class ScaleNorm(nn.Module):
    """Scale Norm.

    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.

    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """

        if torch.onnx.is_in_onnx_export() and TORCH_VERSION >= (1, 12):
            norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
            norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class ChannelWiseScale(nn.Module):
    """Scale vector by element multiplications.

    Args:
        dim (int): The dimension of the scale vector.
        init_value (float, optional): The initial value of the scale vector.
            Defaults to 1.0.
        trainable (bool, optional): Whether the scale vector is trainable.
            Defaults to True.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """Forward function."""

        return x * self.scale
