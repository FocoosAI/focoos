# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autocast
from torch.nn.modules.utils import _pair
from torchvision.ops import nms

from focoos.models.focoos_model import BaseModelNN
from focoos.models.yoloxpose.config import YOLOXPoseConfig
from focoos.models.yoloxpose.loss import KeypointCriterion
from focoos.models.yoloxpose.ports import KeypointOutput, KeypointTargets, YOLOXPoseModelOutput
from focoos.models.yoloxpose.utils import bias_init_with_prob, filter_scores_and_topk, reduce_mean
from focoos.nn.backbone.base import ShapeSpec
from focoos.nn.backbone.build import load_backbone
from focoos.nn.layers.base import get_activation_fn
from focoos.nn.layers.block import SPPF, C2f
from focoos.nn.layers.conv import Conv2d, ConvNormLayer
from focoos.nn.layers.norm import get_norm
from focoos.utils.box import bbox_overlaps

INF = 100000.0
EPS = 1.0e-7


class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
        centralize_points (bool): Whether to centralize the points to
            the center of anchors. Defaults to False.
    """

    def __init__(
        self, strides: Union[List[int], List[Tuple[int, int]]], offset: float = 0.5, centralize_points: bool = False
    ) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.centralize_points = centralize_points
        self.offset = offset if not centralize_points else 0.0

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self) -> List[int]:
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x: Tensor, y: Tensor, row_major: bool = True) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(
        self,
        featmap_sizes: List[Tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
        with_stride: bool = False,
    ) -> List[Tensor]:
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device where the anchors will be
                put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device, with_stride=with_stride
            )
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(
        self,
        featmap_size: Tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
        with_stride: bool = False,
    ) -> Tensor:
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Defaults to torch.float32.
            device (str | torch.device): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        shift_y = shift_y.to(dtype)

        if self.centralize_points:
            shift_x = shift_x + float(stride_w - 1) / 2.0
            shift_y = shift_y + float(stride_h - 1) / 2.0

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0],), stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(
        self,
        featmap_sizes: List[Tuple[int, int]],
        pad_shape: Tuple[int, int],
        device: torch.device = torch.device("cuda"),
    ) -> List[Tensor]:
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                arrange as (h, w).
            device (str | torch.device): The device where the anchors will be
                put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w), device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(
        self, featmap_size: Tuple[int, int], valid_size: Tuple[int, int], device: torch.device = torch.device("cuda")
    ) -> Tensor:
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str | torch.device): The device where the flags will be
            put on. Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(
        self,
        prior_idxs: Tensor,
        featmap_size: Tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ) -> Tensor:
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (str | torch.device): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height + self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris


def cast_tensor_type(x, scale=1.0, dtype=None):
    if dtype == "fp16":
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


class BBoxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, bboxes1, bboxes2, mode: Literal["iou", "iof", "giou"] = "iou", is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                score> format, or be empty. If ``is_aligned `` is ``True``,
                then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == "fp16":
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f"(scale={self.scale}, dtype={self.dtype})"
        return repr_str


class PoseOKS:
    """OKS score Calculator."""

    def __init__(self, metainfo: Optional[str] = "configs/_base_/datasets/coco.py"):
        return
        # if metainfo is not None:
        #     metainfo = parse_pose_metainfo(dict(from_file=metainfo))
        #     sigmas = metainfo.get('sigmas', None)
        #     if sigmas is not None:
        #         self.sigmas = torch.as_tensor(sigmas)

    @torch.no_grad()
    def __call__(
        self, output: Tensor, target: Tensor, target_weights: Tensor, areas: Tensor, eps: float = 1e-8
    ) -> Tensor:
        dist = torch.norm(output - target, dim=-1)
        areas = areas.reshape(*((1,) * (dist.ndim - 2)), -1, 1)
        dist = dist / areas.pow(0.5).clip(min=eps)

        if hasattr(self, "sigmas"):
            if self.sigmas.device != dist.device:
                self.sigmas = self.sigmas.to(dist.device)
            sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)
            dist = dist / (sigmas * 2)

        target_weights = target_weights / torch.sum(target_weights, dim=-1, keepdim=True).clip(min=eps)
        oks = (torch.exp(-dist.pow(2) / 2) * target_weights).sum(dim=-1)
        return oks


class SimOTAAssigner:
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (float): Radius of center area to determine
            if a prior is in the center of a gt. Defaults to 2.5.
        candidate_topk (int): Top-k ious candidates to calculate dynamic-k.
            Defaults to 10.
        iou_weight (float): Weight of bbox iou cost. Defaults to 3.0.
        cls_weight (float): Weight of classification cost. Defaults to 1.0.
        oks_weight (float): Weight of keypoint OKS cost. Defaults to 3.0.
        vis_weight (float): Weight of keypoint visibility cost. Defaults to 0.0
        dynamic_k_indicator (str): Cost type for calculating dynamic-k,
            either 'iou' or 'oks'. Defaults to 'iou'.
        use_keypoints_for_center (bool): Whether to use keypoints to determine
            if a prior is in the center of a gt. Defaults to False.
        iou_calculator (dict): Config of IoU calculation method.
            Defaults to dict(type='BBoxOverlaps2D').
        oks_calculator (dict): Config of OKS calculation method.
            Defaults to dict(type='PoseOKS').
    """

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
        iou_weight: float = 3.0,
        cls_weight: float = 1.0,
        oks_weight: float = 3.0,
        vis_weight: float = 0.0,
        dynamic_k_indicator: str = "iou",
        use_keypoints_for_center: bool = False,
        #  iou_calculator: ConfigType = dict(type='BBoxOverlaps2D'),
        oks_calculator: PoseOKS = PoseOKS(),
    ):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.oks_weight = oks_weight
        self.vis_weight = vis_weight
        assert dynamic_k_indicator in ("iou", "oks"), (
            f"the argument `dynamic_k_indicator` should be either 'iou' or 'oks', but got {dynamic_k_indicator}"
        )
        self.dynamic_k_indicator = dynamic_k_indicator

        self.use_keypoints_for_center = use_keypoints_for_center
        # self.iou_calculator = TASK_UTILS.build(iou_calculator)
        # self.oks_calculator = TASK_UTILS.build(oks_calculator)
        self.iou_calculator = BBoxOverlaps2D()
        self.oks_calculator = oks_calculator

    def assign(self, pred_instances: KeypointTargets, gt_instances: KeypointTargets, **kwargs) -> dict:
        """Assign gt to priors using SimOTA.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
        Returns:
            dict: Assignment result containing assigned gt indices,
                max iou overlaps, assigned labels, etc.
        """
        assert gt_instances is not None, "gt_instances is None"
        assert pred_instances is not None, "pred_instances is None"
        assert gt_instances.bboxes is not None, "gt_instances.bboxes is None"
        assert gt_instances.labels is not None, "gt_instances.labels is None"
        assert gt_instances.keypoints is not None, "gt_instances.keypoints is None"
        assert gt_instances.keypoints_visible is not None, "gt_instances.keypoints_visible is None"
        assert gt_instances.areas is not None, "gt_instances.areas is None"
        assert pred_instances.bboxes is not None, "pred_instances.bboxes is None"
        assert pred_instances.scores is not None, "pred_instances.scores is None"
        assert pred_instances.priors is not None, "pred_instances.priors is None"
        assert pred_instances.keypoints is not None, "pred_instances.keypoints is None"
        assert pred_instances.keypoints_visible is not None, "pred_instances.keypoints_visible is None"

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_areas = gt_instances.areas
        num_gt = gt_bboxes.size(0) if gt_bboxes is not None else 0

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        keypoints = pred_instances.keypoints
        keypoints_visible = pred_instances.keypoints_visible
        num_bboxes = decoded_bboxes.size(0) if decoded_bboxes is not None else 0

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes, gt_keypoints, gt_keypoints_visible
        )
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_pred_kpts = keypoints[valid_mask]
        valid_pred_kpts_vis = keypoints_visible[valid_mask]

        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            # No valid bboxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

        cost_matrix = (~is_in_boxes_and_center) * INF

        # calculate iou
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        if self.iou_weight > 0:
            iou_cost = -torch.log(pairwise_ious + EPS)
            cost_matrix = cost_matrix + iou_cost * self.iou_weight

        # calculate oks
        if self.oks_weight > 0 or self.dynamic_k_indicator == "oks":
            pairwise_oks = self.oks_calculator(
                valid_pred_kpts.unsqueeze(1),  # [num_valid, 1, k, 2]
                target=gt_keypoints.unsqueeze(0),  # [1, num_gt, k, 2]
                target_weights=gt_keypoints_visible.unsqueeze(0),  # [1, num_gt, k]
                areas=gt_areas.unsqueeze(0),  # [1, num_gt]
            )  # -> [num_valid, num_gt]

            oks_cost = -torch.log(pairwise_oks + EPS)
            cost_matrix = cost_matrix + oks_cost * self.oks_weight

        # calculate cls
        if self.cls_weight > 0:
            gt_onehot_label = (
                F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(num_valid, 1, 1)
            )
            valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
            # disable AMP autocast to avoid overflow
            with autocast(device_type="cuda", enabled=False):
                cls_cost = (
                    F.binary_cross_entropy(
                        valid_pred_scores.to(dtype=torch.float32),
                        gt_onehot_label,
                        reduction="none",
                    )
                    .sum(-1)
                    .to(dtype=valid_pred_scores.dtype)
                )
            cost_matrix = cost_matrix + cls_cost * self.cls_weight
        # calculate vis
        if self.vis_weight > 0:
            valid_pred_kpts_vis = valid_pred_kpts_vis.unsqueeze(1).repeat(1, num_gt, 1)  # [num_valid, 1, k]
            gt_kpt_vis = gt_keypoints_visible.unsqueeze(0).float()  # [1, num_gt, k]
            with autocast(device_type="cuda", enabled=False):
                vis_cost = (
                    F.binary_cross_entropy(
                        valid_pred_kpts_vis.to(dtype=torch.float32),
                        gt_kpt_vis.repeat(num_valid, 1, 1),
                        reduction="none",
                    )
                    .sum(-1)
                    .to(dtype=valid_pred_kpts_vis.dtype)
                )
            cost_matrix = cost_matrix + vis_cost * self.vis_weight

        if self.dynamic_k_indicator == "iou":
            matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
        elif self.dynamic_k_indicator == "oks":
            matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_oks, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious.to(max_overlaps)
        return dict(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(
        self,
        priors: Tensor,
        gt_bboxes: Tensor,
        gt_keypoints: Optional[Tensor] = None,
        gt_keypoints_visible: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Get the information of which prior is in gt bboxes and gt center
        priors."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        if self.use_keypoints_for_center and gt_keypoints_visible is not None:
            assert gt_keypoints is not None, "gt_keypoints is None"
            gt_kpts_cts = (gt_keypoints * gt_keypoints_visible.unsqueeze(-1)).sum(dim=-2) / gt_keypoints_visible.sum(
                dim=-1, keepdims=True
            ).clip(min=0)
            gt_kpts_cts = gt_kpts_cts.to(gt_bboxes)
            valid_mask = gt_keypoints_visible.sum(-1) > 0

            gt_cxs[valid_mask] = gt_kpts_cts[valid_mask][..., 0]
            gt_cys[valid_mask] = gt_kpts_cts[valid_mask][..., 1]

        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :]
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(
        self, cost: Tensor, pairwise_ious: Tensor, num_gt: int, valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=int(dynamic_ks[gt_idx]), largest=False)
            # cost[:, gt_idx], k=int(dynamic_ks[gt_idx][0]), largest=False) # fix
            matching_matrix[:, gt_idx][pos_idx] = 1
            # matching_matrix[:, gt_idx][:, pos_idx] = 1 # fix

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds


class YoloNeck(nn.Module):
    def __init__(self, input_shape_specs: list[ShapeSpec], c2f_depth: int, feat_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_features = ["res3", "res4", "res5"]
        # starting from "res2" to "res5"
        self.in_channels = sorted([v.channels for v in input_shape_specs if v is not None and v.channels is not None])
        self.in_strides = sorted([v.stride for v in input_shape_specs if v is not None and v.stride is not None])
        self.out_dim = out_dim
        self.feat_dim = feat_dim

        ch_5 = self.in_channels[-1]
        ch_4 = self.in_channels[-2]
        ch_3 = self.in_channels[-3]

        self.sppf = SPPF(ch_5, ch_5, 5)
        self.c2f12 = C2f(ch_4 + ch_5, ch_4, c2f_depth)
        self.c2f15 = C2f(ch_3 + ch_4, feat_dim, c2f_depth)
        self.cv1 = ConvNormLayer(
            ch_in=feat_dim, ch_out=ch_3, kernel_size=3, stride=2, padding=1, act=nn.SiLU(inplace=True)
        )
        self.c2f18 = C2f(ch_3 + ch_4, feat_dim, c2f_depth)
        self.cv2 = ConvNormLayer(
            ch_in=feat_dim, ch_out=ch_4, kernel_size=3, stride=2, padding=1, act=nn.SiLU(inplace=True)
        )
        self.c2f21 = C2f(ch_4 + ch_5, feat_dim)

        self.cv_out = ConvNormLayer(ch_in=feat_dim, ch_out=out_dim, kernel_size=1, stride=1)

    def forward_features(self, features):
        p3, p4, p5 = (features[f] for f in self.in_features)

        # output1
        h9 = self.sppf(p5)
        h11 = torch.cat([torch.nn.functional.interpolate(h9, p4.shape[-2:]), p4], 1)
        h12 = self.c2f12(h11)
        h14 = torch.cat([torch.nn.functional.interpolate(h12, p3.shape[-2:]), p3], 1)
        o3 = self.c2f15(h14)

        h16 = self.cv1(o3)
        h17 = torch.cat([h12, h16], 1)
        o4 = self.c2f18(h17)

        h19 = self.cv2(o4)
        h20 = torch.cat([h9, h19], dim=1)
        o5 = self.c2f21(h20)

        return self.cv_out(o3), (o5, o4, o3)  # 1/32, 1/16, 1/8

    def forward(self, features):
        return self.forward_features(features)


class YOLOXPoseHeadModule(nn.Module):
    """YOLOXPose head module for one-stage human pose estimation.

    This module predicts classification scores, bounding boxes, keypoint
    offsets and visibilities from multi-level feature maps.

    Args:
        num_keypoints (int): Number of keypoints defined for one instance.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 1.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        feat_channels (int): Number of channels in the classification score
            and objectness prediction branch. Defaults to 256.
        stacked_convs (int): Number of stacked convolutions. Defaults to 2.
        featmap_strides (Sequence[int]): Downsample factor of each feature
            map. Defaults to [8, 16, 32].
        norm (Optional[str]): Normalization layer. Defaults to None.
        activation (Optional[Callable[[Tensor], Tensor]]): Activation function.
            Defaults to None.
    """

    def __init__(
        self,
        num_keypoints: int,
        in_channels: Union[int, Sequence],
        num_classes: int = 1,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        norm: Optional[str] = None,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = int(feat_channels * widen_factor)
        self.stacked_convs = stacked_convs

        self.norm = norm
        self.activation = activation
        self.featmap_strides = featmap_strides

        if isinstance(in_channels, int):
            in_channels = int(in_channels * widen_factor)
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_cls_branch()
        self._init_reg_branch()
        self._init_pose_branch()

    def _init_cls_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    Conv2d(
                        in_channels=chn,
                        out_channels=self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm=get_norm(self.norm, self.feat_channels),
                        activation=self.activation,
                    )
                )
            self.conv_cls.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_cls = nn.ModuleList()
        self.out_obj = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_cls.append(nn.Conv2d(self.feat_channels, self.num_classes, 1))

    def _init_reg_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_reg = nn.ModuleList()
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    Conv2d(
                        in_channels=chn,
                        out_channels=self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm=get_norm(self.norm, self.feat_channels),
                        activation=self.activation,
                    )
                )
            self.conv_reg.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_bbox = nn.ModuleList()
        self.out_obj = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(self.feat_channels, 4, 1))
            self.out_obj.append(nn.Conv2d(self.feat_channels, 1, 1))

    def _init_pose_branch(self):
        self.conv_pose = nn.ModuleList()

        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                in_chn = self.in_channels if i == 0 else self.feat_channels
                stacked_convs.append(
                    Conv2d(
                        in_channels=in_chn,
                        out_channels=self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm=get_norm(self.norm, self.feat_channels),
                        activation=self.activation,
                    )
                )
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_kpt = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_kpt.append(nn.Conv2d(self.feat_channels, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(self.feat_channels, self.num_keypoints, 1))

    def init_weights(self):
        """Initialize weights of the head."""
        # Initialize weights for all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.out_cls, self.out_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls_scores (List[Tensor]): Classification scores for each level.
            objectnesses (List[Tensor]): Objectness scores for each level.
            bbox_preds (List[Tensor]): Bounding box predictions for each level.
            kpt_offsets (List[Tensor]): Keypoint offsets for each level.
            kpt_vis (List[Tensor]): Keypoint visibilities for each level.
        """

        cls_scores, bbox_preds, objectnesses = [], [], []
        kpt_offsets, kpt_vis = [], []

        for i in range(len(x)):
            cls_feat = self.conv_cls[i](x[i])
            reg_feat = self.conv_reg[i](x[i])
            pose_feat = self.conv_pose[i](x[i])

            cls_scores.append(self.out_cls[i](cls_feat))
            objectnesses.append(self.out_obj[i](reg_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            kpt_offsets.append(self.out_kpt[i](pose_feat))
            kpt_vis.append(self.out_kpt_vis[i](pose_feat))

        return cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis


class YOLOXPoseHead(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        num_classes: int = 1,
        in_channels: int = 4,
        widen_factor: float = 1.0,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        featmap_strides: Sequence[int] = [8, 16, 32],
        norm: Optional[str] = "BN",
        activation: Optional[Callable[[Tensor], Tensor]] = F.relu,
        use_aux_loss: bool = False,
        loss_cls: Optional[Callable] = None,
        loss_obj: Optional[Callable] = None,
        loss_bbox: Optional[Callable] = None,
        loss_oks: Optional[Callable] = None,
        loss_vis: Optional[Callable] = None,
        loss_bbox_aux: Optional[Callable] = None,
        loss_kpt_aux: Optional[Callable] = None,
        overlaps_power: float = 1.0,
        score_thr: float = 0.01,
        nms_pre: int = 100000,
        nms_thr: float = 1,
        featmap_strides_pointgenerator: List[Tuple[int, int]] = [(16, 16), (32, 32)],
        centralize_points_pointgenerator: bool = True,
    ):
        super().__init__()
        self.featmap_sizes = None
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.use_aux_loss = use_aux_loss
        self.num_keypoints = num_keypoints
        self.overlaps_power = overlaps_power
        self.score_thr = score_thr
        self.nms_pre = nms_pre
        self.nms_thr = nms_thr
        self.nms = True
        self.criterion = KeypointCriterion()

        self.head_module = YOLOXPoseHeadModule(
            num_keypoints=num_keypoints,
            in_channels=in_channels,
            num_classes=num_classes,
            widen_factor=widen_factor,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            featmap_strides=featmap_strides,
            norm=norm,
            activation=activation,
        )
        self.prior_generator = MlvlPointGenerator(
            strides=featmap_strides_pointgenerator, centralize_points=centralize_points_pointgenerator
        )

        self.assigner = SimOTAAssigner(
            dynamic_k_indicator="oks", oks_calculator=PoseOKS(), use_keypoints_for_center=True
        )

        # build losses
        self.loss_cls = loss_cls
        if loss_obj is not None:
            self.loss_obj = loss_obj
        self.loss_bbox = loss_bbox
        self.loss_oks = loss_oks
        self.loss_vis = loss_vis
        if loss_bbox_aux is not None:
            self.loss_bbox_aux = loss_bbox_aux
        if loss_kpt_aux is not None:
            self.loss_kpt_aux = loss_kpt_aux

    def _forward(self, feats: Tuple[Tensor]):
        assert isinstance(feats, (tuple, list))
        return self.head_module(feats)

    def forward(self, features, targets: Optional[list[KeypointTargets]] = None):
        assert isinstance(features, (tuple, list))
        spatial_features, multi_scale_features = features
        if self.training:
            if targets:
                return None, self.loss(feats=multi_scale_features, targets=targets)
        outputs = self.predict(multi_scale_features)
        return outputs, {}

    def losses(self, predictions, targets):
        (
            flatten_cls_scores,
            flatten_objectness,
            flatten_bbox_decoded,
            flatten_kpt_decoded,
            flatten_kpt_vis,
            flatten_cls_scores,
            flatten_bbox_preds,
            flatten_kpt_offsets,
        ) = predictions

        (
            pos_masks,
            cls_targets,
            obj_targets,
            obj_weights,
            bbox_targets,
            bbox_aux_targets,
            kpt_targets,
            kpt_aux_targets,
            vis_targets,
            vis_weights,
            pos_areas,
            pos_priors,
            group_indices,
            num_fg_imgs,
        ) = targets

        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_scores.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # 3. calculate loss
        # 3.1 objectness loss
        losses = dict()

        obj_preds = flatten_objectness.view(-1, 1)
        losses["loss_obj"] = (
            self.criterion.get_loss("objectness", obj_preds, obj_targets, target_weight=obj_weights) / num_total_samples
        )

        if num_pos > 0:
            # 3.2 bbox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses["loss_bbox"] = self.criterion.get_loss("boxes", bbox_preds, bbox_targets) / num_total_samples

            # 3.3 keypoint loss
            kpt_preds = flatten_kpt_decoded.view(-1, self.num_keypoints, 2)[pos_masks]
            losses["loss_kpt"] = self.criterion.get_loss(
                "oks", kpt_preds, kpt_targets, target_weight=vis_targets, areas=pos_areas
            )

            # 3.4 keypoint visibility loss
            kpt_vis_preds = flatten_kpt_vis.view(-1, self.num_keypoints)[pos_masks]
            losses["loss_vis"] = self.criterion.get_loss(
                "visibility", kpt_vis_preds, vis_targets, target_weight=vis_weights
            )

            # 3.5 classification loss
            cls_preds = flatten_cls_scores.view(-1, self.num_classes)[pos_masks]
            losses["overlaps"] = cls_targets.mean(dim=None)
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            losses["loss_cls"] = self.criterion.get_loss("classification", cls_preds, cls_targets) / num_total_samples

            if self.use_aux_loss:
                if hasattr(self, "loss_bbox_aux"):
                    # 3.6 auxiliary bbox regression loss
                    bbox_preds_raw = flatten_bbox_preds.view(-1, 4)[pos_masks]
                    losses["loss_bbox_aux"] = (
                        self.criterion.get_loss("bbox_aux", bbox_preds_raw, bbox_aux_targets) / num_total_samples
                    )

                if hasattr(self, "loss_kpt_aux"):
                    # 3.7 auxiliary keypoint regression loss
                    kpt_preds_raw = flatten_kpt_offsets.view(-1, self.num_keypoints, 2)[pos_masks]
                    kpt_weights = vis_targets / vis_targets.size(-1)
                    losses["loss_kpt_aux"] = self.loss_kpt_aux(kpt_preds_raw, kpt_aux_targets, kpt_weights)

        return losses

    def loss(self, feats: Tuple[Tensor], targets: List[KeypointTargets]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples

        Returns:
            dict: A dictionary of losses.
        """

        # 1. collect & reform predictions
        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = self._forward(feats)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(  # mlvl_priors.shape = [torch.Size([400, 4]), torch.Size([1600, 4]), torch.Size([6400, 4])]
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True
        )
        flatten_priors = torch.cat(mlvl_priors)  # flatten_priors.shape = [8400, 4], format cxcywh ABS

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = self._flatten_predictions(cls_scores)
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = self._flatten_predictions(objectnesses)
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)

        if (
            flatten_cls_scores is None
            or flatten_bbox_preds is None
            or flatten_objectness is None
            or flatten_kpt_offsets is None
            or flatten_kpt_vis is None
        ):
            raise ValueError("One or more flattened predictions are None")

        flatten_bbox_decoded = self.decode_bbox(
            flatten_bbox_preds,  # flatten_bbox_preds.shape = [batch_size, 8400, 4]
            flatten_priors[..., :2],  # flatten_priors[..., :2].shape = [8400, 2]
            flatten_priors[..., -1],
        )  # flatten_priors[..., -1].shape = [8400]
        flatten_kpt_decoded = self.decode_kpt_reg(
            flatten_kpt_offsets,  # flatten_kpt_offsets.shape = [batch_size, 8400, 34]
            flatten_priors[..., :2],  # flatten_priors[..., :2].shape = [8400, 2]
            flatten_priors[..., -1],
        )  # flatten_priors[..., -1].shape = [8400]
        # flatten_kpt_decoded.shape = [batch_size, 8400, 17, 2], format xy ABS
        # flatten_bbox_decoded output format: XYXY ABS, flatten_bbox_decoded.shape = [batch_size, 8400, 4]

        # 2. generate targets
        _targets = self._get_targets(
            flatten_priors,
            flatten_cls_scores.detach(),
            flatten_objectness.detach(),
            flatten_bbox_decoded.detach(),
            flatten_kpt_decoded.detach(),
            flatten_kpt_vis.detach(),
            targets,
        )

        predictions = (
            flatten_cls_scores,
            flatten_objectness,
            flatten_bbox_decoded,
            flatten_kpt_decoded,
            flatten_kpt_vis,
            flatten_cls_scores,
            flatten_bbox_preds,
            flatten_kpt_offsets,
        )

        losses = self.losses(predictions, _targets)
        return losses

    @torch.no_grad()
    def _get_targets(
        self,
        priors: Tensor,
        batch_cls_scores: Tensor,
        batch_objectness: Tensor,
        batch_decoded_bboxes: Tensor,
        batch_decoded_kpts: Tensor,
        batch_kpt_vis: Tensor,
        targets: List[KeypointTargets],
    ):
        num_imgs = len(targets)

        # use clip to avoid nan
        batch_cls_scores = batch_cls_scores.clip(min=-1e4, max=1e4).sigmoid()
        batch_objectness = batch_objectness.clip(min=-1e4, max=1e4).sigmoid()
        batch_kpt_vis = batch_kpt_vis.clip(min=-1e4, max=1e4).sigmoid()
        batch_cls_scores[torch.isnan(batch_cls_scores)] = 0
        batch_objectness[torch.isnan(batch_objectness)] = 0

        targets_each = []
        for i in range(num_imgs):
            target = self._get_targets_single(
                priors=priors,
                cls_scores=batch_cls_scores[i],
                objectness=batch_objectness[i],
                decoded_bboxes=batch_decoded_bboxes[i],
                decoded_kpts=batch_decoded_kpts[i],
                kpt_vis=batch_kpt_vis[i],
                data_sample=targets[i],
            )
            targets_each.append(target)

        targets = list(zip(*targets_each))
        for i, target in enumerate(targets):
            if torch.is_tensor(target[0]):
                target = tuple(filter(lambda x: x.size(0) > 0, target))
                if len(target) > 0:
                    targets[i] = torch.cat(target)

        (
            foreground_masks,
            cls_targets,
            obj_targets,
            obj_weights,
            bbox_targets,
            kpt_targets,
            vis_targets,
            vis_weights,
            pos_areas,
            pos_priors,
            group_indices,
            num_pos_per_img,
        ) = targets

        # post-processing for targets
        if self.use_aux_loss:
            bbox_cxcy = (bbox_targets[:, :2] + bbox_targets[:, 2:]) / 2.0
            bbox_wh = bbox_targets[:, 2:] - bbox_targets[:, :2]
            bbox_aux_targets = torch.cat(
                [(bbox_cxcy - pos_priors[:, :2]) / pos_priors[:, 2:], torch.log(bbox_wh / pos_priors[:, 2:] + 1e-8)],
                dim=-1,
            )

            kpt_aux_targets = (kpt_targets - pos_priors[:, None, :2]) / pos_priors[:, None, 2:]
        else:
            bbox_aux_targets, kpt_aux_targets = None, None

        return (
            foreground_masks,
            cls_targets,
            obj_targets,
            obj_weights,
            bbox_targets,
            bbox_aux_targets,
            kpt_targets,
            kpt_aux_targets,
            vis_targets,
            vis_weights,
            pos_areas,
            pos_priors,
            group_indices,
            num_pos_per_img,
        )

    @torch.no_grad()
    def _get_targets_single(
        self,
        priors: Tensor,
        cls_scores: Tensor,
        objectness: Tensor,
        decoded_bboxes: Tensor,
        decoded_kpts: Tensor,
        kpt_vis: Tensor,
        data_sample: KeypointTargets,
    ) -> tuple:
        """Compute classification, bbox, keypoints and objectness targets for
        priors in a single image.

        Args:
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            cls_scores (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in xyxy format.
            decoded_kpts (Tensor): Decoded keypoints predictions of one image,
                a 3D-Tensor with shape [num_priors, num_keypoints, 2].
            kpt_vis (Tensor): Keypoints visibility predictions of one image,
                a 2D-Tensor with shape [num_priors, num_keypoints].
            data_sample (PoseDataSample): Data sample that contains the ground
                truth annotations for current image.

        Returns:
            tuple: A tuple containing various target tensors for training:
                - foreground_mask (Tensor): Binary mask indicating foreground
                    priors.
                - cls_target (Tensor): Classification targets.
                - obj_target (Tensor): Objectness targets.
                - obj_weight (Tensor): Weights for objectness targets.
                - bbox_target (Tensor): BBox targets.
                - kpt_target (Tensor): Keypoints targets.
                - vis_target (Tensor): Visibility targets for keypoints.
                - vis_weight (Tensor): Weights for keypoints visibility
                    targets.
                - pos_areas (Tensor): Areas of positive samples.
                - pos_priors (Tensor): Priors corresponding to positive
                    samples.
                - group_index (List[Tensor]): Indices of groups for positive
                    samples.
                - num_pos_per_img (int): Number of positive samples.
        """
        # TODO: change the shape of objectness to [num_priors]
        num_priors = priors.size(0)
        # TODO: avoid using mmpose data structures
        gt_instances = KeypointTargets(
            bboxes=data_sample.bboxes,
            scores=None,
            priors=None,
            labels=data_sample.labels,
            keypoints=data_sample.keypoints,
            keypoints_visible=data_sample.keypoints_visible,
            areas=data_sample.areas,
            keypoints_visible_weights=data_sample.get("keypoints_visible_weights", None),
        )
        gt_fields = data_sample.get("gt_fields", dict())
        num_gts = len(gt_instances)

        # No target
        if num_gts == 0:
            cls_target = cls_scores.new_zeros((0, self.num_classes))
            bbox_target = cls_scores.new_zeros((0, 4))
            obj_target = cls_scores.new_zeros((num_priors, 1))
            obj_weight = cls_scores.new_ones((num_priors, 1))
            kpt_target = cls_scores.new_zeros((0, self.num_keypoints, 2))
            vis_target = cls_scores.new_zeros((0, self.num_keypoints))
            vis_weight = cls_scores.new_zeros((0, self.num_keypoints))
            pos_areas = cls_scores.new_zeros((0,))
            pos_priors = priors[:0]
            foreground_mask = cls_scores.new_zeros(num_priors).bool()
            return (
                foreground_mask,
                cls_target,
                obj_target,
                obj_weight,
                bbox_target,
                kpt_target,
                vis_target,
                vis_weight,
                pos_areas,
                pos_priors,
                [],
                0,
            )

        # assign positive samples
        scores = cls_scores * objectness
        pred_instances = KeypointTargets(
            bboxes=decoded_bboxes,
            scores=scores.sqrt_(),
            priors=priors,
            keypoints=decoded_kpts,
            keypoints_visible=kpt_vis,
            keypoints_visible_weights=None,
            areas=None,
            labels=None,
        )
        assign_result = self.assigner.assign(pred_instances=pred_instances, gt_instances=gt_instances)

        # sampling
        pos_inds = torch.nonzero(assign_result["gt_inds"] > 0, as_tuple=False).squeeze(-1).unique()
        num_pos_per_img = pos_inds.size(0)
        pos_gt_labels = assign_result["labels"][pos_inds]
        pos_assigned_gt_inds = assign_result["gt_inds"][pos_inds] - 1

        # bbox target
        assert gt_instances.bboxes is not None, "gt_instances.bboxes is None"
        bbox_target = gt_instances.bboxes[pos_assigned_gt_inds.long()]

        # cls target
        max_overlaps = assign_result["max_overlaps"][pos_inds]
        cls_target = F.one_hot(pos_gt_labels, self.num_classes) * max_overlaps.unsqueeze(-1)

        # pose targets
        assert gt_instances.keypoints is not None, "gt_instances.keypoints is None"
        assert gt_instances.keypoints_visible is not None, "gt_instances.keypoints_visible is None"
        kpt_target = gt_instances.keypoints[pos_assigned_gt_inds]
        vis_target = gt_instances.keypoints_visible[pos_assigned_gt_inds]
        if gt_instances.keypoints_visible_weights is not None:
            vis_weight = gt_instances.keypoints_visible_weights[pos_assigned_gt_inds]
        else:
            vis_weight = vis_target.new_ones(vis_target.shape)
        pos_areas = gt_instances.areas[pos_assigned_gt_inds] if gt_instances.areas is not None else None

        # obj target
        obj_target = torch.zeros_like(objectness)
        obj_target[pos_inds] = 1
        # TODO: check if this is needed
        invalid_mask = gt_fields.get("heatmap_mask", None)
        if invalid_mask is not None and (invalid_mask != 0.0).any():
            # ignore the tokens that predict the unlabled instances
            pred_vis = (kpt_vis.unsqueeze(-1) > 0.3).float()
            mean_kpts = (decoded_kpts * pred_vis).sum(dim=1) / pred_vis.sum(dim=1).clamp(min=1e-8)
            mean_kpts = mean_kpts.reshape(1, -1, 1, 2)
            wh = invalid_mask.shape[-1]
            grids = mean_kpts / (wh - 1) * 2 - 1
            mask = invalid_mask.unsqueeze(0).float()
            weight = F.grid_sample(mask, grids, mode="bilinear", padding_mode="zeros")
            obj_weight = 1.0 - weight.reshape(num_priors, 1)
        else:
            obj_weight = obj_target.new_ones(obj_target.shape)

        # misc
        foreground_mask = torch.zeros_like(objectness.squeeze()).to(torch.bool)
        foreground_mask[pos_inds] = 1
        pos_priors = priors[pos_inds]
        group_index = [torch.where(pos_assigned_gt_inds == num)[0] for num in torch.unique(pos_assigned_gt_inds)]

        return (
            foreground_mask,
            cls_target,
            obj_target,
            obj_weight,
            bbox_target,
            kpt_target,
            vis_target,
            vis_weight,
            pos_areas,
            pos_priors,
            group_index,
            num_pos_per_img,
        )

    def _fuse(self):
        self.nms = False

    def predict(
        self,
        feats: Tuple[Tensor],
    ) -> KeypointOutput:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor]): The multi-stage features from the backbone network.
                Each element in the tuple is a 4D-tensor with shape (B, C, H, W).

        Returns:
            dict: A dictionary containing the following keys:
                - scores (Tensor): Classification scores for each instance with shape (B, N)
                - labels (Tensor): Predicted class labels with shape (B, N)
                - pred_bboxes (Tensor): Predicted bounding boxes in XYXY format with shape (B, N, 4)
                - bbox_scores (Tensor): Bounding box confidence scores with shape (B, N)
                - pred_keypoints (Tensor): Predicted keypoint coordinates with shape (B, N, K, 2)
                - keypoint_scores (Tensor): Keypoint confidence scores with shape (B, N, K)
                - keypoints_visible (Tensor): Keypoint visibility predictions with shape (B, N, K)

            Where:
                - B is the batch size
                - N is the number of instances
                - K is the number of keypoints per instance
        """

        cls_scores, objectnesses, bbox_preds, kpt_offsets, kpt_vis = self._forward(feats)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
            )
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel(),), stride)
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = self._flatten_predictions(cls_scores)
        flatten_cls_scores = flatten_cls_scores.sigmoid() if flatten_cls_scores is not None else None
        flatten_bbox_preds = self._flatten_predictions(bbox_preds)
        flatten_objectness = self._flatten_predictions(objectnesses)
        flatten_objectness = flatten_objectness.sigmoid() if flatten_objectness is not None else None
        flatten_kpt_offsets = self._flatten_predictions(kpt_offsets)
        flatten_kpt_vis = self._flatten_predictions(kpt_vis)
        flatten_kpt_vis = flatten_kpt_vis.sigmoid() if flatten_kpt_vis is not None else None
        flatten_bbox_preds = (
            self.decode_bbox(flatten_bbox_preds, flatten_priors, flatten_stride)
            if flatten_bbox_preds is not None
            else None
        )
        flatten_kpt_reg = (
            self.decode_kpt_reg(flatten_kpt_offsets, flatten_priors, flatten_stride)
            if flatten_kpt_offsets is not None
            else None
        )

        # Initialize empty lists for each key
        all_scores = []
        all_labels = []
        all_pred_bboxes = []
        all_bbox_scores = []
        all_pred_keypoints = []
        all_keypoint_scores = []
        all_keypoints_visible = []

        assert flatten_bbox_preds is not None, "flatten_bbox_preds is None"
        assert flatten_cls_scores is not None, "flatten_cls_scores is None"
        assert flatten_objectness is not None, "flatten_objectness is None"
        assert flatten_kpt_reg is not None, "flatten_kpt_reg is None"
        assert flatten_kpt_vis is not None, "flatten_kpt_vis is None"

        for bboxes, scores, objectness, kpt_reg, kpt_vis in zip(
            flatten_bbox_preds, flatten_cls_scores, flatten_objectness, flatten_kpt_reg, flatten_kpt_vis
        ):
            score_thr = self.score_thr
            scores = scores.clone() * objectness

            nms_pre = self.nms_pre
            scores, labels = scores.max(1, keepdim=True)
            scores, _, keep_idxs_score, results = filter_scores_and_topk(
                scores, score_thr, nms_pre, results=dict(labels=labels[:, 0])
            )

            labels = results["labels"] if results is not None else labels[:, 0]

            bboxes = bboxes[keep_idxs_score]
            kpt_vis = kpt_vis[keep_idxs_score]
            stride = flatten_stride[keep_idxs_score]
            keypoints = kpt_reg[keep_idxs_score]

            if bboxes.numel() > 0 and self.nms:
                nms_thr = self.nms_thr
                if nms_thr < 1.0:
                    keep_idxs_nms = nms(bboxes, scores, nms_thr)
                    bboxes = bboxes[keep_idxs_nms]
                    stride = stride[keep_idxs_nms]
                    labels = labels[keep_idxs_nms]
                    kpt_vis = kpt_vis[keep_idxs_nms]
                    keypoints = keypoints[keep_idxs_nms]
                    scores = scores[keep_idxs_nms]

            all_scores.append(scores.unsqueeze(0))
            all_labels.append(labels.unsqueeze(0))
            all_pred_bboxes.append(bboxes.unsqueeze(0))
            all_bbox_scores.append(scores.unsqueeze(0))
            all_pred_keypoints.append(keypoints.unsqueeze(0))
            all_keypoint_scores.append(kpt_vis.unsqueeze(0))
            all_keypoints_visible.append(kpt_vis.unsqueeze(0))

        return KeypointOutput(
            scores=torch.cat(all_scores, dim=0),
            labels=torch.cat(all_labels, dim=0),
            pred_bboxes=torch.cat(all_pred_bboxes, dim=0),
            bbox_scores=torch.cat(all_bbox_scores, dim=0),
            pred_keypoints=torch.cat(all_pred_keypoints, dim=0),
            keypoint_scores=torch.cat(all_keypoint_scores, dim=0),
            keypoints_visible=torch.cat(all_keypoints_visible, dim=0),
        )

    def decode_bbox(
        self, pred_bboxes: torch.Tensor, priors: torch.Tensor, stride: Union[torch.Tensor, int]
    ) -> torch.Tensor:
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
        return decoded_bboxes  # decoded_bboxes format = XYXY ABS

    def decode_kpt_reg(
        self, pred_kpt_offsets: torch.Tensor, priors: torch.Tensor, stride: torch.Tensor
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
            *pred_kpt_offsets.shape[:-1], self.num_keypoints, 2
        )  # pred_kpt_offsets.shape = [batch_size, 8400, 17, 2]

        decoded_kpts = pred_kpt_offsets * stride + priors
        return decoded_kpts  # decoded_kpts.shape = [batch_size, 8400, 17, 2], format xy ABS

    def _flatten_predictions(self, preds: List[Tensor]) -> Optional[Tensor]:
        """Flattens the predictions from a list of tensors to a single
        tensor."""
        if len(preds) == 0:
            return None

        preds = [x.permute(0, 2, 3, 1).flatten(1, 2) for x in preds]
        return torch.cat(preds, dim=1)


class YOLOXPose(BaseModelNN):
    def __init__(self, config: YOLOXPoseConfig):
        assert len(config.featmap_strides) == len(config.featmap_strides_pointgenerator), (
            "featmap_strides and featmap_strides_pointgenerator must have the same length"
        )
        super().__init__(config)
        self.config = config

        self.backbone = load_backbone(self.config.backbone_config)
        input_shape_specs = list(self.backbone.output_shape().values())
        self.pixel_decoder = YoloNeck(
            input_shape_specs=input_shape_specs,
            feat_dim=self.config.neck_feat_dim,
            out_dim=self.config.neck_out_dim,
            c2f_depth=self.config.c2f_depth,
        )

        self.head = YOLOXPoseHead(
            num_keypoints=self.config.num_keypoints,
            num_classes=self.config.num_classes,
            in_channels=self.config.in_channels,
            widen_factor=1.0,
            feat_channels=self.config.feat_channels,
            stacked_convs=self.config.stacked_convs,
            featmap_strides=self.config.featmap_strides,
            norm=self.config.norm,
            activation=get_activation_fn(self.config.activation),
            use_aux_loss=self.config.use_aux_loss,
            overlaps_power=self.config.overlaps_power,
            score_thr=self.config.score_thr,
            nms_pre=self.config.nms_pre,
            nms_thr=self.config.nms_thr,
            featmap_strides_pointgenerator=self.config.featmap_strides_pointgenerator,
            centralize_points_pointgenerator=self.config.centralize_points_pointgenerator,
        )

        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def dtype(self):
        return self.pixel_mean.dtype

    def forward(self, images: torch.Tensor, targets: list[KeypointTargets] = []) -> YOLOXPoseModelOutput:
        images = (images - self.pixel_mean) / self.pixel_std  # type: ignore

        features = self.backbone(images)
        features = self.pixel_decoder(features)
        outputs, losses = self.head(features, targets)
        return YOLOXPoseModelOutput(outputs=outputs, loss=losses)
