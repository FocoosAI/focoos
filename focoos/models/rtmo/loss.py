# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autocast
from torch.nn.modules.utils import _pair

from focoos.models.rtmo.ports import KeypointTargets, RTMOKeypointLossLiteral
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

    def __init__(self, num_keypoints: int, sigmas=None):
        if sigmas is None and num_keypoints == 17:
            # Assume is COCO keypoints
            sigmas = [
                0.026,
                0.025,
                0.025,
                0.035,
                0.035,
                0.079,
                0.079,
                0.072,
                0.072,
                0.062,
                0.062,
                0.107,
                0.107,
                0.087,
                0.087,
                0.089,
                0.089,
            ]
        else:
            sigmas = [0.05] * num_keypoints  # When we don't know the sigmas, use a default value 0.05
        self.sigmas = torch.as_tensor(sigmas)

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
        oks_calculator: PoseOKS = PoseOKS(num_keypoints=17),
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
        assert gt_instances.boxes is not None, "gt_instances.bboxes is None"
        assert gt_instances.labels is not None, "gt_instances.labels is None"
        assert gt_instances.keypoints is not None, "gt_instances.keypoints is None"
        assert gt_instances.keypoints_visible is not None, "gt_instances.keypoints_visible is None"
        assert gt_instances.areas is not None, "gt_instances.areas is None"
        assert pred_instances.boxes is not None, "pred_instances.bboxes is None"
        assert pred_instances.scores is not None, "pred_instances.scores is None"
        assert pred_instances.priors is not None, "pred_instances.priors is None"
        assert pred_instances.keypoints is not None, "pred_instances.keypoints is None"
        assert pred_instances.keypoints_visible is not None, "pred_instances.keypoints_visible is None"

        gt_bboxes = gt_instances.boxes
        gt_labels = gt_instances.labels
        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_areas = gt_instances.areas
        num_gt = gt_bboxes.size(0) if gt_bboxes is not None else 0

        decoded_bboxes = pred_instances.boxes
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


class KeypointCriterion(nn.Module):
    def __init__(self, num_keypoints: int):
        super().__init__()
        # self.loss_obj = BCELoss(use_target_weight=True, reduction="sum", loss_weight=1.0) # Not used
        self.loss_bbox = IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0)
        self.loss_oks = OKSLoss(
            reduction="mean", norm_target_weight=True, loss_weight=30.0, num_keypoints=num_keypoints
        )
        self.loss_vis = BCELoss(use_target_weight=True, reduction="mean", loss_weight=1.0)
        # self.loss_cls = BCELoss(reduction="sum", loss_weight=1.0) # Not used
        self.loss_bbox_aux = L1Loss(reduction="sum", loss_weight=1.0)
        self.loss_mle = MLECCLoss(use_target_weight=True, loss_weight=1.0)
        self.loss_cls = VariFocalLoss(reduction="sum", use_target_weight=True, loss_weight=1.0)

    def get_loss(self, loss: RTMOKeypointLossLiteral, outputs, targets, **kwargs) -> torch.Tensor:
        loss_map: dict[RTMOKeypointLossLiteral, nn.Module] = {
            "boxes": self.loss_bbox,
            "oks": self.loss_oks,
            "visibility": self.loss_vis,
            "bbox_aux": self.loss_bbox_aux,
            "mle": self.loss_mle,
            "classification_varifocal": self.loss_cls,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, **kwargs)


class IoULoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self, reduction="mean", mode="log", eps: float = 1e-16, loss_weight=1.0):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), (
            f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        )

        assert mode in ("linear", "square", "log"), (
            f"the argument `reduction` should be either 'linear', 'square' or 'log', but got {mode}"
        )

        self.reduction = reduction
        self.criterion = partial(F.cross_entropy, reduction="none")
        self.loss_weight = loss_weight
        self.mode = mode
        self.eps = eps

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
        """
        ious = bbox_overlaps(output, target, is_aligned=True).clamp(min=self.eps)

        if self.mode == "linear":
            loss = 1 - ious
        elif self.mode == "square":
            loss = 1 - ious.pow(2)
        elif self.mode == "log":
            loss = -ious.log()
        else:
            raise NotImplementedError

        if target_weight is not None:
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


class VariFocalLoss(nn.Module):
    """Varifocal loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        alpha (float): A balancing factor for the negative part of
            Varifocal Loss. Defaults to 0.75.
        gamma (float): Gamma parameter for the modulating factor.
            Defaults to 2.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0, reduction="mean", alpha=0.75, gamma=2.0):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), (
            f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        )

        self.reduction = reduction
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = gamma

    def criterion(self, output, target):
        label = (target > 1e-4).to(target)
        weight = self.alpha * output.sigmoid().pow(self.gamma) * (1 - label) + target
        output = output.clip(min=-10, max=10)
        vfl = F.binary_cross_entropy_with_logits(output, target, reduction="none") * weight
        return vfl

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss = loss * target_weight
        else:
            loss = self.criterion(output, target)

        loss[torch.isinf(loss)] = 0.0
        loss[torch.isnan(loss)] = 0.0

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


class BCELoss(nn.Module):
    """Binary Cross Entropy loss.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of the loss. Default: 1.0.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            before output. Defaults to False.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.0, reduction="mean", use_sigmoid=False):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), (
            f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        )

        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        criterion = F.binary_cross_entropy if use_sigmoid else F.binary_cross_entropy_with_logits
        self.criterion = partial(criterion, reduction="none")
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K]): Output classification.
            target (torch.Tensor[N, K]): Target classification.
            target_weight (torch.Tensor[N, K] or torch.Tensor[N]):
                Weights across different labels.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output, target)
            if target_weight.dim() == 1:
                target_weight = target_weight[:, None]
            loss = loss * target_weight
        else:
            loss = self.criterion(output, target)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


class MLECCLoss(nn.Module):
    """Maximum Likelihood Estimation loss for Coordinate Classification.

    This loss function is designed to work with coordinate classification
    problems where the likelihood of each target coordinate is maximized.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        mode (str): Specifies the mode of calculating loss:
            'linear' | 'square' | 'log'. Default: 'log'.
        use_target_weight (bool): If True, uses weighted loss. Different
            joint types may have different target weights. Defaults to False.
        loss_weight (float): Weight of the loss. Defaults to 1.0.

    Raises:
        AssertionError: If the `reduction` or `mode` arguments are not in the
                        expected choices.
        NotImplementedError: If the selected mode is not implemented.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        mode: Literal["linear", "square", "log"] = "log",
        use_target_weight: bool = False,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), (
            f"`reduction` should be either 'mean', 'sum', or 'none', but got {reduction}"
        )
        assert mode in ("linear", "square", "log"), (
            f"`mode` should be either 'linear', 'square', or 'log', but got {mode}"
        )

        self.reduction = reduction
        self.mode = mode
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, target_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the MLECCLoss.

        Args:
            outputs (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The ground truth targets.
            target_weight (torch.Tensor, optional): Optional tensor of weights
                for each target.

        Returns:
            torch.Tensor: Calculated loss based on the specified mode and
                reduction.
        """

        assert len(outputs) == len(targets), "Outputs and targets must have the same length"

        prob = 1.0
        for o, t in zip(outputs, targets):
            prob *= (o * t).sum(dim=-1)

        if self.mode == "linear":
            loss = 1.0 - prob
        elif self.mode == "square":
            loss = 1.0 - prob.pow(2)
        elif self.mode == "log":
            loss = -torch.log(prob + 1e-4)

        loss[torch.isnan(loss)] = 0.0

        if self.use_target_weight:
            assert target_weight is not None
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == "sum":
            loss = loss.flatten(1).sum(dim=1)
        elif self.reduction == "mean":
            loss = loss.flatten(1).mean(dim=1)

        return torch.mul(loss, self.loss_weight)


class OKSLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.
    (2022).

    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.

    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.

    Args:
        metainfo (Optional[str]): Path to a JSON file containing information
            about the dataset's annotations.
        reduction (str): Options are "none", "mean" and "sum".
        eps (float): Epsilon to avoid log(0).
        loss_weight (float): Weight of the loss. Default: 1.0.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'linear'
        norm_target_weight (bool): whether to normalize the target weight
            with number of visible keypoints. Defaults to False.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        mode: Literal["linear", "square", "log"] = "linear",
        eps: float = 1e-8,
        norm_target_weight: bool = False,
        loss_weight=1.0,
        sigmas: Optional[List[float]] = None,
        num_keypoints: int = 17,
    ):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), (
            f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        )

        assert mode in ("linear", "square", "log"), (
            f"the argument `reduction` should be either 'linear', 'square' or 'log', but got {mode}"
        )

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode
        self.norm_target_weight = norm_target_weight
        self.eps = eps

        if sigmas is None and num_keypoints == 17:
            # Assume is COCO keypoints
            sigmas = [
                0.026,
                0.025,
                0.025,
                0.035,
                0.035,
                0.079,
                0.079,
                0.072,
                0.072,
                0.062,
                0.062,
                0.107,
                0.107,
                0.087,
                0.087,
                0.089,
                0.089,
            ]
        else:
            sigmas = [0.05] * num_keypoints  # When we don't know the sigmas, use a default value 0.05
        self.register_buffer("sigmas", torch.as_tensor(sigmas))

    def forward(self, output, target, target_weight=None, areas=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_labels: K

        Args:
            output (torch.Tensor[N, K, 2]): Output keypoints coordinates.
            target (torch.Tensor[N, K, 2]): Target keypoints coordinates..
            target_weight (torch.Tensor[N, K]): Loss weight for each keypoint.
            areas (torch.Tensor[N]): Instance size which is adopted as
                normalization factor.
        """
        dist = torch.norm(output - target, dim=-1)
        if areas is not None:
            # Clip areas to prevent division by very small numbers
            areas = areas.clamp(min=self.eps)
            dist = dist / areas.pow(0.5).unsqueeze(-1)

        # Reweight by sigmas
        sigmas = self.sigmas.reshape(*((1,) * (dist.ndim - 1)), -1)  # type: ignore
        dist = dist / (sigmas * 2)

        # Clip large distance values to prevent exp(-large_value) from becoming too small
        dist = torch.clamp(dist, max=50.0)
        oks = torch.exp(-dist.pow(2) / 2)

        if target_weight is not None:
            if self.norm_target_weight:
                sum_weights = target_weight.sum(dim=-1, keepdims=True).clamp(min=self.eps)
                target_weight = target_weight / sum_weights
            else:
                target_weight = target_weight / target_weight.size(-1)
            oks = oks * target_weight
        oks = oks.sum(dim=-1)

        if self.mode == "linear":
            loss = 1 - oks
        elif self.mode == "square":
            loss = 1 - oks.pow(2)
        elif self.mode == "log":
            # Add epsilon to prevent log(0)
            oks = torch.clamp(oks, min=self.eps)
            loss = -torch.log(oks)
        else:
            raise NotImplementedError()

        # Handle any NaN values
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss * self.loss_weight


class L1Loss(nn.Module):
    """L1Loss loss."""

    def __init__(self, reduction="mean", use_target_weight=False, loss_weight=1.0):
        super().__init__()

        assert reduction in ("mean", "sum", "none"), (
            f"the argument `reduction` should be either 'mean', 'sum' or 'none', but got {reduction}"
        )

        self.criterion = partial(F.l1_loss, reduction=reduction)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            for _ in range(target.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = self.criterion(output * target_weight, target * target_weight)
        else:
            loss = self.criterion(output, target)

        # Handle any NaN values
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

        return loss * self.loss_weight
