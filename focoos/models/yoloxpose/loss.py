# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from focoos.models.yoloxpose.ports import YoloXPoseKeypointLossLiteral
from focoos.utils.box import bbox_overlaps


class KeypointCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_obj = BCELoss(use_target_weight=True, reduction="sum", loss_weight=1.0)
        self.loss_bbox = IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0)
        self.loss_oks = OKSLoss(reduction="mean", norm_target_weight=True, loss_weight=30.0)
        self.loss_vis = BCELoss(use_target_weight=True, reduction="mean", loss_weight=1.0)
        self.loss_cls = BCELoss(reduction="sum", loss_weight=1.0)
        self.loss_bbox_aux = L1Loss(reduction="sum", loss_weight=1.0)
        # RTMO specific
        self.loss_mle = MLECCLoss(use_target_weight=True, loss_weight=1.0)
        self.loss_cls_varifocal = VariFocalLoss(reduction="sum", use_target_weight=True, loss_weight=1.0)

    def get_loss(self, loss: YoloXPoseKeypointLossLiteral, outputs, targets, **kwargs) -> torch.Tensor:
        loss_map: dict[YoloXPoseKeypointLossLiteral, nn.Module] = {
            "objectness": self.loss_obj,
            "boxes": self.loss_bbox,
            "oks": self.loss_oks,
            "visibility": self.loss_vis,
            "classification": self.loss_cls,
            "bbox_aux": self.loss_bbox_aux,
            # RTMO specific
            "mle": self.loss_mle,
            "classification_varifocal": self.loss_cls_varifocal,
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
        metainfo: Optional[str] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        mode: Literal["linear", "square", "log"] = "linear",
        eps: float = 1e-8,
        norm_target_weight: bool = False,
        loss_weight=1.0,
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
