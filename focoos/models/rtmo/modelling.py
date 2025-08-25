# Copyright (c) OpenMMLab. All rights reserved.
#
# Reimplemented from MMPose: https://github.com/open-mmlab/mmpose

import types
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms

from focoos.models.focoos_model import BaseModelNN
from focoos.models.rtmo.config import RTMOConfig
from focoos.models.rtmo.decoder import HybridEncoder
from focoos.models.rtmo.loss import (
    KeypointCriterion,
    MlvlPointGenerator,
    PoseOKS,
    SimOTAAssigner,
)
from focoos.models.rtmo.ports import KeypointOutput, KeypointTargets, RTMOModelOutput
from focoos.models.rtmo.transformer import SinePositionalEncoding
from focoos.models.rtmo.utils import (
    ChannelWiseScale,
    Scale,
    ScaleNorm,
    bbox_xyxy2cs,
    bias_init_with_prob,
    decode_bbox,
    decode_kpt_reg,
    filter_scores_and_topk,
    flatten_predictions,
    reduce_mean,
)
from focoos.nn.backbone.build import load_backbone
from focoos.nn.layers.misc import DropPath
from focoos.utils.env import TORCH_VERSION

EPS = 1e-8


class GAUEncoder(nn.Module):
    """Gated Attention Unit (GAU) Encoder.

    Args:
        in_token_dims (int): The input token dimension.
        out_token_dims (int): The output token dimension.
        expansion_factor (int, optional): The expansion factor of the
            intermediate token dimension. Defaults to 2.
        s (int, optional): The self-attention feature dimension.
            Defaults to 128.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        drop_path (float, optional): The drop path rate. Defaults to 0.0.
        act_fn (str, optional): The activation function which should be one
            of the following options:

            - 'ReLU': ReLU activation.
            - 'SiLU': SiLU activation.

            Defaults to 'SiLU'.
        bias (bool, optional): Whether to use bias in linear layers.
            Defaults to False.
        pos_enc (bool, optional): Whether to use rotary position
            embedding. Defaults to False.
        spatial_dim (int, optional): The spatial dimension of inputs

    Reference:
        `Transformer Quality in Linear Time
        <https://arxiv.org/abs/2202.10447>`_
    """

    def __init__(
        self,
        in_token_dims,
        out_token_dims,
        expansion_factor=2,
        s=128,
        eps=1e-5,
        dropout_rate=0.0,
        drop_path=0.0,
        act_fn="SiLU",
        bias=False,
        pos_enc: str = "add",
        spatial_dim: int = 1,
    ):
        super(GAUEncoder, self).__init__()
        self.s = s
        self.bias = bias
        self.pos_enc = pos_enc
        self.in_token_dims = in_token_dims
        self.spatial_dim = spatial_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        self._build_layers()

        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == "SiLU":
            assert TORCH_VERSION >= (1, 7), "SiLU activation requires PyTorch version >= 1.7"

            self.act_fn = nn.SiLU(True)
        else:
            self.act_fn = nn.ReLU(True)

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = ChannelWiseScale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_s = np.sqrt(s)
        self.dropout_rate = dropout_rate

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

    def _build_layers(self):
        self.uv = nn.Linear(self.in_token_dims, 2 * self.e + self.s, bias=self.bias)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))

    def _forward(self, x, mask=None, pos_enc: Optional[torch.Tensor] = None):
        """GAU Forward function."""
        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)
        # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
        dim = base.ndim - self.gamma.ndim + 1
        gamma = self.gamma.view(*((1,) * dim), *self.gamma.size())
        beta = self.beta.view(*((1,) * dim), *self.beta.size())
        base = base.unsqueeze(-2) * gamma + beta
        # [B, K, 2, s] -> [B, K, s], [B, K, s]
        q, k = torch.unbind(base, dim=-2)

        if self.pos_enc == "rope":
            q = (
                SinePositionalEncoding.apply_rotary_pos_enc(q, pos_enc, self.spatial_dim)
                if pos_enc is not None
                else None
            )
            k = (
                SinePositionalEncoding.apply_rotary_pos_enc(k, pos_enc, self.spatial_dim)
                if pos_enc is not None
                else None
            )
        elif self.pos_enc == "add":
            pos_enc = pos_enc.reshape(*((1,) * (q.ndim - 2)), q.size(-2), q.size(-1)) if pos_enc is not None else None
            q = q + pos_enc if pos_enc is not None else q
            k = k + pos_enc if pos_enc is not None else k

        # [B, K, s].transpose(-1, -2) -> [B, s, K]
        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.matmul(q, k.transpose(-1, -2)) if q is not None and k is not None else None

        # [B, K, K]
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if mask is not None:
            kernel = kernel * mask

        if self.dropout_rate > 0.0:
            kernel = self.dropout(kernel)

        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.matmul(kernel, v)
        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x, mask=None, pos_enc=None):
        """Forward function."""
        out = self.drop_path(self._forward(x, mask=mask, pos_enc=pos_enc))
        if self.shortcut:
            return self.res_scale(x) + out
        else:
            return out


class RTMOHeadModule(nn.Module):
    """RTMO head module for one-stage human pose estimation.

    This module implements the core prediction heads for RTMO, performing
    joint object detection and pose estimation from multi-level feature maps.
    It consists of two main branches that share the same input features:

    1. Classification Branch: Predicts object confidence scores
    2. Pose Branch: Predicts bounding boxes, keypoint offsets, and visibility

    The module uses a shared feature extraction approach where input features
    are split into two halves, with each half processed by a dedicated branch.

    Args:
        num_keypoints (int): Number of keypoints defined for one instance.
        in_channels (int): Number of channels in the input feature maps.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 1.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        cls_feat_channels (int): Number of channels in the classification score
            and objectness prediction branch. Defaults to 256.
        stacked_convs (int): Number of stacked convolution layers in each branch.
            Defaults to 2.
        num_groups (int): Number of groups for group convolutions in pose branch.
            Defaults to 8.
        channels_per_group (int): Number of channels per group in pose branch.
            Defaults to 36.
        pose_vec_channels (int): Number of output channels for pose vectors.
            If -1, uses the full pose branch output. Defaults to -1.
        featmap_strides (Sequence[int]): Strides of input feature maps.
            Defaults to [16, 32].
    """

    def __init__(
        self,
        num_keypoints: int,
        in_channels: int,
        num_classes: int = 1,
        widen_factor: float = 1.0,
        cls_feat_channels: int = 256,
        stacked_convs: int = 2,
        num_groups: int = 8,
        channels_per_group: int = 36,
        pose_vec_channels: int = -1,
        featmap_strides: Sequence[int] = [16, 32],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_feat_channels = int(cls_feat_channels * widen_factor)
        self.stacked_convs = stacked_convs

        self.featmap_strides = featmap_strides

        self.in_channels = int(in_channels * widen_factor)
        self.num_keypoints = num_keypoints

        self.num_groups = num_groups
        self.channels_per_group = int(widen_factor * channels_per_group)
        self.pose_vec_channels = pose_vec_channels

        self._init_layers()

    def _init_layers(self):
        """Initialize heads for all level feature maps."""
        self._init_cls_branch()
        self._init_pose_branch()

    def _init_cls_branch(self):
        """Initialize classification branch for all level feature maps."""
        self.conv_cls = nn.ModuleList()

        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.cls_feat_channels
                stacked_convs.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(chn, self.cls_feat_channels, 3, padding=1, bias=False)),
                                ("bn", nn.BatchNorm2d(self.cls_feat_channels, eps=0.001, momentum=0.03)),
                                ("activate", nn.SiLU()),
                            ]
                        )
                    )
                )
            self.conv_cls.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_cls = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_cls.append(nn.Conv2d(self.cls_feat_channels, self.num_classes, 1))

    def _init_pose_branch(self):
        """Initialize pose prediction branch for all level feature maps."""
        self.conv_pose = nn.ModuleList()
        out_chn = self.num_groups * self.channels_per_group
        for _ in self.featmap_strides:
            stacked_convs = []
            for i in range(self.stacked_convs * 2):
                # After splitting, each branch gets half of the input channels
                chn = self.in_channels if i == 0 else out_chn
                groups = 1 if i == 0 else self.num_groups
                stacked_convs.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(chn, out_chn, 3, padding=1, groups=groups, bias=False),
                                ),
                                ("bn", nn.BatchNorm2d(out_chn, eps=0.001, momentum=0.03)),
                                ("activate", nn.SiLU()),
                            ]
                        )
                    )
                )
            self.conv_pose.append(nn.Sequential(*stacked_convs))

        # output layers
        self.out_bbox = nn.ModuleList()
        self.out_kpt_reg = nn.ModuleList()
        self.out_kpt_vis = nn.ModuleList()
        for _ in self.featmap_strides:
            self.out_bbox.append(nn.Conv2d(out_chn, 4, 1))
            self.out_kpt_reg.append(nn.Conv2d(out_chn, self.num_keypoints * 2, 1))
            self.out_kpt_vis.append(nn.Conv2d(out_chn, self.num_keypoints, 1))

        if self.pose_vec_channels > 0:
            self.out_pose = nn.ModuleList()
            for _ in self.featmap_strides:
                self.out_pose.append(nn.Conv2d(out_chn, self.pose_vec_channels, 1))

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        bias_init = bias_init_with_prob(0.01)
        for conv_cls in self.out_cls:
            conv_cls.bias.data.fill_(bias_init)  # type: ignore

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            cls_scores (List[Tensor]): Classification scores for each level.
            bbox_preds (List[Tensor]): Bounding box predictions for each level.
            kpt_offsets (List[Tensor]): Keypoint offsets for each level.
            kpt_vis (List[Tensor]): Keypoint visibilities for each level.
            pose_feats (List[Tensor]): Pose features for each level.
        """

        cls_scores, bbox_preds = [], []
        kpt_offsets, kpt_vis = [], []
        pose_feats = []

        for i in range(len(x)):
            cls_feat, reg_feat = x[i].split(x[i].size(1) // 2, 1)  # reg_feat.shape = torch.Size([8, 64, 20, 20])

            cls_feat = self.conv_cls[i](cls_feat)
            reg_feat = self.conv_pose[i](reg_feat)  # reg_feat.shape = torch.Size([8, 144, 20, 20])

            cls_scores.append(self.out_cls[i](cls_feat))
            bbox_preds.append(self.out_bbox[i](reg_feat))
            if self.training:
                # `kpt_offsets` generates the proxy poses for positive sample selection during training
                kpt_offsets.append(self.out_kpt_reg[i](reg_feat))
            kpt_vis.append(self.out_kpt_vis[i](reg_feat))

            if self.pose_vec_channels > 0:
                pose_feats.append(self.out_pose[i](reg_feat))
            else:
                pose_feats.append(reg_feat)

        return cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_feats


class DCC(nn.Module):
    """Dynamic Coordinate Classifier (DCC) for precise keypoint localization.

    DCC is a novel approach that treats keypoint coordinate prediction as a
    classification problem rather than regression. Instead of predicting
    continuous coordinate offsets, DCC predicts discrete bins for x and y
    coordinates, enabling more precise localization through:

    1. Dynamic Bin Allocation: Bins are dynamically adjusted based on
       bounding box scale and center
    2. Sinusoidal Positional Encoding: Provides rich spatial information
       to the coordinate bins
    3. Gated Attention Unit (GAU): Enhances keypoint feature interactions
    4. Learnable Sigma: Adaptive uncertainty estimation for each keypoint

    Args:
        in_channels (int): Number of input feature channels from pose vectors.
        num_keypoints (int): Number of keypoints to predict per person.
        feat_channels (int): Number of feature channels for keypoint processing.
        num_bins (Tuple[int, int]): Number of bins for x and y coordinates.
            Defaults to (192, 256).
        spe_channels (int): Number of channels for sinusoidal positional encoding.
            Defaults to 128.
        spe_temperature (float): Temperature parameter for positional encoding.
            Defaults to 300.
        gau_cfg (dict, optional): Configuration for Gated Attention Unit.
            Defaults to dict(s=128, expansion_factor=2, dropout_rate=0.0,
            drop_path=0.0, act_fn="SiLU", pos_enc="add").
    """

    def __init__(
        self,
        in_channels: int,
        num_keypoints: int,
        feat_channels: int,
        num_bins: Tuple[int, int],
        spe_channels: int = 128,
        spe_temperature: int = 300,
        gau_cfg: Optional[dict] = dict(
            s=128, expansion_factor=2, dropout_rate=0.0, drop_path=0.0, act_fn="SiLU", pos_enc="add"
        ),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        self.feat_channels = feat_channels
        self.num_bins = num_bins
        self.gau_cfg = gau_cfg

        self.spe = SinePositionalEncoding(
            out_channels=spe_channels,
            temperature=spe_temperature,
        )
        self.spe_feat_channels = spe_channels

        self._build_layers()
        self._build_basic_bins()

    def _build_layers(self):
        """Builds layers for the model."""

        # GAU encoder
        if self.gau_cfg is not None:
            gau_cfg = self.gau_cfg.copy()
            gau_cfg["in_token_dims"] = self.feat_channels
            gau_cfg["out_token_dims"] = self.feat_channels
            self.gau = GAUEncoder(**gau_cfg)
            if gau_cfg.get("pos_enc", "none") in ("add", "rope"):
                self.pos_enc = nn.Parameter(torch.randn(self.num_keypoints, gau_cfg["s"]))

        # fully-connected layers to convert pose feats to keypoint feats
        pose_to_kpts = [
            nn.Linear(self.in_channels, self.feat_channels * self.num_keypoints),
            nn.BatchNorm1d(self.feat_channels * self.num_keypoints),
        ]
        self.pose_to_kpts = nn.Sequential(*pose_to_kpts)

        # adapter layers for dynamic encodings
        self.x_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)
        self.y_fc = nn.Linear(self.spe_feat_channels, self.feat_channels)

        # fully-connected layers to predict sigma
        self.sigma_fc = nn.Sequential(nn.Linear(self.in_channels, self.num_keypoints), nn.Sigmoid(), Scale(0.1))

    def _build_basic_bins(self):
        """Builds basic bin coordinates for x and y."""
        self.register_buffer("y_bins", torch.linspace(-0.5, 0.5, self.num_bins[1]))
        self.register_buffer("x_bins", torch.linspace(-0.5, 0.5, self.num_bins[0]))

    def _apply_softmax(self, x_hms, y_hms):
        """Apply softmax on 1-D heatmaps.

        Args:
            x_hms (Tensor): 1-D heatmap in x direction.
            y_hms (Tensor): 1-D heatmap in y direction.

        Returns:
            tuple: A tuple containing the normalized x and y heatmaps.
        """

        x_hms = x_hms.clamp(min=-5e4, max=5e4)
        y_hms = y_hms.clamp(min=-5e4, max=5e4)
        pred_x = x_hms - x_hms.max(dim=-1, keepdims=True).values.detach()
        pred_y = y_hms - y_hms.max(dim=-1, keepdims=True).values.detach()

        exp_x, exp_y = pred_x.exp(), pred_y.exp()
        prob_x = exp_x / (exp_x.sum(dim=-1, keepdims=True) + EPS)
        prob_y = exp_y / (exp_y.sum(dim=-1, keepdims=True) + EPS)

        return prob_x, prob_y

    def _get_bin_enc(self, bbox_cs, grids):
        """Calculate dynamic bin encodings for expanded bounding box.

        This function computes dynamic bin allocations and encodings based
        on the expanded bounding box center-scale (bbox_cs) and grid values.
        The process involves adjusting the bins according to the scale and
        center of the bounding box and then applying a sinusoidal positional
        encoding (spe) followed by a fully connected layer (fc) to obtain the
        final x and y bin encodings.

        Args:
            bbox_cs (Tensor): A tensor representing the center and scale of
                bounding boxes.
            grids (Tensor): A tensor representing the grid coordinates.

        Returns:
            tuple: A tuple containing the encoded x and y bins.
        """
        center, scale = bbox_cs.split(2, dim=-1)
        center = center - grids

        x_bins, y_bins = self.x_bins, self.y_bins

        # dynamic bin allocation
        x_bins = x_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = y_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 1:2] + center[..., 1:2]

        # dynamic bin encoding
        x_bins_enc = self.x_fc(self.spe(position=x_bins))
        y_bins_enc = self.y_fc(self.spe(position=y_bins))

        return x_bins_enc, y_bins_enc

    def _pose_feats_to_heatmaps(self, pose_feats, x_bins_enc, y_bins_enc):
        """Convert pose features to heatmaps using x and y bin encodings.

        This function transforms the given pose features into keypoint
        features and then generates x and y heatmaps based on the x and y
        bin encodings. If Gated attention unit (gau) is used, it applies it
        to the keypoint features. The heatmaps are generated using matrix
        multiplication of pose features and bin encodings.

        Args:
            pose_feats (Tensor): The pose features tensor.
            x_bins_enc (Tensor): The encoded x bins tensor.
            y_bins_enc (Tensor): The encoded y bins tensor.

        Returns:
            tuple: A tuple containing the x and y heatmaps.
        """

        kpt_feats = self.pose_to_kpts(pose_feats)

        kpt_feats = kpt_feats.reshape(*kpt_feats.shape[:-1], self.num_keypoints, self.feat_channels)

        if hasattr(self, "gau"):
            kpt_feats = self.gau(kpt_feats, pos_enc=getattr(self, "pos_enc", None))

        x_hms = torch.matmul(kpt_feats, x_bins_enc.transpose(-1, -2).contiguous())
        y_hms = torch.matmul(kpt_feats, y_bins_enc.transpose(-1, -2).contiguous())

        return x_hms, y_hms

    def _decode_xy_heatmaps(self, x_hms, y_hms, bbox_cs):
        """Decode x and y heatmaps to obtain coordinates.

        This function  decodes x and y heatmaps to obtain the corresponding
        coordinates. It adjusts the x and y bins based on the bounding box
        center and scale, and then computes the weighted sum of these bins
        with the heatmaps to derive the x and y coordinates.

        Args:
            x_hms (Tensor): The normalized x heatmaps tensor.
            y_hms (Tensor): The normalized y heatmaps tensor.
            bbox_cs (Tensor): The bounding box center-scale tensor.

        Returns:
            Tensor: A tensor of decoded x and y coordinates.
        """
        center, scale = bbox_cs.split(2, dim=-1)

        x_bins, y_bins = self.x_bins, self.y_bins

        x_bins = x_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = y_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 1:2] + center[..., 1:2]

        x = (x_hms * x_bins.unsqueeze(1)).sum(dim=-1)
        y = (y_hms * y_bins.unsqueeze(1)).sum(dim=-1)

        return torch.stack((x, y), dim=-1)

    def generate_target_heatmap(self, kpt_targets, bbox_cs, sigmas, areas):
        """Generate target heatmaps for keypoints based on bounding box.

        This function calculates x and y bins adjusted by bounding box center
        and scale. It then computes distances from keypoint targets to these
        bins and normalizes these distances based on the areas and sigmas.
        Finally, it uses these distances to generate heatmaps for x and y
        coordinates under assumption of laplacian error.

        Args:
            kpt_targets (Tensor): Keypoint targets tensor.
            bbox_cs (Tensor): Bounding box center-scale tensor.
            sigmas (Tensor): Learned deviation of grids.
            areas (Tensor): Areas of GT instance assigned to grids.

        Returns:
            tuple: A tuple containing the x and y heatmaps.
        """

        # calculate the error of each bin from the GT keypoint coordinates
        center, scale = bbox_cs.split(2, dim=-1)
        x_bins = self.x_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 0:1] + center[..., 0:1]
        y_bins = self.y_bins.view(*((1,) * (scale.ndim - 1)), -1) * scale[..., 1:2] + center[..., 1:2]

        dist_x = torch.abs(kpt_targets.narrow(2, 0, 1) - x_bins.unsqueeze(1))
        dist_y = torch.abs(kpt_targets.narrow(2, 1, 1) - y_bins.unsqueeze(1))

        # normalize
        areas = areas.pow(0.5).clip(min=1).reshape(-1, 1, 1)
        sigmas = sigmas.clip(min=1e-3).unsqueeze(2)
        dist_x = dist_x / areas / sigmas
        dist_y = dist_y / areas / sigmas

        hm_x = torch.exp(-dist_x / 2) / sigmas
        hm_y = torch.exp(-dist_y / 2) / sigmas

        return hm_x, hm_y

    def forward_train(self, pose_feats, bbox_cs, grids):
        """Forward pass for training.

        This function processes pose features during training. It computes
        sigmas using a fully connected layer, generates bin encodings,
        creates heatmaps from pose features, applies softmax to the heatmaps,
        and then decodes the heatmaps to get pose predictions.

        Args:
            pose_feats (Tensor): The pose features tensor.
            bbox_cs (Tensor): The bounding box in the format of center & scale.
            grids (Tensor): The grid coordinates.

        Returns:
            tuple: A tuple containing pose predictions, heatmaps, and sigmas.
        """
        sigmas = self.sigma_fc(pose_feats)
        x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
        x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc, y_bins_enc)
        x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
        pose_preds = self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)
        return pose_preds, (x_hms, y_hms), sigmas

    @torch.no_grad()
    def forward_test(self, pose_feats, bbox_cs, grids):
        """Forward pass for testing.

        This function processes pose features during testing. It generates
        bin encodings, creates heatmaps from pose features, and then decodes
        the heatmaps to get pose predictions.

        Args:
            pose_feats (Tensor): The pose features tensor.
            bbox_cs (Tensor): The bounding box in the format of center & scale.
            grids (Tensor): The grid coordinates.

        Returns:
            Tensor: Pose predictions tensor.
        """
        x_bins_enc, y_bins_enc = self._get_bin_enc(bbox_cs, grids)
        x_hms, y_hms = self._pose_feats_to_heatmaps(pose_feats, x_bins_enc, y_bins_enc)
        x_hms, y_hms = self._apply_softmax(x_hms, y_hms)
        pose_preds = self._decode_xy_heatmaps(x_hms, y_hms, bbox_cs)
        return pose_preds

    def switch_to_export(self):
        self._convert_pose_to_kpts()
        if hasattr(self, "gau"):
            self._convert_gau()
        self._convert_forward_test()

        self.export = True

    def _convert_pose_to_kpts(self):
        """Merge BatchNorm layer into Fully Connected layer.

        This function merges a BatchNorm layer into the associated Fully
        Connected layer to avoid dimension mismatch during ONNX exportation. It
        adjusts the weights and biases of the FC layer to incorporate the BN
        layer's parameters, and then replaces the original FC layer with the
        updated one.
        """
        fc, bn = self.pose_to_kpts

        # Calculate adjusted weights and biases
        std = (bn.running_var + bn.eps).sqrt()
        weight = fc.weight * (bn.weight / std).unsqueeze(1)
        bias = bn.bias + (fc.bias - bn.running_mean) * bn.weight / std

        # Update FC layer with adjusted parameters
        fc.weight.data = weight.detach()
        fc.bias.data = bias.detach()
        self.pose_to_kpts = fc

    def _convert_gau(self):
        """Reshape and merge tensors for Gated Attention Unit (GAU).

        This function pre-processes the gamma and beta tensors of the GAU and
        handles the position encoding if available. It also redefines the GAU's
        forward method to incorporate these pre-processed tensors, optimizing
        the computation process.
        """
        # Reshape gamma and beta tensors in advance
        gamma_q = self.gau.gamma[0].view(1, 1, 1, self.gau.gamma.size(-1))
        gamma_k = self.gau.gamma[1].view(1, 1, 1, self.gau.gamma.size(-1))
        beta_q = self.gau.beta[0].view(1, 1, 1, self.gau.beta.size(-1))
        beta_k = self.gau.beta[1].view(1, 1, 1, self.gau.beta.size(-1))

        # Adjust beta tensors with position encoding if available
        if hasattr(self, "pos_enc"):
            pos_enc = self.pos_enc.reshape(1, 1, *self.pos_enc.shape)
            beta_q = beta_q + pos_enc
            beta_k = beta_k + pos_enc

        gamma_q = gamma_q.detach()
        gamma_k = gamma_k.detach()
        beta_q = beta_q.detach()
        beta_k = beta_k.detach()

        @torch.no_grad()
        def _forward(self, x, *args, **kwargs):
            norm = torch.linalg.norm(x, dim=-1, keepdim=True) * self.ln.scale
            x = x / norm.clamp(min=self.ln.eps) * self.ln.g

            uv = self.uv(x)
            uv = self.act_fn(uv)

            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=-1)

            if not torch.onnx.is_in_onnx_export():
                q = base * gamma_q.to(base) + beta_q.to(base)
                k = base * gamma_k.to(base) + beta_k.to(base)
            else:
                q = base * gamma_q + beta_q
                k = base * gamma_k + beta_k
            qk = torch.matmul(q, k.transpose(-1, -2))

            kernel = torch.square(torch.nn.functional.relu(qk / self.sqrt_s))
            x = u * torch.matmul(kernel, v)

            x = self.o(x)
            return x

        self.gau._forward = types.MethodType(_forward, self.gau)

    def _convert_forward_test(self):
        """Simplify the forward test process.

        This function precomputes certain tensors and redefines the
        forward_test method for the model. It includes steps for converting
        pose features to keypoint features, performing dynamic bin encoding,
        calculating 1-D heatmaps, and decoding these heatmaps to produce final
        pose predictions.
        """
        x_bins_ = self.x_bins.view(1, 1, -1)  # type: ignore
        y_bins_ = self.y_bins.view(1, 1, -1)  # type: ignore
        dim_t = self.spe.dim_t.view(1, 1, 1, -1)

        @torch.no_grad()
        def _forward_test(self, pose_feats, bbox_cs, grids):
            # step 1: pose features -> keypoint features
            kpt_feats = self.pose_to_kpts(pose_feats)
            kpt_feats = kpt_feats.reshape(*kpt_feats.shape[:-1], self.num_keypoints, self.feat_channels)
            if hasattr(self, "gau"):
                kpt_feats = self.gau(kpt_feats)

            # step 2: dynamic bin encoding
            center, scale = bbox_cs.split(2, dim=-1)
            center = center - grids

            if not torch.onnx.is_in_onnx_export():
                x_bins = x_bins_.to(scale) * scale[..., 0:1] + center[..., 0:1]
                y_bins = y_bins_.to(scale) * scale[..., 1:2] + center[..., 1:2]
                freq_x = x_bins.unsqueeze(-1) / dim_t.to(scale)
                freq_y = y_bins.unsqueeze(-1) / dim_t.to(scale)
            else:
                x_bins = x_bins_ * scale[..., 0:1] + center[..., 0:1]
                y_bins = y_bins_ * scale[..., 1:2] + center[..., 1:2]
                freq_x = x_bins.unsqueeze(-1) / dim_t
                freq_y = y_bins.unsqueeze(-1) / dim_t

            spe_x = torch.cat((freq_x.cos(), freq_x.sin()), dim=-1)
            spe_y = torch.cat((freq_y.cos(), freq_y.sin()), dim=-1)

            x_bins_enc = self.x_fc(spe_x.float()).transpose(-1, -2).contiguous()
            y_bins_enc = self.y_fc(spe_y.float()).transpose(-1, -2).contiguous()

            # step 3: calculate 1-D heatmaps
            x_hms = torch.matmul(kpt_feats, x_bins_enc)
            y_hms = torch.matmul(kpt_feats, y_bins_enc)
            x_hms, y_hms = self._apply_softmax(x_hms, y_hms)

            # step 4: decode 1-D heatmaps through integral
            x1 = (x_hms * x_bins.unsqueeze(-2)).sum(dim=-1)
            y1 = (y_hms * y_bins.unsqueeze(-2)).sum(dim=-1)
            x = x1 + grids[..., 0:1]
            y = y1 + grids[..., 1:2]

            keypoints = torch.stack((x, y), dim=-1)

            if not torch.onnx.is_in_onnx_export():
                keypoints = keypoints.squeeze(0)
            return keypoints

        self.forward_test = types.MethodType(_forward_test, self)


class RTMOHead(nn.Module):
    """One-stage coordinate classification head for RTMO pose estimation.

    This head module performs joint object detection and pose estimation in a
    single forward pass. It combines traditional YOLO-style detection with
    novel dynamic coordinate classification (DCC) for precise keypoint
    localization. The head consists of multiple branches:

    1. Classification Branch: Predicts object confidence scores
    2. Bounding Box Branch: Predicts bounding box coordinates
    3. Keypoint Regression Branch: Predicts keypoint offsets and visibility
    4. Dynamic Coordinate Classifier (DCC): Predicts precise keypoint
       coordinates using discrete bin classification

    Args:
        num_keypoints (int): Number of keypoints to detect per person.
        num_classes (int): Number of object classes (typically 1 for person).
            Defaults to 1.
        in_channels (int): Number of input feature channels. Defaults to -1.
        widen_factor (float): Width multiplier for network channels.
            Defaults to 0.5.
        featmap_strides (Sequence[int]): Strides of feature maps from backbone.
            Defaults to [16, 32].
        featmap_strides_pointgenerator (List[int]): Strides for point generator.
            Defaults to [16, 32].
        centralize_points_pointgenerator (bool): Whether to centralize points
            in point generator. Defaults to True.
        nms_topk (int): Maximum number of detections before NMS.
            Defaults to 100000.
        nms_thr (float): NMS IoU threshold. Defaults to 1.
        score_thr (float): Confidence score threshold for filtering.
            Defaults to 0.01.
        cls_feat_channels (int): Number of channels in classification branch.
            Defaults to 256.
        pose_vec_channels (int): Number of channels for pose feature vectors.
            Defaults to 256.
        proxy_target_cc (bool): Whether to use coordinate classification
            predictions as regression targets. Defaults to False.
        bbox_padding (float): Padding factor for bounding boxes.
            Defaults to 1.25.
        overlaps_power (float): Power factor for overlap scores in loss.
            Defaults to 1.0.
        stacked_convs (int): Number of stacked convolutions in each branch.
            Defaults to 2.
        feat_channels_dcc (int): Number of feature channels in DCC module.
            Defaults to 128.
        num_bins (Tuple[int, int]): Number of bins for x and y coordinates
            in DCC. Defaults to (192, 256).
        spe_channels (int): Number of channels for sinusoidal positional
            encoding in DCC. Defaults to 128.
        gau_cfg (dict): Configuration for Gated Attention Unit in DCC.
            Defaults to dict(s=128, expansion_factor=2, dropout_rate=0.0).
    """

    def __init__(
        self,
        num_keypoints: int,
        num_classes: int = 1,
        in_channels: int = -1,
        widen_factor: float = 0.5,
        featmap_strides: Sequence[int] = [16, 32],
        featmap_strides_pointgenerator: List[int] = [16, 32],
        centralize_points_pointgenerator: bool = True,
        nms_topk: int = 100000,
        nms_thr: float = 1,
        score_thr: float = 0.01,
        # RTMOHead related params
        cls_feat_channels: int = 256,
        pose_vec_channels: int = 256,
        proxy_target_cc: bool = False,
        bbox_padding: float = 1.25,
        overlaps_power: float = 1.0,
        stacked_convs: int = 2,  #
        ## DCC related params
        feat_channels_dcc: int = 128,
        num_bins: Tuple[int, int] = (192, 256),
        spe_channels: int = 128,
        gau_cfg: dict = dict(s=128, expansion_factor=2, dropout_rate=0.0),
    ):
        super().__init__()

        self.featmap_sizes = None
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.stacked_convs = stacked_convs
        self.featmap_strides = featmap_strides
        self.featmap_strides_pointgenerator = featmap_strides_pointgenerator
        self.criterion = KeypointCriterion(num_keypoints=num_keypoints)
        self.overlaps_power = overlaps_power
        self.bbox_padding = bbox_padding
        self.nms_topk = nms_topk
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.nms = True

        # build modules
        self.head_module = RTMOHeadModule(
            num_keypoints=num_keypoints,
            in_channels=in_channels,
            pose_vec_channels=pose_vec_channels,
            num_classes=num_classes,
            widen_factor=widen_factor,
            cls_feat_channels=cls_feat_channels,
            stacked_convs=stacked_convs,
            num_groups=8,
            channels_per_group=36,
            featmap_strides=featmap_strides,
        )

        self.proxy_target_cc = proxy_target_cc
        in_channels_dcc = pose_vec_channels
        self.prior_generator = MlvlPointGenerator(
            strides=featmap_strides_pointgenerator, centralize_points=centralize_points_pointgenerator
        )
        self.assigner = SimOTAAssigner(
            dynamic_k_indicator="oks",
            oks_calculator=PoseOKS(num_keypoints=num_keypoints),
            use_keypoints_for_center=True if widen_factor == 0.5 else False,
        )
        self.dcc = DCC(
            num_keypoints=num_keypoints,
            in_channels=in_channels_dcc,
            feat_channels=feat_channels_dcc,
            num_bins=num_bins,
            spe_channels=spe_channels,
            gau_cfg=gau_cfg,
        )

    def forward(self, features: Tuple[Tensor], targets: Optional[list[KeypointTargets]] = None):
        assert isinstance(features, (tuple, list))

        if self.training and targets is not None:
            return None, self.loss(feats=features, targets=targets)
        outputs = self.predict(features)
        return outputs, None

    def losses(self, predictions, targets):
        (
            flatten_cls_scores,
            flatten_bbox_preds,
            flatten_objectness,
            flatten_kpt_offsets,
            flatten_kpt_vis,
            flatten_pose_vecs,
            flatten_bbox_decoded,
            flatten_kpt_decoded,
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
        num_total_samples = max(reduce_mean(num_pos), 1.0)  # type: ignore

        # 3. calculate loss
        extra_info = dict(num_samples=num_total_samples)
        losses = dict()
        cls_preds_all = flatten_cls_scores.view(-1, self.num_classes)

        if num_pos > 0:
            # 3.1 bbox loss
            bbox_preds = flatten_bbox_decoded.view(-1, 4)[pos_masks]
            losses["loss_bbox"] = self.criterion.get_loss("boxes", bbox_preds, bbox_targets) / num_total_samples

            # 3.2 keypoint visibility loss
            kpt_vis_preds = flatten_kpt_vis.view(-1, self.num_keypoints)[pos_masks]
            losses["loss_vis"] = self.criterion.get_loss(
                "visibility", kpt_vis_preds, vis_targets, target_weight=vis_weights
            )

            # 3.3 keypoint loss
            kpt_reg_preds = flatten_kpt_decoded.view(-1, self.num_keypoints, 2)[pos_masks]

            if hasattr(self.criterion, "loss_mle") and self.criterion.loss_mle.loss_weight > 0:
                pose_vecs = flatten_pose_vecs.view(-1, flatten_pose_vecs.size(-1))[pos_masks]
                bbox_cs = torch.cat(bbox_xyxy2cs(bbox_preds, self.bbox_padding), dim=1)
                # 'cc' refers to 'cordinate classification'
                kpt_cc_preds, pred_hms, sigmas = self.dcc.forward_train(pose_vecs, bbox_cs, pos_priors[..., :2])
                target_hms = self.dcc.generate_target_heatmap(kpt_targets, bbox_cs, sigmas, pos_areas)
                losses["loss_mle"] = self.criterion.get_loss(
                    "mle", pred_hms, target_hms, target_weight=vis_targets
                ).mean()

            if self.proxy_target_cc:
                # form the regression target using the coordinate classification predictions
                with torch.no_grad():
                    diff_cc = torch.norm(kpt_cc_preds - kpt_targets, dim=-1)
                    diff_reg = torch.norm(kpt_reg_preds - kpt_targets, dim=-1)
                    mask = (diff_reg > diff_cc).float()
                    kpt_weights_reg = vis_targets * mask
                    oks = self.assigner.oks_calculator(kpt_cc_preds, kpt_targets, vis_targets, pos_areas)
                    cls_targets = oks.unsqueeze(1)

                losses["loss_oks"] = self.criterion.get_loss(
                    "oks", kpt_reg_preds, kpt_cc_preds.detach(), target_weight=kpt_weights_reg, areas=pos_areas
                )

            else:
                losses["loss_oks"] = self.criterion.get_loss(
                    "oks", kpt_reg_preds, kpt_targets, target_weight=vis_targets, areas=pos_areas
                )

            # update the target for classification loss
            # the target for the positive grids are set to the oks calculated
            # using predictions and assigned ground truth instances
            extra_info["overlaps"] = cls_targets.mean(dim=None)  # type: ignore
            cls_targets = cls_targets.pow(self.overlaps_power).detach()
            obj_targets[pos_masks] = cls_targets.to(obj_targets)

        # 3.4 classification loss
        losses["loss_cls"] = (
            self.criterion.get_loss("classification_varifocal", cls_preds_all, obj_targets, target_weight=obj_weights)
            / num_total_samples
        )

        return losses

    def loss(self, feats: Tuple[Tensor], targets: List[KeypointTargets]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            targets (List[KeypointTargets]): The batch
                data samples

        Returns:
            dict: A dictionary of losses.
        """

        # 1. collect & reform predictions
        # feats.shape = [torch.Size([8, 128, 20, 20]), torch.Size([8, 128, 40, 40]), torch.Size([8, 128, 80, 80])]
        assert isinstance(feats, (tuple, list)), "feats must be a tuple or list"
        cls_scores, bbox_preds, kpt_offsets, kpt_vis, pose_vecs = self.head_module(
            feats
        )  # pose_vecs.shape = [(8, 144, 20, 20), (8, 144, 40, 40), (8, 144, 80, 80)]

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True
        )
        flatten_priors = torch.cat(mlvl_priors)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = flatten_predictions(cls_scores)
        flatten_bbox_preds = flatten_predictions(bbox_preds)
        flatten_objectness = (
            torch.ones_like(flatten_cls_scores).detach().narrow(-1, 0, 1) * 1e4
            if flatten_cls_scores is not None
            else None
        )
        flatten_kpt_offsets = flatten_predictions(kpt_offsets)
        flatten_kpt_vis = flatten_predictions(kpt_vis)
        flatten_pose_vecs = flatten_predictions(pose_vecs)  # flatten_pose_vecs.shape = torch.Size([8, 8400, 144])
        flatten_bbox_decoded = (
            decode_bbox(pred_bboxes=flatten_bbox_preds, priors=flatten_priors[..., :2], stride=flatten_priors[..., -1])
            if flatten_bbox_preds is not None
            else None
        )
        num_keypoints = self.num_keypoints
        flatten_kpt_decoded = (
            decode_kpt_reg(
                pred_kpt_offsets=flatten_kpt_offsets,
                priors=flatten_priors[..., :2],
                stride=flatten_priors[..., -1],
                num_keypoints=num_keypoints,
            )
            if flatten_kpt_offsets is not None
            else None
        )

        # 2. generate targets
        assert flatten_cls_scores is not None, "flatten_cls_scores is None"
        assert flatten_objectness is not None, "flatten_objectness is None"
        assert flatten_bbox_decoded is not None, "flatten_bbox_decoded is None"
        assert flatten_kpt_decoded is not None, "flatten_kpt_decoded is None"
        assert flatten_kpt_vis is not None, "flatten_kpt_vis is None"
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
            flatten_bbox_preds,
            flatten_objectness,
            flatten_kpt_offsets,
            flatten_kpt_vis,
            flatten_pose_vecs,
            flatten_bbox_decoded,
            flatten_kpt_decoded,
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
        num_priors = priors.size(0)
        gt_instances = KeypointTargets(
            boxes=data_sample.boxes,
            scores=None,
            priors=None,
            labels=data_sample.labels,
            keypoints=data_sample.keypoints,
            keypoints_visible=data_sample.keypoints_visible,
            areas=data_sample.areas,
            keypoints_visible_weights=data_sample.get("keypoints_visible_weights", None),
        )
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
            boxes=decoded_bboxes,
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
        assert gt_instances.boxes is not None, "gt_instances.bboxes is None"
        bbox_target = gt_instances.boxes[pos_assigned_gt_inds.long()]

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

    def predict(self, feats: Tuple[Tensor]) -> KeypointOutput:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor]): The multi-stage features from the backbone

        Returns:
            Predictions: A tuple containing:
                - cls_scores (List[Tensor]): Classification scores for each level
                - bbox_preds (List[Tensor]): Bounding box predictions for each level
                - kpt_vis (List[Tensor]): Keypoint visibility predictions for each level
                - pose_vecs (List[Tensor]): Pose feature vectors for each level
        """
        assert isinstance(feats, (tuple, list)), "feats must be a tuple or list"
        head_scores = self.head_module(feats)
        cls_scores, bbox_preds, _, kpt_vis, pose_vecs = head_scores

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

        # flatten predictions
        flatten_cls_scores = flatten_predictions(cls_scores) if cls_scores is not None else None
        flatten_cls_scores = flatten_cls_scores.sigmoid() if flatten_cls_scores is not None else None
        flatten_bbox_preds = flatten_predictions(bbox_preds) if bbox_preds is not None else None
        flatten_kpt_vis = flatten_predictions(kpt_vis) if kpt_vis is not None else None
        flatten_kpt_vis = flatten_kpt_vis.sigmoid() if flatten_kpt_vis is not None else None
        flatten_pose_vecs = flatten_predictions(pose_vecs) if pose_vecs is not None else None
        if flatten_pose_vecs is None:
            flatten_pose_vecs = [None] * len(feats[0])
        assert flatten_bbox_preds is not None, "flatten_bbox_preds is None"
        assert flatten_cls_scores is not None, "flatten_cls_scores is None"
        assert flatten_kpt_vis is not None, "flatten_kpt_vis is None"
        assert flatten_pose_vecs is not None, "flatten_pose_vecs is None"

        flatten_bbox_preds = decode_bbox(pred_bboxes=flatten_bbox_preds, priors=flatten_priors, stride=flatten_stride)

        all_scores = []
        all_labels = []
        all_pred_bboxes = []
        all_bbox_scores = []
        all_pred_keypoints = []
        all_keypoint_scores = []
        all_keypoints_visible = []

        for bboxes, scores, kpt_vis, pose_vecs in zip(
            flatten_bbox_preds, flatten_cls_scores, flatten_kpt_vis, flatten_pose_vecs
        ):
            score_thr = self.score_thr
            # NMS
            nms_topk = self.nms_topk
            scores, labels = scores.max(1, keepdim=True)
            if not torch.onnx.is_in_onnx_export():
                scores, _, keep_idxs_score, results = filter_scores_and_topk(
                    scores, score_thr, nms_topk, results=dict(labels=labels[:, 0])
                )
                labels = results["labels"]  # type: ignore
                bboxes = bboxes[keep_idxs_score]
                kpt_vis = kpt_vis[keep_idxs_score]
                grids = flatten_priors[keep_idxs_score]
                stride = flatten_stride[keep_idxs_score]
            else:
                # When ONNX exporting, we don't filter the scores and labels
                scores = scores.squeeze(dim=1)
                labels = labels[:, 0]
                bboxes = bboxes
                kpt_vis = kpt_vis
                grids = flatten_priors
                stride = flatten_stride

            if bboxes.numel() > 0 and self.nms:
                nms_thr = self.nms_thr
                keep_idxs_nms = nms(bboxes, scores, nms_thr)
                if nms_thr < 1.0:
                    bboxes = bboxes[keep_idxs_nms]
                    stride = stride[keep_idxs_nms]
                    labels = labels[keep_idxs_nms]
                    kpt_vis = kpt_vis[keep_idxs_nms]
                    scores = scores[keep_idxs_nms]

                if not torch.onnx.is_in_onnx_export():
                    pose_vecs = pose_vecs[keep_idxs_score][keep_idxs_nms] if pose_vecs is not None else None
                else:
                    # When ONNX exporting, we don't filter the pose_vecs by keep_idxs_score
                    pose_vecs = pose_vecs[keep_idxs_nms] if pose_vecs is not None else None

                bbox_cs = torch.cat(bbox_xyxy2cs(bboxes, self.bbox_padding), dim=1)
                grids = grids[keep_idxs_nms]
                keypoints = self.dcc.forward_test(pose_vecs, bbox_cs, grids)

            else:
                # empty prediction
                keypoints = bboxes.new_zeros((0, self.num_keypoints, 2))

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
            boxes=torch.cat(all_pred_bboxes, dim=0),
            boxes_scores=torch.cat(all_bbox_scores, dim=0),
            keypoints=torch.cat(all_pred_keypoints, dim=0),
            keypoints_scores=torch.cat(all_keypoint_scores, dim=0),
            keypoints_visible=torch.cat(all_keypoints_visible, dim=0),
        )

    def switch_to_export(self, test_cfg: Dict, device: str = "cuda"):
        """Precompute and save the grid coordinates and strides."""

        self.export = True

        # grid generator
        input_size = test_cfg.get("input_size", (640, 640))
        featmaps = []
        for s in self.featmap_strides:
            featmaps.append(torch.rand(1, 1, input_size[0] // s, input_size[1] // s))
        featmap_sizes = [fmap.shape[2:] for fmap in featmaps]

        self.mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=torch.float32, device=torch.device(device)
        )

        self.flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            self.flatten_priors.new_full((featmap_size.numel(),), stride)
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        self.flatten_stride = torch.cat(mlvl_strides)


class RTMO(BaseModelNN):
    """Real-Time Multi-Person Pose Estimation (RTMO) model.

    RTMO is a one-stage, real-time human pose estimation model that combines
    the efficiency of YOLO-style object detection with precise keypoint
    localization through dynamic coordinate classification (DCC). The model
    consists of three main components:

    1. Backbone: Extracts multi-scale features from input images
    2. Neck (HybridEncoder): Enhances features using transformer and CSP layers
    3. Head (RTMOHead): Performs joint detection and pose estimation

    The model uses a novel coordinate classification approach where keypoint
    coordinates are predicted as discrete bins rather than continuous offsets,
    enabling more precise localization while maintaining real-time performance.

    Args:
        config (RTMOConfig): Configuration object containing all model parameters
            including backbone, neck, and head configurations.

    Architecture:
        - Backbone: Configurable backbone network for feature extraction
        - HybridEncoder: Combines transformer and CSP layers for feature enhancement
        - RTMOHead: One-stage head with:
            - Classification branch for object detection
            - Regression branch for bounding box prediction
            - Keypoint regression branch for pose estimation
            - Dynamic Coordinate Classifier (DCC) for precise keypoint localization

    Key Features:
        - Real-time inference with high accuracy
        - Dynamic coordinate classification for precise keypoint localization
        - Gated Attention Unit (GAU) for enhanced feature processing
        - Multi-scale feature fusion
        - End-to-end training without intermediate supervision

    Example:
        ```python
        from focoos.models.rtmo.config import RTMOConfig
        from focoos.models.rtmo.modelling import RTMO

        # Create configuration
        config = RTMOConfig(
            num_keypoints=17,  # COCO keypoints
            backbone_config=backbone_cfg,
            # ... other config parameters
        )

        # Initialize model
        model = RTMO(config)

        # Forward pass
        images = torch.randn(1, 3, 640, 640)
        outputs = model(images)

        # Access predictions
        keypoints = outputs.keypoints  # Shape: (batch, num_instances, num_keypoints, 2)
        scores = outputs.scores  # Shape: (batch, num_instances)
        boxes = outputs.boxes  # Shape: (batch, num_instances, 4)
        ```
    """

    def __init__(self, config: RTMOConfig):
        assert len(config.featmap_strides) == len(config.featmap_strides_pointgenerator), (
            "featmap_strides and featmap_strides_pointgenerator must have the same length"
        )
        super().__init__(config)
        self.config = config

        self.backbone = load_backbone(self.config.backbone_config)

        self.neck = HybridEncoder(
            shape_specs=self.backbone.output_shape(),
            transformer_embed_dims=self.config.transformer_embed_dims,
            transformer_num_heads=self.config.transformer_num_heads,
            transformer_feedforward_channels=self.config.transformer_feedforward_channels,
            transformer_dropout=self.config.transformer_dropout,
            transformer_encoder_layers=self.config.transformer_encoder_layers,
            csp_layers=self.config.csp_layers,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            pe_temperature=self.config.pe_temperature,
            widen_factor=self.config.widen_factor,
            spe_learnable=self.config.spe_learnable,
            output_indices=self.config.output_indices,
        )

        self.head = RTMOHead(
            num_keypoints=self.config.num_keypoints,
            num_classes=self.config.num_classes,
            in_channels=self.config.in_channels,
            widen_factor=self.config.widen_factor,
            pose_vec_channels=self.config.pose_vec_channels,
            cls_feat_channels=self.config.cls_feat_channels,
            stacked_convs=self.config.stacked_convs,
            featmap_strides=self.config.featmap_strides,
            featmap_strides_pointgenerator=self.config.featmap_strides_pointgenerator,
            centralize_points_pointgenerator=self.config.centralize_points_pointgenerator,
            overlaps_power=self.config.overlaps_power,
            # DCC related params
            feat_channels_dcc=self.config.feat_channels_dcc,
            num_bins=self.config.num_bins,
            spe_channels=self.config.spe_channels,
            # GAU related params
            gau_cfg=dict(
                s=self.config.gau_s,
                expansion_factor=self.config.gau_expansion_factor,
                dropout_rate=self.config.gau_dropout_rate,
                pos_enc="add",
                act_fn="SiLU",
            ),
            score_thr=self.config.score_thr,
            nms_topk=self.config.nms_topk,
            nms_thr=self.config.nms_thr,
        )

        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def dtype(self):
        return self.pixel_mean.dtype

    def forward(self, images: torch.Tensor, targets: list[KeypointTargets] = []) -> RTMOModelOutput:
        images = (images - self.pixel_mean) / self.pixel_std  # type: ignore
        features = self.backbone(images)
        features = self.neck(features)

        outputs, losses = self.head(features, targets)
        if self.training:
            assert targets is not None and len(targets) > 0, "targets should not be None or empty - training mode"
            return RTMOModelOutput(
                scores=torch.zeros(0, 0, 0),
                labels=torch.zeros(0, 0, 0),
                boxes=torch.zeros(0, 0, 4),
                boxes_scores=torch.zeros(0, 0, 0),
                keypoints=torch.zeros(0, 0, 0),
                keypoints_scores=torch.zeros(0, 0, 0),
                keypoints_visible=torch.zeros(0, 0, 0),
                loss=losses,
            )

        return RTMOModelOutput(
            scores=outputs.scores,
            labels=outputs.labels,
            boxes=outputs.boxes,
            boxes_scores=outputs.boxes_scores,
            keypoints=outputs.keypoints,
            keypoints_scores=outputs.keypoints_scores,
            keypoints_visible=outputs.keypoints_visible,
            loss=None,
        )

    def switch_to_export(self, test_cfg: Dict, device: str = "cuda"):
        self.neck.switch_to_export(test_cfg=test_cfg, device=device)
        self.head.switch_to_export(test_cfg=test_cfg, device=device)
        self.head.dcc.switch_to_export()
