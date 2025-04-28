import copy
import logging
import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from PIL import Image
from scipy.optimize import linear_sum_assignment

from focoos.data.mappers.detection_dataset_mapper import DetectionDatasetDict
from focoos.models.fai_model import BaseModelNN
from focoos.models.fai_rtdetr.config import RTDetrConfig
from focoos.models.fai_rtdetr.processor import RTDetrProcessor
from focoos.models.fai_rtdetr.rtdetr_ports import RTDETRModelOutput, RTDETRTargets
from focoos.nn.backbone.base import BaseBackbone
from focoos.nn.backbone.build import load_backbone
from focoos.nn.layers.base import MLP
from focoos.nn.layers.conv import Conv2d
from focoos.nn.layers.deformable import ms_deform_attn_core_pytorch
from focoos.nn.layers.functional import inverse_sigmoid
from focoos.nn.layers.transformer import TransformerEncoder, TransformerEncoderLayer
from focoos.structures import Instances
from focoos.utils.box import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from focoos.utils.distributed.comm import get_world_size
from focoos.utils.distributed.dist import is_dist_available_and_initialized

logger = logging.getLogger(__name__)


def get_activation(act: str, inpace: bool = True):
    """get activation"""
    act = act.lower()

    if act == "silu":
        m = nn.SiLU()

    elif act == "relu":
        m = nn.ReLU()

    elif act == "leaky_relu":
        m = nn.LeakyReLU()

    elif act == "silu":
        m = nn.SiLU()

    elif act == "gelu":
        m = nn.GELU()

    elif act is None:
        m = nn.Identity()

    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError("")

    if hasattr(m, "inplace"):
        m.inplace = inpace  # type: ignore

    return m


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def _fuse(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1, bias=True)
            kernel, bias = self.get_equivalent_kernel_bias()
            self.conv.weight.data = kernel
            if self.conv.bias is not None:
                self.conv.bias.data = bias
            # self.__delattr__('conv1')
            # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        assert running_var is not None, "Error: running_var is None"
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class HybridEncoder(nn.Module):
    def __init__(
        self,
        backbone: BaseBackbone,
        feat_dim: int,
        out_dim: int,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        resolution=640,
    ):
        super().__init__()

        self.backbone = backbone
        self.input_shape = sorted(backbone.output_shape().items(), key=lambda x: x[1].stride)  # type: ignore

        # starting from "res2" to "res5"
        self.in_channels = [v.channels for k, v in self.input_shape]
        self.in_strides = [v.stride for k, v in self.input_shape]
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        if resolution is not None:
            if isinstance(resolution, int):
                self.eval_spatial_size = (resolution, resolution)
            else:
                self.eval_spatial_size = resolution
        else:
            self.eval_spatial_size = None

        self.in_features = ["res3", "res4", "res5"]
        self.in_channels = self.in_channels[1:]
        self.in_strides = self.in_strides[1:]

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in self.in_channels:  # from res3 to res5
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, feat_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(feat_dim),
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            feat_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )

        self.encoder = nn.ModuleList(
            [TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))]
        )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(feat_dim, feat_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    feat_dim * 2,
                    feat_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(ConvNormLayer(feat_dim, feat_dim, 3, 1, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    feat_dim * 2,
                    feat_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        self.mask_dim = out_dim
        self.mask_features = Conv2d(
            feat_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self._reset_parameters()

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return self.backbone.padding_constraints

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.in_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.feat_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """ """
        # FIXME: using int(w) and int(h) leads to a traceble model without dynamic axes!
        # The entire function may be substituted by PositionalEmbeddingSine
        # from anyma.models.layers.position_encoding import PositionEmbeddingSine
        # pe_layer = PositionEmbeddingSine(N_steps, normalize=False/True)
        # pos_embed = pe_layer(proj_feats[enc_ind]).flatten(2).transpose(1, 2)
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)

        feats = [features[f] for f in self.in_features]
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.feat_dim, self.pe_temperature).to(
                        src_flatten.device
                    )
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None)
                    if pos_embed is not None:
                        pos_embed = pos_embed.to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.feat_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):  # 2, 1
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, size=feat_low.shape[-2:], mode="bilinear")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)
        # Inner out is: [[bs, c, h/8, h/8], [bs, c, h/16, h/16], [bs, c, h/32, h/32]]
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = F.interpolate(feat_low, size=feat_height.shape[-2:], mode="bilinear")
            downsample_feat = self.downsample_convs[idx](downsample_feat)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return self.mask_features(outs[0]), outs[::-1]


class DETRHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_dim: int,
        num_classes: int,
        criterion: nn.Module,
        # extra parameters
        transformer_predictor: nn.Module,
        mask_on=False,
        cls_sigmoid=True,
    ):
        """
        Args:
            num_classes: number of classes to predict
            loss_weight: loss weight
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.criterion = criterion
        self.predictor = transformer_predictor
        self.cls_sigmoid = cls_sigmoid

        self.num_classes = num_classes
        self.mask_on = mask_on

    def layers(self, features, targets: list[RTDETRTargets] = []):
        _, multi_scale_features = features
        predictions = self.predictor(feats=multi_scale_features, targets=targets)

        return predictions

    def forward(self, features, targets: list[RTDETRTargets] = []):
        outputs = self.layers(features, targets)

        if self.training:
            return None, self.losses(outputs, targets)
        else:
            boxes = outputs["pred_boxes"]
            boxes = box_cxcywh_to_xyxy(boxes)

            logits = outputs["pred_logits"]
            if self.cls_sigmoid:
                logits = F.sigmoid(logits)
            else:
                logits = F.softmax(logits, -1)[:, :, :-1]

            return (logits, boxes), {}

    def losses(self, predictions, targets: list[RTDETRTargets]):
        losses = self.criterion(predictions, targets)

        return losses


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict,
        losses: list[str],
        eos_coef: float = 0.1,
        num_points: int = 0,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.0,
        deep_supervision: bool = True,
        use_focal: bool = False,  # deprecated
        loss_class_type: str = "ce_loss",  # deprecated
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        cls_sigmoid: bool = False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.deep_supervision = deep_supervision

        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.loss_class_type = loss_class_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cls_sigmoid = cls_sigmoid

    def loss_labels_vfl(self, outputs, targets: list[RTDETRTargets], indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t.boxes[i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        ious = torch.diag(ious).detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t.labels[J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.focal_alpha * pred_score.pow(self.focal_gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        losses = {"loss_vfl": loss}
        # losses["class_error"] = 1 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets: list[RTDETRTargets], indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v.labels) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets: list[RTDETRTargets], indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t.boxes[i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets: list[RTDETRTargets], indices, num_masks, **kwargs):
        loss_map = {
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "vfl": self.loss_labels_vfl,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, **kwargs)

    def forward(self, outputs, targets: list[RTDETRTargets]):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t.labels) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_masks)
            for k in list(l_dict.keys()):
                if k in self.weight_dict:
                    l_dict[k] *= self.weight_dict[k]
                losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs and self.deep_supervision:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    for k in list(l_dict.keys()):
                        if k in self.weight_dict:
                            l_dict[k] *= self.weight_dict[k]

                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}

                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if "dn_aux_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_masks = num_masks * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = (
            dn_meta["dn_positive_idx"],
            dn_meta["dn_num_group"],
        )
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices


class BoxHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets: list[RTDETRTargets]):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a RTDETRTargets containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v.labels for v in targets])
        tgt_bbox = torch.cat([v.boxes for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        # FIXME This linear sum assignment is done on CPU. Can we use GPU?

        sizes = [len(v.boxes) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_bbox: {}".format(self.cost_bbox),
            "cost_giou: {}".format(self.cost_giou),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
    ):
        """
        Multi-Scale Deformable Attention Module
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
        )
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = ms_deform_attn_core_pytorch

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dropout=0.0,
        activation="relu",
        dim_feedforward=1024,
        n_levels=4,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.activation(self.linear1(tgt)))

    def forward(
        self,
        tgt,
        reference_points,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        # if attn_mask is not None:
        #     attn_mask = torch.where(
        #         attn_mask.to(torch.bool),
        #         torch.zeros_like(attn_mask),
        #         torch.full_like(attn_mask, float('-inf'), dtype=tgt.dtype))

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        ref_points = ref_points_detach  # Initialize ref_points before the loop, this is just for linter.
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class RTDETRTransformerPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim,  # not used since no mask_classification yet
        mask_on=True,
        *,
        num_classes: int,
        sigmoid: bool = True,
        hidden_dim: int,
        num_queries: int = 300,
        nhead: int = 8,
        dec_layers: int = 6,
        dim_feedforward: int = 1024,
        position_embed_type: str = "sine",
        num_scales: int = 3,
        num_decoder_points: int = 4,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        eval_idx: int = -1,
    ):
        super().__init__()
        assert position_embed_type in [
            "sine",
            "learned",
        ], f"ValueError: position_embed_type not supported {position_embed_type}!"

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.mask_on = mask_on
        self.sigmoid = sigmoid
        if self.mask_on:
            raise NotImplementedError("mask classification not supported yet!")

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_levels = num_scales
        assert self.num_levels == 3, "num_scales should equal to 3"
        num_classes = num_classes if self.sigmoid else num_classes + 1
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.dec_layers = dec_layers
        if resolution is not None:
            if isinstance(resolution, int):
                self.eval_spatial_size = (resolution, resolution)
            else:
                self.eval_spatial_size = resolution
        else:
            self.eval_spatial_size = None
        self.eps = 1e-2
        # !fixme: generalize to any feat stride.
        self.feat_strides = [32, 16, 8]

        # backbone feature projection
        self.input_proj = self._build_input_proj_layer([in_channels] * num_scales)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            n_levels=num_scales,
            n_points=num_decoder_points,
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, dec_layers, eval_idx)

        # decoder embedding
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
            ),
        )
        self.enc_score_classifier = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_classifier = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_classifier = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(dec_layers)])
        self.dec_bbox_classifier = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(dec_layers)]
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(num_classes=num_classes + 1)

    def _reset_parameters(self, num_classes):
        bias = bias_init_with_prob(1 / num_classes)

        init.constant_(self.enc_score_classifier.bias, bias)
        init.constant_(self.enc_bbox_classifier.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_classifier.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_classifier, self.dec_bbox_classifier):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False),
                            ),
                            (
                                "norm",
                                nn.BatchNorm2d(
                                    self.hidden_dim,
                                ),
                            ),
                        ]
                    )
                )
            )
        return input_proj

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [
            0,
        ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)] for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(  # 40x40 -> 00000, 11111, 2222, 3333, ...,
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)  # index matrix # 40x40x2 e.g. grid_xy[i,j] = [j, i]
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # normalized coords # 1x40x40x2
            # reverse the order of level to match the order of spatial_shapes
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** (2 - lvl))
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))  # This is the inverse of sigmoid.
        anchors = torch.where(valid_mask, anchors, 0.0)

        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes):
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_classifier(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_classifier(output_memory) + anchors

        if self.sigmoid:
            scores = enc_outputs_class.max(-1).values
        else:
            scores = F.softmax(enc_outputs_class, dim=-1)[:, :, :-1].max(-1).values

        _, topk_ind = torch.topk(scores, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]),
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]),
        )

        # extract region features
        target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        target = target.detach()

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, feats, targets: list[RTDETRTargets] = []):
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        (
            target,
            init_ref_points_unact,
            enc_topk_bboxes,
            enc_topk_logits,
        ) = self._get_decoder_input(memory, spatial_shapes)

        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_classifier,
            self.dec_score_classifier,
            self.query_pos_head,
        )

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training:
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["aux_outputs"].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]


class FAIRTDetr(BaseModelNN):
    def __init__(self, config: RTDetrConfig):
        super().__init__(config)
        self._export = False
        self.config = config

        backbone = load_backbone(self.config.backbone_config)

        self.pixel_decoder = HybridEncoder(
            backbone=backbone,
            feat_dim=self.config.pixel_decoder_feat_dim,
            out_dim=self.config.pixel_decoder_out_dim,
            expansion=self.config.pixel_decoder_expansion,
            dropout=self.config.pixel_decoder_dropout,
            nhead=self.config.pixel_decoder_nhead,
            dim_feedforward=self.config.pixel_decoder_dim_feedforward,
            num_encoder_layers=self.config.pixel_decoder_num_encoder_layers,
            resolution=self.config.resolution,
        )
        self.head = DETRHead(
            in_channels=self.config.transformer_predictor_out_dim,
            out_dim=self.config.head_out_dim,
            num_classes=self.config.num_classes,
            criterion=SetCriterion(
                num_classes=self.config.num_classes,
                matcher=BoxHungarianMatcher(
                    cost_class=self.config.matcher_cost_class,
                    cost_bbox=self.config.matcher_cost_bbox,
                    cost_giou=self.config.matcher_cost_giou,
                    use_focal_loss=self.config.matcher_use_focal_loss,
                    alpha=self.config.matcher_alpha,
                    gamma=self.config.matcher_gamma,
                ),
                weight_dict={
                    "loss_vfl": self.config.weight_dict_loss_vfl,
                    "loss_bbox": self.config.weight_dict_loss_bbox,
                    "loss_giou": self.config.weight_dict_loss_giou,
                },
                losses=self.config.criterion_losses,
                eos_coef=self.config.criterion_eos_coef,
                num_points=self.config.criterion_num_points,
                focal_alpha=self.config.criterion_focal_alpha,
                focal_gamma=self.config.criterion_focal_gamma,
            ),
            transformer_predictor=RTDETRTransformerPredictor(
                in_channels=self.config.pixel_decoder_out_dim,
                out_dim=self.config.transformer_predictor_out_dim,
                num_classes=self.config.num_classes,
                hidden_dim=self.config.transformer_predictor_hidden_dim,
                mask_on=False,
                sigmoid=True,
                num_queries=self.config.num_queries,
                nhead=self.config.transformer_predictor_nhead,
                dec_layers=self.config.transformer_predictor_dec_layers,
                dim_feedforward=self.config.transformer_predictor_dim_feedforward,
                resolution=self.config.resolution,
            ),
            mask_on=False,
            cls_sigmoid=True,
        )
        self.resolution = self.config.resolution
        self.top_k = self.config.num_queries
        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)
        self.size_divisibility = self.config.size_divisibility
        self.num_classes = self.config.num_classes
        self.processor = RTDetrProcessor()

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(
        self,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
            list[DetectionDatasetDict],
        ],
    ):
        images, targets = self.processor.preprocess(
            inputs,
            training=self.training,
            device=self.device,  # type: ignore
            dtype=self.pixel_mean.dtype,  # type: ignore
            size_divisibility=self.size_divisibility,
            padding_constraints=self.pixel_decoder.padding_constraints,
            resolution=self.resolution,
        )
        images = (images - self.pixel_mean) / self.pixel_std  # type: ignore

        features = self.pixel_decoder(images)
        outputs, losses = self.head(features, targets)

        if self.training:
            assert targets is not None and len(targets) > 0, "targets should not be None or empty - training mode"
            return losses
        else:
            return RTDETRModelOutput(logits=outputs[0], boxes=outputs[1])

    def post_process(self, outputs, batched_inputs) -> list[dict[str, Instances]]:
        """
        Post-process the outputs of the model.
        This function is used in the evaluation phase to convert raw outputs to Instances.
        """
        return self.processor.eval_postprocess(outputs, batched_inputs, top_k=self.top_k)
