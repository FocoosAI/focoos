import math
from typing import Dict, Optional, Union

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from focoos.models.fai_mf.config import MaskFormerConfig
from focoos.models.fai_mf.loss import MaskHungarianMatcher, SetCriterion
from focoos.models.fai_mf.ports import MaskFormerModelOutput, MaskFormerTargets
from focoos.models.fai_mf.processor import MaskFormerProcessor
from focoos.models.fai_model import BaseModelNN
from focoos.nn.backbone.base import BaseBackbone
from focoos.nn.backbone.build import load_backbone
from focoos.nn.layers.base import MLP
from focoos.nn.layers.conv import Conv2d
from focoos.ports import DatasetEntry
from focoos.structures import Instances
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                f"when normalize is set,scale should be provided and in float or int type, found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # use view as mmdet instead of flatten for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        ffn_type="standard",
    ):
        super().__init__()

        assert ffn_type in [
            "standard",
            "convnext",
        ], "FFN can be of 'standard' or 'convnext' type"
        self.ffn_type = ffn_type
        self.normalize_before = normalize_before

        if self.ffn_type == "standard":
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm = nn.LayerNorm(d_model)

            self.activation = nn.ReLU()

            self._reset_parameters()
        elif self.ffn_type == "convnext":
            if not normalize_before:
                logger.warning(
                    "Pre-normalizazion is applied by default with convnext FFN layer since "
                    "it supports only pre-normalization."
                )
                self.normalize_before = True

            layer_scale_init_value = 1e-6
            drop_path = 0.0

            self.dwconv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)  # depthwise conv
            self.norm = LayerNorm(d_model, eps=1e-6)
            # pointwise/1x1 convs, implemented with linear layers
            self.pwconv1 = nn.Linear(d_model, dim_feedforward)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(dim_feedforward, d_model)
            self.gamma = (
                nn.Parameter(layer_scale_init_value * torch.ones(d_model), requires_grad=True)
                if layer_scale_init_value > 0
                else None
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        if self.ffn_type == "standard":
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout(tgt2)
            return tgt
        elif self.ffn_type == "convnext":
            # tgt shape is -> QxNxC (Q: num. of query - N: batch size - C: num. of channels)
            Q, N, C = tgt.shape
            H = W = int(Q**0.5)
            x = tgt.permute(1, 2, 0).reshape(N, C, H, W)

            tgt2 = self.dwconv(x)
            tgt2 = tgt2.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            tgt2 = self.norm(tgt2)
            tgt2 = self.pwconv1(tgt2)
            tgt2 = self.act(tgt2)
            tgt2 = self.pwconv2(tgt2)
            if self.gamma is not None:
                tgt2 = self.gamma * tgt2
            tgt2 = tgt2.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

            tgt = tgt + self.drop_path(torch.flatten(tgt2, 2, 3).permute(2, 0, 1))
            return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class PredictionHeads(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_classes,
        mask_dim,
        num_heads,
        mask_classification,
        use_attn_masks,
    ):
        super().__init__()
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        # output FFNs
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_classifier = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.num_heads = num_heads
        self.use_attn_masks = use_attn_masks
        self.num_classes = num_classes

    def reset_classifier(self, num_classes: Optional[int] = None):
        _num_classes = num_classes if num_classes else self.num_classes
        self.classifier = nn.Linear(self.classifier.in_features, _num_classes + 1).to(self.classifier.weight.device)

    def forward(self, x, mask_features, sizes=None, process=True):
        decoder_output = self.decoder_norm(x)
        decoder_output = decoder_output.transpose(0, 1)
        # just a linear layer [hidden, n_class + 1]
        outputs_class = self.classifier(decoder_output)
        mask_embed = self.mask_classifier(decoder_output)  # MLP with 3 linear layer
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if sizes is not None:
            if self.use_attn_masks:
                attn_masks = []
                if not isinstance(sizes, list):
                    sizes = [sizes]
                for attn_mask_target_size in sizes:
                    # NOTE: prediction is of higher-resolution
                    # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
                    attn_mask = F.interpolate(
                        outputs_mask,
                        size=attn_mask_target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    if process:
                        # must use bool type
                        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
                        attn_mask = attn_mask.flatten(2) < 0
                    attn_mask = attn_mask.detach() if self.training else attn_mask
                    attn_masks.append(attn_mask)
            else:
                attn_masks = None
            return outputs_class, outputs_mask, attn_masks
        else:
            return outputs_class, outputs_mask

    def forward_class_only(self, x):
        decoder_output = self.decoder_norm(x)
        decoder_output = decoder_output.transpose(0, 1)
        # just a linear layer [hidden, n_class + 1]
        outputs_class = self.classifier(decoder_output)
        return outputs_class


class FPN(nn.Module):
    def __init__(
        self,
        backbone: BaseBackbone,
        feat_dim: int,
        out_dim: int,
        # norm: str = "BN",
    ):
        """
        Args:
            backbone: basic backbones to extract features from images
            feat_dim: number of output channels for the intermediate conv layers.
            out_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.backbone = backbone
        self.input_shape = sorted(backbone.output_shape().items(), key=lambda x: x[1].stride)  # type: ignore
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in self.input_shape]
        # starting from "res2" to "res5"
        self.in_channels = [v.channels for k, v in self.input_shape]
        self.in_strides = [v.stride for k, v in self.input_shape]
        self.out_dim = out_dim
        self.feat_dim = feat_dim

        feature_channels = [v.channels for k, v in self.input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = False
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = nn.BatchNorm2d(feat_dim)
                output_conv = Conv2d(
                    in_channels,
                    feat_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = nn.BatchNorm2d(feat_dim)
                output_norm = nn.BatchNorm2d(feat_dim)

                lateral_conv = Conv2d(
                    in_channels,
                    feat_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv2d(
                    feat_dim,
                    feat_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = out_dim
        self.mask_features = Conv2d(
            feat_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return self.backbone.padding_constraints

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        return self.forward_features(features)

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < 3:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), multi_scale_features


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        num_scales: int = 3,
        pre_norm: bool,
        enforce_input_project: bool,
        use_attn_masks: bool = True,
    ):
        super().__init__()

        self.use_attn_masks = use_attn_masks

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        assert 0 < num_scales <= 3, "num_scales must between 1 and 3"
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.num_scales = num_scales
        self.num_classes = num_classes
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = min(self.num_scales, dec_layers)
        # self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(min(self.num_feature_levels, dec_layers)):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.forward_prediction_heads = PredictionHeads(
            hidden_dim,
            num_classes,
            out_dim,
            nheads,
            mask_classification=True,
            use_attn_masks=use_attn_masks,
        )

    def reset_classifier(self, num_classes: Optional[int] = None):
        self.forward_prediction_heads.reset_classifier(num_classes if num_classes else self.num_classes)

    def forward(self, x, mask_features, targets=None, mask=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        x = x[: self.num_scales]

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            # + self.level_embed(i)[None, :, None])
            src.append(self.input_proj[i](x[i]).flatten(2))

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, sizes=size_list[0]
        )
        attn_mask = attn_mask[0] if self.use_attn_masks else None
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            #  if on a mask is all True (no pixel active), use cross attention (put every pixel at False)
            # B N # 1 if any False, 0 if all True
            if attn_mask is not None:
                m_mask = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
                attn_mask = attn_mask.type_as(output) * m_mask.type_as(output)
                attn_mask = attn_mask.bool().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,  # query
                src[level_index],  # key and value
                memory_mask=attn_mask if self.use_attn_masks else None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                sizes=size_list[(i + 1) % self.num_feature_levels],
            )
            attn_mask = attn_mask[0] if self.use_attn_masks else None
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class,
                predictions_mask,
            ),
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]


class MaskFormerHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_dim: int,
        num_classes: int,
        criterion: nn.Module,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        cls_sigmoid=False,
    ):
        """
        Args:
            num_classes: number of classes to predict
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.ignore_value = ignore_value
        self.criterion = criterion

        self.predictor = transformer_predictor

        self.num_classes = num_classes
        self.metadata = None
        self.cls_sigmoid = cls_sigmoid
        self.mask_threshold = 0

    def reset_classifier(self, num_classes: Optional[int] = None):
        self.predictor.reset_classifier(num_classes if num_classes else self.num_classes)

    def layers(self, features, targets=None, mask=None):
        mask_features, multi_scale_features = features
        predictions = self.predictor(multi_scale_features, mask_features, targets=targets, mask=mask)

        return predictions

    def forward(self, features, targets: list[MaskFormerTargets] = []):
        outputs = self.layers(features, targets=targets)

        loss = None
        if targets is not None and len(targets) > 0:
            loss = self.losses(outputs, targets)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]

        if self.cls_sigmoid:
            mask_cls = mask_cls.sigmoid()[..., :-1]
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()

        return (mask_cls, mask_pred), loss

    def losses(self, predictions, targets):
        if isinstance(predictions, tuple):
            predictions, mask_dict = predictions
            losses = self.criterion(predictions, targets, mask_dict)
        else:
            losses = self.criterion(predictions, targets)

        return losses


class FAIMaskFormer(BaseModelNN):
    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self._export = False
        self.config = config
        accepted_postprocessing_types = ["semantic", "instance", "panoptic"]
        if self.config.postprocessing_type not in accepted_postprocessing_types:
            raise ValueError(
                f"Invalid postprocessing type: {self.config.postprocessing_type}. Must be one of: {accepted_postprocessing_types}"
            )

        backbone = load_backbone(self.config.backbone_config)

        self.pixel_decoder = FPN(
            backbone=backbone,
            feat_dim=self.config.pixel_decoder_feat_dim,
            out_dim=self.config.pixel_decoder_out_dim,
        )
        self.head = MaskFormerHead(
            in_channels=self.config.transformer_predictor_out_dim,
            out_dim=self.config.head_out_dim,
            num_classes=self.config.num_classes,
            ignore_value=255,
            criterion=SetCriterion(
                num_classes=self.config.num_classes,
                matcher=MaskHungarianMatcher(
                    cost_class=self.config.matcher_cost_class,
                    cost_mask=self.config.matcher_cost_mask,
                    cost_dice=self.config.matcher_cost_dice,
                    num_points=self.config.criterion_num_points,
                    cls_sigmoid=self.config.cls_sigmoid,
                ),
                weight_dict={
                    "loss_ce": self.config.weight_dict_loss_ce,
                    "loss_mask": self.config.weight_dict_loss_mask,
                    "loss_dice": self.config.weight_dict_loss_dice,
                },
                deep_supervision=self.config.criterion_deep_supervision,
                eos_coef=self.config.criterion_eos_coef,
                losses=["labels", "masks"],
                num_points=self.config.criterion_num_points,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                loss_class_type="ce_loss" if not self.config.cls_sigmoid else "bce_loss",
                cls_sigmoid=self.config.cls_sigmoid,
            ),
            transformer_predictor=MultiScaleMaskedTransformerDecoder(
                in_channels=self.config.pixel_decoder_out_dim,
                out_dim=self.config.transformer_predictor_out_dim,
                num_classes=self.config.num_classes,
                hidden_dim=self.config.transformer_predictor_hidden_dim,  # this is query dim
                num_queries=self.config.num_queries,
                nheads=8,
                dim_feedforward=self.config.transformer_predictor_dim_feedforward,
                dec_layers=self.config.transformer_predictor_dec_layers,
                num_scales=3,
                pre_norm=True,
                enforce_input_project=True,
                use_attn_masks=True,
            ),
            cls_sigmoid=True,
        )
        self.resolution = self.config.resolution
        self.top_k = self.config.num_queries
        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)
        self.size_divisibility = self.config.size_divisibility
        self.num_classes = self.config.num_classes
        self.processor = MaskFormerProcessor(self.config)

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
            list[DatasetEntry],
        ],
    ) -> MaskFormerModelOutput:
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
        (logits, masks), losses = self.head(features, targets)

        return MaskFormerModelOutput(logits=logits, masks=masks, loss=losses)

    def post_process(self, outputs, batched_inputs) -> list[dict[str, Union[Instances, torch.Tensor]]]:
        """
        Post-process the outputs of the model.
        This function is used in the evaluation phase to convert raw outputs to Instances.
        """
        return self.processor.eval_postprocess(outputs, batched_inputs)
