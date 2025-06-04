from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from focoos.models.fai_mf.config import MaskFormerConfig
from focoos.models.fai_mf.loss import MaskHungarianMatcher, SetCriterion
from focoos.models.fai_mf.ports import MaskFormerModelOutput, MaskFormerTargets
from focoos.models.focoos_model import BaseModelNN
from focoos.nn.backbone.base import BaseBackbone
from focoos.nn.backbone.build import load_backbone
from focoos.nn.layers.base import MLP
from focoos.nn.layers.conv import Conv2d
from focoos.nn.layers.position_encoding import PositionEmbeddingSine
from focoos.nn.layers.transformer import (
    CrossAttentionLayer,
    FFNLayer,
    SelfAttentionLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionHeads(nn.Module):
    """Prediction heads for mask classification and segmentation."""

    def __init__(
        self,
        hidden_dim,
        num_classes,
        mask_dim,
        num_heads,
        mask_classification,
        use_attn_masks,
    ):
        """Initialize prediction heads.

        Args:
            hidden_dim: Dimension of hidden features
            num_classes: Number of classes to predict
            mask_dim: Dimension of mask features
            num_heads: Number of attention heads
            mask_classification: Whether to perform mask classification
            use_attn_masks: Whether to use attention masks
        """
        super().__init__()
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        # output FFNs
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_classifier = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.num_heads = num_heads
        self.use_attn_masks = use_attn_masks
        self.num_classes = num_classes

    def reset_classifier(self, num_classes: Optional[int] = None):
        """Reset the classifier with a new number of classes.

        Args:
            num_classes: New number of classes (optional)
        """
        _num_classes = num_classes if num_classes else self.num_classes
        self.classifier = nn.Linear(self.classifier.in_features, _num_classes + 1).to(self.classifier.weight.device)

    def forward(self, x, mask_features, sizes=None, process=True):
        """Forward pass for prediction heads.

        Args:
            x: Input features
            mask_features: Mask features
            sizes: Target sizes for attention masks
            process: Whether to process attention masks

        Returns:
            Class logits, mask predictions, and optionally attention masks
        """
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
        """Forward pass for class prediction only.

        Args:
            x: Input features

        Returns:
            Class logits
        """
        decoder_output = self.decoder_norm(x)
        decoder_output = decoder_output.transpose(0, 1)
        # just a linear layer [hidden, n_class + 1]
        outputs_class = self.classifier(decoder_output)
        return outputs_class


class TransformerEncoderOnly(nn.Module):
    """Transformer encoder-only architecture."""

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        """Initialize transformer encoder-only architecture.

        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            activation: Activation function
            normalize_before: Whether to apply normalization before layers
        """
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            batch_first=False,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        """Forward pass for transformer encoder.

        Args:
            src: Source tensor
            mask: Attention mask
            pos_embed: Positional embedding

        Returns:
            Output tensor after transformer encoding
        """
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos_embed=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerFPN(nn.Module):
    """Feature Pyramid Network with optional transformer layers."""

    def __init__(
        self,
        backbone: BaseBackbone,
        feat_dim: int,
        out_dim: int,
        transformer_layers: int = 0,
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_pre_norm: bool = True,
    ):
        """Initialize Transformer Feature Pyramid Network.

        Args:
            backbone: Backbone network to extract features
            feat_dim: Number of channels for intermediate conv layers
            out_dim: Number of channels for final conv layer
            transformer_layers: Number of transformer encoder layers
            transformer_dropout: Dropout probability for transformer
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

        if transformer_layers > 0:
            in_channels = feature_channels[len(self.in_features) - 1]
            self.input_proj = Conv2d(in_channels, feat_dim, kernel_size=1)
            # weight_init.c2_xavier_fill(self.input_proj)
            nn.init.kaiming_uniform_(self.input_proj.weight, a=1)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0)
            self.transformer = TransformerEncoderOnly(
                d_model=feat_dim,
                dropout=transformer_dropout,
                nhead=transformer_nheads,
                dim_feedforward=transformer_dim_feedforward,
                num_encoder_layers=transformer_layers,
                normalize_before=transformer_pre_norm,
            )
            N_steps = feat_dim // 2
            self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        else:
            self.input_proj = None
            self.transformer = None
            self.pe_layer = None

        lateral_convs = []
        output_convs = []

        use_bias = False
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = nn.BatchNorm2d(feat_dim)
                output_conv = Conv2d(
                    feat_dim if transformer_layers > 0 else in_channels,
                    feat_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # weight_init.c2_xavier_fill(output_conv)
                nn.init.kaiming_uniform_(output_conv.weight, a=1)
                if output_conv.bias is not None:
                    nn.init.constant_(output_conv.bias, 0)
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
                # weight_init.c2_xavier_fill(lateral_conv)
                nn.init.kaiming_uniform_(lateral_conv.weight, a=1)
                if lateral_conv.bias is not None:
                    nn.init.constant_(lateral_conv.bias, 0)
                # weight_init.c2_xavier_fill(output_conv)
                nn.init.kaiming_uniform_(output_conv.weight, a=1)
                if output_conv.bias is not None:
                    nn.init.constant_(output_conv.bias, 0)
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
        # weight_init.c2_xavier_fill(self.mask_features)
        nn.init.kaiming_uniform_(self.mask_features.weight, a=1)
        if self.mask_features.bias is not None:
            nn.init.constant_(self.mask_features.bias, 0)

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
                if self.transformer is not None and self.input_proj is not None and self.pe_layer is not None:
                    x = self.input_proj(x)
                    x = self.transformer(x, mask=None, pos_embed=self.pe_layer(x))
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
                # weight_init.c2_xavier_fill(self.input_proj[-1])
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)
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
        accepted_postprocessing_types = ["semantic", "instance"]
        if self.config.postprocessing_type not in accepted_postprocessing_types:
            raise ValueError(
                f"Invalid postprocessing type: {self.config.postprocessing_type}. Must be one of: {accepted_postprocessing_types}"
            )

        backbone = load_backbone(self.config.backbone_config)

        self.pixel_decoder = TransformerFPN(
            backbone=backbone,
            feat_dim=self.config.pixel_decoder_feat_dim,
            out_dim=self.config.pixel_decoder_out_dim,
            transformer_layers=self.config.pixel_decoder_transformer_layers,
            transformer_dropout=self.config.pixel_decoder_transformer_dropout,
            transformer_nheads=self.config.pixel_decoder_transformer_nheads,
            transformer_dim_feedforward=self.config.pixel_decoder_transformer_dim_feedforward,
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
            cls_sigmoid=self.config.cls_sigmoid,
        )
        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)
        self.size_divisibility = self.config.size_divisibility
        self.num_classes = self.config.num_classes

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def dtype(self):
        return self.pixel_mean.dtype

    def forward(
        self,
        images: torch.Tensor,
        targets: list[MaskFormerTargets] = [],
    ) -> MaskFormerModelOutput:
        images = (images - self.pixel_mean) / self.pixel_std  # type: ignore

        features = self.pixel_decoder(images)
        (logits, masks), losses = self.head(features, targets)

        if not self.training:
            masks = F.interpolate(masks, size=images.shape[2:], mode="bilinear", align_corners=False)

        return MaskFormerModelOutput(masks=masks, logits=logits, loss=losses)
