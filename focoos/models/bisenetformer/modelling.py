from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from focoos.models.bisenetformer.config import BisenetFormerConfig
from focoos.models.bisenetformer.loss import MaskHungarianMatcher, SetCriterion
from focoos.models.bisenetformer.ports import BisenetFormerOutput, BisenetFormerTargets
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


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__()
        self.proj = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.conv = ConvBNReLU(out_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)

        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(self.proj(x))
        # feat = self.conv(x)
        atten = feat.mean(dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class ContextPath(nn.Module):
    def __init__(self, inplanes, hidden_dim=128, out4=False):
        super().__init__()
        # inplanes -> 0 res2 1/4, 1 res3 1/8, 2 res4 1/16, 3 res5 1/32
        self.arm32 = AttentionRefinementModule(inplanes[3], hidden_dim)
        self.conv_avg = ConvBNReLU(inplanes[3], hidden_dim, ks=1, stride=1, padding=0)
        self.conv_head32 = ConvBNReLU(hidden_dim, hidden_dim, ks=3, stride=1, padding=1)

        self.arm16 = AttentionRefinementModule(inplanes[2], hidden_dim)
        self.conv_head16 = ConvBNReLU(hidden_dim, hidden_dim, ks=3, stride=1, padding=1)

        self.out4 = out4
        if self.out4:
            self.arm8 = AttentionRefinementModule(inplanes[1], hidden_dim)
            self.conv_head8 = ConvBNReLU(hidden_dim, hidden_dim, ks=3, stride=1, padding=1)

    def forward(self, feat4, feat8, feat16, feat32):
        avg = feat32.mean(dim=(2, 3), keepdim=True)

        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = F.interpolate(feat32_sum, size=feat16.shape[-2:], mode="bilinear")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.shape[-2:], mode="bilinear")
        feat16_up = self.conv_head16(feat16_up)

        if self.out4:
            feat8_arm = self.arm8(feat8)
            feat8_sum = feat8_arm + feat16_up
            feat8_up = F.interpolate(feat8_sum, size=feat4.shape[-2:], mode="bilinear")
            feat8_up = self.conv_head8(feat8_up)
        else:
            feat8_sum = feat16_up
            feat8_up = None

        return feat8_up, feat8_sum, feat16_sum, feat32_sum  # x4, x8, x16, x32


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan1, in_chan2, out_chan, *args, **kwargs):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chan1, out_chan, kernel_size=1)
        self.proj2 = nn.Conv2d(in_chan2, out_chan, kernel_size=1)
        self.convblk = ConvBNReLU(out_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        # fcat = torch.cat([fsp, fcp], dim=1)
        # feat = self.convblk(fcat)  # self.proj1(fsp) + self.proj2(fcp))
        feat = self.convblk(self.proj1(fsp) + self.proj2(fcp))
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiseNet(nn.Module):
    def __init__(
        self,
        backbone: BaseBackbone,
        feat_dim: int,
        out_dim: int,
    ):
        """
        Args:
            backbone: basic backbones to extract features from images
            feat_dim: number of output channels for the intermediate conv layers.
            out_dim: number of output channels for the final conv layer.
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
        # from res2 to res5
        self.cp = ContextPath(feature_channels, self.feat_dim)
        self.ffm = FeatureFusionModule(feature_channels[1], self.feat_dim, self.feat_dim)
        self.conv_out = ConvBNReLU(feat_dim, out_dim, ks=3, stride=1, padding=1)

    @property
    def padding_constraints(self) -> Dict[str, int]:
        return self.backbone.padding_constraints

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        return self.forward_features(features)

    def forward_features(self, features):
        res2, res3, res4, res5 = (features[f] for f in self.in_features)
        _, feat_cp8, feat_cp16, feat_cp32 = self.cp(res2, res3, res4, res5)
        feat_fuse = self.ffm(res3, feat_cp8)
        feat_out = self.conv_out(feat_fuse)

        return feat_out, (feat_cp32, feat_cp16, feat_cp8)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        enforce_input_project: bool,
        use_attn_masks: bool = True,
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.use_attn_masks = use_attn_masks
        self.query_init = False

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
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
        if not self.query_init:
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we use 2 scale)
        self.num_feature_levels = min(2, dec_layers)
        # self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(min(self.num_feature_levels, dec_layers)):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                nn.init.kaiming_uniform_(self.input_proj[-1].weight, a=1)  # type: ignore
                if self.input_proj[-1].bias is not None:
                    nn.init.constant_(self.input_proj[-1].bias, 0)  # type: ignore
            else:
                self.input_proj.append(nn.Sequential())

        self.forward_prediction_heads = PredictionHeads(
            hidden_dim, num_classes, out_dim, nheads, mask_classification, use_attn_masks
        )

        if self.query_init:
            self.out_dim = out_dim
            self.kernels = nn.Conv2d(in_channels=out_dim, out_channels=num_queries, kernel_size=1)
            nn.init.kaiming_uniform_(self.kernels.weight, a=1)
            if self.kernels.bias is not None:
                nn.init.constant_(self.kernels.bias, 0)
            self.proj = nn.Linear(out_dim, hidden_dim)

    def forward(self, x, mask_features, targets=None, mask=None):
        # x is a list of multi-scale feature
        # assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        x = x[:-1]  # F1 and F2 only (not F3)

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

        if not self.query_init:
            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        else:
            query_embed = None

        predictions_class = []
        predictions_mask = []

        if self.query_init:
            outputs_mask = self.kernels(mask_features)
            proposal_kernels = self.kernels.weight.clone()

            output = proposal_kernels[None].expand(mask_features.shape[0], *proposal_kernels.size())
            output = output.squeeze((3, 4)).permute(1, 0, 2)

            if output.shape[-1] != self.out_dim:
                output = self.proj(output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, outputs_mask, sizes=size_list[0], compute_masks=False
            )
        else:
            # prediction heads on learnable query features
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, sizes=size_list[0]
            )
        attn_mask = attn_mask[0] if self.use_attn_masks else None
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # if on a mask is all True (no pixel active), use cross attention (put every pixel at False)
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
                output, mask_features, sizes=size_list[(i + 1) % self.num_feature_levels]
            )
            attn_mask = attn_mask[0] if self.use_attn_masks else None
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


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

    def layers(self, features, targets=None, mask=None):
        mask_features, multi_scale_features = features
        predictions = self.predictor(multi_scale_features, mask_features, targets=targets, mask=mask)

        return predictions

    def forward(self, features, targets: list[BisenetFormerTargets] = []):
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


class BisenetFormer(BaseModelNN):
    def __init__(self, config: BisenetFormerConfig):
        super().__init__(config)
        self._export = False
        self.config = config
        accepted_postprocessing_types = ["semantic", "instance"]
        if self.config.postprocessing_type not in accepted_postprocessing_types:
            raise ValueError(
                f"Invalid postprocessing type: {self.config.postprocessing_type}. Must be one of: {accepted_postprocessing_types}"
            )

        backbone = load_backbone(self.config.backbone_config)

        self.pixel_decoder = BiseNet(
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
            transformer_predictor=TransformerDecoder(
                in_channels=self.config.pixel_decoder_out_dim,
                out_dim=self.config.transformer_predictor_out_dim,
                num_classes=self.config.num_classes,
                hidden_dim=self.config.transformer_predictor_hidden_dim,  # this is query dim
                num_queries=self.config.num_queries,
                nheads=8,
                dim_feedforward=self.config.transformer_predictor_dim_feedforward,
                dec_layers=self.config.transformer_predictor_dec_layers,
                pre_norm=True,
                enforce_input_project=True,
                use_attn_masks=True,
            ),
            cls_sigmoid=self.config.cls_sigmoid,
        )
        self.resolution = self.config.resolution
        self.top_k = self.config.num_queries
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
        targets: list[BisenetFormerTargets] = [],
    ) -> BisenetFormerOutput:
        images = (images - self.pixel_mean) / self.pixel_std  # type: ignore

        features = self.pixel_decoder(images)
        (logits, masks), losses = self.head(features, targets)

        if not self.training:
            masks = F.interpolate(masks, size=images.shape[2:], mode="bilinear", align_corners=False)

        return BisenetFormerOutput(masks=masks, logits=logits, loss=losses)
