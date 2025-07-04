# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Any, Dict, Optional, Tuple, cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch.optim.lr_scheduler import LRScheduler

from focoos.models.base_model import BaseModelNN
from focoos.models.eomt.config import EoMTConfig
from focoos.models.eomt.loss import MaskHungarianMatcher, SetCriterion
from focoos.models.eomt.ports import EoMTModelOutput, EoMTTargets


class ViT(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int = 16,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = cast(
            VisionTransformer,
            timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=(592, 592),
                patch_size=patch_size,
                num_classes=0,
            ),
        )
        self.backbone.set_input_size(img_size)
        default_cfg = cast(Dict[str, Any], self.backbone.default_cfg)
        pixel_mean = torch.tensor(default_cfg["mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(default_cfg["std"]).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)


class TwoStageWarmupPolySchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        num_backbone_params: int,
        warmup_steps: tuple[int, int],
        total_steps: int,
        poly_power: float,
        last_epoch=-1,
    ):
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.poly_power = poly_power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        non_vit_warmup, vit_warmup = self.warmup_steps
        for i, base_lr in enumerate(self.base_lrs):
            if i >= self.num_backbone_params:
                if non_vit_warmup > 0 and step < non_vit_warmup:
                    lr = base_lr * (step / non_vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            else:
                if step < non_vit_warmup:
                    lr = 0
                elif step < non_vit_warmup + vit_warmup:
                    lr = base_lr * ((step - non_vit_warmup) / vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup - vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup - vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            lrs.append(lr)
        return lrs


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x


class _EoMT(nn.Module):
    def __init__(
        self,
        encoder: ViT,
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size)

        mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device) > prob
            attn_mask[:, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :][random_queries] = True

        return attn_mask

    def _attn(self, module: nn.Module, x: torch.Tensor, mask: Optional[torch.Tensor]):
        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def forward(self, x: torch.Tensor):
        x = x / 255.0
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        x = self.encoder.backbone.patch_embed(x)
        x = self.encoder.backbone._pos_embed(x)
        x = self.encoder.backbone.patch_drop(x)
        x = self.encoder.backbone.norm_pre(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat((self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)

            if self.masked_attn_enabled and i >= len(self.encoder.backbone.blocks) - self.num_blocks:
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                interpolated = F.interpolate(
                    mask_logits,
                    self.encoder.backbone.patch_embed.grid_size,
                    mode="bilinear",
                )
                interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
                attn_mask[
                    :,
                    : self.num_q,
                    self.num_q + self.encoder.backbone.num_prefix_tokens :,
                ] = interpolated > 0
                attn_mask = self._disable_attn_mask(
                    attn_mask,
                    self.attn_mask_probs[i - len(self.encoder.backbone.blocks) + self.num_blocks],
                )

            x = x + block.drop_path1(block.ls1(self._attn(block.attn, block.norm1(x), attn_mask)))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )


class EoMT(BaseModelNN):
    def __init__(self, config: EoMTConfig):
        super().__init__(config)
        self.config = config
        self.network = _EoMT(
            encoder=ViT(img_size=config.im_size),
            num_classes=config.num_classes,
            num_q=config.num_queries,
            num_blocks=config.num_blocks,
            masked_attn_enabled=config.masked_attn_enabled,
        )
        self.criterion = SetCriterion(
            num_classes=self.config.num_classes,
            matcher=MaskHungarianMatcher(
                cost_class=self.config.criterion_class_coefficient,
                cost_mask=self.config.criterion_mask_coefficient,
                cost_dice=self.config.criterion_dice_coefficient,
                num_points=self.config.criterion_num_points,
                cls_sigmoid=False,
            ),
            weight_dict={
                "loss_ce": self.config.criterion_class_coefficient,
                "loss_mask": self.config.criterion_mask_coefficient,
                "loss_dice": self.config.criterion_dice_coefficient,
            },
            deep_supervision=True,
            eos_coef=self.config.criterion_no_object_coefficient,
            losses=["labels", "masks"],
            num_points=self.config.criterion_num_points,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            loss_class_type="ce_loss",
            cls_sigmoid=False,
        )

    @property
    def device(self) -> torch.device:
        """Get the device where the model is located."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the model parameters."""
        return next(self.parameters()).dtype

    def forward(
        self,
        images: torch.Tensor,
        targets: list[EoMTTargets] = [],
    ) -> EoMTModelOutput:
        """Perform forward pass through the model.

        Args:
            images: Input images tensor
            targets: Training targets (used during training)

        Returns:
            EoMTModelOutput with masks, logits, and optional loss
        """
        # Get raw model outputs
        mask_logits_per_layer, class_logits_per_layer = self.network(images)

        # Use final layer predictions
        masks = mask_logits_per_layer[-1]
        logits = class_logits_per_layer[-1]

        loss = None
        # Calculate loss during training
        if self.training and targets and len(targets) > 0:
            out = {
                "pred_logits": class_logits_per_layer[-1],
                "pred_masks": mask_logits_per_layer[-1],
                "aux_outputs": self._set_aux_loss(
                    class_logits_per_layer,
                    mask_logits_per_layer,
                ),
            }
            loss = self.criterion(out, targets)

        if not self.training:
            masks = F.interpolate(masks, size=images.shape[2:], mode="bilinear", align_corners=False)

        return EoMTModelOutput(masks=masks.sigmoid(), logits=logits[:, :, :-1].softmax(dim=1), loss=loss)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
