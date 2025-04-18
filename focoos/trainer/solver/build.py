import copy
import itertools
from typing import Any, Dict, List, Optional

import torch

from focoos.nn.layers.norm import LayerNorm as ConvNextLayerNorm

from .lr_scheduler import (
    BaseLRScheduler,
    WarmupCosineLR,
    WarmupMultiStepLR,
    WarmupPolyLR,
)

_OPTIMIZERS = {
    "ADAMW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSPROP": torch.optim.RMSprop,
}
_SCHEDULERS = {
    "FIXED": BaseLRScheduler,
    "POLY": WarmupPolyLR,
    "COSINE": WarmupCosineLR,
    "MULTISTEP": WarmupMultiStepLR,
}


def full_model_gradient_clipping(optim, clip_norm_val):
    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    return FullModelGradientClippingOptimizer


def get_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    weight_decay_embed: Optional[float] = None,
    backbone_multiplier: float = 1.0,
    decoder_multiplier: float = 1.0,
    head_multiplier: float = 1.0,
) -> List[Dict[str, Any]]:
    defaults = {}
    defaults["lr"] = base_lr
    defaults["weight_decay"] = weight_decay

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
        ConvNextLayerNorm,
    )
    params = []
    memo = set()
    for module_name, module in model.named_modules():
        # Module name is the outer key (such as backbone.stage1.0)
        for module_param_name, value in module.named_parameters(recurse=False):
            # module_param_name is the inner, such as weight, bias, etc.
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * backbone_multiplier
                if backbone_multiplier == 0:
                    hyperparams["weight_decay"] = 0.0
            if "pixel_decoder" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * decoder_multiplier
                if backbone_multiplier == 0:
                    hyperparams["weight_decay"] = 0.0
            if "head" in module_name and "classifier" not in module_name:
                hyperparams["lr"] = hyperparams["lr"] * head_multiplier
                if head_multiplier == 0:
                    hyperparams["weight_decay"] = 0.0
            if isinstance(module, norm_module_types):
                hyperparams["weight_decay"] = weight_decay_norm
            if isinstance(module, torch.nn.Embedding) or "pos_embed" in module_param_name:  # SegFormer and Swin:
                hyperparams["weight_decay"] = weight_decay_embed
            if "relative_position_bias_table" in module_param_name:  # Swin (or attention in general)
                hyperparams["weight_decay"] = 0.0
            params.append({"params": [value], **hyperparams})

    return params


def build_optimizer(
    name: str,
    learning_rate: float,
    weight_decay: float,
    model: torch.nn.Module,
    weight_decay_norm: float = 0.0,
    weight_decay_embed: float = 0.0,
    backbone_multiplier: float = 0.1,
    decoder_multiplier: float = 1.0,
    head_multiplier: float = 1.0,
    clip_gradients: float = 0.1,
    extra: Optional[dict] = None,
) -> torch.optim.Optimizer:
    params = get_optimizer_params(
        model,
        base_lr=learning_rate,
        weight_decay=weight_decay,
        weight_decay_norm=weight_decay_norm,
        weight_decay_embed=weight_decay_embed,
        backbone_multiplier=backbone_multiplier,
        decoder_multiplier=decoder_multiplier,
        head_multiplier=head_multiplier,
    )

    if name.upper() in _OPTIMIZERS:
        optimizer_class = _OPTIMIZERS[name.upper()]
    else:
        raise NotImplementedError(f"Optimizer {name} is not supported. Use one of {_OPTIMIZERS.keys()}.")

    if clip_gradients > 0.0:
        optimizer_class = full_model_gradient_clipping(optimizer_class, clip_gradients)

    if extra is None:
        extra = {}
    return optimizer_class(params=params, lr=learning_rate, weight_decay=weight_decay, **extra)


def build_lr_scheduler(
    name: str,
    max_iters: int,
    optimizer: torch.optim.Optimizer,
    last_epoch: int = -1,
    extra: Optional[dict] = None,
) -> BaseLRScheduler:
    """
    Build a LR scheduler from config.
    """
    if name.upper() in _SCHEDULERS:
        scheduler_class = _SCHEDULERS[name.upper()]
    else:
        raise NotImplementedError(f"Scheduler {name} is not supported. Use one of {_SCHEDULERS.keys()}.")

    if extra is None:
        extra = {}

    return scheduler_class(max_iters=max_iters, last_epoch=last_epoch, optimizer=optimizer, **extra)
