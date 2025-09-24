from typing import Dict, List

import torch
import torch.nn as nn

from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetDict
from focoos.models.fai_cls.config import ClassificationConfig
from focoos.models.fai_cls.ports import ClassificationModelOutput, ClassificationTargets
from focoos.models.focoos_model import BaseModelNN
from focoos.nn.backbone.build import load_backbone
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationHead(nn.Module):
    """Classification head for image classification models."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        dropout_rate: float = 0.0,
        dense_prediction: bool = False,
    ):
        """Initialize the classification head.

        Args:
            in_features: Number of input features from backbone
            hidden_dim: Hidden dimension for the classifier
            num_classes: Number of output classes
            num_layers: Number of layers in the classifier
            dropout_rate: Dropout rate for regularization
            dense_prediction: Whether to use dense prediction
        """
        super().__init__()

        if num_layers == 2:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1) if not dense_prediction else nn.Identity(),
                # nn.Flatten(),
                nn.Conv2d(in_features, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
                nn.AdaptiveMaxPool2d(1) if dense_prediction else nn.Identity(),
            )
        elif num_layers == 1:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1) if not dense_prediction else nn.Identity(),
                # nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(in_features, num_classes, kernel_size=1),
                nn.AdaptiveMaxPool2d(1) if dense_prediction else nn.Identity(),
            )
        else:
            raise ValueError(f"Invalid number of layers: {num_layers}")

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """Forward pass of the classification head.

        Args:
            features: Features from the backbone [N, C, H, W]

        Returns:
            Classification logits [N, num_classes]
        """
        return self.classifier(features).flatten(start_dim=1)


class ClassificationLoss(nn.Module):
    """Loss module for image classification tasks."""

    def __init__(
        self,
        num_classes: int,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        pos_weight: float = 10.0,
    ):
        """Initialize the loss module.

        Args:
            num_classes: Number of classes
            use_focal_loss: Whether to use focal loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing parameter
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        # Use CrossEntropyLoss if not using focal loss
        if not use_focal_loss:
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight * torch.ones(self.num_classes))

    def forward(self, logits: torch.Tensor, targets: List[ClassificationTargets]) -> Dict[str, torch.Tensor]:
        """Compute the classification loss.

        Args:
            logits: Classification logits [N, num_classes]
            targets: List of classification targets

        Returns:
            Dictionary with loss values
        """
        target_one_hot = torch.stack([target.labels for target in targets]).to(dtype=logits.dtype, device=logits.device)

        if self.use_focal_loss:
            # Compute focal loss manually
            pred = torch.sigmoid(logits)
            # target_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

            # Apply label smoothing if needed
            if self.label_smoothing > 0:
                target_one_hot = target_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

            # Compute focal loss
            pred = torch.clamp(pred, min=1e-6, max=1.0)

            loss = (
                -self.focal_alpha
                * ((1 - pred) ** self.focal_gamma)
                * (target_one_hot * torch.log(pred) + (1 - target_one_hot) * torch.log(1 - pred))
            )
            loss = loss.sum(dim=1).mean()
        else:
            # target_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
            # Use standard cross entropy loss
            loss = self.ce_loss(logits, target_one_hot)

        return {"loss_cls": loss}


class FAIClassification(BaseModelNN):
    """Image classification model that can use any backbone."""

    def __init__(self, config: ClassificationConfig):
        """Initialize the classification model.

        Args:
            config: Model configuration
        """
        super().__init__(config)

        self.config = config

        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)

        # Load backbone
        self.backbone = load_backbone(config.backbone_config)

        # Use the highest level feature by default (e.g., res5 for ResNet)
        assert config.features in self.backbone.output_shape()
        self.in_features = config.features
        self.feature_channels = self.backbone.output_shape()[self.in_features].channels
        assert self.feature_channels is not None

        # Create classification head
        self.cls_head = ClassificationHead(
            in_features=self.feature_channels,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout_rate=config.dropout_rate,
            dense_prediction=config.dense_prediction,
        )

        # Create loss module
        self.criterion = ClassificationLoss(
            num_classes=config.num_classes,
            use_focal_loss=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            pos_weight=config.pos_weight,
        )

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def dtype(self):
        return self.pixel_mean.dtype

    def forward(
        self,
        images: torch.Tensor,
        targets: list[ClassificationTargets] = [],
    ) -> ClassificationModelOutput:
        """Forward pass of the classification model.

        Args:
            inputs: Input images or dataset dictionaries

        Returns:
            Classification model output with logits and optional loss
        """
        # type: ignore

        # Extract features from backbone
        features = self.backbone(images)

        # Extract the highest level feature
        feature_map = features[self.in_features]

        # Apply classification head
        logits = self.cls_head(feature_map)

        # Compute loss if targets are provided (training mode)
        loss = self.criterion(logits, targets) if targets else None

        return ClassificationModelOutput(logits=logits, loss=loss)

    def eval_post_process(
        self, outputs: ClassificationModelOutput, inputs: List[ClassificationDatasetDict]
    ) -> List[Dict]:
        """Post-process model outputs for inference.

        Args:
            outputs: Model outputs
            batched_inputs: Batch input metadata

        Returns:
            Processed results with classification predictions
        """
        return self.processor.eval_postprocess(outputs, inputs)  # type: ignore
