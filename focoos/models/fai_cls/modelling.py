from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from focoos.data.mappers.classification_dataset_mapper import ClassificationDatasetDict
from focoos.models.fai_cls.config import ClassificationConfig
from focoos.models.fai_cls.ports import ClassificationModelOutput, ClassificationTargets
from focoos.models.fai_cls.processor import ClassificationProcessor
from focoos.models.fai_model import BaseModelNN
from focoos.nn.backbone.build import load_backbone
from focoos.ports import FocoosDetections
from focoos.utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationHead(nn.Module):
    """Classification head for image classification models."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, num_layers: int, dropout_rate: float = 0.0):
        """Initialize the classification head.

        Args:
            in_features: Number of input features from backbone
            hidden_dim: Hidden dimension for the classifier
            num_classes: Number of output classes
            num_layers: Number of layers in the classifier
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        if num_layers == 2:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes),
            )
        elif num_layers == 1:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes),
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
        return self.classifier(features)


class ClassificationLoss(nn.Module):
    """Loss module for image classification tasks."""

    def __init__(
        self,
        num_classes: int,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        multi_label: bool = False,
    ):
        """Initialize the loss module.

        Args:
            num_classes: Number of classes
            use_focal_loss: Whether to use focal loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing parameter
            multi_label: Whether to use multi-label loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.multi_label = multi_label

        # Use CrossEntropyLoss if not using focal loss
        if not use_focal_loss and not multi_label:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif not use_focal_loss and multi_label:
            self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: List[ClassificationTargets]) -> Dict[str, torch.Tensor]:
        """Compute the classification loss.

        Args:
            logits: Classification logits [N, num_classes]
            targets: List of classification targets

        Returns:
            Dictionary with loss values
        """
        labels = torch.stack([target.labels for target in targets]).to(logits.device)

        if self.use_focal_loss:
            # Compute focal loss manually
            pred_softmax = F.softmax(logits, dim=1) if not self.multi_label else torch.sigmoid(logits)
            target_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()

            # Apply label smoothing if needed
            if self.label_smoothing > 0:
                target_one_hot = target_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

            # Compute focal loss
            pred_softmax = torch.clamp(pred_softmax, min=1e-6, max=1.0)
            if self.multi_label:
                loss = (
                    -self.focal_alpha
                    * ((1 - pred_softmax) ** self.focal_gamma)
                    * (target_one_hot * torch.log(pred_softmax) + (1 - target_one_hot) * torch.log(1 - pred_softmax))
                )
            else:
                loss = (
                    -self.focal_alpha
                    * ((1 - pred_softmax) ** self.focal_gamma)
                    * target_one_hot
                    * torch.log(pred_softmax)
                )
            loss = loss.sum(dim=1).mean()
        else:
            if self.multi_label:
                target_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
                # Use standard cross entropy loss
                loss = self.ce_loss(logits, target_one_hot)
            else:
                # Use standard cross entropy loss
                loss = self.ce_loss(logits, labels)

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
        self.processor = ClassificationProcessor(config)

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
        )

        # Create loss module
        self.criterion = ClassificationLoss(
            num_classes=config.num_classes,
            use_focal_loss=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            multi_label=config.multi_label,
        )

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def dtype(self):
        return self.pixel_mean.dtype

    def forward(
        self,
        inputs: Union[
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
            Image.Image,
            List[Image.Image],
            List[ClassificationDatasetDict],
        ],
    ) -> ClassificationModelOutput:
        """Forward pass of the classification model.

        Args:
            inputs: Input images or dataset dictionaries

        Returns:
            Classification model output with logits and optional loss
        """

        images, targets = self.processor.preprocess(
            inputs,
            training=self.training,
            device=self.device,
            dtype=self.dtype,
            resolution=self.config.resolution,
            size_divisibility=self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

        images = (images - self.pixel_mean) / self.pixel_std  # type: ignore
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

    def post_process(
        self,
        outputs: ClassificationModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        **kwargs,
    ) -> List[FocoosDetections]:
        """Post-process model outputs for inference.

        Args:
            outputs: Model outputs
            batched_inputs: Batch input metadata

        Returns:
            Processed results with classification predictions
        """
        for k in kwargs:
            logger.warning(f"Unexpected kwarg '{k}' provided to post_process")
        return self.processor.postprocess(outputs, inputs)
