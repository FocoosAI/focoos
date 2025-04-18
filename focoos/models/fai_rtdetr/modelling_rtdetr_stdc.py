import logging
from typing import Union

import numpy as np
import torch
from PIL import Image

from focoos.data.mappers.detection_dataset_mapper import DetectionDatasetDict
from focoos.models.fai_model import BaseModelNN
from focoos.models.fai_rtdetr.config_rtdetr_stdc import RTDetrStdCConfig
from focoos.models.fai_rtdetr.modelling_rtdetr_resnet import (
    BoxHungarianMatcher,
    DETRHead,
    HybridEncoder,
    RTDETRTargets,
    RTDETRTransformerPredictor,
    SetCriterion,
)
from focoos.models.fai_rtdetr.processor import detector_postprocess
from focoos.nn.backbone.stdc import D2STDCnet
from focoos.structures import Boxes, ImageList, Instances
from focoos.utils.box import box_xyxy_to_cxcywh

logger = logging.getLogger(__name__)


class FAIRTDetrStdC(BaseModelNN):
    def __init__(self, config: RTDetrStdCConfig, model_info):
        super().__init__(config, model_info)
        self._export = False
        self.config = config
        self.pixel_decoder = HybridEncoder(
            backbone=D2STDCnet(
                base=self.config.backbone_base,
                layers=self.config.backbone_layers,
                out_features=self.config.backbone_out_features,
            ),
            feat_dim=self.config.pixel_decoder_feat_dim,
            out_dim=self.config.pixel_decoder_out_dim,
            dropout=self.config.pixel_decoder_dropout,
            nhead=self.config.pixel_decoder_nhead,
            dim_feedforward=self.config.pixel_decoder_dim_feedforward,
            num_encoder_layers=self.config.pixel_decoder_num_encoder_layers,
        )
        self.head = DETRHead(
            in_channels=self.config.head_in_channels,
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
                in_channels=self.config.head_in_channels,
                out_dim=self.config.head_out_dim,
                num_classes=self.config.num_classes,
                hidden_dim=self.config.transformer_predictor_hidden_dim,
                mask_on=self.config.transformer_predictor_mask_on,
                sigmoid=self.config.transformer_predictor_sigmoid,
                num_queries=self.config.transformer_predictor_num_queries,
                nhead=self.config.transformer_predictor_nhead,
                dec_layers=self.config.transformer_predictor_dec_layers,
                dim_feedforward=self.config.transformer_predictor_dim_feedforward,
                resolution=self.config.transformer_predictor_resolution,
            ),
            mask_on=self.config.head_mask_on,
            cls_sigmoid=self.config.head_cls_sigmoid,
        )
        self.top_k_masks = self.config.transformer_predictor_num_queries
        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)
        self.size_divisibility = self.config.size_divisibility
        self.ignore_value = self.config.ignore_value
        self.mask_on = self.config.mask_on
        self.num_classes = self.config.num_classes

    def forward(self, inputs: list[DetectionDatasetDict]):
        images = [x.image.to(self.device) for x in inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            tensors=images,
            # FIXME using size_divisibility in eval make detection break due to padding issue (in scaling bboxes)
            size_divisibility=self.size_divisibility if self.training else 0,
            padding_constraints=self.pixel_decoder.padding_constraints,
        )

        targets = []
        if self.training:
            # mask classification target
            gt_instances = [x.instances.to(self.device) for x in inputs]
            targets = self._prepare_targets(gt_instances, images)

        features = self.pixel_decoder(images.tensor)
        outputs, losses = self.head(features, targets)

        if self.training:
            return losses
        else:
            return outputs

    @property
    def device(self):
        return self.pixel_mean.device

    def predict(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
    ):
        # Convert single instances to lists for uniform processing
        if isinstance(inputs, (Image.Image, np.ndarray, torch.Tensor)):
            inputs = [inputs]

        # Process each input based on its type
        processed_inputs = []
        for inp in inputs:
            if isinstance(inp, Image.Image):
                inp = np.array(inp)
            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp).to(self.device)
            elif isinstance(inp, torch.Tensor):
                inp = inp.to(self.device)

            # Ensure input has correct shape and type
            if inp.dim() == 3:  # Add batch dimension if missing
                inp = inp.unsqueeze(0)
            if inp.shape[1] != 3 and inp.shape[-1] == 3:  # Convert HWC to CHW if needed
                inp = inp.permute(0, 3, 1, 2)

            processed_inputs.append(inp)

        # Stack all inputs into a single batch tensor
        # use pixel mean to get dtype -> If fp16, pixel_mean is fp16, so inputs will be fp16
        inputs = torch.cat(processed_inputs, dim=0).to(self.device, self.pixel_mean.dtype)
        print(inputs.shape)
        # Normalize the inputs
        inputs = (inputs - self.pixel_mean) / self.pixel_std

        features = self.pixel_decoder(inputs)
        output, _ = self.head(features, None)
        box_cls, box_pred = output

        return box_cls, box_pred

    def _prepare_targets(self, targets, images) -> list[RTDETRTargets]:
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append(RTDETRTargets(labels=gt_classes, boxes=gt_boxes))

        return new_targets

    def post_process(self, outputs, batched_inputs) -> list[Instances]:
        """
        Post-process the outputs of the model.
        This function is used in the evaluation phase to convert raw outputs to Instances.
        """
        results = []
        box_cls, box_pred = outputs

        for i in range(len(batched_inputs)):
            size = batched_inputs[i].image.shape[-2:]  # reshaped image size
            h = batched_inputs[i].height
            w = batched_inputs[i].width
            out_sizes = (h, w)  # original image size

            # Process results directly within the loop
            scores = box_cls[i]
            # Use dim instead of axis for torch.topk
            scores, index = torch.topk(scores.flatten(0), self.top_k_masks, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            processed_box_pred = box_pred[i].gather(dim=0, index=index.unsqueeze(-1).repeat(1, box_pred[i].shape[-1]))

            result = Instances(size)
            result.pred_boxes = Boxes(processed_box_pred)
            result.pred_boxes.scale(scale_x=size[1], scale_y=size[0])
            result.scores = scores
            result.pred_classes = labels

            result = detector_postprocess(result, output_height=out_sizes[0], output_width=out_sizes[1])

            results.append({"instances": result})

        return results
