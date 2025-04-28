from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from focoos.data.mappers.detection_dataset_mapper import DetectionDatasetDict
from focoos.models.fai_rtdetr.ports import RTDETRModelOutput, RTDETRTargets
from focoos.ports import FocoosDet, FocoosDetections
from focoos.structures import Boxes, ImageList, Instances
from focoos.utils.box import box_xyxy_to_cxcywh


# perhaps should rename to "resize_instance"
def detector_postprocess(
    results: Instances,
    output_height: int,
    output_width: int,
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


class RTDetrProcessor:
    def preprocess(
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
        training: bool,
        device: torch.device,
        dtype: torch.dtype,
        size_divisibility: int = 0,
        padding_constraints: Optional[Dict[str, int]] = None,
        resolution: Optional[int] = 640,
    ) -> tuple[torch.Tensor, list[RTDETRTargets]]:
        targets = []
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], DetectionDatasetDict):
            images = [x.image.to(device) for x in inputs]
            images = ImageList.from_tensors(
                tensors=images,
                # FIXME using size_divisibility in eval make detection break due to padding issue (in scaling bboxes)
                size_divisibility=size_divisibility if training or size_divisibility else 0,
                padding_constraints=padding_constraints,
            )
            images_torch = images.tensor
            if training:
                # mask classification target
                gt_instances = [x.instances.to(device) for x in inputs]
                h, w = images.tensor.shape[-2:]
                targets = []
                for targets_per_image in gt_instances:
                    image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=device)
                    gt_classes = targets_per_image.gt_classes
                    gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
                    gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                    targets.append(RTDETRTargets(labels=gt_classes, boxes=gt_boxes))
        else:
            if training:
                raise ValueError("During training, inputs should be a list of DetectionDatasetDict")
            if isinstance(inputs, (Image.Image, np.ndarray, torch.Tensor)):
                inputs_list = [inputs]
            else:
                inputs_list = inputs

            # Process each input based on its type
            processed_inputs = []
            for inp in inputs_list:
                # todo check for tensor of 4 dimesions.
                if isinstance(inp, Image.Image):
                    inp = np.array(inp)
                if isinstance(inp, np.ndarray):
                    inp = torch.from_numpy(inp)

                # Ensure input has correct shape and type
                if inp.dim() == 3:  # Add batch dimension if missing
                    inp = inp.unsqueeze(0)
                if inp.shape[1] != 3 and inp.shape[-1] == 3:  # Convert HWC to CHW if needed
                    inp = inp.permute(0, 3, 1, 2)

                processed_inputs.append(inp)

            # Stack all inputs into a single batch tensor
            # use pixel mean to get dtype -> If fp16, pixel_mean is fp16, so inputs will be fp16
            # TODO: this will break with different image sizes
            images_torch = torch.cat(processed_inputs, dim=0).to(device, dtype=dtype)
            if resolution is not None:
                images_torch = torch.nn.functional.interpolate(
                    images_torch, size=resolution, mode="bilinear", align_corners=False
                )
            # Normalize the inputs
        return images_torch, targets

    def eval_postprocess(
        self,
        output: RTDETRModelOutput,
        batched_inputs: list[DetectionDatasetDict],
        top_k: int = 300,
    ) -> list[dict[str, Instances]]:
        results = []
        box_cls, box_pred = output.logits, output.boxes
        batch_size = box_cls.shape[0]
        num_classes = box_cls.shape[-1]

        for i in range(batch_size):
            # Process results directly within the loop
            scores, labels, processed_box_pred = self._get_predictions(box_cls[i], box_pred[i], top_k, num_classes)

            result = Instances(image_size=(1, 1))  # we are using normalized boxes
            result.pred_boxes = Boxes(processed_box_pred)
            result.scores = scores
            result.pred_classes = labels
            result = detector_postprocess(
                result, output_height=batched_inputs[i].height, output_width=batched_inputs[i].width
            )
            results.append({"instances": result})

        return results

    def get_image_sizes(
        self,
        inputs: Union[torch.Tensor, np.ndarray, Image.Image, list[Image.Image], list[np.ndarray], list[torch.Tensor]],
    ):
        image_sizes = []

        if isinstance(inputs, (torch.Tensor, np.ndarray)):
            # Single tensor/array input
            if isinstance(inputs, torch.Tensor):
                height, width = inputs.shape[-2:]
            else:  # numpy array
                height, width = inputs.shape[-3:-1] if inputs.ndim > 3 else inputs.shape[:2]
            image_sizes.append((height, width))
        elif isinstance(inputs, Image.Image):
            # Single PIL image
            width, height = inputs.size
            image_sizes.append((height, width))
        elif isinstance(inputs, list):
            # List of inputs
            for img in inputs:
                if isinstance(img, torch.Tensor):
                    height, width = img.shape[-2:]
                elif isinstance(img, np.ndarray):
                    height, width = img.shape[-3:-1] if img.ndim > 3 else img.shape[:2]
                elif isinstance(img, Image.Image):
                    width, height = img.size
                else:
                    raise ValueError(f"Unsupported input type in list: {type(img)}")
                image_sizes.append((height, width))
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")
        return image_sizes

    def _get_predictions(self, scores, boxes, top_k, num_classes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, index = torch.topk(scores.flatten(0), top_k, dim=-1)
        labels = index % num_classes
        index = index // num_classes
        box_pred = boxes.gather(dim=0, index=index.unsqueeze(-1).repeat(1, boxes.shape[-1]))
        return scores, labels, box_pred

    def postprocess(
        self,
        output: RTDETRModelOutput,
        inputs: Union[
            torch.Tensor,
            np.ndarray,
            Image.Image,
            list[Image.Image],
            list[np.ndarray],
            list[torch.Tensor],
        ],
        top_k: int = 300,
        threshold: float = 0.5,
    ) -> list[FocoosDetections]:
        # Extract image sizes from inputs
        image_sizes = self.get_image_sizes(inputs)

        results = []
        batch_size = output.boxes.shape[0]
        num_classes = output.logits.shape[-1]
        assert len(image_sizes) == batch_size, (
            f"Expected image sizes {len(image_sizes)} to match batch size {batch_size}"
        )

        for i in range(batch_size):
            # Process results directly within the loop
            scores, labels, box_pred = self._get_predictions(output.logits[i], output.boxes[i], top_k, num_classes)

            # Apply threshold to filter out low-confidence predictions
            mask = scores > threshold
            box_pred = box_pred[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Multiply boxes by image size
            box_pred[:, 0::2] = box_pred[:, 0::2] * image_sizes[i][0]
            box_pred[:, 1::2] = box_pred[:, 1::2] * image_sizes[i][1]
            # Convert box coordinates to integers for pixel-precise bounding boxes
            box_pred = box_pred.round().to(torch.int32)

            # Convert tensor outputs to Python lists of floats
            py_box_pred = box_pred.detach().cpu().tolist()
            py_scores = scores.detach().cpu().tolist()
            py_labels = labels.detach().cpu().tolist()

            results.append(
                FocoosDetections(
                    detections=[
                        FocoosDet(
                            bbox=py_bp,
                            conf=py_s,
                            cls_id=py_l,
                        )
                        for py_bp, py_s, py_l in zip(py_box_pred, py_scores, py_labels)
                    ]
                )
            )

        return results
