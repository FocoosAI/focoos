# Copyright (c) FocoosAI SRL. All rights reserved.
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from torch import Tensor, autocast, nn

from focoos.models.fai_mf.ports import MaskFormerTargets
from focoos.structures import BitMasks, Boxes
from focoos.utils.distributed.comm import get_world_size
from focoos.utils.distributed.dist import is_dist_available_and_initialized

"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def shapes_to_tensor(x: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all([isinstance(t, torch.Tensor) for t in x]), "Shape should be tensor during tracing!"
        # as_tensor should not be used in tracing because it records a constant
        ret = torch.stack(x)
        if ret.device != device:  # avoid recording a hard-coded device if not necessary
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def generate_regular_grid_point_coords(R, side_size, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        R (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (R, side_size^2, 2) that contains coordinates
            for the regular grids.
    """
    aff = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]], device=device)
    r = F.affine_grid(aff, torch.Size((1, 1, side_size, side_size)), align_corners=False)
    return r.view(1, -1, 2).expand(R, -1, -1)


def get_uncertain_point_coords_with_randomness(
    coarse_logits,
    uncertainty_func,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


def point_sample_fine_grained_features(features_list, feature_scales, boxes, point_coords):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.

    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.
        boxes (list[Boxes]): A list of I Boxes  objects that contain R_1 + ... + R_I = R boxes all
            together.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """
    cat_boxes = Boxes.cat(boxes)
    num_boxes = [b.tensor.size(0) for b in boxes]

    point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes.tensor, point_coords)
    split_point_coords_wrt_image = torch.split(point_coords_wrt_image, num_boxes)

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = shapes_to_tensor([w, h]) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale.to(feature_map.device)
            point_features_per_image.append(
                point_sample(
                    feature_map[idx_img].unsqueeze(0),
                    point_coords_scaled.unsqueeze(0),
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(1, 0)
            )
        point_features.append(cat(point_features_per_image, dim=1))

    return cat(point_features, dim=0), point_coords_wrt_image


def get_point_coords_wrt_image(boxes_coords, point_coords):
    """
    Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.

    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_image = point_coords.clone()
        point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
        point_coords_wrt_image[:, :, 0] += boxes_coords[:, None, 0]
        point_coords_wrt_image[:, :, 1] += boxes_coords[:, None, 1]
    return point_coords_wrt_image


def sample_point_labels(instances, point_coords):
    """
    Sample point labels from ground truth mask given point_coords.

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. So, i_th elememt of the list contains R_i objects and R_1 + ... + R_N is
            equal to R. The ground-truth gt_masks in each instance will be used to compute labels.
        points_coords (Tensor): A tensor of shape (R, P, 2), where R is the total number of
            instances and P is the number of points for each instance. The coordinates are in
            the absolute image pixel coordinate space, i.e. [0, H] x [0, W].

    Returns:
        Tensor: A tensor of shape (R, P) that contains the labels of P sampled points.
    """
    with torch.no_grad():
        gt_mask_logits = []
        point_coords_splits = torch.split(
            point_coords,
            [len(instances_per_image) for instances_per_image in instances],
        )
        for i, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue
            assert isinstance(instances_per_image.gt_masks, BitMasks), (
                "Point head works with GT in 'bitmask' format. Set INPUT.MASK_FORMAT to 'bitmask'."
            )

            gt_bit_masks = instances_per_image.gt_masks.tensor
            h, w = instances_per_image.gt_masks.image_size
            scale = torch.tensor([w, h], dtype=torch.float, device=gt_bit_masks.device)
            points_coord_grid_sample_format = point_coords_splits[i] / scale
            gt_mask_logits.append(
                point_sample(
                    gt_bit_masks.to(torch.float32).unsqueeze(1),
                    points_coord_grid_sample_format,
                    align_corners=False,
                ).squeeze(1)
            )

    point_labels = cat(gt_mask_logits)
    return point_labels


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`. THIS IS IMPLICLTY BASED ON SIGMOID ACTIVATION!
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def softmax_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1) + 1.0
    denominator = inputs.sum(-1) + targets.sum(-1) + 1.0
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def softmax_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    # loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = -inputs * targets

    return loss.mean(1).sum() / num_masks


softmax_dice_loss_jit = torch.jit.script(softmax_dice_loss)  # type: torch.jit.ScriptModule

softmax_ce_loss_jit = torch.jit.script(softmax_ce_loss)  # type: torch.jit.ScriptModule


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 10,
    gamma: float = 2,
    reduction: str = "mean",
    ignore_index: int = 255,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        ignore_index:
    Returns:
        Loss tensor with the reduction option applied.
    """
    ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=ignore_index)
    p_t = torch.exp(-ce_loss)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        loss = alpha * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


focal_loss_jit = torch.jit.script(focal_loss)  # type:  torch.jit.ScriptModule


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)  # type: torch.jit.ScriptModule


def batch_soft_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = (numerator + 1) / (denominator + 1)
    return loss


batch_soft_dice_loss_jit = torch.jit.script(batch_soft_dice_loss)  # type: torch.jit.ScriptModule


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


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
        use_focal: bool = False,
        loss_class_type: str = "ce_loss",  # ce_loss, focal_loss, bce_loss, bce_focal_loss
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
        assert loss_class_type in [
            "ce_loss",
            "focal_loss",
            "bce_loss",
            "bce_focal_loss",
        ], "loss_class_type must be in ['ce_loss', 'focal_loss', 'bce_loss', 'bce_focal_loss']"

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
        self.use_focal = use_focal
        if use_focal:
            warnings.warn(
                "use_focal is deprecated. Use loss_class_type instead",
                DeprecationWarning,
            )
            loss_class_type = "focal_loss" if loss_class_type == "ce_loss" else loss_class_type
            loss_class_type = "bce_focal_loss" if loss_class_type == "bce_loss" else loss_class_type
        self.loss_class_type = loss_class_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cls_sigmoid = cls_sigmoid

    def loss_labels(self, outputs, targets: list[MaskFormerTargets], indices, num_masks, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t.labels[J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        if self.loss_class_type == "focal_loss":
            loss_ce = focal_loss(
                src_logits.transpose(1, 2),
                target_classes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
        elif self.loss_class_type == "ce_loss":
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif self.loss_class_type == "bce_loss":
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
            if self.cls_sigmoid and src_logits.shape[-1] > target.shape[-1]:
                src_logits = src_logits[..., :-1]
            loss = F.binary_cross_entropy_with_logits(src_logits, target * 1.0, reduction="none")
            loss_ce = loss.mean(1).sum() * src_logits.shape[1] / num_masks

        elif self.loss_class_type == "bce_focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
            loss = torchvision.ops.sigmoid_focal_loss(
                src_logits,
                target.float(),
                self.focal_alpha,
                self.focal_gamma,
                reduction="none",
            )
            loss_ce = loss.mean(1).sum() * src_logits.shape[1] / num_masks
        else:
            raise ValueError("loss_class_type must be in ['ce_loss', 'focal_loss', 'bce_loss', 'bce_focal_loss']")

        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets: list[MaskFormerTargets], indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t.masks for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        if self.num_points != 0:
            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)
        else:
            src_masks = F.interpolate(
                src_masks[:, None],
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            point_logits = src_masks[:, 0].flatten(1)

            target_masks = target_masks.flatten(1)
            point_labels = target_masks.view(src_masks.shape)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
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

    def get_loss(self, loss, outputs, targets, indices, num_masks, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, **kwargs)

    def forward(self, outputs, targets: list[MaskFormerTargets]):
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
            "matcher: {}".format(self.matcher),
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


class MaskHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        num_points: int = 0,
        cls_sigmoid: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
            num_points: Number of points to sample from the mask for matching
            cls_sigmoid: Whether to apply sigmoid to the classification logits
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cls_sigmoid = cls_sigmoid

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets: list[MaskFormerTargets]):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            if self.cls_sigmoid:
                out_prob = outputs["pred_logits"][b].sigmoid()
            else:
                out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b].labels

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b].masks.to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(device_type="cuda", enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
