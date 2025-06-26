import torch
import torch.nn as nn
import torch.nn.functional as F

from focoos.models.focoos_model import BaseModelNN
from focoos.models.fomo.config import FOMOConfig
from focoos.models.fomo.ports import FOMOModelOutput, FOMOTargets
from focoos.nn.backbone.build import load_backbone


def resize_mask_to_size(mask, size, num_classes):
    """Resize a mask to target size while preserving class indices at scaled locations.
    
    Args:
        mask (torch.Tensor): Input mask of shape (H, W) containing class indices and -1
        size (tuple): Target size (H', W') for the resized mask
        
    Returns:
        torch.Tensor: Resized mask of shape size containing class indices at scaled locations
    """
    h, w = mask.shape
    target_h, target_w = size[-2:]
    resized_mask = -torch.ones(target_h, target_w, dtype=mask.dtype, device=mask.device)
    y_idxs, x_idxs = torch.where(mask != -1)
    scaled_x = (x_idxs * target_w / w).long()
    scaled_y = (y_idxs * target_h / h).long()
    resized_mask[scaled_y, scaled_x] = mask[y_idxs, x_idxs]
    
    return resized_mask + 1 # map background  -1 => 0, others to c => c+1

class FOMOCriterion(nn.Module):
    def __init__(self, num_classes: int, loss_type: str):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        
    def forward(self, predictions, targets: list[FOMOTargets]):
        loss_dict = {}
        losses_str = self.loss_type.lower().split("+")
        
        if targets is not None and len(targets) > 0:
            class_weights = torch.tensor([1, # background weight
                            50, 250, 320, 
                            550, 390, 1290, 
                            740], 
                            device=predictions.device, dtype=predictions.dtype
                            ) # Aquarium dataset specific
            
            # target.mask one-hot encoding with no background
            target_masks = torch.cat([target.mask.unsqueeze(0) for target in targets], dim=0)  # [B, H, W]
            target_masks = F.one_hot(target_masks.long(), num_classes=self.num_classes + 1)  # [B, H, W, C+1]
            target_masks = target_masks[..., 1:]  # [B, H, W, C]
            target_masks = target_masks.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            # reshape class_weights to [C, 1, 1]
            class_weights = class_weights[1:].reshape(-1, 1, 1) # [C-1, 1, 1]
                
            if "bce_loss" in losses_str:
                """DEBUG"""
                if False:
                    # compute tensor statistics 
                    # ['background', 'fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
                    for batch_idx in range(target_masks.shape[0]):
                        class_counts = torch.zeros(self.num_classes+1, device=target_masks.device, dtype=torch.int32)
                        for class_idx in range(self.num_classes):
                            class_mask = target_masks[batch_idx, class_idx]
                            unique_classes, counts = torch.unique(class_mask, return_counts=True)
                            if counts.shape[0] == 1:
                                class_counts[0] += counts[0] # background
                            if counts.shape[0] > 1:
                                class_counts[0] += counts[0] # background
                                class_counts[class_idx+1] += counts[1]
                        with open("/home/ubuntu/focoos-1/notebooks/debug_outputs/random_stuff/class_imbalance_stats.txt", "a") as f:
                            f.write(f"{class_counts.tolist()}\n")
                """DEBUG"""
                loss = F.binary_cross_entropy_with_logits(predictions, target_masks, 
                                                          reduction='mean', pos_weight=class_weights
                                                          )
                loss_dict["loss_bce"] = loss if loss is not None else 0
            elif "ce_loss" in losses_str:
                target_masks = torch.cat([target.mask.unsqueeze(0) for target in targets], dim=0).long() # [B, H, W]
                loss = F.cross_entropy(predictions, target_masks, 
                                       weight=class_weights, reduction='mean'
                                       )
                loss_dict["loss_ce"] = loss if loss is not None else 0
                
            if "l1" in losses_str:
                loss = F.l1_loss(predictions, target_masks, reduction='mean')
                loss_dict["loss_l1"] = loss if loss is not None else 0
            if "l2" in losses_str:
                loss = F.mse_loss(predictions, target_masks, reduction='mean')
                loss_dict["loss_l2"] = loss if loss is not None else 0
            if "weighted_l1" in losses_str:
                loss = F.l1_loss(predictions, target_masks, reduction='none')
                weighted_l1_loss = (loss * class_weights).mean()
                loss_dict["loss_weighted_l1"] = weighted_l1_loss if weighted_l1_loss is not None else 0
            if "weighted_l2" in losses_str:
                loss = F.mse_loss(predictions, target_masks, reduction='none')
                weighted_l2_loss = (loss * class_weights).mean()
                loss_dict["loss_weighted_l2"] = weighted_l2_loss if weighted_l2_loss is not None else 0
        
        return loss_dict

class FOMOHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, criterion: nn.Module, activation: str = "relu"):
        super().__init__()
        
        self.num_classes = num_classes
        self.criterion = criterion
        self.activation = getattr(F, activation)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        out_channels = num_classes + 1 if "ce_loss" in self.criterion.loss_type.lower().split("+") else num_classes
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        
    def forward(self, features, targets: list[FOMOTargets] = []):
        outputs = self.conv1(features)
        outputs = self.conv2(self.activation(outputs))
        
        loss = None
        if targets is not None and len(targets) > 0:
            for target in targets:
                target.mask = resize_mask_to_size(target.mask, outputs.shape[-2:], self.num_classes)
            loss_dict = self.losses(outputs, targets)
        
        return outputs, loss_dict
    
    def losses(self, predictions, targets: list[FOMOTargets]):
        loss_dict = self.criterion(predictions, targets)

        return loss_dict

class FOMO(BaseModelNN):
    def __init__(self, config: FOMOConfig):
        super().__init__(config)
        self._export = False
        self.config = config
        
        self.backbone = load_backbone(self.config.backbone_config)
        self.head = FOMOHead(
            in_channels=self.backbone.output_shape()[self.config.cut_point].channels,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            criterion=FOMOCriterion(num_classes=self.config.num_classes, loss_type=self.config.loss_type),
            activation=self.config.activation,
        )
        
        if self.config.freeze_backbone:
            print("Freezing backbone weights.")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # TODO: maybe they can be removed, just used for device and dtype properties
        self.register_buffer("pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False)
        
    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def dtype(self):
        return self.pixel_mean.dtype
        
    def forward(
        self, 
        images: torch.Tensor, 
        targets: list[FOMOTargets] = []
    ) -> FOMOModelOutput:
        
        with torch.set_grad_enabled(not self.config.freeze_backbone):
            features = self.backbone(images)
        features_cut = features[self.config.cut_point]
        outputs, loss_dict = self.head(features_cut, targets)
        
        """DEBUG"""
        if False:
            import os
            import shutil
            debug_dir = "/home/ubuntu/focoos-1/notebooks/debug_outputs/train_debug"
            
            # Delete directory contents if it exists
            if os.path.exists(debug_dir):
                if not hasattr(self, '_debug_dir_cleared'):
                    shutil.rmtree(debug_dir)
                    self._debug_dir_cleared = True
                    os.makedirs(debug_dir)
            
                    # Save predictions, ground truth masks and images
                    if targets is not None and len(targets) > 0:
                        target_masks = torch.cat([target.mask.unsqueeze(0) for target in targets], dim=0)  # [B, H, W]
                        target_masks = F.one_hot(target_masks.long(), num_classes=self.config.num_classes + 1)  # [B, H, W, C]
                        target_masks = target_masks[..., :-1]  # [B, H, W, C-1]
                        target_masks = target_masks.permute(0, 3, 1, 2).float()  # [B, C-1, H, W]
                    
                    for i, target in enumerate(targets):
                        # Save predictions
                        pred_save_path = os.path.join(debug_dir, f"{i}_pred_mask_train.pt")
                        torch.save(outputs[i], pred_save_path)
                        # Save ground truth masks
                        gt_save_path = os.path.join(debug_dir, f"{i}_gt_mask_train.pt")
                        torch.save(target_masks[i], gt_save_path)
                        # Save original images
                        image_save_path = os.path.join(debug_dir, f"{i}_image_train.pt")
                        torch.save(images[i], image_save_path)
                        
                    print(f"DEBUG: Training tensors saved to: {debug_dir}")
        """DEBUG"""
        
        if self.training:
            assert targets is not None and len(targets) > 0, "targets should not be None or empty - training mode"
            return FOMOModelOutput(logits=torch.zeros(0, 0, 0), loss=loss_dict)
        
        return FOMOModelOutput(logits=outputs, loss=loss_dict)
