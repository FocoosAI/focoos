import torch
import torch.nn as nn
import torch.nn.functional as F

from focoos.models.focoos_model import BaseModelNN
from focoos.models.fomo.config import FOMOConfig
from focoos.models.fomo.ports import FOMOModelOutput, FOMOTargets
from focoos.nn.backbone.build import load_backbone


def smooth_one_hot_targets(one_hot_masks, sigma=2, kernel_size=11):
    """
    Apply 2D Gaussian smoothing to one-hot encoded masks.
    
    Args:
        one_hot_masks: Tensor of shape [B, C, H, W] where each channel is a 0/1 heatmap
        sigma: Standard deviation of the Gaussian kernel
        kernel_size: Size of the Gaussian kernel (should be odd)
    
    Returns:
        Smoothed masks with same shape as input
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    # Create 2D Gaussian kernel
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.to(one_hot_masks.device)
    
    # Apply convolution to each channel separately
    smoothed_masks = torch.zeros_like(one_hot_masks)
    
    for b in range(one_hot_masks.shape[0]):
        for c in range(one_hot_masks.shape[1]):
            channel = one_hot_masks[b, c].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            smoothed_channel = F.conv2d(
                channel, 
                kernel.unsqueeze(0).unsqueeze(0),  # [1, 1, kernel_size, kernel_size]
                padding=kernel_size // 2
            )
            smoothed_masks[b, c] = smoothed_channel.squeeze()
            
    # Normalize each channel to preserve the original sum
    for b in range(smoothed_masks.shape[0]):
        for c in range(smoothed_masks.shape[1]):
            original_sum = one_hot_masks[b, c].sum()
            current_sum = smoothed_masks[b, c].sum()
            if current_sum > 0:  # Avoid division by zero
                smoothed_masks[b, c] = smoothed_masks[b, c] * (original_sum / current_sum)
    
    return smoothed_masks


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
    
    return resized_mask + 1 # map background  -1 => 0, others c => c+1

class FOMOCriterion(nn.Module):
    def __init__(self, num_classes: int, loss_type: str):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
    '''
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
            if "bce_loss" in losses_str:
                class_weights = class_weights[1:].reshape(-1, 1, 1) # to match target_masks shape
                
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
                target_masks = torch.cat([target.mask.unsqueeze(0) for target in targets], dim=0).long()  # [B, H, W]
                loss = F.cross_entropy(predictions, target_masks, 
                                       weight=class_weights, reduction='mean'
                                       )
                loss_dict["loss_ce"] = loss if loss is not None else 0
            
            # TODO:for below losses only, impplement an if later
            predictions = predictions.sigmoid() # TODO: test also without sigmoid
            if "bce_loss" in losses_str:
                target_masks = target_masks[..., 1:]  # [B, H, W, C]
            target_masks = target_masks.permute(0, 3, 1, 2).float()  # [B, C, H, W]
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
    '''
    def forward(self, predictions, targets: list[FOMOTargets]): # test with heatmaps
        loss_dict = {}
        losses_str = self.loss_type.lower().split("+")
        
        class_weights = torch.tensor([1, # background weight
                            50, 250, 320, 
                            550, 390, 1290, 
                            740], 
                            device=predictions.device, dtype=predictions.dtype
                            ) # Aquarium dataset specific
        
        if targets is not None and len(targets) > 0:
            target_masks = torch.cat([target.mask.unsqueeze(0) for target in targets], dim=0)  # [B, H, W]
            target_masks = F.one_hot(target_masks.long(), num_classes=self.num_classes + 1)  # [B, H, W, C+1]
            target_masks = target_masks[..., 1:]  # [B, H, W, C]
            target_masks = target_masks.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            # Smoothing the target masks
            target_masks_smoothed = smooth_one_hot_targets(target_masks, sigma=2, kernel_size=11)
            
            # For below operations only
            predictions = predictions.sigmoid()
            class_weights = class_weights[1:].reshape(-1, 1, 1) # to match target_masks shape
            target_masks = target_masks_smoothed
            """DEBUG"""
            if False:
                import os
                debug_dir = f"/home/ubuntu/focoos-1/notebooks/debug_outputs/debug_test_fomo_l1_heatmaps"
                os.makedirs(debug_dir, exist_ok=True)
                pred_save_path = os.path.join(debug_dir, "predictions.pt")
                gt_save_path = os.path.join(debug_dir, "gt.pt")
                torch.save(predictions, pred_save_path)
                torch.save(target_masks, gt_save_path)
            """DEBUG"""
            if "l1_heatmaps" in losses_str:
                loss = F.l1_loss(predictions, target_masks, reduction='mean')
                loss_dict["loss_l1"] = loss if loss is not None else 0
            if "l2_heatmaps" in losses_str:
                loss = F.mse_loss(predictions, target_masks, reduction='mean')
                loss_dict["loss_l2"] = loss if loss is not None else 0
            if "weighted_l1_heatmaps" in losses_str:
                loss = F.l1_loss(predictions, target_masks, reduction='none')
                weighted_l1_loss = (loss * class_weights).mean()
                loss_dict["loss_weighted_l1"] = weighted_l1_loss if weighted_l1_loss is not None else 0
            if "weighted_l2_heatmaps" in losses_str:
                loss = F.mse_loss(predictions, target_masks, reduction='none')
                weighted_l2_loss = (loss * class_weights).mean()
                loss_dict["loss_weighted_l2"] = weighted_l2_loss if weighted_l2_loss is not None else 0
                
            # Count losses
            if "count" in losses_str:
                gt_counts_per_class = target_masks.sum(dim=1)
                pred_counts_per_class = predictions.sum(dim=1)
                loss = F.l1_loss(pred_counts_per_class, gt_counts_per_class, reduction='mean')
                loss_dict["loss_count"] = loss if loss is not None else 0
                
        return loss_dict
        

class FOMOHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, criterion: nn.Module, activation: str = "relu"):
        super().__init__()
        
        self.num_classes = num_classes
        self.criterion = criterion
        self.activation = getattr(F, activation)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 
                               kernel_size=1,
                               dilation=1,
                               )
        out_channels = num_classes + 1 if "ce_loss" in self.criterion.loss_type.lower().split("+") else num_classes
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        
    def forward(self, features, targets: list[FOMOTargets] = []):
        outputs = self.conv1(features)
        outputs = self.conv2(self.activation(outputs))
        
        loss_dict = None
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
        
        images = (images - self.pixel_mean) / self.pixel_std
        
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
