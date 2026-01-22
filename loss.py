"""
Loss functions for ResUNet-a
PyTorch implementation converted from TensorFlow/Keras version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TanimotoLoss(nn.Module):
    """
    Tanimoto Loss (also known as Dice Loss)
    Formula: T = intersection / (pred_sum + target_sum - intersection)
    Loss = 1 - T
    """
    def __init__(self, smooth=1e-6):
        super(TanimotoLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W) or (B, H, W)
            target: Ground truth (B, C, H, W) or (B, 1, H, W) or (B, H, W)
        Returns:
            Tanimoto loss
        """
        # Handle different input shapes
        if pred.dim() == 4:  # (B, C, H, W)
            if pred.shape[1] == 1:
                pred = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
            elif pred.shape[1] == 2:
                pred = F.softmax(pred, dim=1)
                pred = pred[:, 1, :, :]  # Take foreground channel (B, H, W)
            else:
                pred = F.softmax(pred, dim=1)
                if target.dim() == 4 and target.shape[1] == 1:
                    target = target.squeeze(1).long()
                    pred = pred.argmax(dim=1).float()
        else:  # (B, H, W)
            pred = torch.sigmoid(pred)
        
        # Handle target shape
        if target.dim() == 4:
            if target.shape[1] == 1:
                target = target.squeeze(1)
            elif target.shape[1] > 1:
                target = target.argmax(dim=1)
        
        target = target.float()
        if target.max() > 1.0:
            target = target / 255.0
        target = (target > 0.5).float()
        
        # Flatten tensors (use reshape to handle non-contiguous tensors)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        union = pred_sum + target_sum - intersection
        
        # Tanimoto coefficient
        tanimoto = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - tanimoto


class TanimotoWithComplementLoss(nn.Module):
    """
    Tanimoto Loss with Complement
    Combines original Tanimoto loss and complement Tanimoto loss
    Formula: alpha * (1 - T) + (1 - alpha) * (1 - T_complement)
    """
    def __init__(self, alpha=0.5, smooth=1e-6):
        super(TanimotoWithComplementLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W) or (B, H, W)
            target: Ground truth (B, C, H, W) or (B, 1, H, W) or (B, H, W)
        Returns:
            Combined Tanimoto loss
        """
        # Handle different input shapes
        if pred.dim() == 4:  # (B, C, H, W)
            if pred.shape[1] == 1:
                # Single channel output - binary segmentation
                pred = torch.sigmoid(pred)
                pred = pred.squeeze(1)  # (B, H, W)
            elif pred.shape[1] == 2:
                # Two channel output - binary segmentation with background/foreground
                # Use softmax and take foreground channel (channel 1)
                pred = F.softmax(pred, dim=1)
                pred = pred[:, 1, :, :]  # Take foreground channel (B, H, W)
            else:
                # Multi-class: use softmax
                pred = F.softmax(pred, dim=1)
                # For multi-class, we need to handle target differently
                if target.dim() == 4 and target.shape[1] == 1:
                    # Convert target to one-hot if needed
                    target = target.squeeze(1).long()  # (B, H, W)
                    # Take the predicted class probabilities
                    pred = pred.argmax(dim=1).float()  # (B, H, W)
        else:  # (B, H, W)
            pred = torch.sigmoid(pred)
        
        # Handle target shape
        if target.dim() == 4:
            if target.shape[1] == 1:
                target = target.squeeze(1)  # (B, H, W)
            elif target.shape[1] > 1:
                # One-hot encoded target - convert to class indices
                target = target.argmax(dim=1)  # (B, H, W)
        
        # Ensure both are float and in [0, 1] range
        target = target.float()
        if target.max() > 1.0:
            target = target / 255.0
        target = (target > 0.5).float()
        
        # Flatten tensors (use reshape to handle non-contiguous tensors)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Original Tanimoto coefficient
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        union = pred_sum + target_sum - intersection
        tanimoto = (intersection + self.smooth) / (union + self.smooth)
        
        # Complement Tanimoto coefficient
        pred_comp = 1 - pred_flat
        target_comp = 1 - target_flat
        intersection_comp = (pred_comp * target_comp).sum()
        pred_sum_comp = pred_comp.sum()
        target_sum_comp = target_comp.sum()
        union_comp = pred_sum_comp + target_sum_comp - intersection_comp
        tanimoto_comp = (intersection_comp + self.smooth) / (union_comp + self.smooth)
        
        # Combined loss
        loss = self.alpha * (1 - tanimoto) + (1 - self.alpha) * (1 - tanimoto_comp)
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss (similar to Tanimoto Loss)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Handle different input shapes
        if pred.dim() == 4:  # (B, C, H, W)
            if pred.shape[1] == 1:
                pred = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
            elif pred.shape[1] == 2:
                pred = F.softmax(pred, dim=1)
                pred = pred[:, 1, :, :]  # Take foreground channel (B, H, W)
            else:
                pred = F.softmax(pred, dim=1)
                if target.dim() == 4 and target.shape[1] == 1:
                    target = target.squeeze(1).long()
                    pred = pred.argmax(dim=1).float()
        else:  # (B, H, W)
            pred = torch.sigmoid(pred)
        
        # Handle target shape
        if target.dim() == 4:
            if target.shape[1] == 1:
                target = target.squeeze(1)
            elif target.shape[1] > 1:
                target = target.argmax(dim=1)
        
        target = target.float()
        if target.max() > 1.0:
            target = target / 255.0
        target = (target > 0.5).float()
        
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


def get_loss_function(loss_name='tanimoto', **kwargs):
    """
    Factory function to get loss function
    
    Args:
        loss_name: Name of loss function ('bce', 'dice', 'tanimoto')
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function module
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'tanimoto':
        return TanimotoWithComplementLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                         f"Supported: 'bce', 'dice', 'tanimoto'")


if __name__ == "__main__":
    # Test loss functions
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    print("Testing loss functions:")
    
    bce_loss = get_loss_function('bce')
    print(f"BCE Loss: {bce_loss(pred, target).item():.4f}")
    
    dice_loss = get_loss_function('dice')
    print(f"Dice Loss: {dice_loss(pred, target).item():.4f}")
    
    tanimoto_loss = get_loss_function('tanimoto')
    print(f"Tanimoto Loss: {tanimoto_loss(pred, target).item():.4f}")
