"""
Utility functions for ResUNet-a
PyTorch implementation converted from TensorFlow/Keras version
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os

# Try to import sklearn metrics, but use numpy fallback if unavailable
try:
    from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, using numpy-based metrics")


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        num_classes: Number of classes
    
    Returns:
        IoU score
    """
    # Convert to numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Handle different input formats
    if pred_np.ndim == 4:  # (B, C, H, W)
        pred_np = pred_np.argmax(axis=1)  # Get class predictions
    else:
        pred_np = (pred_np > 0.5).astype(np.uint8)
    
    if target_np.ndim == 4:  # (B, C, H, W)
        target_np = target_np.argmax(axis=1)
    else:
        target_np = target_np.astype(np.uint8)
    
    # Flatten
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Calculate IoU
    if num_classes == 2:
        # Binary IoU
        intersection = np.logical_and(pred_flat, target_flat).sum()
        union = np.logical_or(pred_flat, target_flat).sum()
        if union == 0:
            return 1.0  # Perfect match when both are empty
        return intersection / union
    else:
        # Multi-class IoU (mean IoU)
        if SKLEARN_AVAILABLE:
            return jaccard_score(target_flat, pred_flat, average='macro', zero_division=0)
        else:
            # Numpy-based multi-class IoU
            ious = []
            for cls in range(num_classes):
                pred_cls = (pred_flat == cls).astype(np.float32)
                target_cls = (target_flat == cls).astype(np.float32)
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum() - intersection
                if union > 0:
                    ious.append(intersection / union)
                else:
                    ious.append(1.0)  # Perfect match when both are empty
            return np.mean(ious)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    """
    Calculate multiple metrics: IoU, Precision, Recall, F1-score
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        num_classes: Number of classes
    
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Handle different input formats
    if pred_np.ndim == 4:  # (B, C, H, W)
        pred_np = pred_np.argmax(axis=1)
    else:
        pred_np = (pred_np > 0.5).astype(np.uint8)
    
    if target_np.ndim == 4:  # (B, C, H, W)
        target_np = target_np.argmax(axis=1)
    else:
        target_np = target_np.astype(np.uint8)
    
    # Flatten
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Calculate metrics
    metrics = {}
    
    if SKLEARN_AVAILABLE:
        # Use sklearn if available
        metrics['iou'] = jaccard_score(target_flat, pred_flat, average='macro', zero_division=0)
        metrics['precision'] = precision_score(target_flat, pred_flat, average='macro', zero_division=0)
        metrics['recall'] = recall_score(target_flat, pred_flat, average='macro', zero_division=0)
        metrics['f1'] = f1_score(target_flat, pred_flat, average='macro', zero_division=0)
        
        # Add diagnostic information for sklearn path too
        if num_classes == 2:
            tp = np.logical_and(pred_flat == 1, target_flat == 1).sum()
            fp = np.logical_and(pred_flat == 1, target_flat == 0).sum()
            fn = np.logical_and(pred_flat == 0, target_flat == 1).sum()
            tn = np.logical_and(pred_flat == 0, target_flat == 0).sum()
            
            total_pixels = len(pred_flat)
            pred_ones = (pred_flat == 1).sum()
            pred_zeros = (pred_flat == 0).sum()
            target_ones = (target_flat == 1).sum()
            target_zeros = (target_flat == 0).sum()
            
            metrics['_diagnostics'] = {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'total_pixels': int(total_pixels),
                'pred_ones': int(pred_ones),
                'pred_zeros': int(pred_zeros),
                'target_ones': int(target_ones),
                'target_zeros': int(target_zeros),
                'pred_ones_ratio': float(pred_ones / total_pixels) if total_pixels > 0 else 0.0,
                'target_ones_ratio': float(target_ones / total_pixels) if total_pixels > 0 else 0.0
            }
    else:
        # Numpy-based metrics calculation
        # For binary classification
        if num_classes == 2:
            tp = np.logical_and(pred_flat == 1, target_flat == 1).sum()
            fp = np.logical_and(pred_flat == 1, target_flat == 0).sum()
            fn = np.logical_and(pred_flat == 0, target_flat == 1).sum()
            tn = np.logical_and(pred_flat == 0, target_flat == 0).sum()
            
            # IoU (Jaccard)
            intersection = tp
            union = tp + fp + fn
            metrics['iou'] = intersection / union if union > 0 else 1.0
            
            # Precision
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1-score
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0.0
            
            # Add diagnostic information
            total_pixels = len(pred_flat)
            pred_ones = (pred_flat == 1).sum()
            pred_zeros = (pred_flat == 0).sum()
            target_ones = (target_flat == 1).sum()
            target_zeros = (target_flat == 0).sum()
            
            metrics['_diagnostics'] = {
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'total_pixels': int(total_pixels),
                'pred_ones': int(pred_ones),
                'pred_zeros': int(pred_zeros),
                'target_ones': int(target_ones),
                'target_zeros': int(target_zeros),
                'pred_ones_ratio': float(pred_ones / total_pixels) if total_pixels > 0 else 0.0,
                'target_ones_ratio': float(target_ones / total_pixels) if total_pixels > 0 else 0.0
            }
        else:
            # Multi-class metrics
            ious = []
            precisions = []
            recalls = []
            f1s = []
            
            for cls in range(num_classes):
                pred_cls = (pred_flat == cls).astype(np.float32)
                target_cls = (target_flat == cls).astype(np.float32)
                
                tp = (pred_cls * target_cls).sum()
                fp = (pred_cls * (1 - target_cls)).sum()
                fn = ((1 - pred_cls) * target_cls).sum()
                
                # IoU
                intersection = tp
                union = tp + fp + fn
                iou = intersection / union if union > 0 else 1.0
                ious.append(iou)
                
                # Precision
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                precisions.append(precision)
                
                # Recall
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                recalls.append(recall)
                
                # F1
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                f1s.append(f1)
            
            metrics['iou'] = np.mean(ious)
            metrics['precision'] = np.mean(precisions)
            metrics['recall'] = np.mean(recalls)
            metrics['f1'] = np.mean(f1s)
    
    return metrics


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   train_losses: Optional[list] = None,
                   val_losses: Optional[list] = None,
                   train_metrics_history: Optional[Dict] = None,
                   val_metrics_history: Optional[Dict] = None,
                   **kwargs):
    """
    Save model checkpoint with full training state
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        train_losses: Training loss history (optional)
        val_losses: Validation loss history (optional)
        train_metrics_history: Training metrics history (optional)
        val_metrics_history: Validation metrics history (optional)
        **kwargs: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    # Save scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save training history if provided
    if train_losses is not None:
        checkpoint['train_losses'] = train_losses
    if val_losses is not None:
        checkpoint['val_losses'] = val_losses
    if train_metrics_history is not None:
        checkpoint['train_metrics_history'] = train_metrics_history
    if val_metrics_history is not None:
        checkpoint['val_metrics_history'] = val_metrics_history
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   filepath: str,
                   device: torch.device,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
    """
    Load model checkpoint with full training state
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None)
        filepath: Path to checkpoint file
        device: Device to load model to
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        Dictionary with checkpoint information including:
        - epoch: Last completed epoch
        - train_losses: Training loss history (if saved)
        - val_losses: Validation loss history (if saved)
        - train_metrics_history: Training metrics history (if saved)
        - val_metrics_history: Validation metrics history (if saved)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {checkpoint.get('epoch', 0) + 1}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_prediction(image: np.ndarray,
                        target: np.ndarray,
                        prediction: np.ndarray,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 5)):
    """
    Visualize original image, ground truth, and prediction
    
    Args:
        image: Original image (H, W, C) or (C, H, W)
        target: Ground truth mask (H, W) or (C, H, W)
        prediction: Predicted mask (H, W) or (C, H, W)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Handle different input formats
    if image.ndim == 3 and image.shape[0] == 3:  # (C, H, W)
        image = image.transpose(1, 2, 0)
    
    if target.ndim == 3:  # (C, H, W)
        target = target[0] if target.shape[0] == 1 else target.argmax(axis=0)
    
    if prediction.ndim == 3:  # (C, H, W)
        prediction = prediction[0] if prediction.shape[0] == 1 else prediction.argmax(axis=0)
    
    # Normalize image if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(target, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(train_losses: list,
                          val_losses: list,
                          train_metrics: Optional[Dict[str, list]] = None,
                          val_metrics: Optional[Dict[str, list]] = None,
                          save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: Optional dictionary of training metrics
        val_metrics: Optional dictionary of validation metrics
        save_path: Optional path to save figure
    """
    num_plots = 1
    if train_metrics and val_metrics:
        num_plots += len(train_metrics)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics if provided
    if train_metrics and val_metrics:
        for idx, (metric_name, train_values) in enumerate(train_metrics.items(), 1):
            if idx < len(axes):
                axes[idx].plot(train_values, label=f'Train {metric_name}')
                if metric_name in val_metrics:
                    axes[idx].plot(val_metrics[metric_name], label=f'Val {metric_name}')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric_name.upper())
                axes[idx].set_title(f'{metric_name.upper()} over Epochs')
                axes[idx].legend()
                axes[idx].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test metrics calculation
    pred = torch.randint(0, 2, (2, 1, 256, 256)).float()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    metrics = calculate_metrics(pred, target)
    print(f"Metrics: {metrics}")
    
    iou = calculate_iou(pred, target)
    print(f"IoU: {iou:.4f}")
    
    print("Utility functions test completed!")
