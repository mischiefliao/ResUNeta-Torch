"""
Data preprocessing and loading for ResUNet-a
PyTorch implementation converted from TensorFlow/Keras version
"""

# IMPORTANT: Set environment variable BEFORE importing torch/numpy
# Fix OpenMP library conflict (common in Anaconda environments)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from glob import glob
from typing import List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Windows compatibility: Set multiprocessing start method
if sys.platform == 'win32':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic segmentation
    """
    def __init__(self, 
                 image_paths: List[str], 
                 mask_paths: List[str], 
                 transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (256, 256),
                 normalize: bool = True):
        """
        Args:
            image_paths: List of paths to input images
            mask_paths: List of paths to ground truth masks
            transform: Optional transform function (albumentations)
            image_size: Target image size (height, width)
            normalize: Whether to normalize images to [0, 1]
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        self.normalize = normalize
        
        # Verify that image and mask lists have same length
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}) must be equal"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Ensure mask is a tensor
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            # Ensure mask has correct shape (1, H, W) if it's (H, W)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # Normalize mask to [0, 1] if needed
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask = (mask > 0.5).float()
            # Ensure mask is (1, H, W) format
            if mask.dim() == 3 and mask.shape[0] != 1:
                # If it's (C, H, W) with C>1, take first channel or convert
                if mask.shape[0] > 1:
                    mask = mask[0:1, :, :]  # Take first channel
                else:
                    mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        else:
            # Basic preprocessing
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize image
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            
            # Normalize mask to [0, 1] and binarize if needed
            mask = mask.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)
            
            # Convert to tensors
            # PyTorch uses channel-first format: (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # Add channel dimension
        
        return image, mask


def get_train_transform(image_size: Tuple[int, int] = (256, 256)):
    """
    Get training data augmentation transforms
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_val_transform(image_size: Tuple[int, int] = (256, 256)):
    """
    Get validation data transforms (no augmentation)
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def load_dataset(image_dir: str,
                 gt_dir: str,
                 validation_split: float = 0.2,
                 batch_size: int = 8,
                 image_size: Tuple[int, int] = (256, 256),
                 num_workers: int = 4,
                 use_augmentation: bool = True):
    """
    Load dataset and create data loaders
    
    Args:
        image_dir: Directory containing input images
        gt_dir: Directory containing ground truth masks
        validation_split: Fraction of data to use for validation
        batch_size: Batch size for data loaders
        image_size: Target image size (height, width)
        num_workers: Number of worker processes for data loading
        use_augmentation: Whether to use data augmentation for training
    
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Supported image extensions
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.TIF', '*.TIFF']
    
    # Get all image paths
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    # Get corresponding mask paths
    mask_paths = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        # Try to find corresponding mask file
        mask_name = img_name  # Assume same name
        mask_path = os.path.join(gt_dir, mask_name)
        
        # Try different extensions if original doesn't exist
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                mask_path = os.path.join(gt_dir, base_name + ext)
                if os.path.exists(mask_path):
                    break
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image: {img_path}")
        
        mask_paths.append(mask_path)
    
    # Split dataset
    num_train = int(len(image_paths) * (1 - validation_split))
    train_images = image_paths[:num_train]
    train_masks = mask_paths[:num_train]
    val_images = image_paths[num_train:]
    val_masks = mask_paths[num_train:]
    
    print(f"Dataset split: {len(train_images)} training, {len(val_images)} validation")
    
    # Create transforms
    train_transform = get_train_transform(image_size) if use_augmentation else get_val_transform(image_size)
    val_transform = get_val_transform(image_size)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        train_images, train_masks, 
        transform=train_transform,
        image_size=image_size
    )
    val_dataset = SegmentationDataset(
        val_images, val_masks,
        transform=val_transform,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python batch_preprocess.py <image_dir> <gt_dir>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    gt_dir = sys.argv[2]
    
    try:
        train_loader, val_loader = load_dataset(
            image_dir=image_dir,
            gt_dir=gt_dir,
            validation_split=0.2,
            batch_size=4,
            image_size=(256, 256)
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test one batch
        for images, masks in train_loader:
            print(f"Batch - Images shape: {images.shape}, Masks shape: {masks.shape}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
