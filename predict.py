"""
Prediction script for ResUNet-a
PyTorch implementation converted from TensorFlow/Keras version
Maintains the same command-line interface as the original TensorFlow version
"""

# IMPORTANT: Set environment variable BEFORE importing torch/numpy
# Fix OpenMP library conflict (common in Anaconda environments)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# Windows compatibility
if sys.platform == 'win32':
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

from model import create_resunet_a
from utils import load_checkpoint


def preprocess_image(image_path, image_size=(256, 256)):
    """
    Preprocess single image for inference
    
    Args:
        image_path: Path to image file
        image_size: Target image size (height, width)
    
    Returns:
        Preprocessed image tensor and original size
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_size = image.shape[:2]  # (H, W)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, image_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats (same as training)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    # PyTorch format: (C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size


def postprocess_prediction(prediction, original_size=None, threshold=0.5):
    """
    Postprocess model prediction
    
    Args:
        prediction: Model output tensor (B, C, H, W) or (B, H, W)
        original_size: Original image size (H, W) for resizing
        threshold: Threshold for binary segmentation
    
    Returns:
        Postprocessed prediction as numpy array (H, W)
    """
    # Apply sigmoid for binary segmentation
    if prediction.dim() == 4:  # (B, C, H, W)
        if prediction.shape[1] > 1:
            # Multi-class: use softmax
            prediction = F.softmax(prediction, dim=1)
            prediction = prediction.argmax(dim=1).float()
        else:
            # Binary: use sigmoid
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > threshold).float()
            prediction = prediction.squeeze(1)  # Remove channel dimension
    else:  # (B, H, W)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > threshold).float()
    
    # Convert to numpy
    prediction = prediction.squeeze(0).cpu().numpy()  # Remove batch dimension
    
    # Resize to original size if provided
    if original_size and prediction.shape != original_size:
        prediction = cv2.resize(prediction, (original_size[1], original_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Convert to uint8
    prediction = (prediction * 255).astype(np.uint8)
    
    return prediction


def predict_single_image(model, image_path, device, image_size=(256, 256)):
    """
    Predict segmentation for a single image
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: Device to run inference on
        image_size: Target image size
    
    Returns:
        Prediction mask and original image size
    """
    # Preprocess
    image_tensor, original_size = preprocess_image(image_path, image_size)
    image_tensor = image_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = postprocess_prediction(output, original_size)
    
    return prediction, original_size


def main(args):
    """Main prediction function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = create_resunet_a(
        in_channels=3,
        out_channels=args.num_classes,
        depth=7,
        layer_norm='batch'
    )
    model.to(device)
    
    # Load checkpoint
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from {args.model_path}")
        else:
            # If it's just state_dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Get image paths
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', 
                       '*.PNG', '*.JPG', '*.JPEG', '*.TIF', '*.TIFF']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(args.image_path, ext)))
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.image_path}")
    
    print(f"Found {len(image_paths)} images")
    
    # Predict for each image
    for image_path in tqdm(image_paths, desc="Predicting"):
        try:
            # Get image name
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            
            # Predict
            prediction, original_size = predict_single_image(
                model, image_path, device,
                image_size=(args.image_size, args.image_size)
            )
            
            # Save prediction
            output_filename = f"{base_name}_pred.png"
            output_path = os.path.join(args.output_path, output_filename)
            cv2.imwrite(output_path, prediction)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"\nPredictions saved to {args.output_path}")
    print(f"Processed {len(image_paths)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with ResUNet-a model")
    
    # Arguments matching the original TensorFlow version
    parser.add_argument('--image_size', type=int, default=256, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='Path to test images directory')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model file (.pth)')
    parser.add_argument('--output_path', type=str, required=True, 
                       help='Path to save predictions')
    
    args = parser.parse_args()
    
    main(args)
