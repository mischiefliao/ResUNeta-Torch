"""
Basic test script to verify model and loss functions work correctly
"""

# IMPORTANT: Set environment variable BEFORE importing torch/numpy
# Fix OpenMP library conflict (common in Anaconda environments)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model import create_resunet_a
from loss import get_loss_function
from utils import calculate_metrics, set_seed


def test_model():
    """Test model forward pass"""
    print("Testing model...")
    set_seed(42)
    
    # Create model
    model = create_resunet_a(in_channels=3, out_channels=2)
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Check output shape
    assert output.shape == (2, 2, 256, 256), f"Expected (2, 2, 256, 256), got {output.shape}"
    print("✓ Model forward pass successful!")
    
    return model, output


def test_loss_functions():
    """Test loss functions"""
    print("\nTesting loss functions...")
    
    # Create dummy predictions and targets
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    # Test BCE loss
    bce_loss = get_loss_function('bce')
    bce_value = bce_loss(pred, target)
    print(f"BCE Loss: {bce_value.item():.4f}")
    assert bce_value.item() > 0, "BCE loss should be positive"
    
    # Test Dice loss
    dice_loss = get_loss_function('dice')
    dice_value = dice_loss(pred, target)
    print(f"Dice Loss: {dice_value.item():.4f}")
    assert dice_value.item() > 0, "Dice loss should be positive"
    
    # Test Tanimoto loss
    tanimoto_loss = get_loss_function('tanimoto')
    tanimoto_value = tanimoto_loss(pred, target)
    print(f"Tanimoto Loss: {tanimoto_value.item():.4f}")
    assert tanimoto_value.item() > 0, "Tanimoto loss should be positive"
    
    print("✓ All loss functions work correctly!")
    return True


def test_metrics():
    """Test metric calculations"""
    print("\nTesting metrics...")
    
    # Create dummy predictions and targets
    pred = torch.randint(0, 2, (2, 1, 256, 256)).float()
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    # Calculate metrics
    metrics = calculate_metrics(pred, target)
    print(f"IoU: {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    # Check metrics are in valid range
    assert 0 <= metrics['iou'] <= 1, "IoU should be between 0 and 1"
    assert 0 <= metrics['precision'] <= 1, "Precision should be between 0 and 1"
    assert 0 <= metrics['recall'] <= 1, "Recall should be between 0 and 1"
    assert 0 <= metrics['f1'] <= 1, "F1 should be between 0 and 1"
    
    print("✓ Metrics calculation successful!")
    return True


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_resunet_a(in_channels=3, out_channels=2)
    model.to(device)
    
    # Create loss and optimizer
    criterion = get_loss_function('tanimoto')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    images = torch.randn(2, 3, 256, 256).to(device)
    masks = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    print("✓ Training step successful!")
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("Running basic tests for ResUNet-a PyTorch implementation")
    print("=" * 50)
    
    try:
        # Test model
        model, output = test_model()
        
        # Test loss functions
        test_loss_functions()
        
        # Test metrics
        test_metrics()
        
        # Test training step
        test_training_step()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
