#!/usr/bin/env python3
"""
PyTorch Installation Test Script
Run this script to verify that PyTorch is properly installed and working.
"""

import sys
import subprocess

def test_pytorch():
    """Test PyTorch installation and basic functionality."""
    
    print("Testing PyTorch installation...")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch successfully imported")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU device count: {torch.cuda.device_count()}")
    
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    try:
        import torch.nn as nn
        print(f"✓ torch.nn successfully imported")
    except ImportError as e:
        print(f"✗ Failed to import torch.nn: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision successfully imported")
        print(f"  Version: {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import torchvision: {e}")
        return False
    
    # Test basic tensor operations
    try:
        print("\nTesting basic tensor operations...")
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.sum()
        y.backward()
        print(f"✓ Tensor creation and autograd working")
        print(f"  Tensor: {x}")
        print(f"  Gradients: {x.grad}")
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False
    
    # Test loss function
    try:
        print("\nTesting loss functions...")
        mse_loss = nn.MSELoss()
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 2.1, 2.9])
        loss = mse_loss(predictions, targets)
        print(f"✓ MSE loss function working")
        print(f"  Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False
    
    # Test other required packages
    required_packages = ['numpy', 'matplotlib', 'sklearn', 'jupyter']
    
    print("\nTesting required packages...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} successfully imported")
        except ImportError:
            print(f"✗ {package} not available")
            return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! PyTorch environment is ready.")
    print("\nYou can now run:")
    print("  python Round2/Gradient-Based-Learning-PyTorch.py")
    print("  jupyter notebook Round2/Round2_SGD-PyTorch.ipynb")
    
    return True

if __name__ == "__main__":
    success = test_pytorch()
    sys.exit(0 if success else 1)
