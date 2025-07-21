#!/usr/bin/env python3
"""
Verification script for PyTorch Deep Learning Environment
"""

import sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd

def verify_environment():
    """Verify that all required packages are installed and working"""
    print("="*60)
    print("PyTorch Deep Learning Environment Verification")
    print("="*60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # Check other packages
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    print("\n" + "="*60)
    print("Testing Basic PyTorch Operations")
    print("="*60)
    
    # Test tensor operations
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y = torch.relu(x - 3)
    print(f"Input tensor: {x}")
    print(f"ReLU(x - 3): {y}")
    
    # Test neural network creation
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    print(f"Model created: {model}")
    
    # Test forward pass
    output = model(x)
    print(f"Forward pass output: {output}")
    
    # Test loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    target = torch.tensor([1.0])
    loss = criterion(output, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("✅ Backward pass completed successfully")
    
    # Test data loading
    from torch.utils.data import DataLoader, TensorDataset
    
    data = torch.randn(100, 5)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    batch_count = len(dataloader)
    print(f"✅ DataLoader created with {batch_count} batches")
    
    # Test Fashion-MNIST dataset
    try:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        # This will download the dataset if not present
        dataset = torchvision.datasets.FashionMNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        print(f"✅ Fashion-MNIST dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"⚠️  Fashion-MNIST download failed: {e}")
    
    print("\n" + "="*60)
    print("Environment Verification Summary")
    print("="*60)
    print("✅ Python environment: OK")
    print("✅ PyTorch installation: OK")
    print("✅ Basic tensor operations: OK")
    print("✅ Neural network creation: OK")
    print("✅ Training components: OK")
    print("✅ Data loading: OK")
    print("✅ All systems ready for deep learning!")
    print("="*60)

if __name__ == "__main__":
    verify_environment()
