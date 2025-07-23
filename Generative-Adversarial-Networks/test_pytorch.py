#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np

def test_pytorch_installation():
    """Test basic PyTorch functionality"""
    print("="*50)
    print("Testing PyTorch Installation")
    print("="*50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test tensor operations
    print("\nTesting tensor operations...")
    x = torch.randn(5, 3, device=device)
    y = torch.randn(3, 4, device=device)
    z = torch.mm(x, y)
    print(f"Matrix multiplication result shape: {z.shape}")
    
    # Test simple neural network
    print("\nTesting neural network...")
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.layer = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.layer(x)
    
    net = SimpleNet().to(device)
    test_input = torch.randn(1, 10, device=device)
    output = net(test_input)
    print(f"Network output: {output.item():.4f}")
    
    # Test optimizer
    print("\nTesting optimizer...")
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    target = torch.tensor([[1.0]], device=device)
    loss = criterion(output, target)
    print(f"Initial loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    print("Backward pass and optimizer step completed successfully!")
    
    print("\n" + "="*50)
    print("PyTorch installation test PASSED!")
    print("="*50)

if __name__ == "__main__":
    test_pytorch_installation()
