# PyTorch migration from Python

## Overview

This document summarizes the migration from TensorFlow/Keras to PyTorch for the Gradient-Based Learning project.

## Files created/modified

### New PyTorch files

1. **Gradient-Based-Learning-PyTorch.py** - PyTorch implementation of all gradient descent algorithms
2. **Round2_SGD-PyTorch.ipynb** - PyTorch version of the Jupyter notebook
3. **requirements.txt** - PyTorch dependencies
4. **setup_pytorch_env.sh** - Environment setup script
5. **Dataset/README.md** - Dataset documentation

## Changes

### From TensorFlow/Keras to PyTorch

#### Imports

```python
# Before (TensorFlow/Keras)
import tensorflow as tf
from tensorflow import keras

# After (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
```

#### Tensors

```python
# Before (NumPy)
x = np.array([1.0, 2.0, 3.0])

# After (PyTorch)
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
```

#### Loss functions
```python
# Before (Manual)
MSE = ((1.0 / m) * (np.sum(np.power(error,2))))

# After (PyTorch)
mse_loss = nn.MSELoss()
loss = mse_loss(predictions, targets)
```

#### Gradients
```python
# Before (Manual)
grad_w = (-2/m)*(error.dot(x))

# After (PyTorch)
loss.backward()  # Automatic gradient computation
grad_w = weight.grad
```

## Environment

### Virtual Environment
```bash
# Create environment
python3 -m venv pytorch_env

# Activate (Linux/Mac)
source pytorch_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio matplotlib numpy scikit-learn jupyter
```

### Setup
```bash
# Use the provided script
./setup_pytorch_env.sh
```

## PyTorch

1. **Automatic Differentiation**: No manual gradient computation required
2. **GPU Support**: Seamless CUDA integration
3. **Dynamic Graphs**: More flexible than static graphs
4. **Better Debugging**: Pythonic debugging experience
5. **Rich Ecosystem**: Pre-trained models, datasets, and utilities

## Code comparison

### Gradient computation
```python
# NumPy (Manual)
def gradient_step(X, y, weight, lrate):
    y_hat = (X @ weight).flatten()
    error = y.flatten() - y_hat
    m = len(y)
    MSE = ((1.0 / m) * (np.sum(np.power(error,2))))
    gradient = ((-2/m*X.T) @ error)
    for i in range(len(weight)):
        weight[i] = (weight[i] - (lrate * gradient[i]))
    return weight, MSE

# PyTorch (Automatic)
def gradient_step_pytorch(X, y, weight, lrate):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    weight = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    
    y_hat = torch.matmul(X, weight).flatten()
    loss = nn.MSELoss()(y_hat, y.flatten())
    loss.backward()
    
    with torch.no_grad():
        weight_new = weight - lrate * weight.grad
    
    return weight_new.detach().numpy(), loss.item()
```

## Files
```
Gradient-Based-Learning/
├── pytorch_env/                 # Virtual environment (git ignored)
├── Dataset/                     # Dataset directory
│   └── README.md               # Dataset documentation
├── Round2/
│   ├── Gradient-Based-Learning.py         # Original NumPy version
│   ├── Gradient-Based-Learning-PyTorch.py # New PyTorch version
│   ├── Round2_SGD.ipynb                   # Original notebook
│   └── Round2_SGD-PyTorch.ipynb           # New PyTorch notebook
├── .gitignore                  # Updated with PyTorch exclusions
├── README.md                   # Updated with PyTorch examples
├── requirements.txt            # PyTorch dependencies
└── setup_pytorch_env.sh        # Environment setup script
```

## Running the code

### Python scripts
```bash
# Activate environment
source pytorch_env/bin/activate

# Run PyTorch version
python Round2/Gradient-Based-Learning-PyTorch.py

# Compare with original
python Round2/Gradient-Based-Learning.py
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open PyTorch notebook
Round2/Round2_SGD-PyTorch.ipynb

# Compare with original
Round2/Round2_SGD.ipynb
```

## Testing and validation

PyTorch implementations maintain the same mathematical behavior as the original NumPy code.

- Same random seeds for reproducibility
- Equivalent gradient computations
- Similar convergence behavior
- Compatible output formats

## Next

1. **Model Classes** Create proper PyTorch nn.Module classes
3. **Visualization** Add TensorBoard integration for training monitoring

## Issues

### Pin Memory error fixed

**Issue** `RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned`

**Cause** Using `pin_memory=True` in DataLoader when tensors are already on GPU.

**Solution** Only use pin_memory for CPU tensors:
```python
# ❌ Wrong - causes error if tensors are on GPU
dataloader = DataLoader(dataset, pin_memory=torch.cuda.is_available())

# ✅ Correct - only pins CPU tensors  
dataloader = DataLoader(dataset, 
                       pin_memory=torch.cuda.is_available() and X.device.type == 'cpu')
```

**Obligatory** Keep tensors on CPU for DataLoader creation, move batches to GPU in training loop.

```python
# Create tensors on CPU
X_tensor = torch.tensor(X, dtype=torch.float32)  # CPU
y_tensor = torch.tensor(y, dtype=torch.float32)  # CPU

# DataLoader with pin_memory (safe for CPU tensors)
dataloader = DataLoader(dataset, pin_memory=torch.cuda.is_available())

# Move batches to GPU during training
for batch_X, batch_y in dataloader:
    batch_X = batch_X.to(device)  # Move to GPU
    batch_y = batch_y.to(device)  # Move to GPU
    # ... training code
```

