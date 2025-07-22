# Gradient-Based-Learning-PyTorch.py
# This file is part of the Deep Learning and PyTorch course
# PyTorch implementation of gradient descent algorithms

import os
import numpy as np                                
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from sklearn.datasets import make_regression

from utils import load_styles

# load_styles()

# =============================================================================
# CUDA DEVICE SETUP AND OPTIMIZATION
# =============================================================================

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device('cuda')
    cuda_device_name = torch.cuda.get_device_name(0)
    cuda_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    print(f"✅ CUDA is available!")
    print(f"   Device: {cuda_device_name}")
    print(f"   Memory: {cuda_memory:.1f} GB")
    print(f"   Using GPU acceleration for PyTorch computations")
else:
    device = torch.device('cpu')
    print(f"⚠️  CUDA is not available - using CPU")
    print(f"   PyTorch will run on CPU (slower but still functional)")

print(f"\nSelected device: {device}")

print(f"\n{'='*60}")
print("WHY PYTORCH + CUDA IS OPTIMIZED FOR DEEP LEARNING")
print(f"{'='*60}")

print("""
🚀 TENSOR OPERATIONS ON GPU:
• Tensors are multi-dimensional arrays perfect for parallel processing
• GPU has thousands of cores vs CPU's few cores
• Matrix operations (dot products, convolutions) are highly parallelizable
• PyTorch tensors can seamlessly move between CPU and GPU

⚡ CUDA ADVANTAGES:
• Massive parallelization: 1000s of threads vs CPU's 8-16 threads  
• Memory bandwidth: GPU memory is 10x faster than system RAM
• Specialized cores: Tensor cores optimized for AI/ML computations
• Automatic memory management and optimization

🎯 PYTORCH CUDA OPTIMIZATIONS:
• Automatic kernel fusion: Combines operations for efficiency
• Memory pooling: Reduces allocation overhead
• Mixed precision: Uses FP16 for speed, FP32 for accuracy
• Asynchronous execution: Overlaps computation with memory transfers

📊 TYPICAL SPEEDUPS:
• Linear algebra operations: 10-50x faster on GPU
• Neural network training: 5-20x faster overall
• Large batch processing: Up to 100x faster
• Gradient computations: Highly parallelized
""")

if cuda_available:
    print("✅ This session will benefit from GPU acceleration!")
else:
    print("💡 For GPU acceleration, ensure CUDA-compatible GPU and drivers are installed")

print(f"{'='*60}\n")

def gradient_step_onefeature_pytorch(x, y, weight, lrate):
    '''
    PyTorch implementation for performing gradient descent step for linear predictor and MSE as loss function.
    
    The inputs to the function are the following parameters:
     - torch.Tensor with feature values x of shape (m,1)
     - torch.Tensor with labels y of shape (m,1)
     - scalar value `weight`, which is the weight used for computing the prediction
     - scalar value `lrate`, which is a coefficient alpha used during weight update (learning rate)

    The function will return a new weight guess (updated weight value) and the current MSE value.   
    '''
    
    # Convert to PyTorch tensors if they aren't already and move to device
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32).to(device)
    else:
        x = x.to(device)
        
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32).to(device)
    else:
        y = y.to(device)
    
    weight_tensor = torch.tensor(weight, dtype=torch.float32, requires_grad=True, device=device)
    
    # 1. compute predictions, given the weight vector w
    y_hat = torch.matmul(x, weight_tensor).flatten()
    
    # 2. compute MSE loss using PyTorch's MSE loss function
    mse_loss = nn.MSELoss()
    loss = mse_loss(y_hat, y.flatten())
    
    # 3. compute the gradient using PyTorch's autograd
    loss.backward()
    grad_w = weight_tensor.grad
    
    # 4. update the weights
    with torch.no_grad():
        weight_new = weight_tensor - lrate * grad_w
    
    return weight_new.item(), loss.item()

def gradient_step_onefeature(x,y,weight,lrate):
    '''
    This is a function for performing gradient descent step for linear predictor and MSE as loss function.
    
    The inputs to the function are the following parameters:

     - numpy array with feature values x of shape (m,1)
     - numpy array with labels y of shape (m,1)
     - scalar value `weight`, which is the weight used for computing the prediction
     - scalar value `lrate`, which is a coefficient alpha used during weight update (learning rate)

    The function will return a new weight guess (updated weight value) and the current MSE value.   
    
    '''

    # performing the Gradient Step:
    # 1. compute predictions, given the weight vector w

    y_hat = x.dot(weight).flatten()
    
    #print('y_hat', y_hat)
    
    # 2. compute MSE loss
    error = y.flatten() - y_hat
    
    m = len(y)

    MSE = ((1.0 / m) * (np.sum(np.power(error,2))))
    
    #print('MSE', MSE)
    
    # 3. compute the average gradient of the loss function
   
    grad_w = (-2/m)*(error.dot(x))
    
    #print('grad_w', grad_w)
    
    # 4. update the weights
    weight = (weight - (lrate * grad_w))[0]
    
    #print('weight', weight)

    return weight, MSE

#from round02 import test_gradient_step_one_feature

#test_gradient_step_one_feature(gradient_step_onefeature)

def GD_onefeature_pytorch(x, y, epochs, lrate):
    '''
    PyTorch implementation for performing gradient descent for linear predictor and MSE as loss function.
    The helper function `gradient_step_onefeature_pytorch` performs gradient step for dataset of size `m`, 
    where each datapoint has only one feature. 
    
    The inputs to the function `GD_onefeature_pytorch()` are the following parameters:
    - numpy array or torch.Tensor with the feature values x of shape (m,1)
    - numpy array or torch.Tensor with the labels y of shape (m,1)
    - scalar value `epochs`, which is the number of epochs 
    - scalar value `lrate`, which is the coefficient alpha used during weight update (learning rate)
    '''
    
    # Convert to PyTorch tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # initialize weight vector randomly
    torch.manual_seed(42)
    weight = torch.rand(1).item()
    
    # create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):
        # run the gradient step for the whole data set
        weight, MSE = gradient_step_onefeature_pytorch(x, y, weight, lrate)
        # store current weight and training loss 
        weights.append(weight)
        loss.append(MSE)
                       
    return weights, loss

def GD_onefeature(x,y,epochs,lrate):  
    
    '''
    This is a function for performing gradient descent for linear predictor and MSE as loss function.
    The helper function `gradient_step_onefeature` performs gradient step for dataset of size `m`, 
    where each datapoint has only one feature. 
    
    The inputs to the function `GD_onefeature()` are the following parameters:
    - numpy array with the feature values x of shape (m,1)
    - numpy array with the labels y of shape (m,1)
    - scalar value `epochs`, which is the number of epochs 
    - scalar value `lrate`, which is the coefficient alpha used during weight update (learning rate)
    
    '''
    
    # initialize weight vector randomly
    np.random.seed(42)
    weight = np.random.rand()    
    # create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):
        # run the gradient step for the whole data set
        weight, MSE = gradient_step_onefeature(x,y,weight,lrate)
        # store current weight and training loss 
        weights.append(weight)
        loss.append(MSE)
                       
    return weights, loss

# Generate dataset for regression problem
x, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42) 
y = y.reshape(-1,1)
x = preprocessing.scale(x)

# set epoches and learning rate
epochs = 100
lrate = 0.1

# Compare PyTorch vs NumPy implementation
print("Comparing PyTorch vs NumPy implementations:")

# NumPy implementation
(weights_numpy, loss_numpy) = GD_onefeature(x, y, epochs, lrate)
print(f"NumPy final weight: {weights_numpy[-1]:.6f}, final loss: {loss_numpy[-1]:.6f}")

# PyTorch implementation
(weights_pytorch, loss_pytorch) = GD_onefeature_pytorch(x, y, epochs, lrate)
print(f"PyTorch final weight: {weights_pytorch[-1]:.6f}, final loss: {loss_pytorch[-1]:.6f}")

# plot loss and weight values
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# NumPy results
ax[0,0].plot(weights_numpy, loss_numpy, 'b-', label='NumPy')
ax[0,0].set_xlabel("weight", fontsize=14)
ax[0,0].set_ylabel("Loss", fontsize=14)
ax[0,0].set_title("Loss vs Weights (NumPy)", fontsize=16)
ax[0,0].legend()

ax[0,1].plot(range(epochs), loss_numpy, 'b-', label='NumPy')
ax[0,1].set_xlabel("epoch", fontsize=14)
ax[0,1].set_ylabel("Loss", fontsize=14)
ax[0,1].set_title("Loss vs Epochs (NumPy)", fontsize=16)
ax[0,1].legend()

# PyTorch results
ax[1,0].plot(weights_pytorch, loss_pytorch, 'r-', label='PyTorch')
ax[1,0].set_xlabel("weight", fontsize=14)
ax[1,0].set_ylabel("Loss", fontsize=14)
ax[1,0].set_title("Loss vs Weights (PyTorch)", fontsize=16)
ax[1,0].legend()

ax[1,1].plot(range(epochs), loss_pytorch, 'r-', label='PyTorch')
ax[1,1].set_xlabel("epoch", fontsize=14)
ax[1,1].set_ylabel("Loss", fontsize=14)
ax[1,1].set_title("Loss vs Epochs (PyTorch)", fontsize=16)
ax[1,1].legend()

plt.tight_layout()
plt.show()

def gradient_step_pytorch(X, y, weight, lrate):
    '''
    PyTorch implementation for performing gradient step with MSE loss function.
      
    The inputs:
    - torch.Tensor (matrix) with feature values X of shape (m,n)
    - torch.Tensor with labels y of shape (m,1)
    - torch.Tensor `weight` of shape (n,1), which is the weight used for computing prediction
    - scalar value `lrate`, which is a coefficient alpha used during weight update

    The function will return a new weight guess (updated weight value) and current MSE value.   
    '''
    
    # Convert to PyTorch tensors if they aren't already
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    else:
        weight = weight.detach().clone().requires_grad_(True)
    
    # 1. Compute predictions
    y_hat = torch.matmul(X, weight).flatten()
    
    # 2. compute MSE loss
    mse_loss = nn.MSELoss()
    loss = mse_loss(y_hat, y.flatten())
    
    # 3. compute gradient using PyTorch's autograd
    loss.backward()
    
    # 4. update the weights
    with torch.no_grad():
        weight_new = weight - lrate * weight.grad
    
    return weight_new.detach().numpy(), loss.item()

def gradient_step(X, y, weight, lrate):
    
    '''
    This is a function for performing gradient step with MSE loss function.
      
     The inputs:
    - numpy array (matrix) with feature values X of shape (m,n)
    - numpy array with labels y of shape (m,1)
    - numpy array `weight` of shape (n,1), which is the weight used for computing prediction
    - scalar value `lrate`, which is a coefficient alpha used during weight update

    The function will return a new weight guess (updated weight value) and current MSE value.   
    
    '''

    # performing Gradient Step:

    # 1. Compute predictions, given the feature matrix X of shape (m,n) and weight vector w of shape (n,1).
    #    Predictions should be stored in an array `y_hat` of shape (m,1).
    y_hat = (X @ weight).flatten()
    
    # 2. compute MSE loss
    
    error = y.flatten() - y_hat
    
    m = len(y)
    
    MSE = ((1.0 / m) * (np.sum(np.power(error,2))))
    
    # 3. compute average gradient of loss function
    gradient = ((-2/m*X.T) @ error)
    
    # 4. update the weights

    for i in range(len(weight)):
        weight[i] = (weight[i] - (lrate * gradient[i]))
    
    return weight, MSE 

# from round02 import test_gradient_step

# test_gradient_step(gradient_step)

def GD_pytorch(X, y, epochs, lrate):
    '''
    PyTorch implementation for performing gradient descent for linear predictor and MSE as loss function.
    The helper function `gradient_step_pytorch` performs gradient step for dataset of size `m`, 
    where each datapoint has `n` features. 
    '''
    
    # Convert to PyTorch tensors
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # initialize the weight vector randomly
    torch.manual_seed(42)
    weight = torch.rand(X.shape[1], 1, dtype=torch.float32)
    
    # create a list to store the loss values 
    weights = []
    loss = []
     
    for i in range(epochs):
        # run the gradient step for the whole dataset
        weight, MSE = gradient_step_pytorch(X, y, weight, lrate)
        # store the MSE loss of each batch of each epoch
        weights.append(weight.copy())
        loss.append(MSE)
                       
    return weights, loss

def GD(X,y,epochs,lrate):  
    '''
    This is a function for performing gradient descent for linear predictor and MSE as loss function.
    The helper function `gradient_step` performs gradient step for dataset of size `m`, 
    where each datapoint has `n` features. 
    
    '''
    
    # initialize the weight vector randomly
    np.random.seed(42)
    weight = np.random.rand(X.shape[1],1)    
    # create a list to store the loss values 
    weights = []
    loss = []
     
    for i in range(epochs):
        # run the gradient step for the whole dataset
        weight, MSE = gradient_step(X, y, weight, lrate)
        # store the MSE loss of each batch of each epoch
        weights.append(weight)
        loss.append(MSE)
                       
    return weights, loss

# generate a dataset for a regression problem and set number of features to four
X2, y2 = make_regression(n_samples=100, n_features=4, noise=20, random_state=42) 
y2 = y2.reshape(-1,1)

X2 = preprocessing.scale(X2)

# set epoch and learning rate
epochs = 100
lrate = 0.1

print("\nComparing multi-feature gradient descent:")

# NumPy implementation
(weights_numpy, loss_numpy) = GD(X2, y2, epochs, lrate)
print(f"NumPy final loss: {loss_numpy[-1]:.6f}")

# PyTorch implementation
(weights_pytorch, loss_pytorch) = GD_pytorch(X2, y2, epochs, lrate)
print(f"PyTorch final loss: {loss_pytorch[-1]:.6f}")

# plot the cost function comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].set_ylabel('Loss', fontsize=16)
ax[0].set_xlabel('epochs', fontsize=16)
ax[0].plot(range(epochs), loss_numpy, 'b.', label='NumPy')
ax[0].set_title('NumPy Implementation')
ax[0].legend()

ax[1].set_ylabel('Loss', fontsize=16)
ax[1].set_xlabel('epochs', fontsize=16)
ax[1].plot(range(epochs), loss_pytorch, 'r.', label='PyTorch')
ax[1].set_title('PyTorch Implementation')
ax[1].legend()

plt.tight_layout()
plt.show()

# =============================================================================
# ADAM OPTIMIZER EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("ADAM OPTIMIZER EXAMPLES")
print("="*60)

class LinearModel(nn.Module):
    """Simple linear model for Adam optimizer demonstration"""
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)
        
    def forward(self, x):
        return self.linear(x)

def train_with_adam(X, y, epochs=100, lr=0.001, betas=(0.9, 0.999)):
    """Train linear model using Adam optimizer with CUDA support"""
    
    print(f"Training with Adam: lr={lr}, betas={betas}")
    print(f"Device: {device}")
    
    # Convert to PyTorch tensors and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Initialize model and move to device
    model = LinearModel(X.shape[1]).to(device)
    
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=1e-8)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    weights = []
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        # Store metrics (move to CPU for numpy conversion)
        losses.append(loss.item())
        weights.append(model.linear.weight.data.cpu().clone().numpy().flatten()[0])
        
        if (epoch + 1) % 25 == 0:
            print(f'  Epoch [{epoch+1:3d}/{epochs}], Loss: {loss.item():.6f}, Weight: {weights[-1]:.6f}')
    
    return model, losses, weights

def train_with_sgd_comparison(X, y, epochs=100, lr=0.01):
    """Train linear model using SGD optimizer for comparison with CUDA support"""
    
    print(f"Training with SGD: lr={lr}")
    print(f"Device: {device}")
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)
    
    model = LinearModel(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    weights = []
    
    for epoch in range(epochs):
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        weights.append(model.linear.weight.data.cpu().clone().numpy().flatten()[0])
        
        if (epoch + 1) % 25 == 0:
            print(f'  Epoch [{epoch+1:3d}/{epochs}], Loss: {loss.item():.6f}, Weight: {weights[-1]:.6f}')
    
    return model, losses, weights

# Example 1: Basic Adam vs SGD comparison
print("\n1. BASIC ADAM vs SGD COMPARISON")
print("-" * 40)

# Use the same dataset
model_adam, losses_adam, weights_adam = train_with_adam(X2, y2, epochs=100, lr=0.001)
model_sgd, losses_sgd, weights_sgd = train_with_sgd_comparison(X2, y2, epochs=100, lr=0.01)

print(f"\nFinal Results:")
print(f"Adam - Final Loss: {losses_adam[-1]:.6f}, Final Weight: {weights_adam[-1]:.6f}")
print(f"SGD  - Final Loss: {losses_sgd[-1]:.6f}, Final Weight: {weights_sgd[-1]:.6f}")

# Example 2: Adam hyperparameter comparison
print("\n2. ADAM HYPERPARAMETER COMPARISON")
print("-" * 40)

adam_configs = [
    {'lr': 0.001, 'betas': (0.9, 0.999), 'name': 'Adam Default'},
    {'lr': 0.01, 'betas': (0.9, 0.999), 'name': 'Adam High LR'},
    {'lr': 0.001, 'betas': (0.5, 0.999), 'name': 'Adam Low β1'},
    {'lr': 0.001, 'betas': (0.9, 0.99), 'name': 'Adam Low β2'},
]

adam_results = {}

for config in adam_configs:
    print(f"\n{config['name']}: lr={config['lr']}, betas={config['betas']}")
    model, losses, weights = train_with_adam(X2, y2, epochs=50, 
                                           lr=config['lr'], 
                                           betas=config['betas'])
    adam_results[config['name']] = {
        'losses': losses,
        'final_loss': losses[-1],
        'final_weight': weights[-1]
    }
    print(f"  Final: Loss={losses[-1]:.6f}, Weight={weights[-1]:.6f}")

# Visualization
print("\n3. VISUALIZATION")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Adam Optimizer Analysis', fontsize=16)

# Adam vs SGD Loss comparison
axes[0, 0].plot(losses_adam, 'r-', label='Adam (lr=0.001)', linewidth=2)
axes[0, 0].plot(losses_sgd, 'b-', label='SGD (lr=0.01)', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Convergence: Adam vs SGD')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Adam vs SGD Weight evolution
axes[0, 1].plot(weights_adam, 'r-', label='Adam', linewidth=2)
axes[0, 1].plot(weights_sgd, 'b-', label='SGD', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Weight Value')
axes[0, 1].set_title('Weight Evolution: Adam vs SGD')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Adam hyperparameter comparison
colors = ['red', 'blue', 'green', 'orange']
for i, (name, result) in enumerate(adam_results.items()):
    axes[1, 0].plot(result['losses'], color=colors[i], 
                   label=name.split()[1], linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Adam Hyperparameter Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Final loss comparison (bar chart)
names = [config['name'].split()[1] for config in adam_configs]
final_losses = [adam_results[config['name']]['final_loss'] for config in adam_configs]
bars = axes[1, 1].bar(names, final_losses, color=colors)
axes[1, 1].set_xlabel('Configuration')
axes[1, 1].set_ylabel('Final Loss')
axes[1, 1].set_title('Final Loss by Configuration')
for bar, loss in zip(bars, final_losses):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{loss:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Example 3: Learning rate sensitivity analysis
print("\n4. LEARNING RATE SENSITIVITY ANALYSIS")
print("-" * 40)

learning_rates = [0.0001, 0.001, 0.01, 0.1]
lr_results = []

for lr in learning_rates:
    print(f"Testing learning rate: {lr}")
    model, losses, weights = train_with_adam(X2, y2, epochs=25, lr=lr)
    lr_results.append({
        'lr': lr,
        'final_loss': losses[-1],
        'converged': losses[-1] < losses[0] * 0.1  # Converged if loss reduced by 90%
    })

print("\nLearning Rate Analysis:")
for result in lr_results:
    status = "✓ Converged" if result['converged'] else "✗ Did not converge"
    print(f"  LR: {result['lr']:6.4f} - Final Loss: {result['final_loss']:8.6f} - {status}")

# Plot learning rate sensitivity
plt.figure(figsize=(10, 6))
lrs = [r['lr'] for r in lr_results]
losses = [r['final_loss'] for r in lr_results]
colors = ['green' if r['converged'] else 'red' for r in lr_results]

plt.semilogx(lrs, losses, 'o-', linewidth=2, markersize=10)
for i, (lr, loss, color) in enumerate(zip(lrs, losses, colors)):
    plt.scatter(lr, loss, color=color, s=100, zorder=5)
    plt.text(lr, loss + 0.5, f'{loss:.3f}', ha='center', va='bottom')

plt.xlabel('Learning Rate (log scale)', fontsize=12)
plt.ylabel('Final Loss (25 epochs)', fontsize=12)
plt.title('Adam Learning Rate Sensitivity Analysis', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(['Learning Rate Sweep', 'Converged', 'Did Not Converge'], loc='upper left')
plt.show()

print("\n" + "="*60)
print("ADAM OPTIMIZER SUMMARY")
print("="*60)
print("Key Adam Advantages:")
print("1. ✓ Adaptive learning rates per parameter")
print("2. ✓ Combines momentum (β₁) and RMSprop (β₂)")
print("3. ✓ Generally faster initial convergence")
print("4. ✓ Works well with default hyperparameters")
print("5. ✓ Handles sparse gradients effectively")
print("\nWhen to use Adam:")
print("• Quick prototyping and experimentation")
print("• When you don't have time to tune SGD hyperparameters")  
print("• Problems with sparse features or gradients")
print("• Non-stationary optimization landscapes")
print("\nPyTorch Adam usage:")
print("  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)")
print("  # Default betas=(0.9, 0.999), eps=1e-8")
print("="*60)

# =============================================================================
# BATCH vs MINI-BATCH IMPLEMENTATION (PyTorch Version)
# =============================================================================

print("\n" + "="*70)
print("PYTORCH BATCH vs MINI-BATCH TRAINING")
print("="*70)

class LinearModelBatch(nn.Module):
    """Linear model for batch vs mini-batch comparison"""
    def __init__(self, input_size):
        super(LinearModelBatch, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=True)
        
    def forward(self, x):
        return self.linear(x)

def create_pytorch_batches(X, y, batch_size, shuffle=True):
    """Create mini-batches using PyTorch DataLoader"""
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=cuda_available and X.device.type == 'cpu'  # Only pin CPU tensors
    )
    return dataloader

# =============================================================================
# ADVANCED DATALOADER EXAMPLES AND ALTERNATIVES
# =============================================================================

def create_advanced_dataloader(X, y, batch_size, shuffle=True, num_workers=0, drop_last=False):
    """
    Advanced DataLoader creation with additional options
    
    Parameters:
    - X, y: Input features and labels
    - batch_size: Number of samples per batch
    - shuffle: Whether to shuffle the data each epoch
    - num_workers: Number of worker processes for data loading (0 = single process)
    - drop_last: Whether to drop the last incomplete batch
    """
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=cuda_available and X.device.type == 'cpu'  # Only pin CPU tensors
    )
    return dataloader

def manual_batch_creation(X, y, batch_size):
    """
    Manual batch creation WITHOUT DataLoader - for comparison
    """
    print(f"Creating batches manually (batch_size={batch_size})")
    
    n_samples = len(X)
    indices = torch.randperm(n_samples)  # Shuffle indices
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_X, batch_y))
    
    return batches

def dataloader_vs_manual_comparison(X, y, batch_size=16):
    """
    Compare DataLoader vs Manual batch creation approaches
    """
    print(f"\n{'='*60}")
    print("DATALOADER vs MANUAL BATCH CREATION COMPARISON")
    print(f"{'='*60}")
    
    import time
    
    # Convert to tensors but keep on CPU for DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(X) // batch_size}")
    
    # Method 1: DataLoader
    print(f"\n📦 METHOD 1: PyTorch DataLoader")
    start_time = time.time()
    dataloader = create_advanced_dataloader(X_tensor, y_tensor, batch_size, shuffle=True)
    
    batch_count = 0
    for batch_X, batch_y in dataloader:
        batch_count += 1
        # Simulate processing
        _ = batch_X.shape, batch_y.shape
    
    dataloader_time = time.time() - start_time
    print(f"   Time: {dataloader_time:.4f} seconds")
    print(f"   Batches processed: {batch_count}")
    print(f"   Memory efficient: ✅")
    print(f"   Shuffle support: ✅")
    print(f"   Multiprocessing support: ✅")
    
    # Method 2: Manual batching
    print(f"\n🔧 METHOD 2: Manual Batch Creation")
    start_time = time.time()
    manual_batches = manual_batch_creation(X_tensor, y_tensor, batch_size)
    
    batch_count = 0
    for batch_X, batch_y in manual_batches:
        batch_count += 1
        # Simulate processing
        _ = batch_X.shape, batch_y.shape
    
    manual_time = time.time() - start_time
    print(f"   Time: {manual_time:.4f} seconds")
    print(f"   Batches processed: {batch_count}")
    print(f"   Memory efficient: ⚠️ (loads all data)")
    print(f"   Shuffle support: ✅")
    print(f"   Multiprocessing support: ❌")
    
    # Comparison
    print(f"\n📊 PERFORMANCE COMPARISON:")
    if dataloader_time < manual_time:
        speedup = manual_time / dataloader_time
        print(f"   DataLoader is {speedup:.2f}x faster")
    else:
        slowdown = dataloader_time / manual_time
        print(f"   Manual method is {slowdown:.2f}x faster")
    
    print(f"\n💡 KEY DIFFERENCES:")
    print(f"   • DataLoader: Built-in shuffling, multiprocessing, memory efficiency")
    print(f"   • Manual: Full control, but requires more implementation")
    print(f"   • DataLoader is preferred for production code")

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class example - alternative to TensorDataset
    Demonstrates how to create custom data loading logic
    """
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample_X = self.X[idx]
        sample_y = self.y[idx]
        
        if self.transform:
            sample_X = self.transform(sample_X)
            
        return sample_X, sample_y

def create_custom_dataset_loader(X, y, batch_size=32, shuffle=True):
    """
    Create DataLoader using custom Dataset class
    """
    print(f"Creating DataLoader with Custom Dataset class")
    
    # Convert to tensors if needed
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Create custom dataset
    dataset = CustomDataset(X, y)
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=cuda_available and X.device.type == 'cpu'  # Only pin CPU tensors
    )
    
    return dataloader

def train_pytorch_full_batch(X, y, epochs=50, lr=0.01):
    """Full batch training in PyTorch with CUDA support"""
    print(f"PyTorch FULL BATCH training (dataset size: {len(X)})")
    print(f"Device: {device}")
    
    # Convert to tensors and move to device (GPU/CPU)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Initialize model and move to device
    model = LinearModelBatch(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training
    losses = []
    weights = []
    
    for epoch in range(epochs):
        # Full batch forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store metrics (move to CPU for numpy conversion)
        losses.append(loss.item())
        weights.append(model.linear.weight.data.cpu().clone().numpy().flatten()[0])
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}, Loss: {loss.item():.6f}, Weight: {weights[-1]:.6f}")
    
    return model, losses, weights

def train_pytorch_mini_batch(X, y, epochs=50, lr=0.01, batch_size=10):
    """Mini-batch training in PyTorch with CUDA support"""
    print(f"PyTorch MINI-BATCH training (batch size: {batch_size})")
    print(f"Device: {device}")
    
    # Convert to tensors but keep on CPU for DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # Initialize model and move to device
    model = LinearModelBatch(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Create data loader (with CPU tensors)
    dataloader = create_pytorch_batches(X_tensor, y_tensor, batch_size)
    
    # Training
    losses = []
    weights = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_X, batch_y in dataloader:
            # Move batch to device if needed (DataLoader might not preserve device)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Mini-batch forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        weights.append(model.linear.weight.data.cpu().clone().numpy().flatten()[0])
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}, Loss: {avg_loss:.6f}, Weight: {weights[-1]:.6f}")
    
    return model, losses, weights

def train_pytorch_sgd(X, y, epochs=25, lr=0.01):
    """Stochastic Gradient Descent in PyTorch (batch_size=1) with CUDA support"""
    print(f"PyTorch SGD training (batch size: 1)")
    print(f"Device: {device}")
    
    # Convert to tensors but keep on CPU for DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # Initialize model and move to device
    model = LinearModelBatch(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Create data loader with batch_size=1 (with CPU tensors)
    dataloader = create_pytorch_batches(X_tensor, y_tensor, batch_size=1)
    
    # Training
    losses = []
    weights = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_X, batch_y in dataloader:
            # Move batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Single sample forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        weights.append(model.linear.weight.data.cpu().clone().numpy().flatten()[0])
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}, Loss: {avg_loss:.6f}, Weight: {weights[-1]:.6f}")
    
    return model, losses, weights

# Generate a larger dataset for PyTorch batch comparison
print("\nGenerating dataset for PyTorch batch comparison...")
np.random.seed(42)
X_pytorch, y_pytorch = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
X_pytorch = preprocessing.scale(X_pytorch)  # Normalize features
y_pytorch = preprocessing.scale(y_pytorch.reshape(-1, 1)).flatten()  # Normalize target

print(f"PyTorch dataset shape: X={X_pytorch.shape}, y={y_pytorch.shape}")

# Run PyTorch batch comparisons
print("\n1. PyTorch FULL BATCH:")
model_full, losses_full, weights_full = train_pytorch_full_batch(X_pytorch, y_pytorch, epochs=50, lr=0.01)

print("\n2. PyTorch MINI-BATCH (batch_size=20):")
model_mini20, losses_mini20, weights_mini20 = train_pytorch_mini_batch(X_pytorch, y_pytorch, epochs=50, lr=0.01, batch_size=20)

print("\n3. PyTorch MINI-BATCH (batch_size=50):")
model_mini50, losses_mini50, weights_mini50 = train_pytorch_mini_batch(X_pytorch, y_pytorch, epochs=50, lr=0.01, batch_size=50)

print("\n4. PyTorch SGD (batch_size=1):")
model_sgd_pt, losses_sgd_pt, weights_sgd_pt = train_pytorch_sgd(X_pytorch, y_pytorch, epochs=25, lr=0.01)

# Advanced PyTorch features: Different optimizers with mini-batches
print("\n5. OPTIMIZER COMPARISON with Mini-Batches:")

def train_with_optimizer(X, y, optimizer_name, epochs=30, batch_size=32):
    """Train with different PyTorch optimizers with CUDA support"""
    # Keep tensors on CPU for DataLoader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    model = LinearModelBatch(X.shape[1]).to(device)
    criterion = nn.MSELoss()
    
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    dataloader = create_pytorch_batches(X_tensor, y_tensor, batch_size)
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        for batch_X, batch_y in dataloader:
            # Move batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        losses.append(np.mean(epoch_losses))
    
    print(f"  {optimizer_name}: Final Loss = {losses[-1]:.6f}")
    return model, losses

# Compare optimizers
optimizers = ['SGD', 'Adam', 'RMSprop']
optimizer_results = {}

for opt_name in optimizers:
    model_opt, losses_opt = train_with_optimizer(X_pytorch, y_pytorch, opt_name, epochs=30, batch_size=32)
    optimizer_results[opt_name] = losses_opt

# Comprehensive PyTorch visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('PyTorch Batch vs Mini-Batch Analysis', fontsize=16)

# Loss convergence comparison
axes[0, 0].plot(losses_full, 'b-', linewidth=2, label='Full Batch')
axes[0, 0].plot(losses_mini20, 'r-', linewidth=2, label='Mini-Batch (20)')
axes[0, 0].plot(losses_mini50, 'g-', linewidth=2, label='Mini-Batch (50)')
axes[0, 0].plot(losses_sgd_pt, 'm-', linewidth=2, label='SGD (1)', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('PyTorch: Loss Convergence')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Weight evolution (first weight only for multi-feature model)
axes[0, 1].plot(weights_full, 'b-', linewidth=2, label='Full Batch')
axes[0, 1].plot(weights_mini20, 'r-', linewidth=2, label='Mini-Batch (20)')
axes[0, 1].plot(weights_mini50, 'g-', linewidth=2, label='Mini-Batch (50)')
axes[0, 1].plot(weights_sgd_pt, 'm-', linewidth=2, label='SGD (1)', alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('First Weight Value')
axes[0, 1].set_title('PyTorch: Weight Evolution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Optimizer comparison
for opt_name, losses in optimizer_results.items():
    axes[0, 2].plot(losses, linewidth=2, label=f'{opt_name}')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].set_title('PyTorch: Optimizer Comparison (Batch=32)')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Memory usage analysis
batch_sizes_analysis = [1, 10, 20, 50, 100, len(X_pytorch)]
memory_usage = [size * X_pytorch.shape[1] * 4 / 1024 for size in batch_sizes_analysis]  # KB

axes[1, 0].semilogx(batch_sizes_analysis, memory_usage, 'o-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Batch Size (log scale)')
axes[1, 0].set_ylabel('Memory Usage (KB)')
axes[1, 0].set_title('Memory Usage vs Batch Size')
axes[1, 0].grid(True, alpha=0.3)

# Training speed simulation (updates per epoch)
updates_per_epoch = [len(X_pytorch) // size for size in batch_sizes_analysis[:-1]] + [1]
axes[1, 1].bar(range(len(batch_sizes_analysis)), updates_per_epoch, color='skyblue', alpha=0.7)
axes[1, 1].set_xlabel('Batch Configuration')
axes[1, 1].set_ylabel('Updates per Epoch')
axes[1, 1].set_title('Gradient Updates per Epoch')
axes[1, 1].set_xticks(range(len(batch_sizes_analysis)))
axes[1, 1].set_xticklabels([f'B={size}' if size < len(X_pytorch) else 'Full' for size in batch_sizes_analysis], rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Final performance comparison
methods = ['Full Batch', 'Mini-20', 'Mini-50', 'SGD']
final_losses_pt = [losses_full[-1], losses_mini20[-1], losses_mini50[-1], losses_sgd_pt[-1]]

bars = axes[1, 2].bar(methods, final_losses_pt, color=['blue', 'red', 'green', 'magenta'], alpha=0.7)
axes[1, 2].set_ylabel('Final Loss')
axes[1, 2].set_title('PyTorch: Final Performance Comparison')
axes[1, 2].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, loss in zip(bars, final_losses_pt):
    height = bar.get_height()
    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{loss:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("PYTORCH BATCH vs MINI-BATCH SUMMARY")
print("="*70)
print("Final Results:")
print(f"Full Batch     - Final Loss: {losses_full[-1]:.6f}")
print(f"Mini-Batch(20) - Final Loss: {losses_mini20[-1]:.6f}")
print(f"Mini-Batch(50) - Final Loss: {losses_mini50[-1]:.6f}")
print(f"SGD (size=1)   - Final Loss: {losses_sgd_pt[-1]:.6f}")

print(f"\nOptimizer Comparison (batch_size=32):")
for opt_name, losses in optimizer_results.items():
    print(f"{opt_name:10s} - Final Loss: {losses[-1]:.6f}")

print("\nPyTorch Advantages for Mini-Batch Training:")
print("✓ Built-in DataLoader for efficient batching")
print("✓ Automatic GPU parallelization within batches")
print("✓ Memory-efficient gradient computation")
print("✓ Advanced optimizers (Adam, RMSprop, etc.)")
print("✓ Dynamic computation graphs")
print("✓ Easy integration with neural networks")

print("\nDataLoader Benefits:")
print("• Automatic shuffling between epochs")
print("• Efficient memory usage with large datasets")
print("• Multi-threading support")
print("• Handles uneven batch sizes automatically")
print("="*70)
