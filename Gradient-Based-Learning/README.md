# The mathematics of optimization for Deep Learning

The mathematics of a neural network is a composition of related mappings, and non-linear (activation) mappings, where we alternate between an affine mapping (weights and biases) and an activation function (applied element wise).

**Optimization algorithms**

For any 𝑓(𝑥) function, if the value of 𝑓(𝑥) at 𝑥 is smaller than the values of 𝑓(𝑥) at any other points in the proximity of 𝑥, then 𝑓(𝑥) could be a local minimum. 

If the value of  𝑓(𝑥) at 𝑥 is the minimum of the 𝑓(𝑥) function over the entire domain, then 𝑓(𝑥) is the global minimum.

We can approximate the local minimum and global minimum of the following 𝑓(𝑥)function.

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Gradient-Based-Learning/function.png?raw=true)

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Gradient-Based-Learning/optimization.svg?raw=true)


## Gradient Based Optimization

Gradient Descent (GD) minimizes the training error by incrementally improving the current guess for the optimal parameters by moving a bit into the direction of the negative gradient.

Gradient Descent is used to tune (adjust) the parameters according to the gradient of the average loss incurred by the Artificial Neural Network (ANN) on a training set.

This average loss is also known as the training error and defines a cost function 𝑓(𝐰) that we want to minimize.

For a given pair of predicted label value ŷ and true label value 𝑦, the loss function 𝐿(𝑦,ŷ) provides a measure for the error, or "loss", incurred in predicting the true label 𝑦 by ŷ.

If the label values are numeric (like a temperature), then the squared error loss 𝐿(𝑦,ŷ)=(𝑦−ŷ)² is often a good choice for the loss function. If the label values are categories (like "cat" and "dog"), we might use the "0/1" loss 𝐿(𝑦,ŷ)=0  if and only if 𝑦=ŷ and 𝐿(𝑦,ŷ)=1 otherwise.

Gradient descent (GD) constructs a sequence of parameter vectors 𝐰(0),𝐰(1),... such that the loss values 𝑓(𝐰(0)),𝑓(𝐰(1)),... pass toward the minimum loss. GD is an iterative algorithm that gradually improves the current guess (approximation) 𝐰(𝑘) for the optimum weight vector.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import make_regression

def gradient_step_onefeature_pytorch(x, y, weight, lrate):
    """
    PyTorch implementation for performing gradient descent step for linear predictor and MSE loss function.
    """
    # Convert to PyTorch tensors if they aren't already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    weight_tensor = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    
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

# Test
# from round02 import test_gradient_step_one_feature
# test_gradient_step_one_feature(gradient_step_onefeature_pytorch)
```
**Gradient Descent algorithm with PyTorch**

```python
def GD_onefeature_pytorch(x, y, epochs, lrate):
    """
    PyTorch implementation for performing gradient descent for linear predictor and MSE loss function.
    """
    # Convert to PyTorch tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # Initialize weight vector randomly
    torch.manual_seed(42)
    weight = torch.rand(1).item()
    
    # Create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):
        # Run the gradient step for the whole data set
        weight, MSE = gradient_step_onefeature_pytorch(x, y, weight, lrate)

        # Store current weight and training loss
        weights.append(weight)
        loss.append(MSE)
                 
    return weights, loss

# Generate dataset for regression problem
x, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42) 
y = y.reshape(-1,1)
x = preprocessing.scale(x)

# Set epoches and learning rate
epochs = 100
lrate = 0.1

# Store results
(weights, loss) = GD_onefeature_pytorch(x, y, epochs, lrate)

# Plot loss and weight values
fig, ax = plt.subplots(1,2, figsize=(10,3), sharey=True)

# Loss vs weights plot
ax[0].plot(weights, loss)
ax[0].set_xlabel("weight", fontsize=16)
ax[0].set_ylabel("Loss", fontsize=16)

# Loss vs epoch plot
ax[1].plot(range(epochs), loss)
ax[1].set_xlabel("epoch", fontsize=16)

plt.show()
```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Gradient-Based-Learning/loss_weight.png?raw=true)

**Learning Rate with PyTorch**

***What is the Learning Rate in neural networks?***

The Learning Rate (LR) is a hyperparameter that controls how quickly the model is adapted to the estimated error each time the model weights are updated. PyTorch optimizers like `torch.optim.SGD` and `torch.optim.Adam` use this parameter.

Smaller Learning Rates require more training epochs given the smaller changes made to the weights each update, whereas larger learning rates result in rapid changes and require fewer training epochs.

In PyTorch's backpropagation, model weights are updated to reduce the error estimates of our loss function using automatic differentiation.

***Learning Rate Scheduling in PyTorch***

PyTorch provides several learning rate schedulers in `torch.optim.lr_scheduler` function.

- **StepLR**: Decays learning rate by a factor every few epochs
- **ExponentialLR**: Decays learning rate exponentially
- **CosineAnnealingLR**: Sets learning rate using cosine annealing schedule
- **ReduceLROnPlateau**: Reduces learning rate when metric has stopped improving

```python
epochs = 100
lrates = [0.001, 0.01, 0.1, 0.9]

fig = plt.figure(figsize=(6,4))
weights_list = []
loss_list = []

for lrate in lrates:
    weight, loss = GD_onefeature_pytorch(x, y, epochs, lrate)
    print('lrate', lrate, 'final weight', weight[-1], 'final loss', loss[-1])
    weights_list.append(weight)
    loss_list.append(loss)

# Plot results
for i, lrate in enumerate(lrates):
    plt.plot(weights_list[i], loss_list[i], label=f"lrate{lrate}")
    plt.legend()

plt.xlabel("weight", fontsize=16)
plt.ylabel("Loss", fontsize=16)  
plt.show()
```


```
epochs = 100
lrates = [0.001, 0.01, 0.1, 0.9]

fig = plt.figure(figsize=(6,4))
weights_list = []
loss_list = []

for lrate in lrates:
    weight, loss = GD_onefeature(x, y, epochs, lrate)
    print('lrate', lrate, 'weight', weight, 'loss', loss)
    weights_list.append(weight)
    loss_list.append(loss)

# Plot results
for i,lrate in enumerate(lrates):
    plt.plot(weights_list[i], loss_list[i], label=f"lrate{lrate}")
    plt.legend()

plt.xlabel("weight", fontsize=16)
plt.ylabel("Loss", fontsize=16)  
plt.show()

```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Gradient-Based-Learning/learning_rate.png?raw=true)

**Gradient step with PyTorch**

The inputs to the function are:
- torch.Tensor (matrix) with feature values X of shape (m,n)
- torch.Tensor with labels y of shape (m,1)
- torch.Tensor `weight` of shape (n,1), which is the weight used for computing prediction
- scalar value `lrate`, which is a coefficient alpha used during weight update

The function will return a new weight guess (updated weight value) and current MSE value. 

```python
def gradient_step_pytorch(X, y, weight, lrate):
    """
    PyTorch implementation for performing gradient step with MSE loss function.
    """
    
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

# Test 
# from round02 import test_gradient_step
# test_gradient_step(gradient_step_pytorch)
```
![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Gradient-Based-Learning/vectorized_gradient_descent.png?raw=true)


**Stochastic Gradient Descent**

Deep learning neural networks are trained using the Stochastic Gradient Descent (SGD) algorithm.

Stochastic Gradient Descent is an optimization algorithm that estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using the back-propagation of errors algorithm.

```
def batch(X,y,batch_size):

	# Creating mini-batches of the dataset.
   
    # Check if the number of data points is equal in feature matrix X and label vector y
    
    np.random.seed(42)
    p = np.random.permutation(len(y))
    X_perm = X[p]
    y_perm = y[p]
    
    # Generate batches
    for i in range(0,X.shape[0],batch_size):
        yield (X_perm[i:i + batch_size], y_perm[i:i + batch_size])

def minibatchSGD(X, y, batch_size, epochs, lrate):  
    
    # Initialize the weight randomly
    np.random.seed(42)
    weight = np.random.rand()  
    
    # Create a list to store the loss values 
    loss = []
    weights = []
     
    for i in range(epochs):

        # Use another for-loop to iterate batch() generator and access batches one-by-one
        for mini_batch in batch(X,y,batch_size):

            X_batch, y_batch = mini_batch

            # Feed current batch to `gradient_step_onefeature()` and get weight and loss values
            weight, MSE = gradient_step_onefeature(X_batch,y_batch,weight,lrate)

            print('weight', weight, 'MSE',MSE)

            # Store current weight and loss values in corresponding lists
            weights.append(weight)
            
            loss.append(MSE)

    	# One epoch is finished when the algorithm goes through all batches
  
    return weights, loss

weights, loss = minibatchSGD(x, y, 50, 2, 0.1)
print('weights', weights, 'loss', loss)

# Set epoches and learning rate
epochs = 100
lrate = 0.02

# Iterate through the values of `batch_sizes` param
batch_sizes = [1, 10, 100]

# List for storing weights and loss for each batch size (length of both lists=3)
weights_batches = []
loss_batches = []

for batch_size in batch_sizes:
    weights, loss = minibatchSGD(x, y, batch_size, epochs, lrate)
    
    weights_batches.append(weights)
    
    loss_batches.append(loss)

    print('batch_size', batch_size, 'weights_batches', weights_batches, 'loss_batches', 

    loss_batches)
    
    print('batch_size', batch_size)

for batch_size, weights, loss in zip(batch_sizes, weights_batches, loss_batches):
    plt.plot(weights, loss, label="batch size"+str(batch_size))
    plt.legend()

plt.xlabel("weight", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.rcParams['figure.figsize'] = [40, 20]
plt.show()

# History of the MSE loss obtained during learning
batch_size1_loss   = loss_batches[0]
batch_size10_loss  = loss_batches[1]
batch_size100_loss = loss_batches[2]

# Create the figure and axes objects
fig, axes = plt.subplots(1,3, sharey=True, figsize=(15,5))

# Create lists of loss values and batch sizes for further iteration in for-loop
batch_loss_list = [batch_size1_loss, batch_size10_loss, batch_size100_loss]
batch_size      = [1,10,100] 

for ax, batch_loss, size in zip(axes, batch_loss_list, batch_size):
    # Plot only first 100 values
    ax.plot(np.arange(len(batch_loss[:100])), batch_loss[:100])
    # Remove top and right subplot's frames 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set subplot's title
    ax.set_title("batch size = "+str(size), fontsize=18)

# Set x- and y-axis labels
axes[0].set_xlabel('batch #', fontsize=18)
axes[0].set_ylabel('Loss', fontsize=18)

plt.ylim(0,10000)
plt.show()
```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Gradient-Based-Learning/sgd_batch_size.png?raw=true)

## Using PyTorch for Deep Learning

This project demonstrates gradient-based optimization using **PyTorch** instead of TensorFlow/Keras. PyTorch provides several advantages:

### PyTorch environment setup

1. **Python Virtual Environment**
   ```bash
   python3 -m venv pytorch_env
   source pytorch_env/bin/activate  # On Linux/Mac
   # pytorch_env\Scripts\activate  # On Windows
   ```

2. **Install PyTorch and Python dependencies**
   ```bash
   pip install torch torchvision torchaudio matplotlib numpy scikit-learn jupyter notebook
   ```

3. **Verify**
   ```python
   import torch
   print("PyTorch version:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   ```

### PyTorch

1. **Automatic Differentiation (Autograd)**
   ```python
   x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
   y = x.sum()
   y.backward()  # Compute gradients automatically
   print(x.grad)  # Access gradients
   ```

2. **Built-in Loss functions**
   ```python
   import torch.nn as nn
   mse_loss = nn.MSELoss()
   loss = mse_loss(predictions, targets)
   ```

3. **Optimizers**
   ```python
   import torch.optim as optim
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   ```

### Running the PyTorch examples

1. **Python scripts**
   ```bash
   source pytorch_env/bin/activate
   python Gradient-Based-Learning-PyTorch.py
   ```

2. **Jupyter Notebooks**
   ```bash
   source pytorch_env/bin/activate
   jupyter notebook Round2_SGD-PyTorch.ipynb
   ```

### Comparison: NumPy vs PyTorch

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| Gradient Computation | Manual | Automatic (Autograd) |
| GPU Support | No | Yes |
| Deep Learning | Limited | Full Support |
| Memory Efficiency | Good | Excellent |
| Ecosystem | Scientific Computing | Deep Learning |

### PyTorch advantages

- **Automatic differentiation** No need to manually compute gradients
- **GPU acceleration** CUDA
- **Dynamic computation graphs** Flexible model architectures  
- **Ecosystem** Pre-trained models, datasets, and tools
- **Production** TorchScript for deployment

***References***

Optimization Algorithms https://d2l.ai/chapter_optimization/index.html

---

## Batch vs Mini-Batch training

### Understanding different training approaches

In gradient-based optimization, we can process training data in different ways.

#### **Full batch training (Batch Gradient Descent)**
- Uses the **full training dataset** for each gradient update
- **Advantages**
  - Most accurate gradient estimation
  - Smoother convergence path
  - Deterministic behavior
  - Better for convex optimization
- **Disadvantages**
  - Memory intensive for large datasets
  - Slower per-epoch training
  - Can get stuck in local minima
  - Poor GPU utilization efficiency

#### **Mini-Batch Training (Mini-Batch Gradient Descent)**
- Uses **small subsets** (typically 32-512 samples) for each gradient update
- **Advantages**
  - Memory efficient - independent of dataset size
  - Faster per-epoch training
  - Good balance of speed and stability
  - Better GPU utilization
  - Regularization effect from gradient noise
  - **Industry standard** for deep learning
- **Disadvantages**
  - Less accurate gradient estimation
  - More hyperparameters to tune
  - Slightly noisy convergence

#### **Stochastic Gradient Descent (SGD)**
- Uses **single samples** for each gradient update
- **Advantages**
  - Minimal memory usage
  - Can escape local minima due to noise
  - Fastest per-update computation
  - Good for online learning
- **Disadvantages**
  - Very noisy convergence
  - Poor GPU utilization
  - Requires careful learning rate tuning

### Recommended batch sizes

| Dataset Size | Recommended Approach | Typical Batch Size |
|-------------|---------------------|-------------------|
| < 1,000 samples | Full Batch or Large Mini-Batch | 64-256 |
| 1K - 100K samples | Mini-Batch | 32-128 |
| > 100K samples | Mini-Batch | 64-512 |
| Very Large (> 1M) | Mini-Batch | 256-1024 |

### Implementation

This repository includes examples of batch vs mini-batch training in multiple frameworks.

#### **NumPy** (`Round2/Gradient-Based-Learning.py`)
- Pure NumPy implementation for educational purposes
- Manual batch creation and gradient computation
- Demonstrates fundamental concepts without framework abstractions

#### **TensorFlow/Keras** (`Batch_vs_MiniBatch_TensorFlow.ipynb`)
- Production-ready implementation with tf.data
- Built-in batching and shuffling
- GPU acceleration support
- Automatic memory management

#### **PyTorch** (`Round2/Gradient-Based-Learning-PyTorch.py`, `Round2/Round2_SGD-PyTorch.ipynb`)
- Dynamic computation graphs
- DataLoader for efficient batching
- Advanced optimizers (Adam, RMSprop)
- Flexible batch size experimentation

### Insights from experiments

1. **Memory usage** Scales linearly with batch size
2. **Training Speed** Mini-batches provide best speed/stability trade-off
3. **Convergence** Full batch is smoothest, SGD is noisiest
4. **Performance** All methods converge to similar solutions
5. **GPU efficiency** Mini-batches (32-128) optimal for most hardware

### When to choose each approach?

**Full Batch**
- Dataset fits comfortably in memory
- Need most accurate gradient estimates
- Working with convex optimization problems
- Final fine-tuning stages

**Mini-Batch**
- Dataset is large (most common scenario)
- Need good balance of speed and stability
- Using GPU acceleration
- Training deep neural networks
- Want regularization effects

**SGD**
- Extremely memory constrained
- Online/streaming learning scenarios
- Very large datasets
- Need to escape local minima

### Framework advantages

**NumPy**
- Best for understanding fundamentals
- Complete control over implementation
- Educational value

**TensorFlow/Keras**
- Production-ready with optimizations
- Excellent ecosystem and tools
- Built-in distributed training support

**PyTorch**
- Research-friendly dynamic graphs
- Intuitive API and debugging
- Flexible experimentation

### Running the Batch vs Mini-Batch examples

1. **TensorFlow/Keras**
   ```bash
   # If TensorFlow is installed
   python batch_vs_minibatch_tensorflow.py
   jupyter notebook Batch_vs_MiniBatch_TensorFlow.ipynb
   ```

2. **NumPy**
   ```bash
   python Round2/Gradient-Based-Learning.py
   ```

3. **PyTorch**
   ```bash
   source pytorch_env/bin/activate
   python Round2/Gradient-Based-Learning-PyTorch.py
   jupyter notebook Round2/Round2_SGD-PyTorch.ipynb
   ```

The choice between batch and mini-batch training significantly impacts both training efficiency and final model performance. Mini-batch training has become the de facto standard in modern deep learning due to its excellent balance of computational efficiency, memory usage, and convergence properties.

## PyTorch DataLoader vs manual batch creation

PyTorch provides multiple ways to handle batching for training. This section explains the key differences between using DataLoader and manual batch creation methods, along with their respective advantages and use cases.

### Method 1: Manual batch creation

Manual batch creation gives you full control over how data is divided into batches:

```python
def manual_batch_creation(X, y, batch_size):
    """Manual batch creation WITHOUT DataLoader"""
    n_samples = len(X)
    indices = torch.randperm(n_samples)  # Shuffle indices
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_X, batch_y))
    
    return batches

# Usage example
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
manual_batches = manual_batch_creation(X_tensor, y_tensor, batch_size=32)

# Training loop with manual batches
for epoch in range(epochs):
    for batch_X, batch_y in manual_batches:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Manual batch creation - Pros & Cons**
- **Pros:** Full control over batching logic, simple to understand, no additional dependencies
- **Cons:** Loads entire dataset into memory, no built-in multiprocessing, manual shuffling required
- **Best for:** Small datasets, custom batching logic, educational purposes

### Method 2: PyTorch DataLoader

DataLoader is PyTorch's built-in solution for efficient data loading and batching:

```python
def create_advanced_dataloader(X, y, batch_size, shuffle=True, num_workers=0, drop_last=False):
    """Advanced DataLoader creation with multiple options"""
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,       # Parallel data loading
        drop_last=drop_last,           # Drop incomplete last batch
        pin_memory=torch.cuda.is_available() and X.device.type == 'cpu'  # Only pin CPU tensors
    )
    return dataloader

# Usage example
dataloader = create_advanced_dataloader(X_tensor, y_tensor, batch_size=32, shuffle=True)

# Training loop with DataLoader
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)  # Move to GPU/CPU
        batch_y = batch_y.to(device)
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**PyTorch DataLoader - Pros & Cons**
- **Pros:** Memory efficient, built-in shuffling, multiprocessing support, GPU optimization
- **Pros:** Handles large datasets, automatic batching, integrates with custom Dataset classes
- **Cons:** Slight learning curve, additional abstraction layer
- **Best for:** Production code, large datasets, performance-critical applications

### Method 3: Custom dataset classes

For advanced data loading scenarios, you can create custom Dataset classes:

```python
class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset for advanced data loading scenarios"""
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

# Create DataLoader with custom dataset
def create_custom_dataset_loader(X, y, batch_size=32, shuffle=True):
    dataset = CustomDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and X.device.type == 'cpu'  # Only pin CPU tensors
    )
    return dataloader
```

### Performance comparison

| Feature | Manual Batching | DataLoader | Custom Dataset |
|---------|----------------|------------|----------------|
| **Memory efficiency** | ❌ Low | ✅ High | ✅ High |
| **Multiprocessing** | ❌ No | ✅ Yes | ✅ Yes |
| **Shuffling** | 🔧 Manual | ✅ Built-in | ✅ Built-in |
| **GPU optimization** | 🔧 Manual | ✅ Automatic | ✅ Automatic |
| **Large Dataset support** | ❌ Limited | ✅ Excellent | ✅ Excellent |
| **Customization** | ✅ Full Control | 🔧 Limited | ✅ Full Control |
| **Complexity** | 🟢 Simple | 🟡 Medium | 🔴 Advanced |

### DataLoader features

**1. Multiprocessing for faster loading:**
```python
# Use multiple workers for parallel data loading
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

**2. Pin Memory for GPU**
```python
# Pin memory for faster GPU transfer (only for CPU tensors!)
dataloader = DataLoader(dataset, batch_size=32, 
                       pin_memory=torch.cuda.is_available() and X.device.type == 'cpu')
```
⚠️ **Important** Only use `pin_memory=True` when your tensors are on CPU. If tensors are already on GPU, setting `pin_memory=True` will cause a RuntimeError.

**3. Drop last batch**
```python
# Drop incomplete last batch for consistent batch sizes
dataloader = DataLoader(dataset, batch_size=32, drop_last=True)
```

**4. Custom collate functions**
```python
def custom_collate_fn(batch):
    # Custom logic for combining batch samples
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])

dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
```

### When to use manual batch, dataset classes or DataLoader methods?

**Manual batch creation**
- Working with small datasets (< 10,000 samples)
- Learning gradient descent fundamentals
- Need complete control over batching logic
- Prototyping simple algorithms

**DataLoader**
- Working with medium to large datasets
- Building production deep learning models
- Need memory efficiency and performance
- Using standard batching approaches

**Custom dataset classes**
- Loading data from files or databases
- Need complex data preprocessing
- Working with non-standard data formats
- Implementing data augmentation pipelines

### Recommended

1. **Always use DataLoader for production code** - it's optimized and battle-tested
2. **Start with num_workers=0** and increase gradually - too many workers can slow things down
3. **Use pin_memory=True ONLY for CPU tensors** - enables faster GPU transfer, but causes errors if tensors are already on GPU
4. **Set drop_last=True** if you need consistent batch sizes across epochs
5. **Profile your data loading** - it shouldn't be the bottleneck in training
6. **Keep tensors on CPU for DataLoader creation** - move batches to GPU inside training loop

### PyTorch usage in the code

In our PyTorch implementation files, you'll find  examples of all data load methods.

- **`Gradient-Based-Learning-PyTorch.py`**: Contains `create_advanced_dataloader()`, `manual_batch_creation()`, and `dataloader_vs_manual_comparison()` functions
- **`Round2_SGD-PyTorch.ipynb`**: Interactive examples with performance comparisons
- All examples include CUDA support for GPU acceleration when available

### DataLoader in Mini-Batch training

Here's how DataLoader integrates into mini-batch training functions.

```python
def train_with_dataloader_minibatch(X, y, batch_size=32, epochs=50, lr=0.01):
    """Mini-batch training using PyTorch DataLoader"""
    
    # Convert to tensors and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Create DataLoader for automatic batching
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,     # Mini-batch size
        shuffle=True,              # Shuffle each epoch
        pin_memory=torch.cuda.is_available() and X_tensor.device.type == 'cpu',  # Only pin CPU tensors
        drop_last=False            # Keep incomplete last batch
    )
    
    # Model and optimizer setup
    model = LinearModel(X.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Mini-batch training loop
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # DataLoader automatically provides mini-batches
        for batch_X, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass (mini-batch gradient update)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, Batches = {batch_count}')
    
    return model, avg_loss

# Usage: Can easily switch between different batch sizes
model_16 = train_with_dataloader_minibatch(X, y, batch_size=16)   # Smaller batches
model_32 = train_with_dataloader_minibatch(X, y, batch_size=32)   # Medium batches
model_64 = train_with_dataloader_minibatch(X, y, batch_size=64)   # Larger batches
```

**Advantages of DataLoader in Mini-Batch functions**

1. **Automatic shuffling**: Data is reshuffled every epoch without manual intervention
2. **Mini-Batch batches**: Handles incomplete last batches automatically
3. **GPU optimization**: `pin_memory` enables faster GPU data transfer
4. **Configuration**: Simple parameter changes for different batch sizes
5. **Memory efficiency**: Only loads one batch at a time, not entire dataset
6. **Built-in randomization**: Ensures proper randomization for better training

### DataLoader vs manual batching

| Aspect | DataLoader | Manual |
|--------|-------------------|-----------------|
| **Complexity** | Minimal - few lines | More code required |
| **Memory usage** | Efficient - one batch at a time | Loads entire dataset |
| **Shuffling** | Automatic per epoch | Manual implementation needed |
| **Performance** | Optimized C++ backend | Pure Python loops |
| **GPU support** | Built-in optimizations | Manual tensor transfers |
| **Production** | Tested | ⚠️ Needs thorough testing |
| **Maintenance** | Low - handled by PyTorch | High - custom code to maintain |

**Conclusions**

DataLoader is the recommended approach for mini-batch training in PyTorch applications.

DataLoader provides better performance, memory efficiency, and ease of use compared to manual batch creation methods.

### Resources

torch.utils.data

https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

LRScheduler

https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html

