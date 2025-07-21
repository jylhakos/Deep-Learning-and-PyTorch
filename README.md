# Deep-Learning-PyTorch

## Technologies
- **Python**: The programming language
- **PyTorch**: Deep learning framework for neural networks
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **CUDA**: GPU acceleration (when available)
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation and analysis

## Project

```
Deep-Learning-and-PyTorch/
├── Artificial-Neural-Networks/
│   ├── Artificial-Neural-Networks-PyTorch.py    # Main PyTorch implementation
│   ├── Artificial-Neural-Networks.py            # Original TensorFlow version
│   ├── actfunctions.py                          # Activation functions
│   ├── Round1_ANN.ipynb                         # Jupyter notebook
│   └── utils/                                   # Utility functions
├── Components-Machine-Learning/                 # Machine Learning fundamentals
│   ├── Components_of_ML.ipynb                   # Interactive notebook
│   ├── Components_of_ML_PyTorch.py              # PyTorch ML components
│   ├── Components of Machine Learning.py        # Scikit-learn version
│   └── R0_data/                                 # Diagrams
├── Dataset/                                     # Training and test datasets
│   ├── cats_and_dogs/
│   └── cats_and_dogs_small/
└── README.md
```

A deep learning project using PyTorch for neural network implementation and training.

**Neural Network**

Neural networks include layers and modules that perform operations on data.

The torch.nn namespace provides all the building blocks you need to build your own neural network.

```

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

```

**Device for training**

We want to be able to train our model on an accelerator such as CUDA. 

If the current accelerator is available, we will use CUDA, otherwise, we use the CPU.

```

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

```

**The Class**

We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__.

```

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

```

We create an instance of NeuralNetwork, and move it to the device, and print NeuralNetwork structure.

```

model = NeuralNetwork().to(device)

print(model)

```

To use the Neural Network model, we pass it the input data. 

This executes the model’s forward, along with some background operations.

```

X = torch.rand(1, 28, 28, device=device)

logits = model(X)

pred_probab = nn.Softmax(dim=1)(logits)

y_pred = pred_probab.argmax(1)

print(f"Predicted class: {y_pred}")

```

Let’s illustrate the layers in the Neural Network model. 

To illustrate layers, we will take a sample minibatch of 3 images of size 28x28 and see what happens to it as we pass it through the network.

```

input_image = torch.rand(3,28,28)

print(input_image.size())

```

**nn.Flatten**

We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is maintained).

```

flatten = nn.Flatten()

flat_image = flatten(input_image)

print(flat_image.size())

```

**nn.Linear**

The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.

```

layer1 = nn.Linear(in_features=28*28, out_features=20)

hidden1 = layer1(flat_image)

print(hidden1.size())

```

**nn.ReLU**

Non-linear activations are what create the mappings between the model’s inputs and outputs.

```

print(f"Before ReLU: {hidden1}\n\n")

hidden1 = nn.ReLU()(hidden1)

print(f"After ReLU: {hidden1}")

```

**nn.Sequential**

The nn.Sequential is an ordered container of modules. 

The data is passed through all the modules in the same order as defined. 

You can use sequential containers to setup network like seq_modules.

```

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

input_image = torch.rand(3,28,28)

logits = seq_modules(input_image)

```

**nn.Softmax**

The last linear layer of the neural network returns logits.

The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class.

```

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

```

The layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training. 

## Setup and Installation

### 1. Create Python Virtual Environment

```bash
# Navigate to project directory
cd /home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate     # On Windows

# Verify activation
which python  # Should show path to .venv/bin/python
```

### 2. Install PyTorch and dependencies

```bash
# Install PyTorch and related packages
pip install torch torchvision torchaudio numpy matplotlib scikit-learn pandas pillow jupyter seaborn

# For CUDA support (if you have compatible GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
```

## Running the code

### Using Python Virtual Environment
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run PyTorch version
python Artificial-Neural-Networks/Artificial-Neural-Networks-PyTorch.py

# Or use full path to virtual environment python
./.venv/bin/python Artificial-Neural-Networks/Artificial-Neural-Networks-PyTorch.py
```

## Features

### 1. Neural network operations
- Manual computation of weighted sums and activations
- Implementation of ReLU and Sigmoid activation functions
- Visualization of neuron outputs and decision boundaries

### 2. MNIST Classification
```python
# PyTorch implementation replaces Keras/TensorFlow approach
class FashionMNISTNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(FashionMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training with PyTorch
model = FashionMNISTNet(128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

### 3. Regression with California housing dataset
```python
class RegressionNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=128):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Differences: TensorFlow/Keras vs PyTorch

| Aspect | TensorFlow/Keras (Original) | PyTorch (New Implementation) |
|--------|----------------------------|------------------------------|
| **Model Definition** | `keras.Sequential([layers.Dense(...)])` | `class Net(nn.Module)` with `forward()` method |
| **Data Loading** | `fashion_mnist.load_data()` | `torchvision.datasets.FashionMNIST()` |
| **Training Loop** | `model.fit(X, y, epochs=20)` | Manual training loop with `loss.backward()` |
| **Optimization** | `optimizer='RMSprop'` | `optim.RMSprop(model.parameters())` |
| **Loss Function** | `loss='sparse_categorical_crossentropy'` | `nn.CrossEntropyLoss()` |
| **Device Management** | Automatic | Explicit `.to(device)` calls |
| **Model Saving** | `model.save('model.h5')` | `torch.save(model.state_dict(), 'model.pth')` |

## Dataset usage

The project uses datasets from the `Dataset/` folder:
- **Fashion-MNIST**: Automatically downloaded by PyTorch
- **California Housing**: Loaded via scikit-learn
- **Custom datasets**: Can be loaded from `Dataset/cats_and_dogs/` folders

## Model architectures

1. **Basic ANN**: Single hidden layer (128 units)
2. **Wide Network**: Single hidden layer (256 units) 
3. **Deep Network**: Four hidden layers (64 units each)
4. **Regression Network**: For continuous value prediction

## Training

- **Learning Rate comparison**: Tests multiple learning rates
- **Model Architecture comparison**: Evaluates different network designs
- **Training history visualization**: Plots loss and accuracy curves
- **GPU**: Automatically uses CUDA if available
- **Model persistence**: Save and load trained models

## Performance

The code includes logging and visualization:
- Training loss and accuracy tracking
- Test set evaluation
- Learning rate optimization curves
- Model architecture comparison charts

## Usage
```python
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = FashionMNISTNet(128).to(device)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
```

## References

Learn the Basics

https://docs.pytorch.org/tutorials/beginner/basics/intro.html

Learning PyTorch with Examples

https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html

## License

This project is licensed under the MIT License - see the LICENSE file for details.
