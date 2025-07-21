# Components-Machine-Learning  - PyTorch

This directory contains files demonstrating the components of Machine Learning using **PyTorch** and ML libraries.

## Contents

- `Components_of_ML.ipynb` - Interactive Jupyter notebook with both scikit-learn and PyTorch implementations
- `Components_of_ML_PyTorch.py` - Complete PyTorch implementation of ML components
- `Components of Machine Learning.py` - Original scikit-learn implementation
- `R0_data/` - The diagrams and datasets

## PyTorch setup & environment

### Prerequisites
Ensure you have the Python virtual environment activated:

```bash
# Navigate to project directory
cd /home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch

# Activate virtual environment
source .venv/bin/activate

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Required libraries

The following libraries are pre-installed in the virtual environment:

- **PyTorch** (`torch`) - Deep learning framework
- **TorchVision** (`torchvision`) - Computer vision utilities
- **NumPy** (`numpy`) - Numerical computations
- **Pandas** (`pandas`) - Data manipulation
- **Matplotlib** (`matplotlib`) - Data visualization  
- **Seaborn** (`seaborn`) - Statistical data visualization
- **Scikit-learn** (`sklearn`) - Traditional ML algorithms
- **Jupyter** (`jupyter`) - Interactive notebooks

## Components of Machine Learning

### 1. **Data**
- **Features**: Measurable properties (e.g., sepal length, width)
- **Labels**: Target values or categories to predict
- **PyTorch**: `TensorDataset` and `DataLoader` for efficient batch processing

### 2. **Hypothesis Space (Model)**
- **Traditional ML**: Linear models, decision trees, SVM
- **PyTorch Neural Networks**: Flexible architectures using `nn.Module`

Example PyTorch model:
```python
import torch.nn as nn

class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))
```

### 3. **Loss Functions**
- **Classification**: Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Binary Classification**: Binary Cross-Entropy (`nn.BCELoss`) 
- **Regression**: Mean Squared Error (`nn.MSELoss`), Mean Absolute Error (`nn.L1Loss`)

## PyTorch Components

### Adam Optimizer

**Adam** (Adaptive Moment Estimation) is one of optimization algorithms in deep learning. Adam combines properties of AdaGrad and RMSProp optimizers.

#### Features:
- **Adaptive Learning Rates**: Automatically adjusts learning rates for each parameter
- **Momentum**: Uses exponentially decaying average of past gradients
- **Bias Correction**: Corrects bias in moment estimates during early training

#### How Adam works?:
```python
import torch.optim as optim

# Initialize Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# Training step
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update parameters
```

#### Adam vs other optimizers:
| Optimizer | Advantages | Disadvantages |
|-----------|------------|---------------|
| **SGD** | Stable | Slow convergence, manual tuning |
| **Adam** | Fast convergence, adaptive | Can overshoot, higher memory usage |
| **RMSProp** | Good for RNNs | Less popular than Adam |

### Neural Network architecture: Linear and ReLU layers

#### **Linear layers (`nn.Linear`)**
Linear layers perform matrix multiplication followed by bias addition:

```python
# Mathematical operation: y = xW^T + b
linear_layer = nn.Linear(in_features=4, out_features=64)

# Input: (batch_size, 4)
# Output: (batch_size, 64)
```

**What it does?:**
- Transforms input features to different dimensional space
- Learns weights (W) and biases (b) during training
- Foundation of neural network learning

#### **ReLU Activation function (`nn.ReLU`)**
ReLU (Rectified Linear Unit) introduces non-linearity:

```python
relu = nn.ReLU()
# f(x) = max(0, x)
```

**Why ReLU is important?:**
- **Non-linearity**: Enables learning complex patterns
- **Computational Efficiency**: Simple max(0, x) operation
- **Gradient Flow**: Helps avoid vanishing gradient problem
- **Sparsity**: Outputs exactly 0 for negative inputs

#### **Building Neural Networks**
```python
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # First transformation
            nn.ReLU(),                           # Non-linearity
            nn.Dropout(0.2),                     # Regularization
            nn.Linear(hidden_dim, hidden_dim//2), # Second transformation
            nn.ReLU(),                           # Non-linearity
            nn.Dropout(0.2),                     # Regularization
            nn.Linear(hidden_dim//2, num_classes) # Output layer
        )
    
    def forward(self, x):
        return self.layers(x)
```

**Layer flow:**
1. **Input** → Linear → ReLU → Dropout
2. **Hidden** → Linear → ReLU → Dropout  
3. **Output** → Linear (no activation for logits)

### Loss functions

#### **CrossEntropyLoss (`nn.CrossEntropyLoss`)**

**Purpose**: Multi-class classification

**Mathematical formula**:
```
CE = -Σ(y_true * log(y_pred))
```

**PyTorch**:
```python
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, targets)

# predictions: (batch_size, num_classes) - logits
# targets: (batch_size,) - class indices
```
- Raw logits (no softmax needed)
- Internally applies softmax + negative log likelihood
- Perfect for multi-class problems
- Penalizes confident wrong predictions heavily

#### **MSELoss (`nn.MSELoss`)**

**Purpose**: Regression tasks

**Mathematical formula**:
```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**PyTorch**:
```python
criterion = nn.MSELoss()
loss = criterion(predictions, targets)

# predictions: (batch_size, output_dim) - continuous values
# targets: (batch_size, output_dim) - continuous targets
```
- Measures average squared differences
- Sensitive to outliers
- Always positive, 0 = perfect fit
- Commonly used for regression

#### **Loss function comparison**
```python
# Example from Components_of_ML_PyTorch.py
def demonstrate_loss_functions():
    y_true = torch.tensor([0, 1, 2, 1, 0])
    y_pred_logits = torch.tensor([[2.0, -1.0, 0.5],
                                  [-0.5, 1.5, 0.2],
                                  [0.1, -0.8, 1.2],
                                  [0.3, 0.9, -0.1],
                                  [1.8, -0.2, 0.0]])
    
    # Cross Entropy for classification
    ce_loss = nn.CrossEntropyLoss()
    ce_value = ce_loss(y_pred_logits, y_true)
    
    # MSE for regression
    y_continuous = torch.tensor([1.0, 2.5, 0.8, 1.9, 0.3])
    y_pred_continuous = torch.tensor([1.1, 2.3, 0.9, 2.1, 0.1])
    mse_loss = nn.MSELoss()
    mse_value = mse_loss(y_pred_continuous, y_continuous)
```

### Training pipeline

Here's how Adam optimizer, neural network layers, and loss functions work together in our PyTorch implementation:

```python
# 1. Define Model Architecture
model = MultiLayerPerceptron(input_dim=4, hidden_dim=64, num_classes=3)

# 2. Choose Loss Function
criterion = nn.CrossEntropyLoss()  # For classification

# 3. Initialize Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training Loop
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        # Forward pass through layers
        outputs = model(batch_X)  # Linear → ReLU → Linear → ReLU → Linear
        
        # Calculate loss
        loss = criterion(outputs, batch_y)  # CrossEntropyLoss
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Adam updates parameters
```

**What happens during training?:**
1. **Forward Pass**: Data flows through Linear→ReLU→Linear→ReLU→Linear layers
2. **Loss Calculation**: CrossEntropyLoss compares predictions with true labels
3. **Backward Pass**: Gradients flow back through all layers
4. **Parameter Update**: Adam optimizer updates weights and biases adaptively

### Visualized layers

```
Input Data (4 features)
         ↓
    Linear Layer (4→64)     ← Learnable weights & biases
         ↓
    ReLU Activation         ← Non-linearity: max(0, x)
         ↓
    Linear Layer (64→32)    ← More learnable parameters
         ↓  
    ReLU Activation         ← More non-linearity
         ↓
    Linear Layer (32→3)     ← Final output layer
         ↓
    Predictions (3 classes)
         ↓
    CrossEntropyLoss        ← Compare with true labels
         ↓
    Gradients              ← Flow backward through layers
         ↓
    Adam Optimizer         ← Update all parameters adaptively
```

**Why this architecture works?:**
- **Linear layers** learn feature transformations
- **ReLU activations** enable complex pattern recognition
- **CrossEntropyLoss** guides learning toward correct classifications
- **Adam optimizer** efficiently finds optimal parameters

## Running the code

### Option 1: Jupyter Notebook (Recommended)
```bash
# Start Jupyter notebook
jupyter notebook

# Navigate to Components-Machine-Learning folder
# Open Components_of_ML.ipynb
# Run cells sequentially to see both scikit-learn and PyTorch implementations
```

### Option 2: Python scripts
```bash
# Run original scikit-learn version
python "Components-Machine-Learning/Components of Machine Learning.py"

# Run PyTorch implementation
python Components-Machine-Learning/Components_of_ML_PyTorch.py
```

## Dataset

**Iris**
- 150 samples, 4 features, 3 classes
- Features: sepal length, sepal width, petal length, petal width
- Classes: Setosa, Versicolor, Virginica
- Perfect for demonstrating classification concepts

**Pandas Data Frames**

```python
import pandas as pd

# Create dictionary
mydict = {'animal':['cat', 'dog','mouse','rat', 'cat'],
         'name':['Fluffy','Chewy','Squeaky','Spotty', 'Diablo'],
         'age, years': [3,5,0.5,1,8]}

# Create dataframe from dictionary
df = pd.DataFrame(mydict, index=['id1','id2','id3','id4','id5'])

# Access row by name with .loc 
print(df.loc['id1'])

# Access row by index with .iloc 
print('\n', df.iloc[0])

# Access column by name with .loc 
print(df.loc[:,'animal'])

# Access column by name without .loc 
print('\n', df['animal'])

# Access column by index with .iloc 
print('\n', df.iloc[:,0])

# Loading from .csv file by using DataFrame
df = pd.read_csv('R0_data/Data.csv')

print("Shape of the dataframe: ",df.shape)
print("Number of dataframe rows: ",df.shape[0])
print("Number of dataframe columns: ",df.shape[1])

# Print first 5 rows 
df.head()
```

1. **Data Representation**: How to structure data for ML with features and labels
2. **Model Architecture**: Difference between traditional ML and neural network approaches
3. **Loss Functions**: How to measure and optimize model performance
4. **Training Process**: Both automated (scikit-learn) and manual (PyTorch) approaches
5. **Evaluation**: Using training, validation, and test sets for model assessment

## Scikit-learn vs PyTorch

| Aspect | Scikit-learn | PyTorch |
|--------|-------------|---------|
| **Complexity** | Simple, high-level | More complex, flexible |
| **Use Case** | Traditional ML | Deep learning & research |
| **Training** | `model.fit(X, y)` | Custom training loops |
| **GPU Support** | Limited | Native support |
| **Customization** | Pre-built algorithms | Fully customizable |

All visualizations are saved as PNG files in the current directory for documentation purposes.

## Questions & troubleshooting

### Q: Why use Adam instead of SGD?
**A:** Adam adapts learning rates automatically and converges faster, especially for complex neural networks. SGD requires manual tuning of learning rate and momentum.

### Q: What happens without ReLU activation?
**A:** Without ReLU, multiple linear layers collapse into a single linear transformation, losing the ability to learn complex patterns. ReLU introduces essential non-linearity.

### Q: When to use CrossEntropyLoss vs MSELoss?
**A:**
- **CrossEntropyLoss**: Classification tasks (predicting categories)
- **MSELoss**: Regression tasks (predicting continuous values)

### Q: How to choose network architecture?
**A:**
- **Input layer size**: Match number of features
- **Hidden layers**: Start with 64-128 neurons, experiment
- **Output layer**: Match number of classes (classification) or 1 (regression)

### Q: My model isn't learning, what to check?
**A:**
1. **Learning rate**: Try 0.001, 0.01, or 0.1
2. **Loss function**: Ensure it matches your task type
3. **Data preprocessing**: Normalize/standardize features
4. **Architecture**: Add more layers or neurons if underfitting

### Comparisons

1. Run `Components_of_ML_PyTorch.py`
2. Try different optimizers: `optim.SGD`, `optim.RMSprop`
3. Modify loss functions

````
