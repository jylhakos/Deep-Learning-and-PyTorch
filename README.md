# Deep-Learning-PyTorch

## Technologies
- **Python**: The programming language
- **PyTorch**: Deep learning framework for neural networks
- **TensorFlow/Keras**: Original implementations (being migrated to PyTorch)
- **NumPy**: Numerical computing and array operations
- **scikit-learn**: Machine learning utilities and traditional ML algorithms
- **CUDA**: GPU acceleration (when available)
- **Matplotlib**: Data visualization and plotting
- **Pandas**: Data manipulation and analysis
- **Seaborn**: Statistical data visualization
- **Transformers**: Hugging Face transformer models (BERT, GPT)
- **NLTK**: Natural language processing toolkit
- **OpenCV**: Computer vision operations
- **Jupyter**: Interactive notebook environment
- **Docker**: Containerization for deployment
- **Flask/FastAPI**: Web APIs for model serving

## Project

```
Deep-Learning-and-PyTorch/
├── Artificial-Neural-Networks/                 # Basic neural networks
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
├── Convolutional-Neural-Nets/                  # CNN implementations
│   ├── R3-PyTorch.py                           # PyTorch CNN implementation
│   ├── R3.py                                   # Original TensorFlow version
│   ├── README.md                               # CNN documentation
│   ├── Round3/                                 # Training notebooks
│   └── utils/                                  # CNN utilities
├── Gradient-Based-Learning/                    # Optimization algorithms
│   ├── Batch_vs_MiniBatch_TensorFlow.ipynb     # Batch learning comparison
│   ├── Round2/                                 # Gradient descent notebooks
│   ├── R2/                                     # Training data and examples
│   ├── utils.py                                # Optimization utilities
│   └── pytorch_env/                            # PyTorch environment
├── Regularization/                             # Regularization techniques
│   ├── DataAugmentation-PyTorch.py             # Data augmentation with PyTorch
│   ├── TransferLearning-PyTorch.py             # Transfer learning implementation
│   ├── Round4.1_PyTorchData-PyTorch.ipynb      # PyTorch data handling
│   ├── Round4.2_DataAugmentation-PyTorch.ipynb # Data augmentation notebook
│   ├── Round4.3_TransferLearning-PyTorch.ipynb # Transfer learning notebook
│   └── pytorch_env/                            # Regularization environment
├── Natural-Language-Processing/                # NLP with PyTorch
│   ├── Natural Language Processing-PyTorch.py  # PyTorch NLP implementation
│   ├── Natural Language Processing.py          # Original TensorFlow version
│   ├── Round 5 - Natural Language Processing (NLP)-PyTorch.ipynb
│   ├── Round 5 - Natural Language Processing (NLP).ipynb
│   └── test_embeddings_path.py                 # Word embeddings testing
├── Recurrent-Neural-Networks/                  # RNN and LSTM networks
│   ├── RNN_LSTM_Electricity_Forecasting.ipynb # Time series forecasting
│   ├── electricity_forecasting.py             # Main forecasting script
│   ├── api_server.py                          # API server for predictions
│   ├── weather_service.py                     # Weather data integration
│   ├── train_models.py                        # Model training utilities
│   └── electricity_forecast_env/              # RNN environment
├── Generative-Adversarial-Networks/           # GANs implementation
│   ├── GAN-PyTorch.py                         # PyTorch GAN implementation
│   ├── GAN.py                                 # Original TensorFlow version
│   ├── Round6_GAN.ipynb                       # GAN training notebook
│   ├── utils-PyTorch.py                       # PyTorch GAN utilities
│   └── utils.py                               # Original utilities
├── Transformer-Tokenizer-Embeddings/          # Modern transformer architectures
│   ├── app.py                                 # Transformer application
│   ├── bert_qa.py                             # BERT question answering
│   ├── fine_tune.py                           # Model fine-tuning
│   ├── INSTALLATION.md                        # Setup instructions
│   ├── scripts/                               # Training and utility scripts
│   ├── docker/                                # Docker containerization
│   └── data/                                  # Transformer datasets
├── Dataset/                                   # Training and test datasets
│   ├── cats_and_dogs/                         # Image classification data
│   ├── cats_and_dogs_small/                   # Smaller image dataset
│   ├── R1/ R2/ R3/ R4/ R5/ R6/                # Course-specific datasets
│   └── utils/                                 # Dataset utilities
├── data/                                      # Additional data files
├── .venv/                                     # Python virtual environment
└── README.md                                  # Project documentation
```

## Project Overview

A comprehensive deep learning project using PyTorch for neural network implementation and training. This repository contains multiple specialized modules covering different aspects of deep learning:

### Neural Networks
- **Artificial-Neural-Networks/**: Basic feedforward neural networks, activation functions, and fundamental concepts
- **Components-Machine-Learning/**: Machine learning fundamentals with both traditional ML and PyTorch implementations

### Architectures
- **Convolutional-Neural-Nets/**: CNN implementations for image classification and computer vision
- **Recurrent-Neural-Networks/**: RNN and LSTM networks for time series forecasting and sequential data
- **Generative-Adversarial-Networks/**: GAN implementations for generative modeling
- **Transformer-Tokenizer-Embeddings/**: Modern transformer architectures, BERT, and attention mechanisms

### Training & optimization
- **Gradient-Based-Learning/**: Optimization algorithms, batch vs mini-batch learning, SGD variants
- **Regularization/**: Techniques to prevent overfitting including data augmentation and transfer learning

### Natural Language Processing
- **Natural-Language-Processing/**: NLP implementations with word embeddings, tokenization, and language models

## Learning Progression

For optimal learning, follow this recommended sequence:

### **Beginner Level**
1. **Components-Machine-Learning/**: Start here to understand ML fundamentals
2. **Artificial-Neural-Networks/**: Basic neural network concepts and PyTorch basics

### **Intermediate Level**  
3. **Gradient-Based-Learning/**: Optimization techniques and training strategies
4. **Convolutional-Neural-Nets/**: Image processing and computer vision
5. **Regularization/**: Preventing overfitting and improving model performance

### **Advanced Level**
6. **Recurrent-Neural-Networks/**: Sequential data and time series
7. **Natural-Language-Processing/**: Text processing and embeddings
8. **Generative-Adversarial-Networks/**: Generative modeling
9. **Transformer-Tokenizer-Embeddings/**: Modern NLP and attention mechanisms

Each module builds upon concepts from previous ones, creating a comprehensive learning path from basic ML to cutting-edge deep learning architectures.

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

## Running the Code

### Using Python Virtual Environment
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run different project modules:

# 1. Basic Neural Networks
python Artificial-Neural-Networks/Artificial-Neural-Networks-PyTorch.py

# 2. Machine Learning Components
python Components-Machine-Learning/Components_of_ML_PyTorch.py

# 3. Convolutional Neural Networks
python Convolutional-Neural-Nets/R3-PyTorch.py

# 4. Natural Language Processing
python Natural-Language-Processing/"Natural Language Processing-PyTorch.py"

# 5. Generative Adversarial Networks
python Generative-Adversarial-Networks/GAN-PyTorch.py

# 6. Recurrent Neural Networks (Time Series)
python Recurrent-Neural-Networks/electricity_forecasting.py

# 7. Transformer Models
python Transformer-Tokenizer-Embeddings/app.py

# Or use full path to virtual environment python
./.venv/bin/python Artificial-Neural-Networks/Artificial-Neural-Networks-PyTorch.py
```

### Running Jupyter Notebooks
```bash
# Start Jupyter notebook server
jupyter notebook

# Navigate to specific project folders and open:
# - Components-Machine-Learning/Components_of_ML.ipynb
# - Artificial-Neural-Networks/Round1_ANN.ipynb
# - Natural-Language-Processing/Round 5 - Natural Language Processing (NLP)-PyTorch.ipynb
# - Convolutional-Neural-Nets/Round3/ (notebooks)
# - And many more...
```

### Docker Deployment
```bash
# For transformer models
cd Transformer-Tokenizer-Embeddings/
docker build -t transformer-app .
docker run -p 5000:5000 transformer-app
```

## Features

### 1. Fundamental Neural Network Operations
- Manual computation of weighted sums and activations
- Implementation of ReLU, Sigmoid, Tanh activation functions
- Visualization of neuron outputs and decision boundaries
- Custom PyTorch nn.Module implementations

### 2. Computer Vision & Image Processing
- **CNN Architectures**: LeNet, AlexNet-style networks for image classification
- **Data Augmentation**: Rotation, scaling, flipping, color jittering
- **Transfer Learning**: Pre-trained model fine-tuning with PyTorch
- **GAN Implementation**: Generative models for image synthesis

### 3. Natural Language Processing
- **Word Embeddings**: Word2Vec, GloVe implementations
- **Transformer Models**: BERT for question answering and text classification
- **Tokenization**: Subword tokenization and vocabulary building
- **Attention Mechanisms**: Self-attention and multi-head attention

### 4. Time Series & Sequential Data
- **RNN/LSTM Networks**: For electricity forecasting and time series prediction
- **Weather Data Integration**: API-based data collection and preprocessing
- **Sequence-to-Sequence Models**: For prediction and forecasting tasks

### 5. Optimization & Training
- **Gradient Descent Variants**: SGD, Adam, RMSprop optimizers
- **Batch Processing**: Comparison of batch vs mini-batch learning
- **Learning Rate Scheduling**: Adaptive learning rate strategies
- **Regularization**: Dropout, batch normalization, weight decay

### 6. Fashion-MNIST Classification
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

## Dataset Usage

The project uses various datasets across different modules:

### Computer Vision Datasets
- **Fashion-MNIST**: Automatically downloaded by PyTorch (clothing classification)
- **CIFAR-10/CIFAR-100**: Image classification datasets
- **Custom Image Datasets**: From `Dataset/cats_and_dogs/` folders
- **Augmented Datasets**: Generated through data augmentation techniques

### NLP Datasets
- **Text Corpora**: For word embedding training
- **Question-Answer Pairs**: For BERT fine-tuning
- **Custom Text Data**: From transformer training

### Time Series Data
- **Electricity Consumption**: Historical power usage data
- **Weather Data**: Temperature, humidity, pressure readings
- **Financial Data**: Stock prices and market indicators

### Traditional ML Datasets
- **California Housing**: Regression task dataset (scikit-learn)
- **Iris Dataset**: Multi-class classification
- **Wine Quality**: Classification and regression tasks

### Generated Data
- **Synthetic Time Series**: For RNN training validation
- **GAN Generated Images**: For model evaluation
- **Noise Data**: For testing model robustness

## Model Architectures

### 1. Feedforward Neural Networks
- **Basic ANN**: Single hidden layer (128 units)
- **Wide Network**: Single hidden layer (256 units) 
- **Deep Network**: Four hidden layers (64 units each)
- **Regression Network**: For continuous value prediction

### 2. Convolutional Neural Networks
- **LeNet-style CNN**: Classic architecture for image classification
- **Custom CNN**: Configurable layers for different image sizes
- **Transfer Learning Models**: Pre-trained ResNet, VGG fine-tuning

### 3. Recurrent Neural Networks
- **Vanilla RNN**: Basic recurrent architecture
- **LSTM Networks**: Long Short-Term Memory for sequence modeling
- **Bidirectional RNN**: For better context understanding

### 4. Generative Models
- **Vanilla GAN**: Basic generator-discriminator architecture
- **DCGAN**: Deep Convolutional GAN for image generation
- **Conditional GAN**: Class-conditional generation

### 5. Transformer Architectures
- **BERT Models**: Pre-trained transformer for NLP tasks
- **Custom Transformers**: Encoder-decoder architectures
- **Attention Mechanisms**: Multi-head self-attention implementations

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
