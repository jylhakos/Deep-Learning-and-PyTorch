# Regularization

To prevent overfitting, the best solution is to use more complete training data. 

The dataset should cover the full range of inputs that the model is expected to handle.

The simplest way to prevent overfitting is to start with a model with a small number of learnable parameters.

## PyTorch Input Pipeline

A sequence of batches that together cover the entire dataset is called an epoch.

We divide the dataset into smaller sets called batches and only store a single batch in the working memory.

After loading new batch of data, we update the neural network parameters (weights and bias) using one iteration of Stochastic Gradient Descent (SGD) variant.

### What is a Pipeline and why is it needed for PyTorch?

A **Pipeline** in PyTorch refers to the sequence of data processing steps that transform raw data into a format suitable for training neural networks. The pipeline is essential because:

1. **Data loading**: Efficiently load data from storage (files, databases, etc.)
2. **Preprocessing**: Apply transformations like normalization, resizing, augmentation
3. **Batching**: Group individual samples into batches for parallel processing
4. **Shuffling**: Randomize data order to improve training stability
5. **Parallel Processing**: Use multiple CPU cores to prepare data while GPU trains the model

**Why PyTorch Pipeline**

- **Dataset Class**: Abstract base class for custom data sources
- **DataLoader**: Handles batching, shuffling, and parallel loading automatically
- **Transforms**: Composable preprocessing operations
- **Memory**: Lazy loading and on-demand processing
- **GPU acceleration**: Seamless integration with CUDA

**Stochastic Gradient Descent**

In SGD, we find out the gradient of the cost function of a single sample at each iteration instead of the sum of the gradient of the cost function of all the samples.

```python

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
import pathlib 

# Create PyTorch Dataset from Python generator
class GeneratorDataset(Dataset):
    def __init__(self, generator_func, length):
        self.generator_func = generator_func
        self.length = length
        self.data = [next(generator_func()) for _ in range(length)]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.int32)

def sequence_generator():
    number = 0
    while True:
        yield number
        number += 1

# Create dataset
generator_dataset = GeneratorDataset(sequence_generator, 100)

# Transformations with DataLoader:
def preprocess(x):
    return x * x

class PreprocessedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        return self.transform(sample)

preprocessed_dataset = PreprocessedDataset(generator_dataset, preprocess)
dataloader = DataLoader(preprocessed_dataset, batch_size=5, shuffle=True, num_workers=2)

# Retrieve first batch
first_batch = next(iter(dataloader))
print(first_batch.numpy())

i = 1

for batch in dataloader:
    if i > 3:
        break
    print(f"batch {i}", batch.numpy())
    i += 1

# Set base directory to Dataset folder (two levels up)
base_dir = pathlib.Path.cwd() / '..' / '..' / 'Dataset' / 'cats_and_dogs_small'

# Custom Image Dataset for PyTorch
class ImageDataset(Dataset):
    def __init__(self, image_paths, class_names, transform=None):
        self.image_paths = image_paths
        self.class_names = class_names
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and transform image, extract label from path
        # Implementation details in actual code files
        pass

# Create PyTorch DataLoaders for training, validation and test
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

train_paths = list((base_dir / 'train').glob('*/*.jpg'))
val_paths = list((base_dir / 'validation').glob('*/*.jpg'))  
test_paths = list((base_dir / 'test').glob('*/*.jpg'))

train_dataset = ImageDataset(train_paths, ['cats', 'dogs'], transform=transform)
val_dataset = ImageDataset(val_paths, ['cats', 'dogs'], transform=transform)
test_dataset = ImageDataset(test_paths, ['cats', 'dogs'], transform=transform)

CLASS_NAMES = ['cats', 'dogs']
IMG_SIZE = 150

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
```

**Data Augmentation**

Data augmentation artificially increases the training set by creating synthetic data points using transformations.

PyTorch provides data augmentation through torchvision.transforms, which can be easily integrated into the data pipeline.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 150
EPOCHS = 20

# PyTorch data augmentation transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# Create datasets with augmentation
train_dataset = ImageDataset(train_paths, CLASS_NAMES, transform=augmentation_transforms)
val_dataset = ImageDataset(val_paths, CLASS_NAMES, transform=base_transforms)
test_dataset = ImageDataset(test_paths, CLASS_NAMES, transform=base_transforms)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

training = True
```

**CNN training with Data Augmentation**

```python
# Define CNN Model using PyTorch
class CNNModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # cv1
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # cv2
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # cv3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # maxpool
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # flatten
            nn.Linear(32 * 75 * 75, 128),  # dense
            nn.ReLU(),
            nn.Linear(128, num_classes),  # output
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create model and training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(num_classes=1).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

# Training loop
if training:
    for epoch in range(2):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'model_pytorch.pth')
else:
    model.load_state_dict(torch.load('model_pytorch.pth'))
```

**Transfer Learning**

Transfer learning is a machine learning technique in which a model trained for one particular task is used as a starting point for a training model for another task.

PyTorch provides excellent support for transfer learning through torchvision.models with pre-trained weights.

```python
from torchvision import models
import torch.nn as nn

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 224  # VGG expects 224x224 input

# Load pre-trained VGG16
class TransferLearningModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super(TransferLearningModel, self).__init__()
        # Load pre-trained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Extract feature layers
        self.features = vgg16.features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
        # Freeze feature layers initially
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create transfer learning model
transfer_model = TransferLearningModel(pretrained=True, num_classes=1).to(device)

print("Model created with pre-trained VGG16 backbone")
print(f"Trainable parameters: {sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)}")
```
**Fine-tuning pre-trained model**

```python
# Data loading with proper transforms for VGG
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(train_paths, CLASS_NAMES, transform=transform_train)
val_dataset = ImageDataset(val_paths, CLASS_NAMES, transform=transform_val)  
test_dataset = ImageDataset(test_paths, CLASS_NAMES, transform=transform_val)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Training with frozen features
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(transfer_model.parameters(), lr=0.001)

if training:
    # Phase 1: Train with frozen backbone
    for epoch in range(10):
        transfer_model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = transfer_model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    
    torch.save(transfer_model.state_dict(), 'transfer_model_pytorch.pth')

# Unfreezing layers for fine-tuning
# Unfreeze the last few layers of VGG16
layers = list(transfer_model.features.children())
for layer in layers[-4:]:  # Unfreeze last 4 layers
    for param in layer.parameters():
        param.requires_grad = True

print("Layers unfrozen for fine-tuning")
for i, layer in enumerate(transfer_model.features.children()):
    trainable = any(p.requires_grad for p in layer.parameters())
    print(f"Layer {i}: {layer.__class__.__name__} - Trainable: {trainable}")

# Fine-tuning with lower learning rate
optimizer_finetune = torch.optim.RMSprop(transfer_model.parameters(), lr=1e-5)

initial_epochs = 5
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

if training:
    # Phase 2: Fine-tune unfrozen layers
    for epoch in range(fine_tune_epochs):
        transfer_model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer_finetune.zero_grad()
            outputs = transfer_model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer_finetune.step()
    
    torch.save(transfer_model.state_dict(), 'transfer_model_finetuned_pytorch.pth')
else:
    transfer_model.load_state_dict(torch.load('transfer_model_finetuned_pytorch.pth'))

# Evaluation
transfer_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = transfer_model(images)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()

test_accuracy = correct / total
print(f'The test set accuracy of model is {test_accuracy:.2f}')
```

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Regularization/transfer_learning.png?raw=true)

## How to Use PyTorch libraries

### Setting up PyTorch

1. **Python Virtual Environment**
```bash
python3 -m venv pytorch_env
source pytorch_env/bin/activate  # Linux/Mac
# or
pytorch_env\Scripts\activate  # Windows
```

2. **Install PyTorch**
```bash
pip install torch torchvision torchaudio matplotlib numpy pandas jupyter ipykernel
```

3. **Activate Python Virtual Environment**
```bash
source activate_pytorch.sh  # Use provided script
# or manually:
source pytorch_env/bin/activate
```

### Running PyTorch code

**Python Scripts**
```bash
python TensorFlowData-PyTorch.py      # Data pipeline examples
python DataAugmentation-PyTorch.py    # Data augmentation examples  
python TransferLearning-PyTorch.py    # Transfer learning examples
```

**Jupyter Notebooks**
```bash
jupyter notebook
# Then open:
# - Round4.1_PyTorchData-PyTorch.ipynb
# - Round4.2_DataAugmentation-PyTorch.ipynb  
# - Round4.3_TransferLearning-PyTorch.ipynb
```

### PyTorch Libraries

1. **torch**: Core PyTorch library with tensors, autograd, neural networks
2. **torchvision**: Computer vision utilities, models, transforms, datasets
3. **torch.utils.data**: Dataset and DataLoader classes for data pipeline
4. **torch.nn**: Neural network layers and loss functions
5. **torch.optim**: Optimization algorithms (SGD, Adam, RMSprop, etc.)

### PyTorch advantages over TensorFlow/Keras

1. **Pythonic API**: More intuitive and easier to debug
2. **Dynamic Computation Graph**: More flexible for research and complex models
3. **GPU integration**: Seamless CUDA support
5. **Production Ready**: TorchScript for deployment

### Dataset

The code expects the dataset to be located at:
```
../../Dataset/cats_and_dogs_small/
├── train/
│   ├── cats/
│   └── dogs/
├── validation/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

If the dataset is not available, the code will create dummy data for demonstration.

### Migration from TensorFlow to PyTorch

This project includes both TensorFlow (original) and PyTorch (converted) versions:

- **TensorFlow files**: `*.py` (original)
- **PyTorch files**: `*-PyTorch.py` (converted)
- **TensorFlow notebooks**: `Round4.*.ipynb` (original)  
- **PyTorch notebooks**: `*-PyTorch.ipynb` (converted)
