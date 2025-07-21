# Artificial-Neural-Networks-PyTorch.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Basic neural network operations
# input/feature vector
x = np.array([0.14, 0.2, -5]).reshape(3, 1)
# bias
b = 0.5
# weight vector
w = np.array([10, 5, 0.3]).reshape(3, 1)

# compute weighted sum
# 𝑧=𝑏+𝑤1𝑥1+𝑤2𝑥2+𝑤3𝑥3
z = b + (w[0]*x[0]) + (w[1]*x[1]) + (w[2]*x[2])

# apply activation function to weighted sum z
# 𝑔(𝑧)=1/(1+𝑒−𝑧)
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    g_grad = g*(1-g)
    return g, g_grad

g = 1/(1+np.exp(-z))

# print the results
print("The output is: ", g, "Shape", g.shape)

# input - feature values of 100 data points
x = np.linspace(-10, 10, 100).reshape(100, 1)

# weight and bias of first hidden neuron
w11, b11 = -1, -2
# weight and bias of second hidden neuron
w12, b12 = 1, -2

# weights and bias of output neuron
b21 = 0.5
w21, w22 = 5, 3

# compute weighted sum for two hidden neurons
z1 = x.dot(w11) + b11
z2 = x.dot(w12) + b12
# compute weighted sum of hidden neurons' outputs (without activation)
h = z1.dot(w21) + z2.dot(w22) + b21
print('h[0]', h[0], 'h[42]', h[42], 'h.shape', h.shape)

# plot outputs of neurons
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8, 2))

# output of first hidden neuron
axes[0].plot(x, z1)
# output of second hidden neuron
axes[1].plot(x, z2)
# output of output neuron
axes[2].plot(x, h)

axes[0].set_title('$z_{1} = b^{(1)}_{1} + w^{(1)}_{1}x$', fontsize=12)
axes[1].set_title('$z_{1} = b^{(1)}_{2} + w^{(1)}_{2}x$', fontsize=12)
axes[2].set_title('$h(x) = b^{(2)}_{1} + w^{(2)}_{1}z_{1} + w^{(2)}_{2}z_{2}$', fontsize=12)

plt.show()

def ReLU(z):
    return np.where(z > 0, z, 0)

def sigmoid_np(z):
    return 1 / (1 + np.exp(-z))

# use weighted sum z1 and z2 computed in previous task and 
# apply ReLU activation function to z1
g1 = ReLU(z1)
# apply ReLU activation function to z2
g2 = ReLU(z2)
# compute weighted sum of hidden neurons' outputs 
h = b11 + w11*g1 + w12*g2

print('h.shape', h.shape, 'h[42]', h[42], 'h[0]', h[0])

g3 = sigmoid_np(z1)
g4 = sigmoid_np(z2)
h2 = b11 + w11*g3 + w12*g4

print('h2.shape', h2.shape, 'h2[42]', h2[42], 'h2[0]', h2[0])

# plot outputs of neurons
fig, axes = plt.subplots(2, 3, figsize=(7, 4))

axes[0, 0].plot(x, z1)  # weighted sum of first hidden neuron
axes[0, 1].plot(x, z2)  # weighted sum of second hidden neuron
axes[0, 2].axis('off')  # hide axis of extra subplot

axes[1, 0].plot(x, g1)  # activation of first hidden neuron
axes[1, 1].plot(x, g2)  # activation of second hidden neuron
axes[1, 2].plot(x, h)   # output 
axes[1, 0].plot(x, g3)  # activation of first hidden neuron
axes[1, 1].plot(x, g3)  # activation of second hidden neuron
axes[1, 2].plot(x, h2)  # output 

axes[0, 0].set_title('$z_{1} = b^{(1)}_{1} + w^{(1)}_{1}x$', fontsize=12)
axes[0, 1].set_title('$z_{1} = b^{(1)}_{2} + w^{(1)}_{2}x$', fontsize=12)
axes[1, 0].set_title('$g(z_{1}) = max(0,z_{1})$', fontsize=12)
axes[1, 1].set_title('$g(z_{2}) = max(0,z_{2})$', fontsize=12)
axes[1, 2].set_title('$h(x) = b^{(2)}_{1} + w^{(2)}_{1}g(z_{1}) + w^{(2)}_{2}g(z_{2})$', fontsize=12)

fig.tight_layout()
plt.show()

# Fashion-MNIST Dataset with PyTorch
# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

# Use subset for faster training (similar to original code)
subset_indices = list(range(16000))
train_subset = torch.utils.data.Subset(train_dataset, subset_indices)

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get some sample data
trainval_images = train_dataset.data[:16000].numpy()
trainval_labels = train_dataset.targets[:16000].numpy()
test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

# shape of train and test image
print(f'Number of training and validation examples {trainval_images.shape}')
print(f'Number of test examples {test_images.shape}')
print(f'Min feature value {trainval_images.min()}')
print(f'Max feature value {trainval_images.max()}')
print(f'Data type {type(trainval_images.min())}')

labels = np.unique(test_labels)
print(labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# display numeric label and corresponding class name 
print('label value \t category')
for class_name, label in zip(class_names, labels):
    print(f'{label} \t\t {class_name}')

# visualize 10 first images from test set
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.xticks([])  # remove ticks on x-axis
    plt.yticks([])  # remove ticks on y-axis
    plt.imshow(test_images[i], cmap='binary')  # set the colormap to 'binary' 
    plt.xlabel(class_names[test_labels[i]])
plt.tight_layout()    
plt.show()

# select the image to visualize
img = test_images[0]
# create figure and axis objects
fig, ax = plt.subplots(1, 1, figsize=(10, 10)) 
# display image
ax.imshow(img, cmap='gray')
width, height = img.shape
# this value will be needed in order to change the color of annotations
thresh = img.max()/2.5

# display grayscale value of each pixel
for x in range(width):
    for y in range(height):
        val = (img[x][y])
        ax.annotate(str(val), xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    # if a pixel is black set the color of annotation as white
                    color='white' if img[x][y] < thresh else 'black')
plt.show()

# Define Neural Network Models using PyTorch
class FashionMNISTNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(FashionMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=20, device='cpu'):
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return history

# Evaluation function
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

# Create and train the basic model
training = True  # Set to False to load pre-trained model

if training:
    model = FashionMNISTNet(128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    print("Training Fashion-MNIST classification model...")
    history = train_model(model, train_loader, criterion, optimizer, epochs=20, device=device)
    
    # Save the model
    torch.save(model.state_dict(), '/home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch/Artificial-Neural-Networks/fashion_mnist_model.pth')
    
    # Plot training history
    plt.figure(figsize=(7, 4))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    model = FashionMNISTNet(128).to(device)
    model.load_state_dict(torch.load('/home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch/Artificial-Neural-Networks/fashion_mnist_model.pth'))

# Evaluate the model
test_loss, test_accuracy = evaluate_model(model, test_loader, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Regression Model for California Housing Dataset
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

# Load and prepare California Housing dataset
def load_housing_dataset():
    X, y = fetch_california_housing(return_X_y=True)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale feature values
    scaler = StandardScaler()
    X_trainval = scaler.fit_transform(X_trainval)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_trainval = torch.FloatTensor(X_trainval)
    y_trainval = torch.FloatTensor(y_trainval).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return X_trainval, y_trainval, X_test, y_test

# Load housing dataset
X_reg_trainval, y_reg_trainval, X_reg_test, y_reg_test = load_housing_dataset()

print(f'Number of training and validation examples {X_reg_trainval.shape}')
print(f'Number of test examples {X_reg_test.shape}')

# Create regression data loader
reg_dataset = TensorDataset(X_reg_trainval, y_reg_trainval)
reg_loader = DataLoader(reg_dataset, batch_size=32, shuffle=True)

if training:
    # Create and train regression model
    model_reg = RegressionNet().to(device)
    criterion_reg = nn.MSELoss()
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.001)
    
    print("Training regression model...")
    model_reg.train()
    history_reg = {'loss': []}
    
    for epoch in range(20):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(reg_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer_reg.zero_grad()
            outputs = model_reg(data)
            loss = criterion_reg(outputs, target)
            loss.backward()
            optimizer_reg.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(reg_loader)
        history_reg['loss'].append(epoch_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/20], Loss: {epoch_loss:.4f}')
    
    # Save regression model
    torch.save(model_reg.state_dict(), '/home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch/Artificial-Neural-Networks/regression_model.pth')
    
    # Plot training history
    plt.figure(figsize=(6, 3))
    plt.plot(history_reg['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    model_reg = RegressionNet().to(device)
    model_reg.load_state_dict(torch.load('/home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch/Artificial-Neural-Networks/regression_model.pth'))

# Evaluate regression model
model_reg.eval()
with torch.no_grad():
    X_reg_test = X_reg_test.to(device)
    y_reg_test = y_reg_test.to(device)
    predictions = model_reg(X_reg_test)
    test_loss_reg = nn.MSELoss()(predictions, y_reg_test).item()

print(f'MSE loss on test dataset: {test_loss_reg:.4f}')

# Model architecture comparison
print("\n=== Model Architecture Comparison ===")

# Model 1: Single hidden layer with 256 units
class Net256(nn.Module):
    def __init__(self):
        super(Net256, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model 2: Four hidden layers with 64 units each
class Net4x64(nn.Module):
    def __init__(self):
        super(Net4x64, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Train and evaluate different architectures
models_to_compare = [
    ('Model 256 units', Net256()),
    ('Model 4x64 units', Net4x64())
]

if training:
    for name, model_arch in models_to_compare:
        print(f"\nTraining {name}...")
        model_arch = model_arch.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_arch.parameters(), lr=0.001)
        
        # Quick training for comparison
        model_arch.train()
        for epoch in range(10):  # Fewer epochs for comparison
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                outputs = model_arch(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        test_loss, test_accuracy = evaluate_model(model_arch, test_loader, device)
        print(f'{name} - Test Accuracy: {test_accuracy:.4f}')

# Learning rate comparison
print("\n=== Learning Rate Comparison ===")

def test_learning_rates(learning_rates, device='cpu'):
    results = []
    
    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        
        # Create new model
        model = FashionMNISTNet(128).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Quick training
        model.train()
        for epoch in range(10):  # Fewer epochs for comparison
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        results.append(test_accuracy)
        print(f"Learning rate {lr}: Test Accuracy = {test_accuracy:.4f}")
    
    return results

if training:
    lrates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    test_acc_results = test_learning_rates(lrates, device)
    
    # Plot learning rate vs accuracy
    plt.figure(figsize=(5, 3))
    plt.plot(lrates, test_acc_results, 'o-')
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.xscale('log')
    plt.grid(True)
    plt.title("Learning Rate vs Test Accuracy")
    plt.show()
    
    best_lr = lrates[np.argmax(test_acc_results)]
    print(f"Best learning rate: {best_lr} with accuracy: {max(test_acc_results):.4f}")

print("\nPyTorch conversion complete! All models have been successfully converted from TensorFlow/Keras to PyTorch.")
