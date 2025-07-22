# Convolutional Neural Networks (CNN)

Convolutional layers learn to "detect" a specific pattern in an image.

The densely connected layers does not preserve the structure or spatial information of the image.

There are three main types of layers in CNN:

1. Convolutional layer (conv)

2. Pooling layer (pooling)

3. Fully connected (or dense) Layer (FC)

A convolutional layer performs convolution operation between the image and kernels (also called filters).

Convolution of the image is a process where the kernel is sliding across the image and computing the weighted sum of the small area (patch) of the image.

The Convolutional layer has the following hyperparameters.

1. Number of kernels, K

2. Stride length, S

3. Zero padding size, P

The pooling layer reduces the number of parameters and has following hyperparameters kernel size, F and Stride length, S

The pooling layer has operations like Max pooling and Average pooling.

The outputs of a Max pooling layer are the largest values of the corresponding patch of the input.

```
import numpy as np

def padding(X, p):
    Z = np.pad(X, ((p,p),(p,p)), 'constant')
    return Z

def convolution(image, kernel, padding, strides):
    kernel = np.flipud(np.fliplr(kernel))
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    if padding != 0:
        imagePadded = np.pad(image, ((padding,padding),(padding,padding)),'constant')
    else:
        imagePadded = image

    for y in range(image.shape[1]):
        if y > image.shape[1] - yKernShape:
            break
        if y % strides == 0:
            for x in range(image.shape[0]):
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def cross_correlation(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def convolution_operation(window, kernel, bias):
    product = np.multiply(window, kernel)
    scalar = np.sum(product)
    scalar = scalar + bias
    return scalar

image = np.array([[1,-1,0], [2,4,-1], [6,0,1]])

kernel = np.array([[1,0], [0.5,-1]])

bias = 0

print('1. stride', stride)
window = image[0:2,0:2]
x1 = convolution_operation(window, kernel, bias)
x1_cc = cross_correlation(window, kernel)
print("x1 =", x1, x1_cc[0][0])

print('2. window', window)
window = np.array(image[0:2,1:3])
x2 = convolution_operation(window, kernel, bias)
x2_cc = cross_correlation(window, kernel)
print("x2 =", x2, x2_cc[0][0])

print('3. window', window)
window = np.array(image[1:3,0:2])
x3 = convolution_operation(window, kernel, bias)
x3_cc = cross_correlation(window, kernel)
print("x3 =", x3, x3_cc[0][0])

print('4. window', window)
window = np.array(image[1:3,1:3])
x4 = convolution_operation(window, kernel, bias)
x4_cc = cross_correlation(window, kernel)
print("x4 =", x4, x4_cc[0][0])

```

**Zero Padding**

Padding is a technique in which we add zero-valued pixels around the image symmetrically.

We call several pixels (or step size) by which kernel traversed in each slide a stride.

The size of the output from convolution operation, i.e feature map, is smaller than the input image size.

This means that we are losing some pixel values around the perimeter of the image.

To get the input-sized output, we employ zero padding.

zero padding size = (kernel size - 1) / 2

```
np.random.seed(1)

X = np.random.randn(28, 28)

print('X.shape', X.shape)

K = np.random.randn(3,3)
P = 1
S = 1

Y_1 = convolution(X,K,P,S)

print('Y_1.shape', Y_1.shape)

P = 0
S = 1

Y_2 = convolution(Y_1,K,P,S)

print('Y_2.shape', Y_2.shape)

P = 0
S = 1

Y_3 = convolution(Y_2,K,P,S)

print('Y_3.shape', Y_3.shape)

plt.rcParams['figure.figsize'] = [40, 20]
figure, plot_matrix = plt.subplots(1,3)
plot_matrix[0].imshow(Y_1)
plot_matrix[1].imshow(Y_2)
plot_matrix[2].imshow(Y_3[0:28])

# The first convolutional layer
pad_cv1, stride_cv1 = 1,1

# The second convolutional layer
pad_cv2, stride_cv2 = 0,1

# The third convolutional layer
pad_cv3, stride_cv3 = 0,1

```

**Fully connected layer**

The feature map from the last convolution or pooling layer is flattened into a single vector of values and fed into a fully connected layer.

After passing through the fully connected layers, the final layer uses the softmax activation function which gives the probabilities of the input belonging to a particular class.

**Number of parameters in CNN layer**

(𝑘𝑒𝑟𝑛𝑒𝑙_𝑤𝑖𝑑𝑡ℎ ∗ 𝑘𝑒𝑟𝑛𝑒𝑙_ℎ𝑒𝑖𝑔ℎ𝑡 ∗ 𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠_𝑖𝑛 + 1 (for bias)) ∗ 𝑐ℎ𝑎𝑛𝑛𝑒𝑙𝑠_𝑜𝑢𝑡

**CNN in PyTorch**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

# Load Fashion-MNIST dataset using PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
])

# Load dataset
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
                                                 download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, 
                                                download=True, transform=transform)

# Get data as numpy arrays
X_trainval = train_dataset.data.numpy()
y_trainval = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

print("X Train {} and X Test size {}".format(X_trainval.shape[0], X_test.shape[0]))

# Split trainval set into training and validation datasets
X_train = X_trainval[0:10000]
y_train = y_trainval[0:10000]
X_val = X_test[:6000]
y_val = y_test[:6000]

print('X_test.shape:', X_test.shape)
print('X_val.shape:', X_val.shape)
print('y_test.shape:', y_test.shape)
print('y_val.shape:', y_val.shape)

# Normalize and convert to PyTorch tensors with channel dimension
X_train = torch.FloatTensor(X_train).unsqueeze(1) / 255.0
X_val = torch.FloatTensor(X_val).unsqueeze(1) / 255.0
X_test = torch.FloatTensor(X_test).unsqueeze(1) / 255.0
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(16 * 27 * 27, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # First conv layer with ReLU
        x = F.relu(self.conv2(x))  # Second conv layer with ReLU
        x = self.maxpool(x)        # Max pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))    # Dense layer with ReLU
        x = self.dropout(x)        # Dropout
        x = self.fc2(x)            # Output layer
        return x

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()  # Equivalent to sparse_categorical_crossentropy
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

print(model)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()           # Zero gradients
            output = model(data)            # Forward pass
            loss = criterion(output, target)  # Calculate loss
            loss.backward()                 # Backward pass
            optimizer.step()                # Update weights
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        # Print training progress
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Save model
torch.save(model.state_dict(), 'model_pytorch.pth')

# Load model (for inference)
# model.load_state_dict(torch.load('model_pytorch.pth'))

# Evaluate model
model.eval()
predicted_classes = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        predicted_classes.extend(predicted.cpu().numpy())

y_true = y_test.numpy()
correct = np.nonzero(np.array(predicted_classes) == y_true)[0]
incorrect = np.nonzero(np.array(predicted_classes) != y_true)[0]

print("Correct predicted classes:", correct.shape[0])
print("Incorrect predicted classes:", incorrect.shape[0])
```

## PyTorch vs TensorFlow/Keras Key Differences

### Data loading and preprocessing
- **TensorFlow/Keras**: `tf.keras.datasets.fashion_mnist.load_data()`
- **PyTorch**: `torchvision.datasets.FashionMNIST()` with `DataLoader`

### Model
- **TensorFlow/Keras**: `keras.Sequential()` with predefined layers
- **PyTorch**: Custom class inheriting from `nn.Module` with explicit forward pass

### Layers
- **Conv2D**: `layers.Conv2D()` → `nn.Conv2d()`
- **MaxPooling**: `layers.MaxPooling2D()` → `nn.MaxPool2d()`
- **Dense/Linear**: `layers.Dense()` → `nn.Linear()`
- **Flatten**: `layers.Flatten()` → `x.view(x.size(0), -1)`

### Training
- **TensorFlow/Keras**: `model.compile()` + `model.fit()`
- **PyTorch**: Manual training loop with forward/backward passes

### Loss and optimization
- **TensorFlow/Keras**: `sparse_categorical_crossentropy` + `RMSprop`
- **PyTorch**: `nn.CrossEntropyLoss()` + `optim.RMSprop()`

### DataLoader and batching
- **TensorFlow/Keras**: Built into `model.fit()`
- **PyTorch**: `DataLoader` for efficient batch processing and data loading

![alt text](https://github.com/jylhakos/Deep-Learning-and-PyTorch/blob/main/Convolutional-Neural-Nets/cnn.png?raw=true)
