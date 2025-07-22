import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
from scipy.signal import convolve2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import pandas as pd
from numpy.random import seed
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
        #imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        #imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        #print(imagePadded)
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

def conv_step(matrix_slice, kernel, b):
    S = np.multiply(matrix_slice, kernel)
    Z = np.sum(S)
    Z = Z + b
    return Z

def plot_images(data_index):
    '''
        This is a function to plot first 9 images.
        data_index: indices of images.
    
    '''
    # plot the sample images
    f, ax = plt.subplots(3,3, figsize=(7,7))

    for i, indx in enumerate(data_index[:9]):
        ax[i//3, i%3].imshow(X_test[indx].reshape(28,28), cmap='gray')
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title("True:{}  Pred:{}".format(class_names[y_test[indx]],class_names[predicted_classes[indx]]), fontsize=8)
    plt.show()

img = mpimg.imread('Round3/R3/st1.png')
imgplot = plt.imshow(img)
plt.show()

input_matrix = np.array([[1, -1, 0], [2, 4, -1], [6, 0, 1]])
kernel = np.array([[1, 0], [0.5, -1]])
b = 0

matrix_slice = input_matrix[0:2,0:2]
#print('1. matrix_slice', matrix_slice)
x1 = conv_step(matrix_slice, kernel, b)
print("x1=", x1)

matrix_slice = np.array(input_matrix[0:2,1:3])
#print('2. matrix_slice', matrix_slice)
x2 = conv_step(matrix_slice, kernel, b)
print("x2=", x2)

matrix_slice = np.array(input_matrix[1:3,0:2])
#print('3. matrix_slice', matrix_slice)
x3 = conv_step(matrix_slice, kernel, b)
print("x3=", x3)

matrix_slice = np.array(input_matrix[1:3,1:3])
#print('4. matrix_slice', matrix_slice)
x4 = conv_step(matrix_slice, kernel, b)
print("x4=", x4)

#output_matrix = np.array([[x1,x2],[x3,x4]])
#print(output_matrix)

image = np.array([[1,-1,0], [2,4,-1], [6,0,1]])
kernel = np.array([[1,0], [0.5,-1]])
bias = 0

window = image[0:2,0:2]
#print('1. stride', stride)
x1 = convolution_operation(window, kernel, bias)
x1_cc = cross_correlation(window, kernel)
print("x1 =", x1, x1_cc[0][0])

window = np.array(image[0:2,1:3])
#print('2. window', window)
x2 = convolution_operation(window, kernel, bias)
x2_cc = cross_correlation(window, kernel)
print("x2 =", x2, x2_cc[0][0])

window = np.array(image[1:3,0:2])
#print('3. window', window)
x3 = convolution_operation(window, kernel, bias)
x3_cc = cross_correlation(window, kernel)
print("x3 =", x3, x3_cc[0][0])

window = np.array(image[1:3,1:3])
#print('4. window', window)
x4 = convolution_operation(window, kernel, bias)
x4_cc = cross_correlation(window, kernel)
print("x4 =", x4, x4_cc[0][0])


fname = Path('Round3') / 'R3' / 'guitar.png' # file name - Updated to use correct R3 folder path
image = Image.open(str(fname)).convert("L") # open image with python Image library
arr = np.asarray(image) # convert image to array

# define kernel values
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# perform convolution operation
conv_im1 = convolve2d(arr, kernel, mode='valid')

fig, axes = plt.subplots(1,3, figsize=(12,6))

axes[0].imshow(arr, cmap='gray')
axes[1].imshow(kernel, cmap='gray')
axes[2].imshow(conv_im1, cmap='gray', vmin=0, vmax=50)

axes[0].set_title('image', fontsize=20)
axes[1].set_title('kernel', fontsize=20)
axes[2].set_title('convolution', fontsize=20)

[ax.axis("off") for ax in axes]

plt.show()

np.random.seed(1)
X = np.random.randn(28, 28)
print('X.shape', X.shape)
#X = np.zeros((28, 28))
K = np.random.randn(3,3)
#K = np.zeros((3, 3))
#P = 1
#S = 1
#X_P = padding(X,P)
#print('X_P.shape', X_P.shape)
#Y = cross_correlation(X_P, K)
#print('Y.shape', Y.shape)

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

# X = 28
# K = 3
# P = (F - 1) / 2
# S = 1
# [(X - K + 2P) / S] + 1)
# 28 = (28 - 3 + 2*1) / 1 + 1
# 26 = (28 - 3 + 2*0) / 1 + 1
# 24 = (24 - 3 + 2*0) / 1 + 1

pad_cv1, stride_cv1 = 1,1
pad_cv2, stride_cv2 = 0,1
pad_cv3, stride_cv3 = 0,1

output_1 = 28-3+1
output_2 = output_1-3+1
output_3 = output_2-3+1
print('Outputs of convolutional layers', output_1, output_2, output_3)
params_conv_num = [((3*3*16) + 1) * output_1] + [((3*3*32) + 1) * output_2] + [((3*3*64) + 1) * output_3]
print('Number of convolutional layer parameters are ', params_conv_num)
params_num = (((3*3*16) + 1) * 26) + (((3*3*32) + 1) * 24) + (((3*3*64) + 1) * 22 )
print("Total number of parameters is", params_num)

seed(1)
torch.manual_seed(1)

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

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# For validation, we'll use part of the test set like in the original
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

print('X_test.shape',X_test.shape)
print('X_val.shape',X_val.shape)
print('y_test.shape',y_test.shape)
print('y_val.shape', y_val.shape)

# Normalize test set by 255
X_test = X_test / 255.0

# Convert to PyTorch tensors and add channel dimension
X_train = torch.FloatTensor(X_train).unsqueeze(1) / 255.0  # Add channel dimension
X_val = torch.FloatTensor(X_val).unsqueeze(1) / 255.0
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)

# Create datasets and data loaders
train_dataset_custom = TensorDataset(X_train, y_train)
val_dataset_custom = TensorDataset(X_val, y_val)
test_dataset_custom = TensorDataset(X_test, y_test)

train_loader_custom = DataLoader(train_dataset_custom, batch_size=32, shuffle=True)
val_loader_custom = DataLoader(val_dataset_custom, batch_size=32, shuffle=False)
test_loader_custom = DataLoader(test_dataset_custom, batch_size=32, shuffle=False)

# Shape of train, validation and test datasets
print(f'Number of training examples: {X_train.shape}')
print(f'Number of validation examples: {X_val.shape}')
print(f'Number of test examples: {X_test.shape}')

# Define the CNN model in PyTorch
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 27 * 27, 64)  # Adjusted for the actual output size
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)

# Print model summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model has {count_parameters(model)} trainable parameters")
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, train_accuracies, val_losses, val_accuracies

training = True

if training:
    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader_custom, val_loader_custom, criterion, optimizer, num_epochs=20
    )
    
    # Save model
    torch.save(model.state_dict(), 'model_pytorch.pth')
    
    # Plot training history
    history_df = pd.DataFrame({
        'loss': train_losses,
        'accuracy': [acc/100 for acc in train_accuracies],  # Convert to fraction
        'val_loss': val_losses,
        'val_accuracy': [acc/100 for acc in val_accuracies]
    })
    
    history_df[['loss', 'val_loss']].plot(figsize=(6,4))
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.grid(True)
    plt.show()
    
    history_df[['accuracy', 'val_accuracy']].plot(figsize=(6,4))
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.grid(True)
    plt.show()
else:
    # Load model
    model = CNNModel().to(device)
    model.load_state_dict(torch.load('model_pytorch.pth'))

# Define a functional model equivalent for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.maxpool = model.maxpool
        
    def forward(self, x):
        features = []
        x = F.relu(self.conv1(x))
        features.append(x)  # First conv layer output
        x = F.relu(self.conv2(x))
        features.append(x)  # Second conv layer output
        x = self.maxpool(x)
        features.append(x)  # Maxpool layer output
        return features

# Create feature extractor
feature_extractor = FeatureExtractor(model)

# Visualize feature maps for the first test image
model.eval()
with torch.no_grad():
    # Get a test image
    test_image = X_test[0].unsqueeze(0).to(device)  # Add batch dimension
    
    # Show the original image
    plt.figure(figsize=(4, 4))
    plt.imshow(test_image.cpu().squeeze(), cmap='gray')
    plt.title('Test Image')
    plt.axis('off')
    plt.show()
    
    # Extract features
    features = feature_extractor(test_image)
    first_layer_activation = features[0]
    
    print(f"First layer activation shape: {first_layer_activation.shape}")
    
    # Visualize feature maps from first convolutional layer
    plt.figure(figsize=(16,16))
    
    num_filters = first_layer_activation.shape[1]
    for i in range(min(num_filters, 64)):  # Show up to 64 feature maps
        plt.subplot(8,8,i+1)
        plt.axis('off')
        plt.imshow(first_layer_activation[0, i].cpu().numpy(), cmap='gray')
        plt.title('act. map '+ str(i+1))
    
    plt.tight_layout()
    plt.show()

# Get predictions for the test data
model.eval()
predicted_classes = []
with torch.no_grad():
    for data, _ in test_loader_custom:
        data = data.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        predicted_classes.extend(predicted.cpu().numpy())

predicted_classes = np.array(predicted_classes)

# Get true test labels
y_true = y_test.numpy()

# Calculate correct and incorrect predictions
correct = np.nonzero(predicted_classes == y_true)[0]
incorrect = np.nonzero(predicted_classes != y_true)[0]

print("Correct predicted classes:", correct.shape[0])
print("Incorrect predicted classes:", incorrect.shape[0])

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

target_names = [f"Class {i} ({class_names[i]}) :" for i in range(10)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

# Display correctly classified images
plot_images(correct)
