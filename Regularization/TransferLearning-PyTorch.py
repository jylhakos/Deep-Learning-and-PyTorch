import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import numpy as np
import os

# Set training=True, when training network
training = True

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 224  # VGG expects 224x224 input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset class for image loading
class ImageDataset(Dataset):
    def __init__(self, image_paths, class_names, transform=None):
        self.image_paths = image_paths
        self.class_names = class_names
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Extract label from path
            label = self._get_label_from_path(image_path)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data if image not found
            if self.transform:
                dummy_image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
                # Apply normalization to dummy image
                dummy_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(dummy_image)
            else:
                dummy_image = torch.zeros(3, IMG_SIZE, IMG_SIZE)
            return dummy_image, 0
    
    def _get_label_from_path(self, image_path):
        parts = str(image_path).split(os.path.sep)
        class_name = parts[-2]
        return self.class_names.index(class_name) if class_name in self.class_names else 0

# Define transforms with VGG preprocessing (ImageNet normalization)
def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

# Dataset paths - Updated to use Dataset folder
base_dir = pathlib.Path.cwd() / '..' / 'Dataset' / 'cats_and_dogs_small'

# Directories for training, validation and test sets
train_dir = base_dir / 'train' 
validation_dir = base_dir / 'validation'
test_dir = base_dir / 'test'

# Load image paths
try:
    train_paths = list(train_dir.glob('*/*.jpg'))
    val_paths = list(validation_dir.glob('*/*.jpg'))
    test_paths = list(test_dir.glob('*/*.jpg'))
    
    print(f"Found {len(train_paths)} training images")
    print(f"Found {len(val_paths)} validation images")
    print(f"Found {len(test_paths)} test images")
except Exception as e:
    print(f"Warning: Could not load image paths: {e}")
    # Create dummy paths for demonstration
    train_paths = val_paths = test_paths = []

# Create datasets
train_transform = get_transforms(augment=True)
val_transform = get_transforms(augment=False)

if train_paths:
    train_dataset = ImageDataset(train_paths, CLASS_NAMES, transform=train_transform)
    val_dataset = ImageDataset(val_paths, CLASS_NAMES, transform=val_transform)
    test_dataset = ImageDataset(test_paths, CLASS_NAMES, transform=val_transform)
else:
    # Create dummy datasets for demonstration
    class DummyDataset(Dataset):
        def __init__(self, size, transform=None):
            self.size = size
            self.transform = transform
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            dummy_image = torch.randn(3, IMG_SIZE, IMG_SIZE)
            if self.transform:
                dummy_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(dummy_image)
            return dummy_image, idx % 2  # Alternate between class 0 and 1
    
    train_dataset = DummyDataset(1000, transform=val_transform)
    val_dataset = DummyDataset(500, transform=val_transform)
    test_dataset = DummyDataset(500, transform=val_transform)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ===== Pre-trained VGG16 Model Setup =====

class VGGFeatureExtractor(nn.Module):
    """VGG16 feature extractor without classifier"""
    def __init__(self, pretrained=True):
        super(VGGFeatureExtractor, self).__init__()
        # Load pre-trained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        
        # Extract only the feature extraction part (convolutional layers)
        self.features = vgg16.features
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Freeze all parameters initially
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class VGGClassifier(nn.Module):
    """Custom classifier head for VGG features"""
    def __init__(self, input_features=25088, hidden_size=4096, num_classes=1):
        super(VGGClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class TransferLearningModel(nn.Module):
    """Complete transfer learning model"""
    def __init__(self, pretrained=True, num_classes=1):
        super(TransferLearningModel, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(pretrained=pretrained)
        self.classifier = VGGClassifier(input_features=25088, num_classes=num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
    
    def freeze_features(self):
        """Freeze feature extractor weights"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_features(self, layer_start=0):
        """Unfreeze feature extractor from specified layer"""
        layers = list(self.feature_extractor.features.children())
        for i, layer in enumerate(layers[layer_start:], layer_start):
            for param in layer.parameters():
                param.requires_grad = True

# Create pre-trained VGG16 feature extractor
print("=== Loading Pre-trained VGG16 ===")
feature_extractor = VGGFeatureExtractor(pretrained=True).to(device)

print("Feature extractor architecture:")
print(feature_extractor)

# Test feature extraction
print("\n=== Testing Feature Extraction ===")
sample_batch = next(iter(train_dataloader))
sample_images, sample_labels = sample_batch[0].to(device), sample_batch[1]

with torch.no_grad():
    features = feature_extractor(sample_images)
    print(f"Input shape: {sample_images.shape}")
    print(f"Feature shape: {features.shape}")

# ===== Feature Extraction Approach =====
print("\n=== Feature Extraction Approach ===")

def extract_features(dataloader, feature_extractor):
    """Extract features for all data"""
    feature_extractor.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)  # Flatten
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

# Extract features for all datasets
print("Extracting features from training set...")
train_features, train_labels = extract_features(train_dataloader, feature_extractor)

print("Extracting features from validation set...")
val_features, val_labels = extract_features(val_dataloader, feature_extractor)

print("Extracting features from test set...")
test_features, test_labels = extract_features(test_dataloader, feature_extractor)

print(f"Train features shape: {train_features.shape}")
print(f"Validation features shape: {val_features.shape}")
print(f"Test features shape: {test_features.shape}")

# Create classifier head
classifier_head = VGGClassifier(input_features=train_features.shape[1], num_classes=1).to(device)

print("\nClassifier head:")
print(classifier_head)

# Training function for classifier head
def train_classifier(model, train_features, train_labels, val_features, val_labels, epochs=20):
    """Train only the classifier head"""
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    # Convert to tensors and move to device
    train_features = train_features.to(device)
    train_labels = train_labels.float().to(device).unsqueeze(1)
    val_features = val_features.to(device)
    val_labels = val_labels.float().to(device).unsqueeze(1)
    
    # Create dataset for batching
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_labels).item()
            val_predicted = (val_outputs > 0.5).float()
            val_acc = (val_predicted == val_labels).sum().item() / len(val_labels)
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history

# Train classifier head on extracted features
if training:
    print("\n=== Training Classifier Head on Extracted Features ===")
    classifier_history = train_classifier(
        classifier_head, train_features, train_labels, 
        val_features, val_labels, epochs=20
    )
    torch.save(classifier_head.state_dict(), 'classifier_head_pytorch.pth')
else:
    try:
        classifier_head.load_state_dict(torch.load('classifier_head_pytorch.pth'))
        print("Loaded pre-trained classifier head")
        classifier_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    except:
        print("No pre-trained classifier found")
        classifier_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

# Plot classifier training history
def plot_history(history, title="Training History"):
    """Plot training history"""
    if not history['loss']:
        print("No training history to plot")
        return
        
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax[0].plot(history['accuracy'], label='Training Accuracy')
    ax[0].plot(history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0.5, 1.05)
    ax[0].legend()
    ax[0].grid(True)
    
    # Loss plot
    ax[1].plot(history['loss'], label='Training Loss')
    ax[1].plot(history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_ylim(0, max(max(history['loss']), max(history['val_loss'])) * 1.1)
    ax[1].legend()
    ax[1].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if classifier_history['loss']:
    plot_history(classifier_history, "Classifier Head Training")

# Evaluate classifier head
def evaluate_model(model, test_features, test_labels):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        test_features = test_features.to(device)
        test_labels = test_labels.float().to(device).unsqueeze(1)
        
        outputs = model(test_features)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    
    print(f'The test set accuracy of classifier head is {accuracy:.2f}')
    return accuracy

evaluate_model(classifier_head, test_features, test_labels)

# ===== Fine-tuning Approach =====
print("\n=== Fine-tuning Pre-trained Model ===")

# Create complete transfer learning model
transfer_model = TransferLearningModel(pretrained=True, num_classes=1).to(device)

# Initially freeze feature extractor
transfer_model.freeze_features()

print("Trainable parameters (classifier only):")
trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in transfer_model.parameters())
print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")

# Training function for complete model
def train_transfer_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train the transfer learning model"""
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history

# Train with frozen features
if training:
    print("Training with frozen feature extractor...")
    frozen_history = train_transfer_model(transfer_model, train_dataloader, val_dataloader, epochs=10)
    torch.save(transfer_model.state_dict(), 'transfer_model_frozen_pytorch.pth')
    plot_history(frozen_history, "Transfer Learning - Frozen Features")

# Fine-tuning: Unfreeze some layers
print("\n=== Fine-tuning: Unfreezing Last Layers ===")

# Unfreeze the last few layers of VGG16
transfer_model.unfreeze_features(layer_start=15)  # Unfreeze from layer 15 onwards

print("Trainable parameters after unfreezing:")
trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")

# Print which layers are trainable
print("\nLayer trainability status:")
for i, layer in enumerate(transfer_model.feature_extractor.features):
    trainable = any(p.requires_grad for p in layer.parameters() if hasattr(layer, 'parameters'))
    print(f"Layer {i}: {layer.__class__.__name__} - Trainable: {trainable}")

# Fine-tuning with lower learning rate
if training:
    print("Fine-tuning with unfrozen layers...")
    
    # Use lower learning rate for fine-tuning
    fine_tune_history = train_transfer_model(
        transfer_model, train_dataloader, val_dataloader, 
        epochs=10, lr=1e-5
    )
    torch.save(transfer_model.state_dict(), 'transfer_model_finetuned_pytorch.pth')
    
    # Combine histories
    combined_history = {
        'loss': frozen_history['loss'] + fine_tune_history['loss'],
        'accuracy': frozen_history['accuracy'] + fine_tune_history['accuracy'],
        'val_loss': frozen_history['val_loss'] + fine_tune_history['val_loss'],
        'val_accuracy': frozen_history['val_accuracy'] + fine_tune_history['val_accuracy']
    }
    
    plot_history(combined_history, "Complete Transfer Learning")
    
    # Final evaluation
    def evaluate_transfer_model(model, test_loader):
        """Evaluate transfer learning model"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()
        
        accuracy = correct / total
        print(f'Final test set accuracy: {accuracy:.2f}')
        return accuracy
    
    final_accuracy = evaluate_transfer_model(transfer_model, test_dataloader)
else:
    try:
        transfer_model.load_state_dict(torch.load('transfer_model_finetuned_pytorch.pth'))
        print("Loaded fine-tuned model")
    except:
        print("No fine-tuned model found")

print("\n=== Transfer Learning with PyTorch Summary ===")
print("PyTorch Transfer Learning advantages:")
print("1. **Pre-trained Models**: Easy access to pre-trained models via torchvision.models")
print("2. **Flexible Architecture**: Easy to modify and combine different model parts")
print("3. **Fine-grained Control**: Precise control over which layers to freeze/unfreeze")
print("4. **Efficient Training**: Automatic gradient computation only for trainable parameters")
print("5. **State Management**: Easy model saving/loading with state_dict()")
print("6. **GPU Acceleration**: Seamless GPU utilization for faster training")

print("\n=== Transfer Learning Strategies ===")
print("1. **Feature Extraction**: Freeze pre-trained layers, train only classifier")
print("2. **Fine-tuning**: Unfreeze some layers and train with lower learning rate")
print("3. **Full Fine-tuning**: Unfreeze all layers (requires more data)")
print("4. **Layer-wise Learning Rates**: Different learning rates for different layers")
print("5. **Progressive Unfreezing**: Gradually unfreeze layers during training")
