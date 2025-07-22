import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

# PyTorch transforms for data augmentation
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(150, scale=(0.9, 1.1)),
])

# The path to the dataset - Updated to use Dataset folder
base_dir = pathlib.Path.cwd() / '..' / 'Dataset' / 'cats_and_dogs_small'

# directories for training, validation and test splits
train_dir = base_dir / 'train'
validation_dir = base_dir / 'validation'
test_dir = base_dir / 'test'

CLASS_NAMES = ['cats', 'dogs']
BATCH_SIZE = 32
IMG_SIZE = 150
EPOCHS = 20

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
            dummy_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor()
            ])
            dummy_image = dummy_transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white'))
            return dummy_image, 0
    
    def _get_label_from_path(self, image_path):
        parts = str(image_path).split(os.path.sep)
        class_name = parts[-2]
        return self.class_names.index(class_name) if class_name in self.class_names else 0

def create_transforms(augment=False):
    """Create transforms with or without augmentation"""
    base_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ]
    
    if augment:
        augment_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0))
        ]
        return transforms.Compose(augment_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)

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
    train_paths = val_paths = test_paths = []

# Create datasets with data augmentation for training
train_transform = create_transforms(augment=True)
val_transform = create_transforms(augment=False)

if train_paths:
    train_dataset = ImageDataset(train_paths, CLASS_NAMES, transform=train_transform)
    val_dataset = ImageDataset(val_paths, CLASS_NAMES, transform=val_transform)
    test_dataset = ImageDataset(test_paths, CLASS_NAMES, transform=val_transform)
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
else:
    print("Warning: No datasets created due to missing image paths")
    train_dataloader = val_dataloader = test_dataloader = None

# Define CNN Model
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
        
        # Calculate the size after conv layers
        # After MaxPool2d with kernel_size=2: 150 -> 75
        # 75 * 75 * 32 = 180000
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

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = CNNModel(num_classes=1).to(device)

# Print model summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model has {count_parameters(model):,} trainable parameters")
print(model)

# Training configuration
training = True
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=2):
    """Train the model"""
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    if train_loader is None:
        print("Warning: No training data available")
        return history
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.float().to(device)
            labels = labels.unsqueeze(1)  # Add dimension for BCE loss
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            if val_loader:
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.float().to(device)
                    labels = labels.unsqueeze(1)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
            else:
                val_loss = val_acc = 0.0
        
        # Store history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history

def plot_history(history):
    """Plot training history"""
    if not history['loss']:
        print("No training history to plot")
        return
        
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    df_accuracy = pd.DataFrame({
        'accuracy': history['accuracy'],
        'val_accuracy': history['val_accuracy']
    })
    
    df_loss = pd.DataFrame({
        'loss': history['loss'],
        'val_loss': history['val_loss']
    })
    
    df_accuracy.plot(ax=ax[0])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    
    df_loss.plot(ax=ax[1])
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

def check_accuracy(model, test_loader, expected_accuracy=None):
    """Evaluate model accuracy"""
    if test_loader is None:
        print("No test data available")
        return
    
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
    print(f'The test set accuracy of model is {accuracy:.2f}')
    
    if expected_accuracy and accuracy < expected_accuracy:
        print(f"Warning: Accuracy {accuracy:.2f} is below expected {expected_accuracy:.2f}")

# Train the base model
if training and train_dataloader:
    print("Training base model...")
    history = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs=2)
    torch.save(model.state_dict(), 'model_pytorch.pth')
    plot_history(history)
    check_accuracy(model, test_dataloader)
else:
    try:
        model.load_state_dict(torch.load('model_pytorch.pth'))
        print("Loaded pre-trained model")
    except:
        print("No pre-trained model found and no training data available")

# Data Augmentation Examples
print("\n=== Data Augmentation Examples ===")

# Show augmented images
if train_dataloader:
    try:
        # Get a batch of images
        images, labels = next(iter(train_dataloader))
        
        plt.figure(figsize=(12, 8))
        
        # Show original and augmented versions
        for i in range(min(9, len(images))):
            # Original image
            plt.subplot(3, 6, i*2 + 1)
            img = images[i].permute(1, 2, 0)
            img = torch.clamp(img, 0, 1)
            plt.imshow(img)
            plt.title(f'Original {CLASS_NAMES[labels[i]]}')
            plt.axis('off')
            
            # Augmented image (apply augmentation again)
            aug_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor()
            ])
            
            plt.subplot(3, 6, i*2 + 2)
            aug_img = aug_transform(images[i])
            aug_img = aug_img.permute(1, 2, 0)
            aug_img = torch.clamp(aug_img, 0, 1)
            plt.imshow(aug_img)
            plt.title(f'Augmented {CLASS_NAMES[labels[i]]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display augmentation examples: {e}")

# Individual augmentation examples
print("\n=== Individual Augmentation Techniques ===")

def visualize_augmentation(original_tensor, augmented_tensor, title):
    """Visualize original vs augmented image"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    orig_img = original_tensor.permute(1, 2, 0)
    orig_img = torch.clamp(orig_img, 0, 1)
    plt.imshow(orig_img)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f'{title}')
    aug_img = augmented_tensor.permute(1, 2, 0)
    aug_img = torch.clamp(aug_img, 0, 1)
    plt.imshow(aug_img)
    plt.axis("off")
    plt.show()

if train_dataloader:
    try:
        # Get sample image
        sample_images, sample_labels = next(iter(train_dataloader))
        sample_image = sample_images[0]
        
        # Convert to PIL for augmentations
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        pil_image = to_pil(sample_image)
        
        # Horizontal flip
        flip_transform = transforms.RandomHorizontalFlip(p=1.0)
        flipped_image = to_tensor(flip_transform(pil_image))
        visualize_augmentation(sample_image, flipped_image, 'Horizontally Flipped')
        
        # Rotation
        rotation_transform = transforms.RandomRotation(degrees=45)
        rotated_image = to_tensor(rotation_transform(pil_image))
        visualize_augmentation(sample_image, rotated_image, 'Rotated')
        
        # Color jitter (brightness, saturation)
        color_transform = transforms.ColorJitter(brightness=0.5, saturation=2.0)
        colored_image = to_tensor(color_transform(pil_image))
        visualize_augmentation(sample_image, colored_image, 'Color Adjusted')
        
        # Center crop
        crop_transform = transforms.CenterCrop(100)
        resize_transform = transforms.Resize((IMG_SIZE, IMG_SIZE))
        cropped_image = to_tensor(resize_transform(crop_transform(pil_image)))
        visualize_augmentation(sample_image, cropped_image, 'Center Cropped')
        
    except Exception as e:
        print(f"Could not display individual augmentations: {e}")

print("\n=== Data Augmentation Pipeline Benefits ===")
print("PyTorch provides several advantages for data augmentation:")
print("1. Built-in transforms in torchvision.transforms")
print("2. Efficient GPU-accelerated augmentations")
print("3. Easy integration with DataLoader")
print("4. Composable transforms for complex pipelines")
print("5. Automatic batching and parallel processing")

# Create augmented dataset loader function
def load_image_with_augmentation(image_path, class_names, img_size=150):
    """Load and augment a single image"""
    augmentation_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(120),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = augmentation_transforms(image)
        
        # Get label from path
        parts = str(image_path).split(os.path.sep)
        class_name = parts[-2]
        label = class_names.index(class_name) if class_name in class_names else 0
        
        return image, torch.tensor(label, dtype=torch.long)
    except:
        # Return dummy data
        dummy_image = torch.zeros(3, img_size, img_size)
        return dummy_image, torch.tensor(0, dtype=torch.long)

# Show augmented batch
if train_paths:
    print("\n=== Augmented Training Batch ===")
    try:
        augmented_images = []
        augmented_labels = []
        
        for i, path in enumerate(train_paths[:9]):  # Get first 9 images
            img, label = load_image_with_augmentation(path, CLASS_NAMES)
            augmented_images.append(img)
            augmented_labels.append(label)
        
        plt.figure(figsize=(10, 10))
        
        for i in range(min(9, len(augmented_images))):
            ax = plt.subplot(3, 3, i + 1)
            img = augmented_images[i].permute(1, 2, 0)
            img = torch.clamp(img, 0, 1)
            plt.imshow(img.numpy())
            plt.title(CLASS_NAMES[augmented_labels[i].item()])
            plt.axis("off")
        
        plt.suptitle('Augmented Training Images')
        plt.show()
        
    except Exception as e:
        print(f"Could not display augmented batch: {e}")

# Display info about PyTorch data pipeline benefits
print("\n=== PyTorch Data Pipeline Benefits ===")
print("1. **DataLoader**: Efficient batching, shuffling, and parallel loading")
print("2. **Custom Datasets**: Flexible data loading from any source")
print("3. **Transforms**: Powerful augmentation and preprocessing pipeline")
print("4. **GPU Acceleration**: Seamless integration with CUDA")
print("5. **Memory Efficiency**: Lazy loading and on-the-fly transformations")
