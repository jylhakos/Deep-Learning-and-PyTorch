# $ source pytorch_env/bin/activate && python3 TensorFlowData-PyTorch.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt 

# Create tensor from list - PyTorch equivalent of tf.data.Dataset.from_tensor_slices
tensor_data = torch.tensor([1, 2, 3, 4, 5, 6])

# PyTorch tensors can be iterated directly
for value in tensor_data:
    print(value)
    print(value.numpy())

# Repeat tensor - PyTorch equivalent
tensor_data = torch.tensor([1, 2, 3])
repeated_tensor = tensor_data.repeat(2)

for value in repeated_tensor:
    print(value.numpy())

def preprocess(x):
    return x * x

# Apply function to tensor
tensor_data = torch.tensor([1, 2, 3])
processed_tensor = preprocess(tensor_data)

for value in processed_tensor:
    print(value.numpy())

# PyTorch DataLoader for batch processing
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Create dataset with preprocessing
dataset = CustomDataset(torch.arange(10).repeat(100), transform=preprocess)

# Create DataLoader for batching, shuffling, and prefetching
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

# PyTorch equivalent of taking batches
for i, batch in enumerate(dataloader):
    if i >= 3:  # Take only 3 batches
        break
    print(batch.numpy())

# Generators with PyTorch Dataset

def sequence_generator():
    number = 0
    while True:
        yield number
        number += 1

numbers = []

for number in sequence_generator():
    numbers.append(number)
    if number > 9:
        break

print(numbers)

a = sequence_generator()

print("The variable is type of ", type(a))

b = next(a)

print(b)

# Create PyTorch Dataset from python generator
class GeneratorDataset(Dataset):
    def __init__(self, generator_func, length):
        self.generator_func = generator_func
        self.length = length
        self.data = [next(generator_func()) for _ in range(length)]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.int32)

# Create dataset from generator
generator_dataset = GeneratorDataset(sequence_generator, 100)

# Create DataLoader with transformations
generator_dataloader = DataLoader(
    generator_dataset,
    batch_size=5,
    shuffle=True,
    num_workers=2
)

# Apply preprocessing to dataset
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
preprocessed_dataloader = DataLoader(preprocessed_dataset, batch_size=5, shuffle=True)

# Retrieve first batch
first_batch = next(iter(preprocessed_dataloader))
print(first_batch.numpy())

# Retrieve batches with a for-loop
i = 1
for batch in preprocessed_dataloader:
    if i > 3:
        break
    print("batch", i, batch.numpy())
    i += 1

# Text file reading with PyTorch
file_path = '../Dataset/R4/poem.txt'  # Updated path to Dataset folder

# PyTorch text dataset
class TextDataset(Dataset):
    def __init__(self, file_path):
        self.lines = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Creating sample data.")
            self.lines = ["Sample line 1\n", "Sample line 2\n", "Sample line 3\n"]
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        return self.lines[idx].strip()

# Create dataset from txt file
text_dataset = TextDataset(file_path)
text_dataloader = DataLoader(text_dataset, batch_size=1)

# Print samples from dataset
for i, line in enumerate(text_dataloader):
    if i >= 5:
        break
    print(line[0])

# Shuffle and batch text data
shuffled_dataloader = DataLoader(text_dataset, batch_size=5, shuffle=True)

for i, batch in enumerate(shuffled_dataloader):
    if i >= 2:
        break
    print("\nbatch:\n", batch)

# Image dataset with PyTorch
# The path to the dataset - Updated to use Dataset folder
base_dir = pathlib.Path.cwd() / '..' / 'Dataset' / 'cats_and_dogs'

print(base_dir)

print(type(base_dir.glob('*')))

for file in base_dir.glob('*'):
    print(file)

# Count jpg files in all subdirectories
try:
    image_count = len(list(base_dir.glob('*/*/*.jpg')))
    print(f'Total number of images in the dataset: {image_count}')
except:
    print("Warning: Dataset not found. Please ensure Dataset folder exists two levels up.")
    image_count = 0

# Custom Image Dataset for PyTorch
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
        except:
            # Return dummy data if image not found
            dummy_image = torch.zeros(3, 150, 150)
            return dummy_image, 0
    
    def _get_label_from_path(self, image_path):
        parts = str(image_path).split(os.path.sep)
        class_name = parts[-2]
        return self.class_names.index(class_name) if class_name in self.class_names else 0

# Define transforms
IMG_SIZE = 150
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ['cats', 'dogs']

# Create image datasets
try:
    train_paths = list((base_dir / 'train').glob('*/*.jpg'))
    val_paths = list((base_dir / 'validation').glob('*/*.jpg'))
    test_paths = list((base_dir / 'test').glob('*/*.jpg'))
    
    train_dataset = ImageDataset(train_paths, CLASS_NAMES, transform=transform)
    val_dataset = ImageDataset(val_paths, CLASS_NAMES, transform=transform)
    test_dataset = ImageDataset(test_paths, CLASS_NAMES, transform=transform)
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f'Number of images in the training set:\t {len(train_dataset)}')
    print(f'Number of images in the validation set:\t {len(val_dataset)}')
    print(f'Number of images in the test set:\t {len(test_dataset)}')
    
    # Get first batch and display images
    try:
        image_batch, label_batch = next(iter(train_dataloader))
        
        plt.figure(figsize=(10, 10))
        
        # Convert tensor to numpy for display
        def denormalize(tensor, mean, std):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return tensor
        
        for i in range(min(9, len(image_batch))):
            ax = plt.subplot(3, 3, i + 1)
            
            # Denormalize and convert to displayable format
            img = image_batch[i].clone()
            img = denormalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0)
            
            plt.imshow(img.numpy())
            label = label_batch[i].item()
            plt.title(CLASS_NAMES[label] if label < len(CLASS_NAMES) else 'Unknown')
            plt.axis("off")
        
        plt.show()
    except Exception as e:
        print(f"Could not display images: {e}")

except Exception as e:
    print(f"Warning: Could not create image datasets: {e}")
    print("Please ensure the Dataset folder structure exists: Dataset/cats_and_dogs/train|validation|test/cats|dogs/")
