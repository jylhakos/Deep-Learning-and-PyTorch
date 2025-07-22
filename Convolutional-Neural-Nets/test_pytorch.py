import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Test basic PyTorch functionality
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test basic tensor operations
x = torch.randn(3, 3)
print("Random tensor:")
print(x)

# Test image path
image_path = Path('Round3') / 'R3' / 'volume.png'
print(f"Image path exists: {image_path.exists()}")

# Test dataset loading
try:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    print("Fashion-MNIST dataset loaded successfully!")
    print("Test dataset size:", len(test_dataset))
    
    # Get a sample
    sample_image, sample_label = test_dataset[0]
    print("Sample image shape:", sample_image.shape)
    print("Sample label:", sample_label)
    
    # Test image loading from R3 directory
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(str(image_path))
        print(f"Successfully loaded image from {image_path}")
        print(f"Image shape: {img.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

print("PyTorch setup test completed successfully!")
