import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test basic operations
x = torch.tensor([1, 2, 3, 4, 5])
y = x * 2
print("Tensor test successful:", x.tolist(), "->", y.tolist())

print("PyTorch installation is working!")
