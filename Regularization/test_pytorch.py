import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch installation successful!")

# Test basic tensor operations
x = torch.tensor([1, 2, 3, 4, 5])
y = x * 2
print("Tensor test:", x, "->", y)
