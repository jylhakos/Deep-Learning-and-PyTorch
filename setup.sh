#!/bin/bash

# Setup script for Deep Learning PyTorch project
echo "Setup Deep Learning PyTorch project."

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing PyTorch and dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying PyTorch installation..."
python -c "
import torch
import torchvision
print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name()}')
print('Setup completed successfully!')
"

echo ""
echo "Setup completed: To activate the environment in future sessions, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the PyTorch scripts:"
echo "  python Artificial-Neural-Networks/Artificial-Neural-Networks-PyTorch.py"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
