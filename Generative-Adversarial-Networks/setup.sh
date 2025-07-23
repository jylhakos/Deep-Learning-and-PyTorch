#!/bin/bash

# Setup script for GANs PyTorch project
echo "Setting up GANs PyTorch project..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing PyTorch and dependencies..."
pip install -r requirements.txt

# Test installation
echo "Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"

# Create Dataset directory if it doesn't exist
if [ ! -d "../Dataset" ]; then
    echo "Creating Dataset directory..."
    mkdir -p ../Dataset
fi

echo ""
echo "Setup complete! To get started:"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the PyTorch GAN script: python GAN-PyTorch.py"
echo "3. Or start Jupyter notebook: jupyter notebook Round6_GAN.ipynb"
echo ""
echo "Available files:"
echo "  - GAN.py (Original TensorFlow implementation)"
echo "  - GAN-PyTorch.py (PyTorch implementation)"
echo "  - utils.py (TensorFlow utilities)"
echo "  - utils-PyTorch.py (PyTorch utilities)"
echo "  - Round6_GAN.ipynb (Jupyter notebook with both implementations)"
echo "  - test_pytorch.py (Test PyTorch installation)"
