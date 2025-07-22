#!/bin/bash

# PyTorch Environment setup script
# This script sets up a Python virtual environment for PyTorch Deep Learning

echo "Setting up PyTorch environment for Deep Learning."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv pytorch_env

# Check if virtual environment was created successfully
if [ ! -d "pytorch_env" ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source pytorch_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch and dependencies
echo "Installing PyTorch and dependencies..."
pip install torch torchvision torchaudio matplotlib numpy scikit-learn jupyter notebook

# Install additional requirements if file exists
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
fi

# Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "    source pytorch_env/bin/activate"
echo ""
echo "To run the PyTorch examples:"
echo "    python Round2/Gradient-Based-Learning-PyTorch.py"
echo ""
echo "To start Jupyter Notebook:"
echo "    jupyter notebook"
