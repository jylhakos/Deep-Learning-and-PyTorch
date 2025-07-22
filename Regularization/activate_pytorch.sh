#!/bin/bash
# Script to activate PyTorch virtual environment

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source pytorch_env/bin/activate

echo "PyTorch virtual environment activated!"
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
echo ""
echo "Available commands:"
echo "  python TensorFlowData-PyTorch.py     - Run PyTorch data pipeline demo"
echo "  python DataAugmentation-PyTorch.py   - Run data augmentation demo"
echo "  python TransferLearning-PyTorch.py   - Run transfer learning demo"
echo "  jupyter notebook                     - Start Jupyter notebook server"
echo ""
echo "To deactivate: deactivate"
