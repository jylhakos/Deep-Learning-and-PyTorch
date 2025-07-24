#!/bin/bash

# BERT QA Environment Setup Script
# This script sets up the Python virtual environment and installs all required packages

echo "=== BERT Question Answering Setup Script ==="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Set environment name
ENV_NAME="bert_qa_env"

# Create virtual environment
echo "Creating virtual environment: $ENV_NAME"
python3 -m venv $ENV_NAME

# Check if virtual environment was created successfully
if [ ! -d "$ENV_NAME" ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully"

# Activate virtual environment
echo "Activating virtual environment..."
source $ENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for lighter resource usage)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Hugging Face libraries
echo "Installing Hugging Face libraries..."
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install tokenizers>=0.14.0
pip install peft>=0.6.0
pip install optimum>=1.14.0
pip install evaluate>=0.4.0
pip install accelerate>=0.24.0

# Install Flask for API
echo "Installing Flask for RESTful API..."
pip install flask>=2.3.0
pip install flask-restful>=0.3.10

# Install additional dependencies
echo "Installing additional dependencies..."
pip install requests>=2.31.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.66.0

# Install quantization libraries
echo "Installing quantization libraries..."
pip install bitsandbytes
pip install intel-extension-for-pytorch>=2.0.0

# Install evaluation metrics
echo "Installing evaluation libraries..."
pip install rouge-score>=0.1.2

# Install development tools
echo "Installing development tools..."
pip install jupyter>=1.0.0
pip install notebook>=6.5.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0

# Install utility libraries
pip install python-dotenv>=1.0.0

# Install from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data models logs results

# Set up Hugging Face cache directory
export HF_HOME="./cache/huggingface"
mkdir -p $HF_HOME

echo "=== Setup Complete ==="
echo ""
echo "To activate the environment, run:"
echo "source $ENV_NAME/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "deactivate"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source $ENV_NAME/bin/activate"
echo "2. Run the fine-tuning script: python scripts/train_model.py"
echo "3. Start the API server: python app.py"
echo "4. Test with cURL: curl -X POST http://localhost:5000/api/health"
