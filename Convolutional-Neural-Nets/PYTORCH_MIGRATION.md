# PyTorch migration

## Overview
This document summarizes the migration from TensorFlow/Keras to PyTorch for the Convolutional Neural Networks project.

## Files

### 1. `R3-PyTorch.py` 
- **Purpose**: PyTorch version of the original `R3.py` file
- **Changes**:
  - Replaced `tensorflow` imports with `torch` imports
  - Replaced `tf.keras.datasets.fashion_mnist` with `torchvision.datasets.FashionMNIST`
  - Replaced `keras.Sequential` model with custom `nn.Module` class
  - Replaced TensorFlow layers with PyTorch equivalents:
    - `layers.Conv2D` → `nn.Conv2d`
    - `layers.MaxPooling2D` → `nn.MaxPool2d`
    - `layers.Dense` → `nn.Linear`
    - `layers.Flatten` → `x.view(x.size(0), -1)`
  - Implemented manual training loop instead of `model.fit()`
  - Used `DataLoader` for batch processing
  - Created feature extractor for visualization

### 2. `Round3/Round3_CNN-PyTorch.ipynb`
- **Purpose**: PyTorch version of the original Jupyter notebook
- **Changes**:
  - Updated all import statements to use PyTorch
  - Converted all code cells to use PyTorch syntax
  - Updated markdown cells to reference PyTorch documentation
  - Added comprehensive comparison between TensorFlow/Keras and PyTorch
  - Maintained the same educational structure and explanations

### 3. Updated `README.md`
  - Added PyTorch examples
  - Included PyTorch vs TensorFlow/Keras comparison table
  - Added data loading examples with `DataLoader`
  - Included training loop examples
  - Added model evaluation examples

### 4. Updated `.gitignore`
  - Allow PNG, JPG, and JPEG files (needed for documentation)
  - Keep virtual environment ignored
  - Add PyTorch model files (*.pth, *.pt) to ignore list
  - Add data directory to ignore list

## Migration

### Data loading
- **From**: `tf.keras.datasets.fashion_mnist.load_data()`
- **To**: `torchvision.datasets.FashionMNIST()` with `DataLoader`

### Model architecture
- **From**: `keras.Sequential()` with predefined layers
- **To**: Custom class inheriting from `nn.Module`

### Layers
| TensorFlow/Keras | PyTorch |
|------------------|---------|
| `layers.Conv2D()` | `nn.Conv2d()` |
| `layers.MaxPooling2D()` | `nn.MaxPool2d()` |
| `layers.Dense()` | `nn.Linear()` |
| `layers.Flatten()` | `x.view(x.size(0), -1)` |
| `layers.Dropout()` | `nn.Dropout()` |

### Training
- **From**: `model.compile()` + `model.fit()`
- **To**: Manual training loop with:
  - `optimizer.zero_grad()`
  - `loss = criterion(output, target)`
  - `loss.backward()`
  - `optimizer.step()`

### Loss and optimization
- **From**: `keras.losses.sparse_categorical_crossentropy` + `keras.optimizers.RMSprop()`
- **To**: `nn.CrossEntropyLoss()` + `optim.RMSprop()`

### Model saving/loading
- **From**: `model.save('model.h5')` / `tf.keras.models.load_model('model.h5')`
- **To**: `torch.save(model.state_dict(), 'model.pth')` / `model.load_state_dict(torch.load('model.pth'))`

## Dataset

### PyTorch DataLoader
- **Automatic batching**: Handles batch creation automatically
- **Shuffling**: Built-in data shuffling for training
- **Parallel loading**: Supports multi-threaded data loading
- **Memory efficiency**: Loads data on-demand

### Usage
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for batch_idx, (data, target) in enumerate(train_loader):
    # Training code here
    pass
```

## Python Virtual Environment

The project uses a Python virtual environment (`.venv`) with the following key packages:
- `torch`: Main PyTorch library
- `torchvision`: Computer vision utilities and datasets
- `torchaudio`: Audio processing (for completeness)
- `numpy`: Numerical computing
- `pandas`: Data analysis
- `matplotlib`: Plotting
- `scikit-learn`: Machine learning metrics
- `pillow`: Image processing
- `jupyter`: Notebook environment

## Image Paths

### **Complete Solution Applied**

1. **PyTorch Notebook (`Round3_CNN-PyTorch.ipynb`)**:
   - **Fixed**: All image paths updated to `./R3/` (since notebook runs from `/Round3/` directory)
   - **Verified**: All images now load correctly in both markdown and Python cells
   - **Tested**: Image loading cells execute successfully

2. **PyTorch Python Script (`R3-PyTorch.py`)**:
   - **Fixed**: `mpimg.imread('R3/st1.png')` → `mpimg.imread('Round3/R3/st1.png')`
   - **Fixed**: `Path('R3') / 'guitar.png'` → `Path('Round3') / 'R3' / 'guitar.png'`
   - **Result**: Script now runs from main directory with correct paths

3. **Original TensorFlow Script (`R3.py`)** - *Also Fixed for Consistency*:
   - **Fixed**: `mpimg.imread('R3/st1.png')` → `mpimg.imread('Round3/R3/st1.png')` 
   - **Fixed**: `Path('..') /'..' /'Dataset' / 'R3' / 'guitar.png'` → `Path('Round3') / 'R3' / 'guitar.png'`

### **Directories & paths**
```
/Convolutional-Neural-Nets/           # Main project directory
├── R3.py                            # Uses: Round3/R3/
├── R3-PyTorch.py                    # Uses: Round3/R3/
├── test_pytorch.py                  # Tests: Round3/R3/volume.png
├── Round3/                          # Subdirectory
│   ├── Round3_CNN-PyTorch.ipynb    # Uses: ./R3/
│   ├── Round3_CNN.ipynb            # Original notebook
│   └── R3/                         # Images directory
│       ├── volume.png
│       ├── c1.png, c2.png, c3.png, c4.png
│       ├── guitar.png, st1.png
│       └── ... (other images)
```

### **Path resolution**
- **From main directory** (`/Convolutional-Neural-Nets/`): Use `Round3/R3/filename.png`
- **From Round3 directory** (`/Round3/`): Use `./R3/filename.png`

### **Verification**
- Notebook images display correctly in markdown cells
- Notebook Python cells execute without FileNotFoundError
- PyTorch script can access images from main directory
- Original script paths also corrected for consistency
- Test script confirms path accessibility

## PyTorch migration

1. **Explicit control**: PyTorch provides more explicit control over the training process
2. **Dynamic graphs**: PyTorch uses dynamic computation graphs (more flexible)
3. **Debugging**: Easier to debug with standard Python debugging tools
4. **Research**: More popular in research communities
5. **Memory management**: More control over GPU memory usage
6. **Pythonic**: More intuitive Python-like syntax

## Testing

- Created `test_pytorch.py` to verify PyTorch installation and basic functionality
- Verified notebook execution with proper kernel configuration
- Tested image loading and display functionality

## Original files

The original TensorFlow/Keras files are preserved:
- `R3.py` (original TensorFlow version)
- `Round3/Round3_CNN.ipynb` (original TensorFlow version)

## Usage

1. **Activate virtual environment**: The `.venv` environment is already configured
2. **Run PyTorch version**: Use `R3-PyTorch.py` for the PyTorch implementation
3. **Use PyTorch notebook**: Open `Round3/Round3_CNN-PyTorch.ipynb` for interactive learning
4. **Compare implementations**: Compare original and PyTorch versions to understand differences

## Performance

- PyTorch models should perform similarly to TensorFlow versions
- GPU acceleration available if CUDA is installed
