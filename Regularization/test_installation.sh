#!/bin/bash
# Simple PyTorch test script

echo "=== PyTorch Installation Test ==="
echo ""

cd "$(dirname "$0")"
source pytorch_env/bin/activate

echo "Virtual environment activated"
echo "Python path: $(which python)"
echo ""

echo "Testing PyTorch import..."
python -c "
try:
    import torch
    import torchvision
    print('✅ PyTorch imported successfully')
    print('✅ PyTorch version:', torch.__version__)
    print('✅ Torchvision version:', torchvision.__version__)
    print('✅ CUDA available:', torch.cuda.is_available())
    
    # Test basic tensor operations
    x = torch.tensor([1, 2, 3])
    y = x * 2
    print('✅ Tensor operations working:', x.tolist(), '->', y.tolist())
    
    print('')
    print('🎉 PyTorch installation is working perfectly!')
    
except ImportError as e:
    print('❌ Import error:', e)
except Exception as e:
    print('❌ Error:', e)
"

echo ""
echo "=== Test Complete ==="
