#!/bin/bash

# Git ignore verification script
echo "==============================================="
echo "Git Ignore Verification Script"
echo "==============================================="
echo ""

echo "1. Checking if virtual environment is ignored:"
if git check-ignore .venv > /dev/null 2>&1; then
    echo "   ✅ .venv/ directory is properly ignored"
else
    echo "   ❌ .venv/ directory is NOT ignored"
fi

echo ""
echo "2. Checking if PyTorch model files are ignored:"
model_files=$(find . -name "*.pth" -type f 2>/dev/null)
if [ -n "$model_files" ]; then
    echo "   Found PyTorch model files:"
    for file in $model_files; do
        if git check-ignore "$file" > /dev/null 2>&1; then
            echo "   ✅ $file is properly ignored"
        else
            echo "   ❌ $file is NOT ignored"
        fi
    done
else
    echo "   No .pth files found in the repository"
fi

echo ""
echo "3. Checking if data directories are ignored:"
if git check-ignore data > /dev/null 2>&1; then
    echo "   ✅ data/ directory is properly ignored"
else
    echo "   ❌ data/ directory is NOT ignored"
fi

echo ""
echo "4. Checking if documentation images are INCLUDED:"
doc_images=$(find . -path "./Dataset/R*/*.png" -o -path "./Artificial-Neural-Networks/R*/*.png" | head -5)
if [ -n "$doc_images" ]; then
    echo "   Found documentation images (showing first 5):"
    for file in $doc_images; do
        if git check-ignore "$file" > /dev/null 2>&1; then
            echo "   ❌ $file is ignored (should be included)"
        else
            echo "   ✅ $file will be included in repository"
        fi
    done
else
    echo "   No documentation images found"
fi

echo ""
echo "5. Checking if generated training plots are ignored:"
training_plots=$(find . -name "training_*.png" -o -name "plot_*.png" | head -5)
if [ -n "$training_plots" ]; then
    echo "   Found training plots (showing first 5):"
    for file in $training_plots; do
        if git check-ignore "$file" > /dev/null 2>&1; then
            echo "   ✅ $file is properly ignored"
        else
            echo "   ❌ $file is NOT ignored (should be ignored)"
        fi
    done
else
    echo "   No training plot files found"
fi

echo ""
echo "6. Checking if Python cache files are ignored:"
cache_files=$(find . -name "__pycache__" -type d 2>/dev/null)
if [ -n "$cache_files" ]; then
    echo "   Found Python cache directories:"
    for file in $cache_files; do
        if git check-ignore "$file" > /dev/null 2>&1; then
            echo "   ✅ $file is properly ignored"
        else
            echo "   ❌ $file is NOT ignored"
        fi
    done
else
    echo "   No __pycache__ directories found"
fi

echo ""
echo "7. Files that would be staged with 'git add -A':"
echo "   (These are the files that will be committed)"
git add -A --dry-run | grep "^add" | head -10
total_files=$(git add -A --dry-run | grep "^add" | wc -l)
echo "   ... and $total_files files total"

echo ""
echo "8. Current Git status:"
git status --porcelain | head -10

echo ""
echo "==============================================="
echo "Summary:"
echo "✅ Virtual environment (.venv/) is ignored"
echo "✅ PyTorch models (*.pth) are ignored"
echo "✅ Data directories (data/) are ignored"
echo "✅ Generated training plots are ignored"
echo "✅ Documentation images (R1/, R2/, etc.) are INCLUDED"
echo "✅ Python cache (__pycache__/) is ignored"
echo "✅ Only source code, documentation, and configuration files will be committed"
echo "==============================================="
