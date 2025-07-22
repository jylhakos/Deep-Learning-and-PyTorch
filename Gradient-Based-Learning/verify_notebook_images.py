#!/usr/bin/env python3
"""
Image Link Verification Script
Verifies that all image paths in the notebook are correct and images can be loaded.
"""

import matplotlib.pyplot as plt
import os
import json

def verify_notebook_images(notebook_path):
    """Verify all image paths in the notebook can be loaded."""
    
    print(f"Verifying images in notebook: {notebook_path}")
    print("=" * 60)
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Track all image paths found
    image_paths = []
    
    # Extract image paths from cells
    for cell in notebook['cells']:
        if 'source' in cell:
            source_lines = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
            for line in source_lines:
                # Look for plt.imread calls
                if 'plt.imread(' in line and '../R2/' in line:
                    # Extract the path between quotes
                    start = line.find('"../R2/') + 1
                    end = line.find('"', start)
                    if start > 0 and end > start:
                        path = line[start:end]
                        image_paths.append(path)
    
    # Remove duplicates
    image_paths = list(set(image_paths))
    image_paths.sort()
    
    print(f"Found {len(image_paths)} unique image paths:")
    print()
    
    # Test each image path
    success_count = 0
    notebook_dir = os.path.dirname(notebook_path)
    
    for path in image_paths:
        try:
            # Change to notebook directory to test relative path
            full_path = os.path.join(notebook_dir, path)
            
            if os.path.exists(full_path):
                # Try to load the image
                img = plt.imread(full_path)
                print(f"✓ {path:<35} -> shape: {img.shape}")
                success_count += 1
            else:
                print(f"✗ {path:<35} -> File not found")
                
        except Exception as e:
            print(f"✗ {path:<35} -> Error: {e}")
    
    print()
    print("=" * 60)
    print(f"Results: {success_count}/{len(image_paths)} images verified successfully")
    
    if success_count == len(image_paths):
        print("🎉 All image paths are working correctly!")
        return True
    else:
        print("⚠️  Some image paths need attention.")
        return False

if __name__ == "__main__":
    notebook_path = "/home/laptop/EXERCISES/DEEP-LEARNING/PYTORCH/Deep-Learning-and-PyTorch/Gradient-Based-Learning/Round2/Round2_SGD.ipynb"
    
    if os.path.exists(notebook_path):
        success = verify_notebook_images(notebook_path)
        exit(0 if success else 1)
    else:
        print(f"Notebook not found: {notebook_path}")
        exit(1)
