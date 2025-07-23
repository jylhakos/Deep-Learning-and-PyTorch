#!/usr/bin/env python3
"""
Test script to verify the embeddings path checking in the original TensorFlow/Keras script
"""

import numpy as np
from pathlib import Path
import pickle

# Extract just the path checking logic from the original script
def test_embeddings_path():
    """Test the embeddings path checking logic"""
    
    # Try different possible paths for the embeddings file
    possible_paths = [
        Path().cwd() / '..' / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
        Path().cwd() / '..' / '..' / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
        Path().cwd() / '..' / '..' / '..' / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
        Path().cwd() / '..' / '..' / '..' / 'coursedata' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
        Path().cwd() / 'Dataset' / 'R5' / '20newsgroups_subset_vocabulary_embeddings.p',
        Path().cwd() / '20newsgroups_subset_vocabulary_embeddings.p'
    ]

    embeddings = None
    embeddings_path = None

    print("Testing embeddings file path checking...")
    print("=" * 60)
    print("Searching for embeddings file in possible locations:")
    
    for i, path in enumerate(possible_paths, 1):
        print(f"{i}. {path}")
        if path.exists():
            print(f"   ✓ Found!")
            embeddings_path = path
            break
        else:
            print(f"   ✗ Not found")

    if embeddings_path and embeddings_path.exists():
        try:
            with open(embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
                vocabulary = list(embeddings.keys())
            print(f'\n✓ Successfully loaded embeddings for {len(vocabulary)} words')
            return True, f"Embeddings loaded successfully from {embeddings_path}"
        except Exception as e:
            print(f"\n❌ Error loading embeddings file: {e}")
            return False, f"Error loading embeddings: {e}"
    else:
        print("\n⚠️  No embeddings file found in any location.")
        print("This is expected if you don't have the original dataset.")
        print("The scripts will use random embeddings as fallback.")
        return False, "No embeddings file found (this is normal)"

if __name__ == "__main__":
    success, message = test_embeddings_path()
    print("\n" + "=" * 60)
    print("RESULT:")
    print(f"Status: {'SUCCESS' if success else 'NO EMBEDDINGS FOUND'}")
    print(f"Message: {message}")
    print("\nNote: Both scripts (TensorFlow/Keras and PyTorch) will work")
    print("without embeddings, but with potentially lower accuracy.")
