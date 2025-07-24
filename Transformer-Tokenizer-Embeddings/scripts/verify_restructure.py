#!/usr/bin/env python3
"""
Quick verification script to test that all imports work correctly
after the project restructuring.
"""

import sys
import os

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports after restructuring...")
    
    try:
        # Test BERT QA import
        print("✓ Testing bert_qa import...")
        import bert_qa
        print("  - QuantizedBertQA class found")
        
        # Test fine-tune import
        print("✓ Testing fine_tune import...")
        import fine_tune
        print("  - BertQAFineTuner class found")
        
        # Test app import
        print("✓ Testing app import...")
        import app
        print("  - Flask app found")
        
        print("\n🎉 All imports successful! Project restructuring worked correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_file_structure():
    """Check that all expected files exist."""
    print("\nChecking file structure...")
    
    expected_files = [
        "app.py",
        "bert_qa.py", 
        "fine_tune.py",
        "requirements.txt",
        "README.md",
        "INSTALLATION.md",
        ".gitignore"
    ]
    
    expected_dirs = [
        "scripts",
        "data",
        "docker"
    ]
    
    # Check files
    for file in expected_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ Missing: {file}")
    
    # Check directories
    for dir in expected_dirs:
        if os.path.isdir(dir):
            print(f"✓ {dir}/")
        else:
            print(f"❌ Missing directory: {dir}/")
    
    # Check that old src/ structure is gone
    if not os.path.exists("src"):
        print("✓ Old src/ directory properly removed")
    else:
        print("⚠️  Old src/ directory still exists")

def main():
    print("=" * 50)
    print("Project Restructuring Verification")
    print("=" * 50)
    
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    check_file_structure()
    
    print("\n" + "=" * 50)
    imports_ok = test_imports()
    print("=" * 50)
    
    if imports_ok:
        print("\n✅ Project restructuring verification PASSED!")
        print("\nYou can now run:")
        print("  python app.py                    # Start the API")
        print("  python scripts/train_model.py    # Run training")
        print("  python scripts/test_api.py       # Test the API")
    else:
        print("\n❌ Project restructuring verification FAILED!")
        print("Check the import errors above and fix them.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
