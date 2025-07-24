#!/usr/bin/env python3
"""
Simple verification script to check project structure without heavy imports.
"""

import os
import ast

def check_syntax(filename):
    """Check if a Python file has valid syntax."""
    try:
        with open(filename, 'r') as file:
            source = file.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 50)
    print("Simple Project Structure Verification")
    print("=" * 50)
    
    # Check Python files
    python_files = ["app.py", "bert_qa.py", "fine_tune.py"]
    
    print("Checking Python file syntax...")
    all_good = True
    
    for file in python_files:
        if os.path.exists(file):
            valid, error = check_syntax(file)
            if valid:
                print(f"✓ {file} - syntax OK")
            else:
                print(f"❌ {file} - syntax error: {error}")
                all_good = False
        else:
            print(f"❌ {file} - missing")
            all_good = False
    
    # Check other important files
    other_files = [
        "requirements.txt",
        "README.md", 
        "INSTALLATION.md",
        ".gitignore"
    ]
    
    print("\nChecking other files...")
    for file in other_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"❌ {file} - missing")
            all_good = False
    
    # Check directories
    dirs = ["scripts", "data", "docker"]
    print("\nChecking directories...")
    for dir in dirs:
        if os.path.isdir(dir):
            print(f"✓ {dir}/")
        else:
            print(f"❌ {dir}/ - missing")
    
    # Check that old structure is gone
    if not os.path.exists("src"):
        print("✓ Old src/ structure removed")
    else:
        print("⚠️  Old src/ structure still exists")
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ Project restructuring verification PASSED!")
        print("\nProject structure is clean and ready to use!")
        print("\nNext steps:")
        print("1. Run: ./scripts/setup_environment.sh")
        print("2. Activate: source bert_qa_env/bin/activate") 
        print("3. Start API: python app.py")
    else:
        print("❌ Some issues found. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
