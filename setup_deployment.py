#!/usr/bin/env python3
"""
Melanoma Detection App - Deployment Setup Script
This script helps is for the project structure to verify everything is ready.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_project_structure():
    """Create the required project structure"""
    print("Creating project structure...")
    
    directories = [
        "models",
        ".streamlit"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def check_model_files():
    """Check if all required model files are present"""
    print("\nChecking model files...")
    
    required_files = [      # changed all of these to keras files so they are compatible with tensorflow
        "models/cnn_model.keras",
        "models/abcd_model.keras", 
        "models/combined_model.keras",
        "models/abcd_scaler.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✓ Found: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"✗ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} model files are missing.")
        print("You need to export your trained models before running the app.")
        print("See the model_export.py script for guidance.")
        return False
    
    print("✓ All model files are present!")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        "streamlit",
        "tensorflow", 
        "cv2",  # opencv-python-headless imports as cv2
        "PIL",  # Pillow imports as PIL
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn"  # scikit-learn imports as sklearn
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special import names
            import_name = package
            if package == "cv2":
                import_name = "cv2"
            elif package == "PIL":
                import_name = "PIL"
            elif package == "sklearn":
                import_name = "sklearn"
            
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Warning: {len(missing_packages)} packages are missing.")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✓ All required packages are installed!")
    return True

def test_streamlit_installation():
    """Test if Streamlit is properly installed"""
    print("\nTesting Streamlit installation...")
    
    try:
        result = subprocess.run(
            ["streamlit", "--version"], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Streamlit is installed: {version}")
            return True
        else:
            print("✗ Streamlit test failed")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ Streamlit is not properly installed or not in PATH")
        return False

def provide_next_steps(models_ready, deps_ready):
    """Provide next steps based on the setup status"""
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    if models_ready and deps_ready:
        print("🎉 Everything is ready!")
        print("\nTo start the application:")
        print("   streamlit run app.py")
        print("\nThen open your browser to: http://localhost:8501")
        
    else:
        print("⚠️  Setup is incomplete. Please address the following:")
        
        if not deps_ready:
            print("\n1. Install dependencies:")
            print("   pip install -r requirements.txt")
        
        if not models_ready:
            print("\n2. Add your trained models to the models/ directory:")
            print("   - cnn_model.h5")
            print("   - abcd_model.h5") 
            print("   - combined_model.h5")
            print("   - abcd_scaler.pkl")
            print("\n   Use the model_export.py script to export from your training environment.")
        
        print("\nAfter fixing these issues, run this script again to verify.")

def main():
    """Main setup function"""
    print("Melanoma Detection App - Deployment Setup")
    print("="*50)
    
    # Create project structure
    create_project_structure()
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("\n⚠️  Warning: app.py not found in current directory.")
        print("Make sure you're in the correct project directory.")
        return
    
    # Check dependencies
    deps_ready = check_dependencies()
    
    # Test Streamlit
    if deps_ready:
        test_streamlit_installation()
    
    # Check model files
    models_ready = check_model_files()
    
    # Provide next steps
    provide_next_steps(models_ready, deps_ready)

if __name__ == "__main__":
    main()