#!/usr/bin/env python3
"""
Wisp Automation Launcher
Handles dependencies and provides user-friendly error messages
"""

import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('cv2', 'opencv-python'),
        ('PIL', 'pillow'),
        ('numpy', 'numpy'),
        ('pyautogui', 'pyautogui'),
    ]
    
    missing_packages = []
    
    for module_name, package_name in required_packages:
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please install manually.")
            return False
    return True

def main():
    """Main launcher function"""
    print("🤖 AI-Powered Wisp Automation Launcher")
    print("="*50)
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        response = input("Install missing dependencies? (y/n): ")
        
        if response.lower() == 'y':
            if not install_dependencies(missing):
                print("Failed to install dependencies. Please install manually:")
                print(f"pip install {' '.join(missing)}")
                return
        else:
            print("Cannot run without required dependencies.")
            return
    
    # Import and run the main automation
    try:
        print("Dependencies OK. Starting automation...")
        print("Loading AI models and initializing system...")
        
        # Import the automation module
        import final_wisp_automation
        
        # Run the main function with original sys.argv
        final_wisp_automation.main()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all files are in the correct location.")
        print("Make sure final_wisp_automation.py is in the same directory.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the error message above.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()