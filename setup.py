#!/usr/bin/env python3
"""
Setup script for Virtual Camera project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed all requirements!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def check_images():
    """Check if required image files exist"""
    required_images = ["dog.png", "trump.jpg", "musk.jpg", "sea.jpg"]
    missing_images = []
    
    for img in required_images:
        if not os.path.exists(img):
            missing_images.append(img)
    
    if missing_images:
        print(f"Missing image files: {', '.join(missing_images)}")
        print("Please add these image files to the project directory before running the application.")
        return False
    else:
        print("All required image files found!")
        return True

def main():
    print("Virtual Camera Setup")
    print("=" * 30)
    
    print("\n1. Installing requirements...")
    if not install_requirements():
        return
    
    print("\n2. Checking for required image files...")
    check_images()
    
    print("\nSetup complete!")
    print("\nTo run the application:")
    print("  python run.py")
    print("\nTo run motion detection:")
    print("  python g.py")
    print("\nPress 'q' to quit any running application.")

if __name__ == "__main__":
    main()
