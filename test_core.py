#!/usr/bin/env python3
"""
Core functionality test - tests AI detection without GUI dependencies
"""

import os
import sys
import json
import numpy as np
from PIL import Image

def test_ai_imports():
    """Test if AI dependencies can be imported"""
    try:
        import torch
        import transformers
        import cv2
        print("✅ AI dependencies imported successfully")
        
        # Test device detection
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            print("✅ Apple Silicon MPS available")
        else:
            device = "cpu"
            print("✅ Using CPU (no GPU detected)")
        
        return True, device
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False, None

def test_ai_detector():
    """Test the AI detector without GUI"""
    try:
        # Import the GPU optimized detector
        sys.path.append('.')
        from gpu_optimized_detector import GPUOptimizedDetector
        
        print("✅ AI detector imported successfully")
        
        # Initialize detector
        detector = GPUOptimizedDetector()
        print(f"✅ Detector initialized on device: {detector.device}")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_image)
        
        # Test detection
        result = detector.detect_letter(test_pil)
        print(f"✅ Detection test completed: {result}")
        
        return True
    except Exception as e:
        print(f"❌ AI detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        with open('final_wisp_config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        print(f"   Keystroke delay: {config.get('keystroke_delay_min', 0.08)}-{config.get('keystroke_delay_max', 0.10)}s")
        print(f"   Confidence threshold: {config.get('confidence_threshold', 0.4)}")
        print(f"   Valid letters: {config.get('valid_letters', ['X', 'Z', 'V'])}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run core functionality tests"""
    print("🧪 AI-Powered Wisp Automation - Core Tests")
    print("="*50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: AI imports
    print("\n1. Testing AI Dependencies...")
    if test_ai_imports()[0]:
        tests_passed += 1
    
    # Test 2: AI detector
    print("\n2. Testing AI Detector...")
    if test_ai_detector():
        tests_passed += 1
    
    # Test 3: Configuration
    print("\n3. Testing Configuration...")
    if test_configuration():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All core tests passed! System is ready for deployment.")
        print("\nTo run the full automation:")
        print("1. Ensure you have a display/GUI environment")
        print("2. Run: python final_wisp_automation.py")
        print("3. Or use the batch file: run_automation.bat")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()