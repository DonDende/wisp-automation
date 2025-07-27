#!/usr/bin/env python3
"""
Test the improved letter detection on actual detection images
"""

import cv2
import numpy as np
from gpu_optimized_detector import GPUOptimizedWispDetector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_detection_on_image(image_path: str):
    """Test detection on a specific image"""
    print(f"\n🔍 Testing detection on: {image_path}")
    
    # Load the detection image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📏 Image shape: {image.shape}")
    
    # Initialize detector
    detector = GPUOptimizedWispDetector()
    
    # Test the new OCR detection method directly
    print("\n🔤 Testing OCR detection method:")
    letters_ocr = detector._detect_letters_with_ocr(image)
    print(f"OCR detected letters: {letters_ocr}")
    
    # Test improved template matching
    print("\n📋 Testing improved template matching:")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    letters_template = detector._detect_letters_by_improved_templates(gray)
    print(f"Template detected letters: {letters_template}")
    
    # Test the full detection pipeline
    print("\n🚀 Testing full detection pipeline:")
    result = detector.detect_wisp_box_optimized(image)
    print(f"Full pipeline result: {result}")
    
    return result

def main():
    """Test on available detection images"""
    import glob
    
    # Find all detection images
    detection_images = glob.glob("/workspace/wisp_automation/detection_*.png")
    
    if not detection_images:
        print("❌ No detection images found!")
        return
    
    print(f"📸 Found {len(detection_images)} detection images")
    
    # Test on the first few images
    for i, image_path in enumerate(detection_images[:3]):
        result = test_detection_on_image(image_path)
        
        if result and result['detected']:
            print(f"✅ Successfully detected: {result['letters']}")
        else:
            print(f"❌ Detection failed")
        
        print("-" * 50)

if __name__ == "__main__":
    main()