#!/usr/bin/env python3
"""
Debug Detection Tool
Shows exactly what the AI detector is analyzing
"""

import cv2
import numpy as np
import pyautogui
import json
import time
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

def debug_detection_visual():
    """Create visual debug output showing detection regions"""
    print("🔍 Debug Detection - Visual Analysis")
    print("="*40)
    
    # Load configuration
    try:
        with open('final_wisp_config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    print("📸 Capturing screen in 3 seconds...")
    print("Make sure the wisp dialog box is visible!")
    time.sleep(3)
    
    # Capture full screen
    full_screenshot = pyautogui.screenshot()
    full_array = np.array(full_screenshot)
    
    # Get detection region
    detection_region = config.get('detection_region')
    
    if detection_region:
        x, y, w, h = detection_region
        print(f"🎯 Using configured region: {detection_region}")
        
        # Capture region
        region_screenshot = pyautogui.screenshot(region=(x, y, w, h))
        region_array = np.array(region_screenshot)
        
        # Create visualization
        vis_img = full_array.copy()
        
        # Draw detection region rectangle
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(vis_img, f"Detection Region: {x},{y},{w},{h}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save images
        cv2.imwrite('debug_full_screen.png', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite('debug_detection_region.png', cv2.cvtColor(region_array, cv2.COLOR_RGB2BGR))
        
        print("✅ Saved debug_full_screen.png (shows detection region)")
        print("✅ Saved debug_detection_region.png (shows what AI analyzes)")
        
        # Analyze the region
        analyze_region(region_array, config)
        
    else:
        print("❌ No detection region configured. Using full screen.")
        cv2.imwrite('debug_full_screen.png', cv2.cvtColor(full_array, cv2.COLOR_RGB2BGR))
        analyze_region(full_array, config)

def analyze_region(image_array, config):
    """Analyze the detection region for letters"""
    print("\n🔍 Analyzing detection region...")
    
    # Convert to different formats for analysis
    img_pil = Image.fromarray(image_array)
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Save grayscale version
    cv2.imwrite('debug_grayscale.png', img_gray)
    print("✅ Saved debug_grayscale.png")
    
    # Apply thresholding to highlight text
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite('debug_threshold.png', thresh)
    print("✅ Saved debug_threshold.png")
    
    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)
    cv2.imwrite('debug_edges.png', edges)
    print("✅ Saved debug_edges.png")
    
    # Try to detect text regions
    detect_text_regions(image_array)
    
    # Test AI detection if available
    test_ai_detection(img_pil, config)

def detect_text_regions(image_array):
    """Detect potential text regions"""
    print("\n📝 Detecting text regions...")
    
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Find contours
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on original image
    contour_img = image_array.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
    
    # Filter contours by size (potential letters)
    letter_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter by size - letters should be reasonably sized
        if 100 < area < 5000 and 10 < w < 100 and 10 < h < 100:
            letter_contours.append(contour)
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(contour_img, f"{w}x{h}", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite('debug_text_regions.png', cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))
    print(f"✅ Found {len(letter_contours)} potential letter regions")
    print("✅ Saved debug_text_regions.png")

def test_ai_detection(img_pil, config):
    """Test AI detection if available"""
    print("\n🤖 Testing AI detection...")
    
    try:
        from gpu_optimized_detector import GPUOptimizedWispDetector
        
        detector = GPUOptimizedWispDetector()
        print(f"✅ AI detector loaded on device: {detector.device}")
        
        # Test detection
        result = detector.detect_letter(img_pil)
        print(f"🎯 AI Detection Result: {result}")
        
        # Test with different confidence thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        print("\n📊 Testing different confidence thresholds:")
        
        for threshold in thresholds:
            detector.confidence_threshold = threshold
            result = detector.detect_letter(img_pil)
            print(f"   Threshold {threshold}: {result}")
        
    except Exception as e:
        print(f"❌ AI detection test failed: {e}")
        print("Make sure gpu_optimized_detector.py is available")

def main():
    """Main debug function"""
    print("🔍 Wisp Detection Debug Tool")
    print("="*40)
    print("This tool helps debug detection issues by:")
    print("1. Showing exactly what region is being analyzed")
    print("2. Creating visual representations of the detection process")
    print("3. Testing AI detection with different settings")
    print("4. Identifying potential text regions")
    
    input("\nPress Enter when ready...")
    
    debug_detection_visual()
    
    print("\n📋 Debug files created:")
    print("- debug_full_screen.png: Full screen with detection region marked")
    print("- debug_detection_region.png: Just the detection region")
    print("- debug_grayscale.png: Grayscale version for text analysis")
    print("- debug_threshold.png: Thresholded image highlighting text")
    print("- debug_edges.png: Edge detection")
    print("- debug_text_regions.png: Detected text regions")
    
    print("\n💡 Tips:")
    print("- Check if the detection region captures the dialog box correctly")
    print("- Look at the grayscale image to see if letters are visible")
    print("- Use the threshold image to see if text stands out")
    print("- If text regions are detected, the letters should be visible")

if __name__ == "__main__":
    main()