#!/usr/bin/env python3
"""
Quick Fix for Wisp Detection
Addresses the issue of detecting letters on sides of screen
"""

import json
import pyautogui
import cv2
import numpy as np
from PIL import Image

def apply_quick_fix():
    """Apply quick fix for detection region"""
    print("🔧 Applying Quick Fix for Wisp Detection")
    print("="*45)
    
    # Based on user-provided coordinates:
    # - Top left corner: x=747, y=180
    # - Bottom right corner: x=1164, y=325
    # - Width: 1164 - 747 = 417
    # - Height: 325 - 180 = 145
    
    optimal_region = [747, 180, 417, 145]
    
    print(f"🎯 Setting optimal detection region: {optimal_region}")
    print("   This focuses on the center-upper area where the dialog appears")
    
    # Update configuration
    try:
        with open('final_wisp_config.json', 'r') as f:
            config = json.load(f)
        
        config['detection_region'] = optimal_region
        config['confidence_threshold'] = 0.3  # Lower threshold for better detection
        config['debug_mode'] = True  # Enable debug mode
        config['save_detection_images'] = True  # Save images for analysis
        
        with open('final_wisp_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration updated successfully")
        print(f"   Detection region: {optimal_region}")
        print(f"   Confidence threshold: 0.3 (lowered for better detection)")
        print(f"   Debug mode: enabled")
        
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return False
    
    # Test the new region
    print("\n🧪 Testing new detection region...")
    test_new_region(optimal_region)
    
    return True

def test_new_region(region):
    """Test the new detection region"""
    try:
        print("📸 Capturing test screenshot in 3 seconds...")
        print("Make sure the wisp dialog is visible!")
        
        import time
        time.sleep(3)
        
        x, y, w, h = region
        
        # Capture the specific region
        region_screenshot = pyautogui.screenshot(region=(x, y, w, h))
        region_array = np.array(region_screenshot)
        
        # Save the test image
        cv2.imwrite('quick_fix_test_region.png', cv2.cvtColor(region_array, cv2.COLOR_RGB2BGR))
        print("✅ Saved quick_fix_test_region.png")
        
        # Also capture full screen with region marked
        full_screenshot = pyautogui.screenshot()
        full_array = np.array(full_screenshot)
        
        # Draw the detection region
        cv2.rectangle(full_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(full_array, f"Detection Region: {x},{y},{w},{h}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite('quick_fix_full_screen.png', cv2.cvtColor(full_array, cv2.COLOR_RGB2BGR))
        print("✅ Saved quick_fix_full_screen.png")
        
        # Analyze the region
        analyze_region_content(region_array)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def analyze_region_content(image_array):
    """Analyze what's in the detection region"""
    print("\n🔍 Analyzing region content...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Check if there's text-like content
    # Look for high contrast areas (white text on dark background)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    white_ratio = white_pixels / total_pixels
    
    print(f"📊 Region analysis:")
    print(f"   White pixel ratio: {white_ratio:.3f}")
    print(f"   Total pixels: {total_pixels}")
    print(f"   White pixels: {white_pixels}")
    
    if white_ratio > 0.05:  # More than 5% white pixels
        print("✅ Good: Region contains white text-like content")
    else:
        print("⚠️  Warning: Region may not contain visible text")
    
    # Check for letter-like shapes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_like_contours = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Letters should be reasonably sized
        if 50 < area < 2000 and 5 < w < 50 and 10 < h < 50:
            letter_like_contours += 1
    
    print(f"   Letter-like shapes: {letter_like_contours}")
    
    if letter_like_contours >= 1:
        print("✅ Good: Found letter-like shapes in region")
    else:
        print("⚠️  Warning: No clear letter shapes detected")

def main():
    """Main quick fix function"""
    print("🔧 Wisp Detection Quick Fix Tool")
    print("="*40)
    print("This tool addresses the issue of detecting letters on the sides of the screen")
    print("by setting a precise detection region focused on the dialog box area.")
    print()
    print("Based on your screenshot analysis, the optimal region is:")
    print("  X: 580 (left edge of dialog)")
    print("  Y: 150 (top edge of dialog)")  
    print("  Width: 290 (dialog width)")
    print("  Height: 100 (dialog height)")
    print()
    
    response = input("Apply this quick fix? (y/n): ").strip().lower()
    
    if response == 'y':
        if apply_quick_fix():
            print("\n✅ Quick fix applied successfully!")
            print("\n📋 Next steps:")
            print("1. Run the main automation script")
            print("2. Check the saved test images to verify the region is correct")
            print("3. If needed, use the calibration tool to fine-tune")
            print("\nFiles created:")
            print("- quick_fix_test_region.png (shows what AI will analyze)")
            print("- quick_fix_full_screen.png (shows detection region on full screen)")
        else:
            print("❌ Quick fix failed. Please check the error messages above.")
    else:
        print("Quick fix cancelled.")

if __name__ == "__main__":
    main()