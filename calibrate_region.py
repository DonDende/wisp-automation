#!/usr/bin/env python3
"""
Region Calibration Tool
Helps determine the exact coordinates for the translucent dialog box
"""

import cv2
import numpy as np
import pyautogui
import json
import time
from PIL import Image, ImageDraw, ImageFont

def capture_and_show_regions():
    """Capture screen and show different detection regions"""
    print("📸 Capturing screen in 3 seconds...")
    print("Make sure the wisp dialog box is visible!")
    time.sleep(3)
    
    # Capture full screen
    screenshot = pyautogui.screenshot()
    img_array = np.array(screenshot)
    
    # Convert to OpenCV format
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Define potential regions based on user feedback and analysis
    regions = {
        "user_provided": [747, 180, 417, 145],  # User-provided correct coordinates
        "center_wide": [700, 150, 500, 200],    # Wider area around correct region
        "center_narrow": [770, 200, 370, 100], # Narrower focus within correct region
        "upper_third": [600, 100, 640, 250],   # Upper third of screen
    }
    
    # Create visualization
    img_vis = img_array.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, (name, (x, y, w, h)) in enumerate(regions.items()):
        color = colors[i % len(colors)]
        # Draw rectangle
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), color, 2)
        # Add label
        cv2.putText(img_vis, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Save visualization
    cv2.imwrite('region_calibration.png', cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    print("✅ Saved region_calibration.png")
    
    # Test each region
    for name, (x, y, w, h) in regions.items():
        region_img = img_array[y:y+h, x:x+w]
        region_filename = f"region_{name}.png"
        cv2.imwrite(region_filename, cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved {region_filename} - Region: {x}, {y}, {w}, {h}")
    
    return regions

def interactive_calibration():
    """Interactive calibration tool"""
    print("🎯 Interactive Region Calibration")
    print("="*40)
    
    regions = capture_and_show_regions()
    
    print("\n📋 Available regions:")
    for i, (name, coords) in enumerate(regions.items(), 1):
        print(f"{i}. {name}: {coords}")
    
    print("\nOptions:")
    print("1-4: Select a predefined region")
    print("c: Custom coordinates")
    print("t: Test current config")
    print("q: Quit")
    
    while True:
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 't':
            test_current_config()
        elif choice == 'c':
            custom_coordinates()
        elif choice.isdigit() and 1 <= int(choice) <= 4:
            region_names = list(regions.keys())
            selected_region = region_names[int(choice) - 1]
            coords = regions[selected_region]
            update_config(coords)
            print(f"✅ Updated config with {selected_region}: {coords}")
        else:
            print("Invalid choice. Please try again.")

def custom_coordinates():
    """Get custom coordinates from user"""
    print("\n📐 Custom Coordinates")
    print("Enter coordinates for the dialog box region:")
    
    try:
        x = int(input("X (left edge): "))
        y = int(input("Y (top edge): "))
        w = int(input("Width: "))
        h = int(input("Height: "))
        
        coords = [x, y, w, h]
        update_config(coords)
        print(f"✅ Updated config with custom coordinates: {coords}")
        
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")

def update_config(coords):
    """Update the configuration file"""
    try:
        with open('final_wisp_config.json', 'r') as f:
            config = json.load(f)
        
        config['detection_region'] = coords
        
        with open('final_wisp_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration updated: detection_region = {coords}")
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")

def test_current_config():
    """Test the current configuration"""
    try:
        with open('final_wisp_config.json', 'r') as f:
            config = json.load(f)
        
        region = config.get('detection_region')
        if not region:
            print("❌ No detection region configured")
            return
        
        print(f"🧪 Testing region: {region}")
        
        # Capture the region
        x, y, w, h = region
        screenshot = pyautogui.screenshot(region=(x, y, w, h))
        
        # Save test image
        screenshot.save('test_region_current.png')
        print("✅ Saved test_region_current.png")
        print("Check this image to see if it captures the dialog box correctly")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Main calibration function"""
    print("🎯 Wisp Dialog Box Region Calibration Tool")
    print("="*50)
    print("This tool helps you find the exact coordinates for the translucent dialog box")
    print("\nInstructions:")
    print("1. Make sure your game is running and the wisp dialog is visible")
    print("2. Run this tool to capture and analyze different regions")
    print("3. Select the region that best captures the dialog box")
    print("4. The configuration will be automatically updated")
    
    input("\nPress Enter when ready...")
    
    interactive_calibration()
    
    print("\n✅ Calibration complete!")
    print("You can now run the main automation script.")

if __name__ == "__main__":
    main()