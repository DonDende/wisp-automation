#!/usr/bin/env python3
"""
Test detection on the specific image we know has X C C
"""

import cv2
import numpy as np
from gpu_optimized_detector import GPUOptimizedWispDetector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_image():
    """Test on the specific image we saw in browser"""
    image_path = "/workspace/wisp_automation/detection_1753658935.png"
    
    print(f"🔍 Testing the specific image: {image_path}")
    print("Expected: X (top center), C (bottom left), C (bottom right)")
    
    # Load the detection image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📏 Image shape: {image.shape}")
    
    # Initialize detector
    detector = GPUOptimizedWispDetector()
    
    # Test template matching with debug info
    print("\n🔍 Testing template matching with debug:")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Test each letter individually
    templates = {}
    for letter in ['X', 'Z', 'V', 'C']:
        template = np.zeros((40, 30), dtype=np.uint8)
        cv2.putText(template, letter, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        templates[letter] = template
    
    detected_letters = []
    
    for letter, template in templates.items():
        # Multi-scale template matching
        best_score = 0
        best_pos = None
        
        for scale in [0.8, 1.0, 1.2]:
            h, w = template.shape
            scaled_template = cv2.resize(template, (int(w * scale), int(h * scale)))
            
            if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                continue
            
            result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_pos = max_loc
        
        print(f"Letter {letter}: score={best_score:.3f}, pos={best_pos}")
        
        if best_score > 0.5:  # Use higher threshold
            detected_letters.append((best_pos[0], best_pos[1], letter, best_score))
    
    # Sort by absolute left to right (x-coordinate only)
    detected_letters.sort(key=lambda x: x[0])
    print(f"\nDetected letters left to right: {[(letter, x, y) for x, y, letter, _ in detected_letters]}")
    print(f"Letter sequence: {[letter for x, y, letter, _ in detected_letters]}")
    
    # Test finding ALL instances of C
    print(f"\n🔍 Looking for ALL instances of C:")
    c_template = np.zeros((40, 30), dtype=np.uint8)
    cv2.putText(c_template, 'C', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    
    result = cv2.matchTemplate(gray, c_template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= 0.5)
    
    c_positions = []
    for pt in zip(*locations[::-1]):
        x, y = pt
        score = result[y, x]
        c_positions.append((x, y, score))
    
    # Remove duplicates that are too close
    filtered_c = []
    for x, y, score in c_positions:
        too_close = False
        for fx, fy, _ in filtered_c:
            if abs(x - fx) < 20 and abs(y - fy) < 20:
                too_close = True
                break
        if not too_close:
            filtered_c.append((x, y, score))
    
    filtered_c.sort(key=lambda x: x[0])  # Sort by x-coordinate
    print(f"All C instances: {[(x, y, f'{score:.3f}') for x, y, score in filtered_c]}")
    
    # Test improved template matching directly
    print("\n📋 Testing improved template matching method:")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    improved_result = detector._detect_letters_by_improved_templates(gray)
    print(f"Improved template result: {improved_result}")
    
    # Test full pipeline
    print("\n🚀 Testing full detection pipeline:")
    result = detector.detect_wisp_box_optimized(image)
    print(f"Full pipeline result: {result}")

if __name__ == "__main__":
    test_specific_image()