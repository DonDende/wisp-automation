#!/usr/bin/env python3
"""
Debug script to help troubleshoot dialog detection issues
"""

import cv2
import numpy as np
import json
from pathlib import Path
import logging
import os

# Disable pyautogui for headless mode
try:
    import pyautogui
    HEADLESS_MODE = False
except ImportError:
    HEADLESS_MODE = True
    pyautogui = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def capture_and_analyze_dialog_region():
    """Capture the dialog region and analyze it"""
    
    # Load comprehensive data
    data_path = Path(__file__).parent / "wisp_comprehensive_data.json"
    if data_path.exists():
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        dialog_config = data.get('automation_config', {}).get('dialog_detection', {})
        dialog_patterns = dialog_config.get('dialog_patterns', [])
        
        logger.info("📊 Dialog detection configuration:")
        for i, pattern in enumerate(dialog_patterns):
            logger.info(f"  Pattern {i+1}: {pattern}")
    else:
        logger.warning("No comprehensive data found")
        return
    
    # Capture current screen
    if HEADLESS_MODE or pyautogui is None:
        logger.error("❌ Cannot capture screen in headless mode")
        logger.info("💡 This script needs to run on a system with display access")
        return
    
    logger.info("📸 Capturing current screen...")
    try:
        screenshot = pyautogui.screenshot()
        screen_array = np.array(screenshot)
        screen_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"❌ Failed to capture screen: {e}")
        return
    
    # Define dialog regions to test
    regions_to_test = [
        {"name": "Original (top-left)", "coords": [0, 0, 400, 200]},
        {"name": "Updated (center-right)", "coords": [800, 200, 800, 600]},
        {"name": "Full screen center", "coords": [400, 150, 800, 500]},
        {"name": "Expected position area", "coords": [1132, 259, 200, 200]}  # Around [1232, 359] with tolerance
    ]
    
    for region in regions_to_test:
        logger.info(f"\n🔍 Testing region: {region['name']}")
        coords = region['coords']
        x, y, w, h = coords
        
        # Extract region
        roi = screen_bgr[y:y+h, x:x+w]
        
        if roi.size > 0:
            # Analyze the region
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray_roi, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Standard deviation (text/content indicator)
            std_dev = np.std(gray_roi)
            
            # Mean brightness
            mean_brightness = np.mean(gray_roi)
            
            logger.info(f"  📈 Edge density: {edge_density:.4f}")
            logger.info(f"  📈 Std deviation: {std_dev:.2f}")
            logger.info(f"  📈 Mean brightness: {mean_brightness:.2f}")
            
            # Save region for visual inspection
            output_path = f"debug_region_{region['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
            cv2.imwrite(output_path, roi)
            logger.info(f"  💾 Saved region to: {output_path}")
            
            # Check if this looks like a dialog
            dialog_score = 0
            if edge_density > 0.01:
                dialog_score += 1
                logger.info("  ✅ Has significant edges (dialog borders)")
            if std_dev > 15:
                dialog_score += 1
                logger.info("  ✅ Has content variation (text/elements)")
            if 50 < mean_brightness < 200:
                dialog_score += 1
                logger.info("  ✅ Has reasonable brightness (not too dark/bright)")
            
            logger.info(f"  🎯 Dialog likelihood score: {dialog_score}/3")
        else:
            logger.warning(f"  ❌ Failed to extract region {coords}")
    
    logger.info("\n🎯 Analysis complete! Check the saved PNG files to see what each region contains.")

if __name__ == "__main__":
    logger.info("🚀 Starting dialog detection debug analysis...")
    logger.info("📋 Make sure your game is visible and the wisp dialog might be open")
    input("Press Enter when ready to capture and analyze...")
    
    try:
        capture_and_analyze_dialog_region()
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()