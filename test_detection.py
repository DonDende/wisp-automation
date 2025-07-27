#!/usr/bin/env python3
"""
Test script to verify dialog detection region and show what we're capturing
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_video_frame_extraction():
    """Test extracting frames from the video to see what we should be looking for"""
    video_path = "/workspace/wisp summon example.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video FPS: {fps}")
    
    # Extract frames at key timestamps where dialogs should appear
    key_times = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]  # seconds
    
    for timestamp in key_times:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if ret:
            # Save full frame for analysis
            output_path = f"test_frame_{timestamp:.0f}s_full.png"
            cv2.imwrite(output_path, frame)
            logger.info(f"Saved full frame: {output_path}")
            
            # Also save different regions for comparison
            height, width = frame.shape[:2]
            
            # Top region (where dialogs might be)
            top_region = frame[0:300, 0:width]
            cv2.imwrite(f"test_frame_{timestamp:.0f}s_top.png", top_region)
            
            # Center region
            center_y = height // 2
            center_region = frame[center_y-150:center_y+150, 0:width]
            cv2.imwrite(f"test_frame_{timestamp:.0f}s_center.png", center_region)
            
            # Bottom region
            bottom_region = frame[height-300:height, 0:width]
            cv2.imwrite(f"test_frame_{timestamp:.0f}s_bottom.png", bottom_region)
            
            logger.info(f"Frame {timestamp}s: {width}x{height} - saved regions")
    
    cap.release()
    logger.info("Frame extraction complete. Check the generated images to see where dialogs appear.")

def analyze_existing_captures():
    """Analyze the existing captured frames to understand what we're looking at"""
    analysis_dir = Path("analysis_output")
    
    if not analysis_dir.exists():
        logger.error("Analysis output directory not found")
        return
    
    # Look at a few captured frames
    frame_files = list(analysis_dir.glob("dialog_frame_*.png"))[:5]
    
    for frame_file in frame_files:
        logger.info(f"Analyzing: {frame_file}")
        
        image = cv2.imread(str(frame_file))
        if image is None:
            continue
        
        height, width = image.shape[:2]
        logger.info(f"  Size: {width}x{height}")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for text-like regions
        # Apply threshold to find text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Size range for text
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio for text
                    text_regions += 1
        
        logger.info(f"  Found {text_regions} potential text regions")
        
        # Check brightness/contrast
        mean_brightness = np.mean(gray)
        logger.info(f"  Average brightness: {mean_brightness:.1f}")

def main():
    """Main function"""
    print("Testing dialog detection...")
    print("1. Extracting test frames from video...")
    test_video_frame_extraction()
    
    print("\n2. Analyzing existing captures...")
    analyze_existing_captures()
    
    print("\nTest complete. Please check the generated test_frame_*.png files")
    print("to see what the video actually contains and where dialogs appear.")

if __name__ == "__main__":
    main()