#!/usr/bin/env python3
"""
Video Analyzer for Wisp Summoning
Extracts frames from the wisp summoning video to analyze dialog box patterns
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
import easyocr
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WispVideoAnalyzer:
    """Analyzes wisp summoning video to extract dialog box information"""
    
    def __init__(self, video_path: str, output_dir: str = "analysis_output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Dialog box detection parameters
        self.dialog_region = (100, 100, 600, 200)  # x, y, width, height - approximate dialog area
        self.target_fps = 15  # Extract frames at 15 fps as requested
        
    def extract_frames_at_fps(self, target_fps: int = 15) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video at specified FPS"""
        logger.info(f"Extracting frames from {self.video_path} at {target_fps} FPS")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        
        logger.info(f"Video: {original_fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
        
        # Calculate frame interval for target FPS
        frame_interval = int(original_fps / target_fps)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / original_fps
                frames.append((timestamp, frame.copy()))
                logger.info(f"Extracted frame at {timestamp:.2f}s")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def detect_dialog_box(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Detect dialog box in frame and extract text"""
        # Focus on the dialog region (top-left area where dialogs typically appear)
        x, y, w, h = self.dialog_region
        dialog_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale for better OCR
        gray_roi = cv2.cvtColor(dialog_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to enhance text
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use OCR to detect text
        ocr_results = self.ocr_reader.readtext(thresh)
        
        # Look for single letters (X, Z, V) which are the keystroke indicators
        detected_letters = []
        for (bbox, text, confidence) in ocr_results:
            text = text.strip().upper()
            if len(text) == 1 and text in ['X', 'Z', 'V'] and confidence > 0.5:
                detected_letters.append({
                    'letter': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        # Also check for dialog box visual characteristics
        dialog_detected = self._detect_dialog_visual(dialog_roi)
        
        result = {
            'timestamp': timestamp,
            'dialog_detected': dialog_detected,
            'letters': detected_letters,
            'ocr_results': [(bbox, text, conf) for bbox, text, conf in ocr_results]
        }
        
        # Save debug image if dialog or letters detected
        if dialog_detected or detected_letters:
            debug_path = self.output_dir / f"dialog_frame_{timestamp:.2f}s.png"
            cv2.imwrite(str(debug_path), dialog_roi)
            result['debug_image'] = str(debug_path)
        
        return result
    
    def _detect_dialog_visual(self, roi: np.ndarray) -> bool:
        """Detect dialog box using visual characteristics"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Look for rectangular borders (dialog boxes typically have borders)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Check if contour is rectangular and large enough
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area for dialog box
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If it's roughly rectangular (4 corners), it might be a dialog
                if len(approx) >= 4:
                    return True
        
        return False
    
    def analyze_video(self) -> Dict:
        """Complete video analysis"""
        logger.info("Starting comprehensive video analysis")
        
        # Extract frames
        frames = self.extract_frames_at_fps(self.target_fps)
        
        # Analyze each frame for dialog boxes
        analysis_results = []
        dialog_sequences = []
        
        for timestamp, frame in frames:
            result = self.detect_dialog_box(frame, timestamp)
            analysis_results.append(result)
            
            # Track dialog sequences
            if result['dialog_detected'] or result['letters']:
                dialog_sequences.append(result)
                logger.info(f"Dialog detected at {timestamp:.2f}s: {[l['letter'] for l in result['letters']]}")
        
        # Generate comprehensive analysis
        analysis = {
            'video_path': self.video_path,
            'analysis_timestamp': timestamp,
            'total_frames_analyzed': len(frames),
            'dialog_sequences': dialog_sequences,
            'keystroke_patterns': self._analyze_keystroke_patterns(dialog_sequences),
            'timing_data': self._analyze_timing(dialog_sequences),
            'all_results': analysis_results
        }
        
        # Save analysis results
        output_file = self.output_dir / "wisp_video_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis complete. Results saved to {output_file}")
        return analysis
    
    def _analyze_keystroke_patterns(self, dialog_sequences: List[Dict]) -> Dict:
        """Analyze patterns in detected keystrokes"""
        patterns = {
            'detected_letters': [],
            'letter_frequencies': {},
            'sequences': []
        }
        
        for seq in dialog_sequences:
            for letter_data in seq['letters']:
                letter = letter_data['letter']
                patterns['detected_letters'].append(letter)
                patterns['letter_frequencies'][letter] = patterns['letter_frequencies'].get(letter, 0) + 1
        
        # Look for common sequences
        if len(patterns['detected_letters']) >= 3:
            for i in range(len(patterns['detected_letters']) - 2):
                sequence = ''.join(patterns['detected_letters'][i:i+3])
                patterns['sequences'].append(sequence)
        
        return patterns
    
    def _analyze_timing(self, dialog_sequences: List[Dict]) -> Dict:
        """Analyze timing between dialog appearances"""
        if len(dialog_sequences) < 2:
            return {'intervals': [], 'average_interval': 0}
        
        intervals = []
        for i in range(1, len(dialog_sequences)):
            interval = dialog_sequences[i]['timestamp'] - dialog_sequences[i-1]['timestamp']
            intervals.append(interval)
        
        return {
            'intervals': intervals,
            'average_interval': sum(intervals) / len(intervals) if intervals else 0,
            'min_interval': min(intervals) if intervals else 0,
            'max_interval': max(intervals) if intervals else 0
        }

def main():
    """Main function to run video analysis"""
    video_path = "/workspace/wisp summon example.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Create analyzer and run analysis
    analyzer = WispVideoAnalyzer(video_path)
    results = analyzer.analyze_video()
    
    # Print summary
    print("\n" + "="*50)
    print("WISP VIDEO ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total frames analyzed: {results['total_frames_analyzed']}")
    print(f"Dialog sequences found: {len(results['dialog_sequences'])}")
    
    if results['keystroke_patterns']['detected_letters']:
        print(f"Detected letters: {results['keystroke_patterns']['detected_letters']}")
        print(f"Letter frequencies: {results['keystroke_patterns']['letter_frequencies']}")
    
    if results['timing_data']['intervals']:
        print(f"Average interval between dialogs: {results['timing_data']['average_interval']:.2f}s")
    
    print(f"\nDetailed results saved to: analysis_output/wisp_video_analysis.json")

if __name__ == "__main__":
    main()