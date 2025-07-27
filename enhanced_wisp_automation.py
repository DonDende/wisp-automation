#!/usr/bin/env python3
"""
Enhanced Wisp Automation Script
Integrates AI-powered dialog detection with 80-100ms keystroke delays
Uses computer vision and pattern matching for reliable dialog detection
"""

import cv2
import numpy as np
import pyautogui
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import threading
from dataclasses import dataclass
import easyocr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DialogDetection:
    """Result of dialog detection"""
    detected: bool
    letters: List[str]
    confidence: float
    timestamp: float
    method: str

class EnhancedDialogDetector:
    """Enhanced dialog detector using multiple detection methods"""
    
    def __init__(self):
        """Initialize the enhanced dialog detector"""
        logger.info("Initializing Enhanced Dialog Detector...")
        
        # Initialize OCR reader (CPU optimized)
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.warning(f"OCR initialization failed: {e}")
            self.ocr_reader = None
        
        # Dialog detection region (where dialogs typically appear)
        self.dialog_region = (50, 50, 700, 300)  # x, y, width, height
        
        # Valid keystroke letters
        self.valid_letters = {'X', 'Z', 'V'}
        
        # Template matching setup
        self.templates = self._load_letter_templates()
        
        # Detection parameters
        self.min_confidence = 0.6
        self.template_threshold = 0.7
        
    def _load_letter_templates(self) -> Dict[str, np.ndarray]:
        """Load or create letter templates for template matching"""
        templates = {}
        
        # Create simple letter templates (you could load actual templates from files)
        for letter in ['X', 'Z', 'V']:
            # Create a simple template (in practice, you'd load actual game screenshots)
            template = np.zeros((30, 20), dtype=np.uint8)
            cv2.putText(template, letter, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            templates[letter] = template
            
        return templates
    
    def capture_dialog_region(self) -> np.ndarray:
        """Capture the dialog region from screen"""
        try:
            x, y, w, h = self.dialog_region
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return np.zeros((300, 700, 3), dtype=np.uint8)
    
    def detect_with_template_matching(self, image: np.ndarray) -> List[Dict]:
        """Detect letters using template matching"""
        detected_letters = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for letter, template in self.templates.items():
            # Perform template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= self.template_threshold)
            
            for pt in zip(*locations[::-1]):
                detected_letters.append({
                    'letter': letter,
                    'confidence': float(result[pt[1], pt[0]]),
                    'method': 'template_matching',
                    'position': pt
                })
        
        return detected_letters
    
    def detect_with_ocr(self, image: np.ndarray) -> List[Dict]:
        """Detect letters using OCR"""
        if not self.ocr_reader:
            return []
        
        detected_letters = []
        
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Use OCR
            results = self.ocr_reader.readtext(thresh, detail=1)
            
            for (bbox, text, confidence) in results:
                text = text.strip().upper()
                
                # Look for single valid letters
                if len(text) == 1 and text in self.valid_letters and confidence > self.min_confidence:
                    detected_letters.append({
                        'letter': text,
                        'confidence': confidence,
                        'method': 'ocr',
                        'bbox': bbox
                    })
                
                # Look for multiple letters in sequence
                elif len(text) > 1:
                    for char in text:
                        if char in self.valid_letters:
                            detected_letters.append({
                                'letter': char,
                                'confidence': confidence * 0.8,
                                'method': 'ocr_sequence',
                                'bbox': bbox
                            })
        
        except Exception as e:
            logger.warning(f"OCR detection failed: {e}")
        
        return detected_letters
    
    def detect_with_color_analysis(self, image: np.ndarray) -> List[Dict]:
        """Detect dialog boxes using color analysis"""
        detected_letters = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for typical dialog elements
        color_ranges = [
            # White text on dark background
            ([0, 0, 200], [180, 30, 255]),
            # Yellow/gold text (common in games)
            ([20, 100, 100], [30, 255, 255])
        ]
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # Size range for single letters
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract the region and try to identify the letter
                    letter_roi = image[y:y+h, x:x+w]
                    
                    # Simple heuristic based on shape
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.3 < aspect_ratio < 1.5:  # Reasonable aspect ratio for letters
                        # This is a simplified approach - in practice you'd use more sophisticated analysis
                        detected_letters.append({
                            'letter': 'UNKNOWN',
                            'confidence': 0.5,
                            'method': 'color_analysis',
                            'position': (x, y),
                            'size': (w, h)
                        })
        
        return detected_letters
    
    def detect_dialog_and_letters(self) -> DialogDetection:
        """Main detection method combining multiple approaches"""
        start_time = time.time()
        
        # Capture screen region
        image = self.capture_dialog_region()
        
        # Try multiple detection methods
        all_detections = []
        
        # Method 1: Template matching
        template_results = self.detect_with_template_matching(image)
        all_detections.extend(template_results)
        
        # Method 2: OCR
        ocr_results = self.detect_with_ocr(image)
        all_detections.extend(ocr_results)
        
        # Method 3: Color analysis (as fallback)
        color_results = self.detect_with_color_analysis(image)
        all_detections.extend(color_results)
        
        # Combine and filter results
        final_letters = self._combine_detections(all_detections)
        
        # Calculate overall confidence
        if final_letters:
            avg_confidence = sum(d.get('confidence', 0) for d in all_detections) / len(all_detections)
        else:
            avg_confidence = 0.0
        
        detection = DialogDetection(
            detected=len(final_letters) > 0,
            letters=final_letters,
            confidence=avg_confidence,
            timestamp=time.time(),
            method='combined'
        )
        
        if detection.detected:
            logger.info(f"Dialog detected: {final_letters} (confidence: {avg_confidence:.2f})")
        
        return detection
    
    def _combine_detections(self, detections: List[Dict]) -> List[str]:
        """Combine detections from multiple methods"""
        letter_votes = {}
        
        for detection in detections:
            letter = detection.get('letter', 'UNKNOWN')
            if letter in self.valid_letters:
                confidence = detection.get('confidence', 0)
                method = detection.get('method', 'unknown')
                
                if letter not in letter_votes:
                    letter_votes[letter] = []
                
                letter_votes[letter].append({
                    'confidence': confidence,
                    'method': method
                })
        
        # Select letters with highest confidence
        final_letters = []
        for letter, votes in letter_votes.items():
            avg_confidence = sum(v['confidence'] for v in votes) / len(votes)
            if avg_confidence > self.min_confidence:
                final_letters.append(letter)
        
        return final_letters

class EnhancedWispAutomation:
    """Enhanced wisp automation with AI-powered dialog detection"""
    
    def __init__(self, config_file: str = "wisp_config.json"):
        """Initialize the enhanced wisp automation"""
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Initialize dialog detector
        self.dialog_detector = EnhancedDialogDetector()
        
        # Automation parameters
        self.keystroke_delay = 0.09  # 90ms delay (80-100ms range)
        self.detection_timeout = 10.0  # seconds
        self.max_attempts = 3
        
        # Statistics
        self.stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0,
            'avg_response_time': 0.0
        }
        
        logger.info("Enhanced Wisp Automation initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            'dialog_region': [50, 50, 700, 300],
            'keystroke_delay': 0.09,
            'detection_timeout': 10.0,
            'valid_letters': ['X', 'Z', 'V'],
            'automation_enabled': True
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def wait_for_dialog(self, timeout: float = None) -> DialogDetection:
        """Wait for dialog to appear"""
        timeout = timeout or self.detection_timeout
        start_time = time.time()
        
        logger.info(f"Waiting for dialog (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            detection = self.dialog_detector.detect_dialog_and_letters()
            
            if detection.detected and detection.letters:
                logger.info(f"Dialog found: {detection.letters}")
                return detection
            
            time.sleep(0.1)  # Check every 100ms
        
        logger.warning("Dialog detection timeout")
        return DialogDetection(False, [], 0.0, time.time(), 'timeout')
    
    def execute_keystrokes(self, letters: List[str]) -> bool:
        """Execute keystrokes with proper timing"""
        if not letters:
            return False
        
        logger.info(f"Executing keystrokes: {letters}")
        
        try:
            for i, letter in enumerate(letters):
                if i > 0:
                    time.sleep(self.keystroke_delay)  # 80-100ms delay between keystrokes
                
                # Press the key
                pyautogui.press(letter.lower())
                logger.info(f"Pressed key: {letter}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute keystrokes: {e}")
            return False
    
    def run_automation_cycle(self) -> bool:
        """Run a single automation cycle"""
        self.stats['attempts'] += 1
        start_time = time.time()
        
        logger.info(f"Starting automation cycle #{self.stats['attempts']}")
        
        try:
            # Wait for dialog to appear
            detection = self.wait_for_dialog()
            
            if not detection.detected:
                logger.warning("No dialog detected")
                self.stats['failures'] += 1
                return False
            
            # Execute keystrokes
            success = self.execute_keystrokes(detection.letters)
            
            if success:
                self.stats['successes'] += 1
                response_time = time.time() - start_time
                self.stats['avg_response_time'] = (
                    (self.stats['avg_response_time'] * (self.stats['successes'] - 1) + response_time) 
                    / self.stats['successes']
                )
                logger.info(f"Automation cycle completed successfully in {response_time:.2f}s")
                return True
            else:
                self.stats['failures'] += 1
                logger.error("Failed to execute keystrokes")
                return False
                
        except Exception as e:
            logger.error(f"Automation cycle failed: {e}")
            self.stats['failures'] += 1
            return False
    
    def run_continuous_automation(self, max_cycles: int = None):
        """Run continuous automation"""
        logger.info("Starting continuous wisp automation...")
        
        cycle_count = 0
        
        try:
            while True:
                if max_cycles and cycle_count >= max_cycles:
                    break
                
                success = self.run_automation_cycle()
                cycle_count += 1
                
                if success:
                    logger.info("Waiting before next cycle...")
                    time.sleep(2.0)  # Wait 2 seconds between successful cycles
                else:
                    logger.info("Retrying after failure...")
                    time.sleep(1.0)  # Shorter wait after failures
                
                # Print statistics every 10 cycles
                if cycle_count % 10 == 0:
                    self.print_statistics()
        
        except KeyboardInterrupt:
            logger.info("Automation stopped by user")
        
        finally:
            self.print_statistics()
    
    def print_statistics(self):
        """Print automation statistics"""
        success_rate = (self.stats['successes'] / self.stats['attempts'] * 100) if self.stats['attempts'] > 0 else 0
        
        print("\n" + "="*50)
        print("WISP AUTOMATION STATISTICS")
        print("="*50)
        print(f"Total attempts: {self.stats['attempts']}")
        print(f"Successes: {self.stats['successes']}")
        print(f"Failures: {self.stats['failures']}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average response time: {self.stats['avg_response_time']:.2f}s")
        print("="*50)
    
    def test_detection(self):
        """Test dialog detection without automation"""
        logger.info("Testing dialog detection...")
        
        detection = self.dialog_detector.detect_dialog_and_letters()
        
        print(f"Detection result:")
        print(f"  Detected: {detection.detected}")
        print(f"  Letters: {detection.letters}")
        print(f"  Confidence: {detection.confidence:.2f}")
        print(f"  Method: {detection.method}")
        
        return detection

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Wisp Automation")
    parser.add_argument("--test", action="store_true", help="Test detection only")
    parser.add_argument("--cycles", type=int, help="Maximum number of cycles to run")
    parser.add_argument("--config", default="wisp_config.json", help="Configuration file")
    
    args = parser.parse_args()
    
    # Create automation instance
    automation = EnhancedWispAutomation(args.config)
    
    if args.test:
        # Test mode
        automation.test_detection()
    else:
        # Run automation
        automation.run_continuous_automation(args.cycles)

if __name__ == "__main__":
    main()