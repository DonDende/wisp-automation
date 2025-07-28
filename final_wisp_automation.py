#!/usr/bin/env python3
"""
Final Wisp Automation Script
Integrates GPU-optimized AI detection with 80-100ms keystroke delays
Complete automation system for wisp summoning
"""

import cv2
import numpy as np
import torch
import os
import time

# Set display for headless environments
if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

import pyautogui
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import threading
from dataclasses import dataclass
from enum import Enum
import random

# Import our optimized detector
from gpu_optimized_detector import GPUOptimizedWispDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable pyautogui failsafe for automation
pyautogui.FAILSAFE = False

@dataclass
class DetectionResult:
    """Result of wisp detection"""
    detected: bool
    letters: List[str]
    confidence: float
    timestamp: float
    processing_time: float
    bbox: Optional[Tuple[int, int, int, int]] = None

class AutomationState(Enum):
    """States for automation state machine"""
    IDLE = "idle"
    DETECTING = "detecting"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"

class FinalWispAutomation:
    """Complete wisp automation system with AI detection"""
    
    def __init__(self, config_file: str = "final_wisp_config.json"):
        """Initialize the final automation system"""
        self.config_file = Path(config_file)
        self.config = self._load_config()
        
        # Initialize AI detector
        logger.info("Initializing AI-powered wisp detector...")
        self.ai_detector = GPUOptimizedWispDetector()
        
        # Automation parameters
        self.keystroke_delay_range = (0.08, 0.10)  # 80-100ms range
        self.detection_timeout = self.config.get('detection_timeout', 10.0)
        self.max_attempts = self.config.get('max_attempts', 5)
        
        # State management
        self.state = AutomationState.IDLE
        self.running = False
        self.stats = {
            'total_attempts': 0,
            'successful_detections': 0,
            'successful_executions': 0,
            'failed_detections': 0,
            'failed_executions': 0,
            'avg_detection_time': 0.0,
            'avg_execution_time': 0.0,
            'letters_detected': {},
            'session_start': time.time()
        }
        
        logger.info("Final Wisp Automation initialized successfully")
        logger.info(f"AI Device: {self.ai_detector.device}")
        logger.info(f"GPU Acceleration: {'Enabled' if self.ai_detector.use_gpu else 'Disabled'}")
    
    def _load_config(self) -> Dict:
        """Load configuration with intelligent defaults"""
        default_config = {
            'detection_timeout': 8.0,
            'max_attempts': 5,
            'keystroke_delay_min': 0.08,
            'keystroke_delay_max': 0.10,
            'detection_region': None,  # Auto-detect
            'confidence_threshold': 0.4,
            'automation_enabled': True,
            'debug_mode': False,
            'save_detection_images': True,
            'cycle_delay': 2.0,
            'valid_letters': ['X', 'Z', 'V']
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        else:
            # Save default config
            self.save_config(default_config)
        
        return default_config
    
    def save_config(self, config: Dict = None):
        """Save current configuration"""
        config_to_save = config or self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def capture_screen_region(self, use_corner_only=False) -> np.ndarray:
        """Capture screen region for detection"""
        try:
            if use_corner_only:
                # Capture only the corner region for letter detection
                corner_region = [753, 193, 165, 142]  # Your specific corner
                x, y, w, h = corner_region
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
            elif self.config.get('detection_region'):
                # Capture full box region for wisp box detection
                x, y, w, h = self.config['detection_region']
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
            else:
                # Capture full screen and let AI detector find the region
                screenshot = pyautogui.screenshot()
            
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return np.zeros((600, 800, 3), dtype=np.uint8)
    
    def detect_wisp_box(self) -> DetectionResult:
        """Two-stage detection: First detect full box, then analyze corner for letters"""
        self.state = AutomationState.DETECTING
        start_time = time.time()
        
        try:
            # Stage 1: Detect full wisp box to confirm it exists
            full_screen_image = self.capture_screen_region(use_corner_only=False)
            full_ai_result = self.ai_detector.detect_wisp_box_optimized(full_screen_image)
            
            if not full_ai_result['detected']:
                # No wisp box detected in full region
                result = DetectionResult(
                    detected=False,
                    letters=[],
                    confidence=0.0,
                    timestamp=time.time(),
                    processing_time=time.time() - start_time,
                    bbox=None
                )
            else:
                # Stage 2: Analyze corner region for letters
                corner_image = self.capture_screen_region(use_corner_only=True)
                corner_ai_result = self.ai_detector.detect_wisp_box_optimized(corner_image)
                
                # Combine results: box detection from full image, letters from corner
                result = DetectionResult(
                    detected=True,  # We know box exists from stage 1
                    letters=corner_ai_result['letters'],  # Letters from corner analysis
                    confidence=full_ai_result['confidence'],  # Confidence from full box detection
                    timestamp=time.time(),
                    processing_time=time.time() - start_time,
                    bbox=full_ai_result.get('bbox')
                )
            
            # Update statistics
            if result.detected:
                self.stats['successful_detections'] += 1
                for letter in result.letters:
                    self.stats['letters_detected'][letter] = self.stats['letters_detected'].get(letter, 0) + 1
                
                logger.info(f"AI detected wisp box: {result.letters} (confidence: {result.confidence:.3f})")
                
                # Save debug images if enabled
                if self.config.get('save_detection_images', False):
                    timestamp = int(result.timestamp)
                    full_debug_path = f"detection_full_{timestamp}.png"
                    corner_debug_path = f"detection_corner_{timestamp}.png"
                    cv2.imwrite(full_debug_path, full_screen_image)
                    cv2.imwrite(corner_debug_path, corner_image)
                    logger.info(f"Debug images saved: {full_debug_path}, {corner_debug_path}")
            else:
                self.stats['failed_detections'] += 1
                logger.info("No wisp box detected")
            
            # Update average detection time
            total_detections = self.stats['successful_detections'] + self.stats['failed_detections']
            self.stats['avg_detection_time'] = (
                (self.stats['avg_detection_time'] * (total_detections - 1) + result.processing_time) 
                / total_detections
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            self.state = AutomationState.ERROR
            return DetectionResult(False, [], 0.0, time.time(), time.time() - start_time)
    
    def execute_keystrokes(self, letters: List[str]) -> bool:
        """Execute keystrokes with precise 80-100ms delays"""
        if not letters:
            return False
        
        self.state = AutomationState.EXECUTING
        start_time = time.time()
        
        try:
            logger.info(f"Executing keystrokes: {letters}")
            
            for i, letter in enumerate(letters):
                if i > 0:
                    # Random delay between 80-100ms
                    delay = random.uniform(*self.keystroke_delay_range)
                    time.sleep(delay)
                    logger.info(f"Delay: {delay*1000:.1f}ms")
                
                # Press the key - try multiple methods for reliability
                success = False
                
                # Method 1: PyAutoGUI (original)
                try:
                    pyautogui.press(letter.lower())
                    logger.info(f"Pressed: {letter} (pyautogui)")
                    success = True
                except Exception as e:
                    logger.warning(f"PyAutoGUI failed for {letter}: {e}")
                
                # Method 2: keyboard library fallback
                if not success:
                    try:
                        import keyboard
                        keyboard.press_and_release(letter.lower())
                        logger.info(f"Pressed: {letter} (keyboard)")
                        success = True
                    except Exception as e:
                        logger.warning(f"keyboard library failed for {letter}: {e}")
                
                # Method 3: pynput fallback
                if not success:
                    try:
                        from pynput.keyboard import Controller
                        controller = Controller()
                        controller.press(letter.lower())
                        controller.release(letter.lower())
                        logger.info(f"Pressed: {letter} (pynput)")
                        success = True
                    except Exception as e:
                        logger.warning(f"pynput failed for {letter}: {e}")
                
                if not success:
                    logger.error(f"All keystroke methods failed for {letter}")
                    raise Exception(f"Could not send keystroke: {letter}")
            
            execution_time = time.time() - start_time
            
            # Update statistics
            self.stats['successful_executions'] += 1
            self.stats['avg_execution_time'] = (
                (self.stats['avg_execution_time'] * (self.stats['successful_executions'] - 1) + execution_time)
                / self.stats['successful_executions']
            )
            
            logger.info(f"Keystrokes executed successfully in {execution_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Keystroke execution failed: {e}")
            self.stats['failed_executions'] += 1
            self.state = AutomationState.ERROR
            return False
    
    def wait_for_wisp_box(self, timeout: float = None) -> DetectionResult:
        """Wait for wisp box to appear with timeout"""
        timeout = timeout or self.detection_timeout
        start_time = time.time()
        
        logger.info(f"Waiting for wisp box (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            if not self.running:
                break
            
            result = self.detect_wisp_box()
            
            if result.detected and result.confidence > self.config.get('confidence_threshold', 0.4):
                logger.info(f"Wisp box found after {time.time() - start_time:.2f}s")
                return result
            
            # Short pause between detection attempts
            time.sleep(0.1)
        
        logger.warning("Wisp box detection timeout")
        return DetectionResult(False, [], 0.0, time.time(), time.time() - start_time)
    
    def run_automation_cycle(self) -> bool:
        """Run a single automation cycle"""
        self.stats['total_attempts'] += 1
        cycle_start = time.time()
        
        logger.info(f"Starting automation cycle #{self.stats['total_attempts']}")
        
        try:
            # Wait for wisp box to appear
            detection = self.wait_for_wisp_box()
            
            if not detection.detected:
                logger.warning("Cycle failed: No wisp box detected")
                return False
            
            # Execute keystrokes
            success = self.execute_keystrokes(detection.letters)
            
            if success:
                cycle_time = time.time() - cycle_start
                logger.info(f"Cycle completed successfully in {cycle_time:.2f}s")
                return True
            else:
                logger.error("Cycle failed: Keystroke execution failed")
                return False
                
        except Exception as e:
            logger.error(f"Automation cycle failed: {e}")
            return False
        finally:
            self.state = AutomationState.IDLE
    
    def run_continuous_automation(self, max_cycles: int = None):
        """Run continuous automation"""
        logger.info("Starting continuous wisp automation...")
        logger.info(f"AI Device: {self.ai_detector.device}")
        logger.info(f"Keystroke delay range: {self.keystroke_delay_range[0]*1000:.0f}-{self.keystroke_delay_range[1]*1000:.0f}ms")
        
        self.running = True
        cycle_count = 0
        
        try:
            while self.running:
                if max_cycles and cycle_count >= max_cycles:
                    logger.info(f"Reached maximum cycles: {max_cycles}")
                    break
                
                success = self.run_automation_cycle()
                cycle_count += 1
                
                if success:
                    logger.info("Waiting before next cycle...")
                    time.sleep(self.config.get('cycle_delay', 2.0))
                else:
                    logger.info("Retrying after failure...")
                    time.sleep(1.0)
                
                # Print statistics every 5 cycles
                if cycle_count % 5 == 0:
                    self.print_statistics()
        
        except KeyboardInterrupt:
            logger.info("Automation stopped by user")
        except Exception as e:
            logger.error(f"Automation error: {e}")
        finally:
            self.running = False
            self.print_final_statistics()
    
    def print_statistics(self):
        """Print current statistics"""
        runtime = time.time() - self.stats['session_start']
        success_rate = (self.stats['successful_executions'] / self.stats['total_attempts'] * 100) if self.stats['total_attempts'] > 0 else 0
        
        print("\n" + "="*60)
        print("WISP AUTOMATION STATISTICS")
        print("="*60)
        print(f"Runtime: {runtime:.1f}s")
        print(f"Total attempts: {self.stats['total_attempts']}")
        print(f"Successful executions: {self.stats['successful_executions']}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"AI Device: {self.ai_detector.device}")
        print(f"Avg detection time: {self.stats['avg_detection_time']:.3f}s")
        print(f"Avg execution time: {self.stats['avg_execution_time']:.3f}s")
        if self.stats['letters_detected']:
            print(f"Letters detected: {self.stats['letters_detected']}")
        print("="*60)
    
    def print_final_statistics(self):
        """Print final session statistics"""
        self.print_statistics()
        
        # Save session stats
        session_file = f"session_stats_{int(time.time())}.json"
        try:
            with open(session_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            logger.info(f"Session statistics saved to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session stats: {e}")
    
    def test_detection_only(self):
        """Test detection without automation"""
        logger.info("Testing detection only...")
        
        result = self.detect_wisp_box()
        
        print(f"\nDetection Test Results:")
        print(f"  Detected: {result.detected}")
        print(f"  Letters: {result.letters}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        print(f"  AI Device: {self.ai_detector.device}")
        
        return result

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Final Wisp Automation with AI Detection")
    parser.add_argument("--test", action="store_true", help="Test detection only")
    parser.add_argument("--cycles", type=int, help="Maximum number of cycles")
    parser.add_argument("--config", default="final_wisp_config.json", help="Configuration file")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # Create automation instance
    automation = FinalWispAutomation(args.config)
    
    if args.benchmark:
        # Run benchmark
        test_image = automation.capture_screen_region()
        benchmark = automation.ai_detector.benchmark_performance(test_image, 10)
        print(f"\nBenchmark Results: {benchmark}")
    elif args.test:
        # Test mode
        automation.test_detection_only()
    else:
        # Run automation
        automation.run_continuous_automation(args.cycles)

if __name__ == "__main__":
    main()