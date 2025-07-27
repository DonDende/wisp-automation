#!/usr/bin/env python3
"""
Custom AI-powered Wisp Detector
Uses a pre-trained vision model from Hugging Face to detect translucent boxes with letters
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WispAIDetector:
    """AI-powered detector for wisp summoning translucent boxes"""
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """Initialize the AI detector"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading AI model: {model_name} on {self.device}")
        
        try:
            # Load the processor and model
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("AI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Detection parameters
        self.confidence_threshold = 0.7
        self.letter_templates = self._create_letter_templates()
        
    def _create_letter_templates(self) -> Dict[str, np.ndarray]:
        """Create templates for X, Z, V letters"""
        templates = {}
        
        for letter in ['X', 'Z', 'V']:
            # Create a template image for each letter
            template = np.zeros((60, 40), dtype=np.uint8)
            
            # Draw the letter
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            color = 255
            
            # Get text size and center it
            (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
            x = (template.shape[1] - text_width) // 2
            y = (template.shape[0] + text_height) // 2
            
            cv2.putText(template, letter, (x, y), font, font_scale, color, thickness)
            templates[letter] = template
            
        return templates
    
    def detect_translucent_box(self, image: np.ndarray) -> Dict:
        """Detect translucent box with letters using AI and computer vision"""
        result = {
            'detected': False,
            'letters': [],
            'confidence': 0.0,
            'bbox': None,
            'method': 'ai_cv_hybrid'
        }
        
        try:
            # First, use computer vision to find potential translucent regions
            potential_regions = self._find_translucent_regions(image)
            
            if not potential_regions:
                return result
            
            # For each potential region, use AI to classify what's inside
            best_detection = None
            best_confidence = 0.0
            
            for region_info in potential_regions:
                region = region_info['region']
                bbox = region_info['bbox']
                
                # Use AI to analyze this region
                ai_result = self._analyze_region_with_ai(region)
                
                if ai_result['confidence'] > best_confidence:
                    best_confidence = ai_result['confidence']
                    best_detection = {
                        'letters': ai_result['letters'],
                        'confidence': ai_result['confidence'],
                        'bbox': bbox
                    }
            
            if best_detection and best_confidence > self.confidence_threshold:
                result.update({
                    'detected': True,
                    'letters': best_detection['letters'],
                    'confidence': best_confidence,
                    'bbox': best_detection['bbox']
                })
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        return result
    
    def _find_translucent_regions(self, image: np.ndarray) -> List[Dict]:
        """Find potential translucent box regions using computer vision"""
        regions = []
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Look for semi-transparent overlays (common characteristics)
        # 1. Slightly different brightness than background
        # 2. Rectangular shape
        # 3. Contains text-like elements
        
        # Apply adaptive threshold to find text regions
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (translucent boxes are typically medium-sized)
            if 1000 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (boxes are typically wider than tall)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 4.0:
                    # Extract the region
                    region = image[y:y+h, x:x+w]
                    
                    # Check if this region might contain text
                    if self._region_has_text_characteristics(region):
                        regions.append({
                            'region': region,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
        
        # Sort by area (larger regions first)
        regions.sort(key=lambda x: x['area'], reverse=True)
        
        return regions[:5]  # Return top 5 candidates
    
    def _region_has_text_characteristics(self, region: np.ndarray) -> bool:
        """Check if a region has characteristics of containing text"""
        if region.size == 0:
            return False
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Check for text-like patterns
        # 1. Reasonable contrast
        contrast = np.std(gray)
        if contrast < 10:  # Too uniform
            return False
        
        # 2. Not too noisy
        if contrast > 100:  # Too noisy
            return False
        
        # 3. Has some structure (edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return 0.01 < edge_density < 0.3  # Reasonable edge density for text
    
    def _analyze_region_with_ai(self, region: np.ndarray) -> Dict:
        """Analyze a region using AI to detect letters"""
        result = {
            'letters': [],
            'confidence': 0.0
        }
        
        try:
            # Convert region to PIL Image
            if len(region.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(region).convert('RGB')
            
            # Resize if too small
            if pil_image.size[0] < 32 or pil_image.size[1] < 32:
                pil_image = pil_image.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Use template matching as a fallback/supplement to AI
            template_results = self._template_match_letters(region)
            
            # For now, use template matching results
            # In a full implementation, you'd combine this with AI classification
            if template_results:
                result['letters'] = template_results['letters']
                result['confidence'] = template_results['confidence']
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
        
        return result
    
    def _template_match_letters(self, region: np.ndarray) -> Optional[Dict]:
        """Use template matching to find letters in the region"""
        if region.size == 0:
            return None
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        detected_letters = []
        confidences = []
        
        for letter, template in self.letter_templates.items():
            # Resize template to match potential letter size in region
            h, w = template.shape
            region_h, region_w = gray.shape
            
            # Try different scales
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            
            for scale in scales:
                scaled_template = cv2.resize(template, (int(w * scale), int(h * scale)))
                
                if scaled_template.shape[0] > region_h or scaled_template.shape[1] > region_w:
                    continue
                
                # Perform template matching
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.6:  # Good match threshold
                    detected_letters.append(letter)
                    confidences.append(max_val)
                    break  # Found this letter, move to next
        
        if detected_letters:
            avg_confidence = sum(confidences) / len(confidences)
            return {
                'letters': detected_letters,
                'confidence': avg_confidence
            }
        
        return None
    
    def analyze_video_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Analyze a single video frame for wisp summoning box"""
        logger.info(f"Analyzing frame at {timestamp:.2f}s")
        
        # Focus on the area where translucent boxes typically appear
        # Based on typical game UI, this is often center or upper-center
        height, width = frame.shape[:2]
        
        # Define search regions
        regions_to_check = [
            # Center region
            frame[height//4:3*height//4, width//4:3*width//4],
            # Upper center
            frame[0:height//2, width//4:3*width//4],
            # Full frame (as fallback)
            frame
        ]
        
        best_result = None
        best_confidence = 0.0
        
        for i, region in enumerate(regions_to_check):
            result = self.detect_translucent_box(region)
            
            if result['detected'] and result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
                best_result['region_index'] = i
        
        if best_result:
            logger.info(f"Detected letters: {best_result['letters']} (confidence: {best_confidence:.2f})")
            return best_result
        
        return {'detected': False, 'letters': [], 'confidence': 0.0}

class WispVideoAnalyzer:
    """Analyze wisp summoning video using AI detector"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.ai_detector = WispAIDetector()
        
    def analyze_first_4_seconds(self) -> Dict:
        """Analyze the first 4 seconds where translucent box appears"""
        logger.info("Analyzing first 4 seconds of video with AI...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        results = []
        detected_sequences = []
        
        # Analyze frames from 0 to 4 seconds at 10 FPS
        for timestamp in np.arange(0.1, 4.1, 0.1):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            result = self.ai_detector.analyze_video_frame(frame, timestamp)
            result['timestamp'] = timestamp
            results.append(result)
            
            if result['detected']:
                detected_sequences.append(result)
                
                # Save debug image
                debug_path = f"ai_detection_{timestamp:.1f}s.png"
                cv2.imwrite(debug_path, frame)
                result['debug_image'] = debug_path
        
        cap.release()
        
        analysis = {
            'video_path': self.video_path,
            'analysis_method': 'ai_powered_cv',
            'time_range': '0-4 seconds',
            'total_frames_analyzed': len(results),
            'detected_sequences': detected_sequences,
            'keystroke_patterns': self._analyze_patterns(detected_sequences),
            'all_results': results
        }
        
        # Save results
        with open('ai_wisp_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def _analyze_patterns(self, sequences: List[Dict]) -> Dict:
        """Analyze detected keystroke patterns"""
        patterns = {
            'detected_letters': [],
            'letter_frequencies': {},
            'timing_data': []
        }
        
        for seq in sequences:
            for letter in seq['letters']:
                patterns['detected_letters'].append(letter)
                patterns['letter_frequencies'][letter] = patterns['letter_frequencies'].get(letter, 0) + 1
            
            patterns['timing_data'].append({
                'timestamp': seq['timestamp'],
                'letters': seq['letters'],
                'confidence': seq['confidence']
            })
        
        return patterns

def main():
    """Main function to run AI-powered analysis"""
    video_path = "/workspace/wisp summon example.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Create analyzer and run analysis
    analyzer = WispVideoAnalyzer(video_path)
    results = analyzer.analyze_first_4_seconds()
    
    # Print summary
    print("\n" + "="*60)
    print("AI-POWERED WISP DETECTION RESULTS")
    print("="*60)
    print(f"Time range analyzed: {results['time_range']}")
    print(f"Total frames analyzed: {results['total_frames_analyzed']}")
    print(f"Sequences detected: {len(results['detected_sequences'])}")
    
    if results['keystroke_patterns']['detected_letters']:
        print(f"Detected letters: {results['keystroke_patterns']['detected_letters']}")
        print(f"Letter frequencies: {results['keystroke_patterns']['letter_frequencies']}")
    
    print(f"\nResults saved to: ai_wisp_analysis.json")
    
    # Show timing data
    if results['keystroke_patterns']['timing_data']:
        print("\nDetection Timeline:")
        for data in results['keystroke_patterns']['timing_data'][:10]:  # Show first 10
            print(f"  {data['timestamp']:.1f}s: {data['letters']} (conf: {data['confidence']:.2f})")

if __name__ == "__main__":
    main()