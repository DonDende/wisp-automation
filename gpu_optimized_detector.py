#!/usr/bin/env python3
"""
GPU-Optimized Wisp Detector
Automatically uses GPU when available, falls back to optimized CPU processing
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import pipeline
from PIL import Image
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import warnings
import re

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUOptimizedWispDetector:
    """GPU-optimized wisp detector with automatic hardware detection"""
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """Initialize with automatic GPU/CPU detection"""
        self.model_name = model_name
        
        # Detect available hardware
        self.device = self._detect_best_device()
        self.use_gpu = self.device.type == 'cuda'
        
        logger.info(f"Hardware detected: {self.device}")
        logger.info(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled (CPU only)'}")
        
        # Load model with appropriate optimizations
        self._load_model()
        
        # Initialize vision pipeline for faster inference
        self._setup_vision_pipeline()
        
    def _detect_best_device(self) -> torch.device:
        """Detect the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU found: {gpu_name} ({gpu_memory:.1f}GB)")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon GPU support
            logger.info("Apple Silicon GPU (MPS) detected")
            return torch.device('mps')
        else:
            logger.info("No GPU detected, using CPU with optimizations")
            return torch.device('cpu')
    
    def _load_model(self):
        """Load model with device-specific optimizations"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load processor
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate settings
            if self.use_gpu:
                # GPU optimizations
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Use half precision for speed
                    device_map="auto"
                )
            else:
                # CPU optimizations
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable optimizations
            if self.use_gpu:
                # GPU-specific optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                # CPU-specific optimizations
                torch.set_num_threads(4)  # Optimize for multi-core CPU
            
            logger.info("Model loaded and optimized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_vision_pipeline(self):
        """Setup optimized vision pipeline"""
        try:
            # Create a classification pipeline for faster inference
            self.vision_pipeline = pipeline(
                "image-classification",
                model=self.model,
                image_processor=self.processor,
                device=0 if self.use_gpu else -1,  # 0 for GPU, -1 for CPU
                torch_dtype=torch.float16 if self.use_gpu else torch.float32
            )
            logger.info("Vision pipeline initialized")
        except Exception as e:
            logger.warning(f"Pipeline setup failed, using direct model: {e}")
            self.vision_pipeline = None
    
    def detect_wisp_box_optimized(self, image: np.ndarray) -> Dict:
        """Optimized detection using GPU acceleration when available"""
        start_time = time.time()
        
        result = {
            'detected': False,
            'letters': [],
            'confidence': 0.0,
            'processing_time': 0.0,
            'device_used': str(self.device)
        }
        
        try:
            # Preprocess image for optimal performance
            processed_regions = self._preprocess_for_detection(image)
            
            best_detection = None
            best_confidence = 0.0
            
            for region_info in processed_regions:
                region = region_info['region']
                bbox = region_info['bbox']
                
                # Use optimized AI analysis
                detection = self._analyze_region_optimized(region)
                
                if detection['confidence'] > best_confidence:
                    best_confidence = detection['confidence']
                    best_detection = {
                        'letters': detection['letters'],
                        'confidence': detection['confidence'],
                        'bbox': bbox
                    }
            
            if best_detection and best_confidence > 0.5:
                result.update({
                    'detected': True,
                    'letters': best_detection['letters'],
                    'confidence': best_confidence,
                    'bbox': best_detection['bbox']
                })
            
        except Exception as e:
            logger.error(f"Optimized detection failed: {e}")
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _preprocess_for_detection(self, image: np.ndarray) -> List[Dict]:
        """Optimized preprocessing for detection"""
        regions = []
        
        # Use efficient image processing
        height, width = image.shape[:2]
        
        # Define strategic regions based on typical game UI layouts
        search_regions = [
            # Center region (most common for dialog boxes)
            {
                'name': 'center',
                'coords': (width//4, height//4, width//2, height//2),
                'priority': 1
            },
            # Upper center (common for notifications)
            {
                'name': 'upper_center', 
                'coords': (width//4, 0, width//2, height//2),
                'priority': 2
            },
            # Lower center (common for action prompts)
            {
                'name': 'lower_center',
                'coords': (width//4, height//2, width//2, height//2),
                'priority': 3
            }
        ]
        
        for region_info in search_regions:
            x, y, w, h = region_info['coords']
            region = image[y:y+h, x:x+w]
            
            # Quick quality check
            if self._is_region_worth_analyzing(region):
                regions.append({
                    'region': region,
                    'bbox': (x, y, w, h),
                    'name': region_info['name'],
                    'priority': region_info['priority']
                })
        
        # Sort by priority
        regions.sort(key=lambda x: x['priority'])
        return regions
    
    def _is_region_worth_analyzing(self, region: np.ndarray) -> bool:
        """Quick check if region is worth detailed analysis"""
        if region.size == 0:
            return False
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Check for sufficient contrast (indicates potential text/UI elements)
        contrast = np.std(gray)
        mean_brightness = np.mean(gray)
        
        # Good regions have moderate contrast and aren't too dark/bright
        return (10 < contrast < 80 and 20 < mean_brightness < 200)
    
    def _analyze_region_optimized(self, region: np.ndarray) -> Dict:
        """Optimized AI analysis of region"""
        result = {'letters': [], 'confidence': 0.0}
        
        try:
            # Convert to PIL Image
            if len(region.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(region).convert('RGB')
            
            # Optimize image size for processing
            target_size = 224  # Standard size for most vision models
            if pil_image.size[0] != target_size or pil_image.size[1] != target_size:
                pil_image = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Use optimized inference
            if self.vision_pipeline:
                # Use pipeline for faster inference
                predictions = self.vision_pipeline(pil_image, top_k=5)
                
                # Analyze predictions for letter-like patterns
                for pred in predictions:
                    label = pred['label'].upper()
                    score = pred['score']
                    
                    # Look for letter patterns in the classification
                    if any(letter in label for letter in ['X', 'Z', 'V', 'C']):
                        result['confidence'] = max(result['confidence'], score)
                        if score > 0.3:  # Lower threshold for detection
                            for letter in ['X', 'Z', 'V', 'C']:
                                if letter in label and letter not in result['letters']:
                                    result['letters'].append(letter)
            else:
                # Direct model inference
                inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    if self.use_gpu:
                        with torch.cuda.amp.autocast():  # Mixed precision for speed
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)
                    
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    confidence = torch.max(probabilities).item()
                    
                    # Simple heuristic: if confidence is reasonable, assume detection
                    if confidence > 0.1:
                        result['confidence'] = confidence
                        # Use OCR to detect actual letters in the region
                        result['letters'] = self._detect_letters_with_ocr(region)
            
            # Supplement with improved template matching for better accuracy
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            improved_letters = self._detect_letters_by_improved_templates(gray_region)
            if improved_letters:
                logger.info(f"OCR letters: {result['letters']} ({len(result['letters'])})")
                logger.info(f"Improved template letters: {improved_letters} ({len(improved_letters)})")
                # Prioritize improved template matching if it finds multiple letters
                if len(improved_letters) > len(result['letters']):
                    logger.info("Using improved template matching result")
                    result['letters'] = improved_letters
                    # Keep the AI confidence since template matching doesn't provide one
            
        except Exception as e:
            logger.warning(f"Optimized analysis failed: {e}")
        
        return result
    
    def _fast_template_matching(self, region: np.ndarray) -> Optional[Dict]:
        """Fast template matching optimized for performance"""
        if region.size == 0:
            return None
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Create simple templates for X, Z, V, C
        templates = {}
        for letter in ['X', 'Z', 'V', 'C']:
            template = np.zeros((40, 30), dtype=np.uint8)
            cv2.putText(template, letter, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
            templates[letter] = template
        
        detected_letters = []
        max_confidence = 0.0
        
        for letter, template in templates.items():
            # Multi-scale template matching
            for scale in [0.8, 1.0, 1.2]:
                h, w = template.shape
                scaled_template = cv2.resize(template, (int(w * scale), int(h * scale)))
                
                if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.5:  # Higher threshold for better accuracy
                    # For this scale, find ALL instances above threshold
                    locations = np.where(result >= 0.5)
                    for pt in zip(*locations[::-1]):
                        x, y = pt
                        score = result[y, x]
                        
                        # Check if too close to existing detections
                        too_close = False
                        for existing_x, existing_y, _, _ in detected_letters:
                            if abs(x - existing_x) < 20 and abs(y - existing_y) < 20:
                                too_close = True
                                break
                        
                        if not too_close:
                            detected_letters.append((x, y, letter, score))
                            max_confidence = max(max_confidence, score)
                    break
        
        if detected_letters:
            # Sort by absolute left to right (x-coordinate only)
            detected_letters.sort(key=lambda x: x[0])  # Sort by x-coordinate only
            return {
                'letters': [letter for _, _, letter, _ in detected_letters],
                'confidence': max_confidence
            }
        
        return None
    
    def _detect_letters_with_ocr(self, region: np.ndarray) -> List[str]:
        """Detect letters using OCR and image processing"""
        try:
            # Convert to grayscale if needed
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()
            
            # Enhance contrast for better OCR
            gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try to detect letters using contour analysis
            letters = self._detect_letters_by_contours(binary)
            if letters:
                return letters
            
            # Fallback: try template matching with improved templates
            return self._detect_letters_by_improved_templates(gray)
            
        except Exception as e:
            logger.warning(f"OCR detection failed: {e}")
            return []
    
    def _detect_letters_by_contours(self, binary_image: np.ndarray) -> List[str]:
        """Detect letters by analyzing contours and shapes"""
        try:
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and aspect ratio
            letter_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by reasonable letter dimensions
                if (10 < w < 100 and 15 < h < 100 and area > 50):
                    letter_contours.append((x, contour))
            
            # Sort by x-coordinate (left to right)
            letter_contours.sort(key=lambda x: x[0])
            
            detected_letters = []
            for x, contour in letter_contours:
                # Extract letter region
                x, y, w, h = cv2.boundingRect(contour)
                letter_roi = binary_image[y:y+h, x:x+w]
                
                # Classify the letter shape
                letter = self._classify_letter_shape(letter_roi, contour)
                if letter:
                    detected_letters.append(letter)
            
            return detected_letters
            
        except Exception as e:
            logger.warning(f"Contour detection failed: {e}")
            return []
    
    def _classify_letter_shape(self, letter_roi: np.ndarray, contour) -> Optional[str]:
        """Classify letter based on shape characteristics"""
        try:
            # Get contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                return None
                
            # Calculate shape descriptors
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Analyze the letter shape using template matching
            best_match = None
            best_score = 0
            
            for letter in ['X', 'Z', 'V', 'C']:
                # Create template
                template = np.zeros((40, 30), dtype=np.uint8)
                cv2.putText(template, letter, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
                
                # Resize letter_roi to match template size
                resized_roi = cv2.resize(letter_roi, (30, 40))
                
                # Template matching
                result = cv2.matchTemplate(resized_roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = letter
            
            # Return best match if confidence is high enough
            if best_score > 0.3:
                return best_match
                
            return None
            
        except Exception as e:
            logger.warning(f"Letter classification failed: {e}")
            return None
    
    def _detect_letters_by_improved_templates(self, gray_image: np.ndarray) -> List[str]:
        """Improved template matching with multiple scales and positions"""
        try:
            detected_letters = []
            used_positions = []  # Track used positions to avoid duplicates
            
            # Check each possible letter and find ALL instances
            for letter in ['X', 'Z', 'V', 'C']:
                # Create template
                template = np.zeros((50, 40), dtype=np.uint8)
                cv2.putText(template, letter, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
                
                if template.shape[0] > gray_image.shape[0] or template.shape[1] > gray_image.shape[1]:
                    continue
                
                # Find ALL matches above threshold
                result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                
                # Find all locations where match is above threshold
                locations = np.where(result >= 0.5)  # Higher threshold
                
                for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                    score = result[pt[1], pt[0]]
                    
                    # Check for overlap with existing detections
                    overlap = False
                    for used_pos in used_positions:
                        if abs(pt[0] - used_pos[0]) < 15 and abs(pt[1] - used_pos[1]) < 15:
                            overlap = True
                            break
                    
                    if not overlap:
                        detected_letters.append((pt[0], pt[1], letter))
                        used_positions.append(pt)
            
            # Sort by absolute left to right (x-coordinate only)
            detected_letters.sort(key=lambda x: x[0])  # Sort by x-coordinate only
            return [letter for _, _, letter in detected_letters]
            
        except Exception as e:
            logger.warning(f"Improved template matching failed: {e}")
            return []
    
    def benchmark_performance(self, test_image: np.ndarray, iterations: int = 10) -> Dict:
        """Benchmark detection performance"""
        logger.info(f"Running performance benchmark ({iterations} iterations)...")
        
        times = []
        for i in range(iterations):
            start = time.time()
            result = self.detect_wisp_box_optimized(test_image)
            end = time.time()
            times.append(end - start)
        
        benchmark = {
            'device': str(self.device),
            'gpu_enabled': self.use_gpu,
            'iterations': iterations,
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps_estimate': 1.0 / np.mean(times)
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Average time: {benchmark['avg_time']:.3f}s")
        logger.info(f"  Estimated FPS: {benchmark['fps_estimate']:.1f}")
        
        return benchmark

def main():
    """Test the GPU-optimized detector"""
    logger.info("Testing GPU-Optimized Wisp Detector...")
    
    # Initialize detector
    detector = GPUOptimizedWispDetector()
    
    # Load a test image
    video_path = "/workspace/wisp summon example.mp4"
    if Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)  # Frame at 1 second
        ret, test_frame = cap.read()
        cap.release()
        
        if ret:
            # Test detection
            result = detector.detect_wisp_box_optimized(test_frame)
            print(f"\nDetection result: {result}")
            
            # Run benchmark
            benchmark = detector.benchmark_performance(test_frame, 5)
            print(f"\nBenchmark results: {benchmark}")
        else:
            logger.error("Could not read test frame from video")
    else:
        logger.error("Video file not found")

if __name__ == "__main__":
    main()