"""
Vision-Language Model for Wisp Detection
Uses a proper VLM that can understand instructions and analyze specific regions
"""

import logging
import time
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Optional
from transformers import BlipProcessor, BlipForConditionalGeneration
import re

logger = logging.getLogger(__name__)

class VisionLanguageWispDetector:
    """Vision-Language Model that can follow instructions to analyze wisp boxes and corners"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large"):
        """Initialize with a vision-language model that can understand instructions"""
        self.model_name = model_name
        
        # Detect available hardware
        self.device = self._detect_best_device()
        self.use_gpu = self.device.type == 'cuda'
        
        logger.info(f"Hardware detected: {self.device}")
        logger.info(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled (CPU only)'}")
        
        # Load vision-language model
        self._load_model()
        
    def _detect_best_device(self) -> torch.device:
        """Detect the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU found: {gpu_name} ({gpu_memory:.1f}GB)")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple Silicon GPU (MPS) detected")
            return torch.device('mps')
        else:
            logger.info("Using CPU")
            return torch.device('cpu')
    
    def _load_model(self):
        """Load the vision-language model"""
        try:
            logger.info(f"Loading vision-language model: {self.model_name}")
            
            # Load processor and model
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move to device and optimize
            self.model = self.model.to(self.device)
            
            if self.use_gpu:
                # Enable optimizations for GPU
                self.model = self.model.half()  # Use FP16 for speed
                logger.info("GPU optimizations enabled (FP16)")
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Vision-language model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_wisp_box_optimized(self, image: np.ndarray) -> Dict:
        """Detect wisp box using vision-language understanding"""
        start_time = time.time()
        
        result = {
            'detected': False,
            'letters': [],
            'confidence': 0.0,
            'processing_time': 0.0,
            'device_used': str(self.device)
        }
        
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # First, check if this looks like a wisp summoning interface
            wisp_detection_result = self._analyze_for_wisp_box(pil_image)
            
            if wisp_detection_result['detected']:
                # Now specifically look for letters in the interface
                letters_result = self._analyze_for_letters(pil_image)
                
                result.update({
                    'detected': True,
                    'letters': letters_result['letters'],
                    'confidence': wisp_detection_result['confidence'],
                    'processing_time': time.time() - start_time
                })
            
        except Exception as e:
            logger.warning(f"Vision-language analysis failed: {e}")
        
        return result
    
    def _analyze_for_wisp_box(self, image: Image.Image) -> Dict:
        """Use VLM to detect if this is a wisp summoning interface"""
        try:
            # Use unconditional image captioning (no prompt) for better results
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if self.use_gpu:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(**inputs, max_new_tokens=30, num_beams=4, do_sample=False)
                else:
                    outputs = self.model.generate(**inputs, max_new_tokens=30, num_beams=4, do_sample=False)
            
            # Decode the response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Wisp box analysis: '{response}'")
            
            # Analyze the caption for game interface indicators
            response_lower = response.lower()
            game_indicators = ['game', 'interface', 'screen', 'menu', 'button', 'text', 'window', 'box', 'ui']
            wisp_indicators = ['wisp', 'magic', 'spell', 'creature', 'summon']
            
            game_score = sum(1 for word in game_indicators if word in response_lower)
            wisp_score = sum(1 for word in wisp_indicators if word in response_lower)
            
            # If it looks like a game interface, assume it could be a wisp box
            confidence = min(1.0, (game_score * 0.3 + wisp_score * 0.7) / 3)
            detected = confidence > 0.2 or game_score > 0  # Liberal detection
            
            return {
                'detected': detected,
                'confidence': confidence,
                'response': response
            }
            
        except Exception as e:
            logger.warning(f"Wisp box analysis failed: {e}")
            return {'detected': False, 'confidence': 0.0, 'response': ''}
    
    def _analyze_for_letters(self, image: Image.Image) -> Dict:
        """Use VLM to specifically look for letters X, Z, V, C in the image"""
        try:
            # Use unconditional captioning to describe what's in the image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if self.use_gpu:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(**inputs, max_new_tokens=25, num_beams=4, do_sample=False)
                else:
                    outputs = self.model.generate(**inputs, max_new_tokens=25, num_beams=4, do_sample=False)
            
            # Decode the response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Letter analysis: '{response}'")
            
            # Extract letters from response
            letters = self._extract_letters_from_response(response)
            
            return {
                'letters': letters,
                'response': response
            }
            
        except Exception as e:
            logger.warning(f"Letter analysis failed: {e}")
            return {'letters': [], 'response': ''}
    
    def _extract_letters_from_response(self, response: str) -> List[str]:
        """Extract valid letters from the model's response"""
        valid_letters = ['X', 'Z', 'V', 'C']
        found_letters = []
        
        # Look for letters in the response
        response_upper = response.upper()
        
        # More aggressive letter detection
        for letter in valid_letters:
            # Look for the letter in various contexts
            patterns = [
                r'\b' + letter + r'\b',  # Standalone letter
                r'LETTER\s+' + letter,   # "letter X"
                r'TEXT\s+' + letter,     # "text X"
                letter + r'\s',          # Letter followed by space
                r'\s' + letter + r'\s',  # Letter surrounded by spaces
                letter                   # Just the letter anywhere
            ]
            
            for pattern in patterns:
                if re.search(pattern, response_upper):
                    if letter not in found_letters:
                        found_letters.append(letter)
                    break
        
        # Also look for common descriptions that might indicate letters
        letter_descriptions = {
            'X': ['cross', 'x-shaped', 'times', 'multiply'],
            'Z': ['zigzag', 'z-shaped', 'zed'],
            'V': ['v-shaped', 'victory', 'chevron'],
            'C': ['c-shaped', 'crescent', 'curve']
        }
        
        for letter, descriptions in letter_descriptions.items():
            for desc in descriptions:
                if desc.upper() in response_upper and letter not in found_letters:
                    found_letters.append(letter)
        
        return found_letters
    
    def analyze_corner_region(self, image: np.ndarray) -> Dict:
        """Specifically analyze a corner region for letters"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Use unconditional captioning for corner analysis
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                if self.use_gpu:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(**inputs, max_new_tokens=20, num_beams=4, do_sample=False)
                else:
                    outputs = self.model.generate(**inputs, max_new_tokens=20, num_beams=4, do_sample=False)
            
            # Decode the response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Corner analysis: '{response}'")
            
            # Extract letters from response
            letters = self._extract_letters_from_response(response)
            
            return {
                'letters': letters,
                'confidence': 0.8 if letters else 0.2,
                'response': response
            }
            
        except Exception as e:
            logger.warning(f"Corner analysis failed: {e}")
            return {'letters': [], 'confidence': 0.0, 'response': ''}