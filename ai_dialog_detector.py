#!/usr/bin/env python3
"""
AI-Powered Dialog Detection for Wisp Summoning
Uses Hugging Face vision-language models to detect and read dialog boxes
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIDialogDetector:
    """AI-powered dialog detection using Hugging Face vision-language models"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """Initialize the AI model for dialog detection"""
        self.model_name = model_name
        logger.info(f"Loading AI model: {model_name}")
        
        try:
            # Use BLIP for visual question answering
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def detect_dialog_and_letters(self, image: np.ndarray, timestamp: float) -> Dict:
        """Use AI to detect dialog box and extract keystroke letters"""
        # Convert OpenCV image to PIL
        if len(image.shape) == 3:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = Image.fromarray(image)
        
        # Resize image if too large (for efficiency)
        if image_pil.size[0] > 800 or image_pil.size[1] > 600:
            image_pil = image_pil.resize((800, 600), Image.Resampling.LANCZOS)
        
        results = {
            'timestamp': timestamp,
            'dialog_detected': False,
            'letters': [],
            'ai_description': '',
            'confidence': 0.0
        }
        
        try:
            # Ask the AI specific questions about the image
            questions = [
                "What letters or keys are shown in this dialog box?",
                "Are there any single letters like X, Z, or V visible in this image?",
                "What text appears in the dialog or popup window?",
                "Describe any keystroke indicators or button prompts you see."
            ]
            
            best_response = ""
            best_confidence = 0.0
            
            for question in questions:
                try:
                    # Process the image with the question
                    inputs = self.processor(image_pil, question, return_tensors="pt").to(self.device)
                    
                    # Generate response
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, max_length=50, num_beams=3)
                    
                    # Decode the response
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean up the response (remove the question if it's repeated)
                    if question.lower() in response.lower():
                        response = response.replace(question, "").strip()
                    
                    logger.info(f"AI Response for '{question}': {response}")
                    
                    # Check if this response contains useful information
                    confidence = self._evaluate_response_quality(response)
                    if confidence > best_confidence:
                        best_response = response
                        best_confidence = confidence
                    
                    # Look for specific letters in the response
                    detected_letters = self._extract_letters_from_response(response)
                    if detected_letters:
                        results['letters'].extend(detected_letters)
                        results['dialog_detected'] = True
                    
                except Exception as e:
                    logger.warning(f"Error processing question '{question}': {e}")
                    continue
            
            results['ai_description'] = best_response
            results['confidence'] = best_confidence
            
            # If we found letters, mark dialog as detected
            if results['letters']:
                results['dialog_detected'] = True
                # Remove duplicates while preserving order
                seen = set()
                unique_letters = []
                for letter in results['letters']:
                    if letter['letter'] not in seen:
                        unique_letters.append(letter)
                        seen.add(letter['letter'])
                results['letters'] = unique_letters
            
        except Exception as e:
            logger.error(f"Error in AI dialog detection: {e}")
            results['ai_description'] = f"Error: {str(e)}"
        
        return results
    
    def _evaluate_response_quality(self, response: str) -> float:
        """Evaluate how useful an AI response is for our task"""
        if not response or len(response.strip()) < 3:
            return 0.0
        
        confidence = 0.1  # Base confidence
        
        # Higher confidence for responses mentioning specific letters
        target_letters = ['X', 'Z', 'V', 'x', 'z', 'v']
        for letter in target_letters:
            if letter in response:
                confidence += 0.3
        
        # Higher confidence for dialog/UI related terms
        ui_terms = ['dialog', 'box', 'button', 'key', 'press', 'letter', 'text', 'prompt']
        for term in ui_terms:
            if term.lower() in response.lower():
                confidence += 0.1
        
        # Lower confidence for generic descriptions
        generic_terms = ['image', 'picture', 'scene', 'background']
        for term in generic_terms:
            if term.lower() in response.lower() and len(response.split()) < 5:
                confidence -= 0.1
        
        return min(confidence, 1.0)
    
    def _extract_letters_from_response(self, response: str) -> List[Dict]:
        """Extract keystroke letters from AI response"""
        letters = []
        target_letters = ['X', 'Z', 'V']
        
        # Look for explicit mentions of target letters
        for letter in target_letters:
            if letter in response.upper():
                letters.append({
                    'letter': letter,
                    'confidence': 0.8,
                    'source': 'ai_detection'
                })
        
        # Also check for lowercase versions
        for letter in ['x', 'z', 'v']:
            if letter in response and letter.upper() not in [l['letter'] for l in letters]:
                letters.append({
                    'letter': letter.upper(),
                    'confidence': 0.7,
                    'source': 'ai_detection'
                })
        
        return letters

class WispVideoAIAnalyzer:
    """Analyzes wisp video using AI-powered dialog detection"""
    
    def __init__(self, video_path: str, output_dir: str = "ai_analysis_output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize AI detector
        self.ai_detector = AIDialogDetector()
        
        # Focus on the dialog region (top-left area where dialogs appear)
        self.dialog_region = (50, 50, 700, 300)  # x, y, width, height
        self.target_fps = 5  # Analyze fewer frames for efficiency with AI
        
    def extract_key_frames(self, start_time: float = 15.0, end_time: float = 20.0) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from the critical time period where dialogs appear"""
        logger.info(f"Extracting key frames from {start_time}s to {end_time}s")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
        
        # Calculate frame numbers for the time range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_interval = int(fps / self.target_fps)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = current_frame / fps
            
            # Extract dialog region
            x, y, w, h = self.dialog_region
            dialog_roi = frame[y:y+h, x:x+w]
            
            frames.append((timestamp, dialog_roi))
            logger.info(f"Extracted key frame at {timestamp:.2f}s")
            
            # Skip to next frame
            current_frame += frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} key frames")
        return frames
    
    def analyze_video_with_ai(self) -> Dict:
        """Analyze video using AI-powered dialog detection"""
        logger.info("Starting AI-powered video analysis")
        
        # Extract key frames from the critical time period
        frames = self.extract_key_frames(15.0, 22.0)  # Focus on 15-22 second range
        
        analysis_results = []
        detected_sequences = []
        
        for timestamp, frame in frames:
            logger.info(f"Analyzing frame at {timestamp:.2f}s with AI...")
            
            result = self.ai_detector.detect_dialog_and_letters(frame, timestamp)
            analysis_results.append(result)
            
            if result['dialog_detected'] or result['letters']:
                detected_sequences.append(result)
                logger.info(f"AI detected at {timestamp:.2f}s: {[l['letter'] for l in result['letters']]} - {result['ai_description']}")
                
                # Save debug image
                debug_path = self.output_dir / f"ai_dialog_frame_{timestamp:.2f}s.png"
                cv2.imwrite(str(debug_path), frame)
                result['debug_image'] = str(debug_path)
        
        # Generate comprehensive analysis
        analysis = {
            'video_path': self.video_path,
            'analysis_method': 'ai_powered',
            'model_used': self.ai_detector.model_name,
            'total_frames_analyzed': len(frames),
            'detected_sequences': detected_sequences,
            'keystroke_patterns': self._analyze_ai_patterns(detected_sequences),
            'timing_data': self._analyze_timing(detected_sequences),
            'all_results': analysis_results
        }
        
        # Save analysis results
        output_file = self.output_dir / "ai_wisp_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"AI analysis complete. Results saved to {output_file}")
        return analysis
    
    def _analyze_ai_patterns(self, sequences: List[Dict]) -> Dict:
        """Analyze patterns in AI-detected keystrokes"""
        patterns = {
            'detected_letters': [],
            'letter_frequencies': {},
            'ai_descriptions': [],
            'confidence_scores': []
        }
        
        for seq in sequences:
            patterns['ai_descriptions'].append(seq['ai_description'])
            patterns['confidence_scores'].append(seq['confidence'])
            
            for letter_data in seq['letters']:
                letter = letter_data['letter']
                patterns['detected_letters'].append(letter)
                patterns['letter_frequencies'][letter] = patterns['letter_frequencies'].get(letter, 0) + 1
        
        return patterns
    
    def _analyze_timing(self, sequences: List[Dict]) -> Dict:
        """Analyze timing between AI detections"""
        if len(sequences) < 2:
            return {'intervals': [], 'average_interval': 0}
        
        intervals = []
        for i in range(1, len(sequences)):
            interval = sequences[i]['timestamp'] - sequences[i-1]['timestamp']
            intervals.append(interval)
        
        return {
            'intervals': intervals,
            'average_interval': sum(intervals) / len(intervals) if intervals else 0,
            'recommended_delay': 0.1  # 100ms delay as requested
        }

def main():
    """Main function to run AI-powered video analysis"""
    video_path = "/workspace/wisp summon example.mp4"
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Create AI analyzer and run analysis
    analyzer = WispVideoAIAnalyzer(video_path)
    results = analyzer.analyze_video_with_ai()
    
    # Print summary
    print("\n" + "="*60)
    print("AI-POWERED WISP VIDEO ANALYSIS SUMMARY")
    print("="*60)
    print(f"Model used: {results['model_used']}")
    print(f"Total frames analyzed: {results['total_frames_analyzed']}")
    print(f"Dialog sequences found: {len(results['detected_sequences'])}")
    
    if results['keystroke_patterns']['detected_letters']:
        print(f"AI detected letters: {results['keystroke_patterns']['detected_letters']}")
        print(f"Letter frequencies: {results['keystroke_patterns']['letter_frequencies']}")
        print(f"Average confidence: {np.mean(results['keystroke_patterns']['confidence_scores']):.2f}")
    
    if results['timing_data']['intervals']:
        print(f"Average interval between detections: {results['timing_data']['average_interval']:.2f}s")
        print(f"Recommended keystroke delay: {results['timing_data']['recommended_delay']}s")
    
    print(f"\nDetailed results saved to: ai_analysis_output/ai_wisp_analysis.json")
    
    # Show some AI descriptions
    if results['detected_sequences']:
        print("\nAI Descriptions:")
        for i, seq in enumerate(results['detected_sequences'][:5]):  # Show first 5
            print(f"  {seq['timestamp']:.2f}s: {seq['ai_description']}")

if __name__ == "__main__":
    main()