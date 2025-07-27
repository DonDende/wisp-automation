#!/usr/bin/env python3
"""
Train a custom AI model for wisp summoning translucent box detection
Downloads and fine-tunes a vision model from Hugging Face specifically for this task
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from PIL import Image
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WispDetectionDataset(Dataset):
    """Custom dataset for wisp summoning detection"""
    
    def __init__(self, image_paths: List[str], labels: List[int], processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        
        return inputs

class WispDetectorTrainer:
    """Trainer for wisp detection model"""
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """Initialize the trainer with a base vision model"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        logger.info(f"Loading model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=4,  # 0: no_wisp, 1: has_X, 2: has_Z, 3: has_V
            ignore_mismatched_sizes=True
        )
        
        self.model.to(self.device)
        
    def extract_training_data(self) -> Tuple[List[str], List[int]]:
        """Extract training data from the first 4 seconds of video analysis"""
        logger.info("Extracting training data from first 4 seconds...")
        
        # Get frames from 0-4 seconds (where the translucent box appears)
        analysis_dir = Path("analysis_output")
        training_images = []
        labels = []
        
        # Look for frames in the first 4 seconds
        for timestamp in np.arange(0.4, 4.1, 0.1):  # 0.4s to 4.0s
            frame_file = analysis_dir / f"dialog_frame_{timestamp:.2f}s.png"
            if frame_file.exists():
                training_images.append(str(frame_file))
                
                # For now, label all early frames as having wisp summoning box
                # In practice, you'd manually label these or use heuristics
                labels.append(1)  # Assume has_X for training
                logger.info(f"Added training image: {frame_file}")
        
        # Also extract frames directly from video for the first 4 seconds
        video_path = "/workspace/wisp summon example.mp4"
        if Path(video_path).exists():
            additional_images, additional_labels = self._extract_video_frames(video_path, 0, 4)
            training_images.extend(additional_images)
            labels.extend(additional_labels)
        
        logger.info(f"Total training samples: {len(training_images)}")
        return training_images, labels
    
    def _extract_video_frames(self, video_path: str, start_time: float, end_time: float) -> Tuple[List[str], List[int]]:
        """Extract frames from video for training"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return [], []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        images = []
        labels = []
        
        # Create training data directory
        training_dir = Path("training_data")
        training_dir.mkdir(exist_ok=True)
        
        for frame_num in range(start_frame, end_frame, int(fps / 10)):  # 10 FPS sampling
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                timestamp = frame_num / fps
                
                # Focus on the area where translucent box appears
                # Based on your description, it's likely in the center or upper area
                height, width = frame.shape[:2]
                
                # Extract different regions to find the translucent box
                regions = [
                    ("center", frame[height//4:3*height//4, width//4:3*width//4]),
                    ("upper", frame[0:height//2, width//4:3*width//4]),
                    ("full", frame)
                ]
                
                for region_name, region in regions:
                    filename = training_dir / f"frame_{timestamp:.2f}s_{region_name}.png"
                    cv2.imwrite(str(filename), region)
                    images.append(str(filename))
                    
                    # Label based on timestamp and region
                    # Early frames (0-2s) likely have the translucent box
                    if timestamp < 2.0:
                        labels.append(1)  # has_wisp_box
                    else:
                        labels.append(0)  # no_wisp_box
        
        cap.release()
        logger.info(f"Extracted {len(images)} frames from video")
        return images, labels
    
    def create_synthetic_training_data(self):
        """Create synthetic training data based on the screenshot description"""
        logger.info("Creating synthetic training data...")
        
        # Create synthetic images that simulate the translucent box with letters
        training_dir = Path("training_data")
        training_dir.mkdir(exist_ok=True)
        
        synthetic_images = []
        labels = []
        
        # Create different variations of translucent boxes with letters
        letters = ['X', 'Z', 'V']
        
        for i, letter in enumerate(letters):
            for variation in range(5):  # 5 variations per letter
                # Create a synthetic image
                img = np.zeros((300, 400, 3), dtype=np.uint8)
                
                # Add game-like background (dark with some texture)
                img[:] = (20, 25, 30)  # Dark background
                
                # Add translucent box (semi-transparent overlay)
                overlay = img.copy()
                cv2.rectangle(overlay, (100, 100), (300, 200), (50, 50, 50, 128), -1)
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                
                # Add the letter in the center
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.0
                color = (255, 255, 255)  # White text
                thickness = 3
                
                # Get text size and center it
                (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
                x = (img.shape[1] - text_width) // 2
                y = (img.shape[0] + text_height) // 2
                
                cv2.putText(img, letter, (x, y), font, font_scale, color, thickness)
                
                # Add some noise/variation
                noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Save synthetic image
                filename = training_dir / f"synthetic_{letter}_{variation}.png"
                cv2.imwrite(str(filename), img)
                synthetic_images.append(str(filename))
                labels.append(i + 1)  # 1=X, 2=Z, 3=V
        
        logger.info(f"Created {len(synthetic_images)} synthetic training images")
        return synthetic_images, labels
    
    def train_model(self, training_images: List[str], labels: List[int]):
        """Train the model on the prepared data"""
        logger.info("Starting model training...")
        
        # Create dataset
        dataset = WispDetectionDataset(training_images, labels, self.processor)
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./wisp_model",
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        logger.info("Training started...")
        trainer.train()
        
        # Save the trained model
        trainer.save_model("./wisp_detector_model")
        self.processor.save_pretrained("./wisp_detector_model")
        
        logger.info("Training completed and model saved!")
    
    def test_model(self, test_image_path: str):
        """Test the trained model on a sample image"""
        logger.info(f"Testing model on: {test_image_path}")
        
        # Load image
        image = Image.open(test_image_path).convert('RGB')
        
        # Process image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
        
        class_names = ["no_wisp", "has_X", "has_Z", "has_V"]
        confidence = predictions[0][predicted_class].item()
        
        logger.info(f"Prediction: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        return predicted_class, confidence

def main():
    """Main training function"""
    logger.info("Starting Wisp Detector Training...")
    
    # Initialize trainer
    trainer = WispDetectorTrainer()
    
    # Extract training data from video analysis
    real_images, real_labels = trainer.extract_training_data()
    
    # Create synthetic training data
    synthetic_images, synthetic_labels = trainer.create_synthetic_training_data()
    
    # Combine real and synthetic data
    all_images = real_images + synthetic_images
    all_labels = real_labels + synthetic_labels
    
    if len(all_images) == 0:
        logger.error("No training data found!")
        return
    
    logger.info(f"Total training samples: {len(all_images)}")
    
    # Train the model
    trainer.train_model(all_images, all_labels)
    
    # Test the model on a sample image
    if real_images:
        trainer.test_model(real_images[0])
    
    logger.info("Training complete! Model saved to ./wisp_detector_model")

if __name__ == "__main__":
    main()