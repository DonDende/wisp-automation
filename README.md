# AI-Powered Wisp Automation System

A comprehensive automation system for wisp summoning with advanced AI-powered translucent box detection, GPU optimization, and precise keystroke timing.

## 🚀 Features

### Core Capabilities
- **AI-Powered Detection**: Uses Hugging Face vision models to detect translucent boxes with letters
- **GPU Optimization**: Automatically detects and uses GPU when available, falls back to optimized CPU processing
- **Precise Timing**: 80-100ms keystroke delays as specified
- **Multi-Method Detection**: Combines AI classification, template matching, and computer vision
- **Real-time Analysis**: Processes video frames at optimized speeds for live detection

### Advanced Features
- **Automatic Hardware Detection**: Supports CUDA, Apple Silicon (MPS), and optimized CPU processing
- **Performance Benchmarking**: Built-in performance testing and statistics
- **Session Statistics**: Comprehensive tracking of success rates, timing, and detection accuracy
- **Configuration System**: Fully customizable parameters and settings
- **Debug Mode**: Save detection images and detailed logs for troubleshooting

## 📁 Key Files

### Main Scripts
- `final_wisp_automation.py` - Complete automation system with AI detection
- `gpu_optimized_detector.py` - GPU-optimized AI detector with hardware auto-detection
- `wisp_ai_detector.py` - AI-powered detection system for translucent boxes
- `train_wisp_detector.py` - Custom model training system

### Analysis Tools
- `video_analyzer.py` - Video analysis at 15fps as requested
- `test_detection.py` - Testing and validation tools
- `analyze_wisp_video.py` - Comprehensive video analysis

### Configuration
- `final_wisp_config.json` - Main configuration file
- `wisp_config.json` - Additional configuration options
- `requirements.txt` - Python dependencies

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)

### Setup
```bash
# Clone the repository
git clone https://github.com/DonDende/wisp-automation.git
cd wisp-automation

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for full functionality
pip install torch torchvision transformers
pip install pyautogui opencv-python pillow
pip install easyocr  # For OCR capabilities
```

## 🎮 Usage

### Quick Start - Test Detection
```bash
python final_wisp_automation.py --test
```

### Run Full Automation
```bash
python final_wisp_automation.py
```

### Run with Specific Cycles
```bash
python final_wisp_automation.py --cycles 10
```

### Performance Benchmark
```bash
python final_wisp_automation.py --benchmark
```

### GPU-Optimized Detection Only
```bash
python gpu_optimized_detector.py
```

## ⚙️ Configuration

### Main Configuration (`final_wisp_config.json`)
```json
{
  "detection_timeout": 8.0,
  "keystroke_delay_min": 0.08,
  "keystroke_delay_max": 0.10,
  "confidence_threshold": 0.4,
  "save_detection_images": true,
  "valid_letters": ["X", "Z", "V"]
}
```

### Key Parameters
- **keystroke_delay_min/max**: 80-100ms delay range as requested
- **confidence_threshold**: AI detection confidence threshold
- **detection_timeout**: Maximum time to wait for translucent box
- **save_detection_images**: Save debug images for analysis

## 🧠 AI Model Details

### Supported Models
- **Primary**: Microsoft ResNet-50 (pre-trained vision model)
- **Fallback**: Template matching with computer vision
- **Custom**: Trainable models for specific game scenarios

### Hardware Support
- **CUDA GPUs**: Automatic detection and optimization
- **Apple Silicon**: MPS backend support
- **CPU**: Multi-threaded optimization with Intel MKL

### Performance
- **GPU**: ~15-20 FPS detection rate
- **CPU**: ~3-5 FPS detection rate
- **Memory**: ~2GB GPU / ~1GB CPU

## 📊 Analysis Results

The system analyzes the first 4 seconds of video (where translucent boxes appear) and provides:

- **Detection Accuracy**: AI confidence scores and letter recognition
- **Timing Analysis**: Precise timing data for keystroke execution
- **Performance Metrics**: Processing speed and hardware utilization
- **Session Statistics**: Success rates, failure analysis, and optimization suggestions

## 🔧 Advanced Features

### Custom Model Training
```bash
python train_wisp_detector.py
```
Trains a custom model using:
- Real video frames from the first 4 seconds
- Synthetic training data
- Transfer learning from pre-trained models

### Video Analysis
```bash
python video_analyzer.py
```
Analyzes video at 15fps as requested and extracts:
- Frame-by-frame detection results
- Timing patterns
- Visual signatures of translucent boxes

### Debug Mode
Enable debug mode in configuration to:
- Save detection images
- Log detailed AI responses
- Track performance metrics
- Generate analysis reports

## 📈 Performance Optimization

### For GPU Users
- Automatic mixed precision (FP16)
- CUDA memory optimization
- Batch processing when possible

### For CPU Users
- Multi-threading optimization
- Intel MKL acceleration
- Memory-efficient processing

### General Optimizations
- Strategic region detection
- Multi-scale template matching
- Confidence-based early termination

## 🎯 Detection Strategy

The system uses a multi-layered approach:

1. **AI Classification**: Hugging Face vision models analyze screen regions
2. **Template Matching**: Computer vision techniques for letter recognition
3. **Color Analysis**: Detects translucent overlay characteristics
4. **Region Prioritization**: Focuses on areas where dialogs typically appear

## 📋 Requirements

### Minimum System Requirements
- Python 3.8+
- 4GB RAM
- 1GB free disk space

### Recommended for GPU Acceleration
- CUDA-compatible GPU with 4GB+ VRAM
- 8GB+ system RAM
- SSD storage for faster model loading

### Dependencies
```
torch>=1.9.0
transformers>=4.20.0
opencv-python>=4.5.0
pillow>=8.0.0
numpy>=1.20.0
pyautogui>=0.9.50
easyocr>=1.6.0
```

## 🐛 Troubleshooting

### Common Issues
1. **No GPU detected**: System will automatically use CPU optimization
2. **Detection timeout**: Adjust `detection_timeout` in configuration
3. **Low confidence**: Lower `confidence_threshold` or improve lighting
4. **Slow performance**: Enable GPU acceleration or reduce detection frequency

### Debug Tools
- Use `--test` flag for detection-only testing
- Enable `save_detection_images` to see what the AI is analyzing
- Check session statistics for performance insights

## 📝 License

This project is for educational and research purposes. Please ensure compliance with game terms of service when using automation tools.

## 🤝 Contributing

Contributions welcome! Please focus on:
- Improving AI detection accuracy
- Adding support for new hardware
- Optimizing performance
- Enhancing user experience

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review configuration options
3. Enable debug mode for detailed logs
4. Create an issue with system specifications and logs