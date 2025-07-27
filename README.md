# Wisp Automation Enhanced

An advanced automation script for wisp summoning in games, featuring sophisticated dialog detection and UI element recognition.

## Overview

This project automates the wisp summoning process by:
1. **Detecting inventory state** - Recognizes when inventory is open/closed
2. **Confirming wisp presence** - Identifies metal wisp items in inventory
3. **Detecting dialog boxes** - Advanced multi-modal dialog detection system
4. **Executing keystrokes** - Automated keystroke sequences for summoning

## Key Features

### Advanced Dialog Detection
- **Multi-modal detection**: Combines region analysis, text recognition, and visual signatures
- **False positive filtering**: Excludes UI elements that aren't actual dialogs
- **Actual dialog focus**: Targets the correct top-left area where dialogs appear
- **Priority detection system**: Uses most accurate methods first with fallbacks

### Comprehensive Data Analysis
- **Video-based training**: Extracted patterns from actual gameplay footage
- **UI exclusion zones**: Prevents false detection of game interface elements
- **Timing intelligence**: Focuses detection during optimal timeframes
- **Adaptive thresholds**: Optimized confidence levels for different detection types

### Robust Error Handling
- **Graceful degradation**: Falls back to simpler methods if advanced detection fails
- **Comprehensive logging**: Detailed debug information for troubleshooting
- **Progress tracking**: Real-time feedback on automation progress
- **Timeout protection**: Prevents infinite waiting loops

## Files

### Core Scripts
- `wisp_automation_enhanced.py` - Main automation script with advanced detection
- `debug_dialog_detection.py` - Debug tool for analyzing dialog detection
- `requirements.txt` - Python dependencies

### Data Files
- `actual_dialog_data.json` - Dialog patterns from correct video timeframe
- `dialog_enhanced_data.json` - Enhanced dialog detection patterns
- `ui_exclusion_data.json` - UI elements to exclude from dialog detection
- `wisp_comprehensive_data.json` - Comprehensive video analysis data

### Utilities
- `run_automation.bat` - Windows batch file to run automation
- `analyze_video.bat` - Batch file for video analysis
- `analyze_wisp_video.py` - Video analysis utility

### Debug Images
- `debug_region_*.png` - Screenshots of detection regions for debugging

## Usage

### Basic Usage
```bash
python wisp_automation_enhanced.py
```

### Debug Mode
```bash
python wisp_automation_enhanced.py --debug
```

### Syntax Test
```bash
python wisp_automation_enhanced.py --test-syntax
```

### Debug Dialog Detection
```bash
python debug_dialog_detection.py
```

## Configuration

The script uses multiple data sources for optimal detection:

1. **Actual Dialog Data** (highest priority)
   - Based on analysis of 18-25 second timeframe
   - Focuses on top-left area [186.5, 146.375]
   - Optimized for real dialog characteristics

2. **Enhanced Dialog Data** (fallback)
   - Comprehensive video analysis patterns
   - Multiple detection regions and methods
   - Advanced visual signature recognition

3. **UI Exclusion Data** (false positive prevention)
   - 18 exclusion zones for UI elements
   - Text patterns to avoid (equipment, items, etc.)
   - Visual signatures of game interface

## Detection Process

1. **Inventory Detection** - Confirms inventory is open
2. **Wisp Confirmation** - Verifies metal wisp presence
3. **Dialog Detection** - Multi-stage dialog detection:
   - Actual dialog regions (top-left focused)
   - Single letter detection (X, Z, V keys)
   - Enhanced pattern matching
   - Comprehensive fallback detection
4. **False Positive Filtering** - Removes UI element detections
5. **Keystroke Execution** - Automated key sequence

## Troubleshooting

### Common Issues

**Dialog not detected:**
- Check if looking in correct region (top-left area)
- Verify timing (dialogs appear after keystroke sequence)
- Review debug images for actual dialog location

**False positives:**
- UI exclusion system should filter these automatically
- Check exclusion zones if UI elements are detected as dialogs

**Timeout errors:**
- Increase timeout values in configuration
- Verify game state matches expected conditions

### Debug Tools

**Debug Dialog Detection:**
```bash
python debug_dialog_detection.py
```
Generates debug images showing detection regions.

**Verbose Logging:**
Enable debug logging to see detailed detection attempts and confidence scores.

## Development History

This project went through several major iterations:

1. **Basic Detection** - Simple shape and color matching
2. **Enhanced Detection** - Video analysis and pattern extraction
3. **False Positive Filtering** - UI exclusion system
4. **Actual Dialog Focus** - Correct timeframe and region analysis

Each iteration improved accuracy and reduced false positives while maintaining detection sensitivity for actual dialogs.

## Dependencies

- Python 3.7+
- OpenCV (cv2)
- PyTesseract (OCR)
- PyAutoGUI (automation)
- NumPy (numerical processing)
- Pillow (image processing)

Install with:
```bash
pip install -r requirements.txt
```

## Future Improvements

- Dynamic threshold adjustment based on success rates
- Machine learning integration for pattern recognition
- Real-time adaptation to different game UI themes
- Multi-game compatibility framework

## Contributing

When modifying the detection system:
1. Test with actual gameplay footage
2. Verify false positive filtering still works
3. Update documentation for any new detection methods
4. Include debug logging for troubleshooting

## License

This project is for educational and personal use. Ensure compliance with game terms of service when using automation tools.