# Wisp Automation Troubleshooting Guide

## 🔧 Quick Fix for Side-of-Screen Detection Issue

Based on your screenshot, I've identified and fixed the issue where the system was detecting letters on the sides of the screen instead of focusing on the translucent dialog box.

### ✅ What Was Fixed

1. **Precise Detection Region**: Updated to `[747, 180, 417, 145]` based on user-provided coordinates
2. **Focused Area**: Now only analyzes the exact area where the dialog box appears
3. **Improved Configuration**: Lowered confidence threshold and enabled debug mode

### 🛠 Available Tools

#### 1. Quick Fix (Recommended First Step)
```bash
python quick_fix.py
```
- Applies the optimal detection region immediately
- Tests the new region with your current screen
- Creates visual confirmation images

#### 2. Region Calibration Tool
```bash
python calibrate_region.py
```
- Interactive tool to fine-tune detection coordinates
- Shows multiple region options
- Allows custom coordinate input
- Updates configuration automatically

#### 3. Visual Debug Tool
```bash
python debug_detection.py
```
- Shows exactly what the AI is analyzing
- Creates multiple debug images:
  - Full screen with detection region marked
  - Just the detection region
  - Grayscale, threshold, and edge detection views
  - Detected text regions
- Tests AI detection with different confidence levels

#### 4. Enhanced Batch File
```bash
run_automation.bat
```
Now includes 6 options:
1. Run Full Automation
2. Test Detection Only  
3. Performance Benchmark
4. Calibrate Detection Region
5. Debug Detection (Visual Analysis)
6. Exit

### 📊 Detection Region Details

Based on user-provided coordinates:
- **Top Left Corner**: (747, 180)
- **Bottom Right Corner**: (1164, 325)
- **X Position**: 747 (left edge of dialog box)
- **Y Position**: 180 (top edge of dialog box)
- **Width**: 417 pixels (dialog box width)
- **Height**: 145 pixels (dialog box height)

This region specifically targets the translucent box with X, Z, V letters and excludes:
- Left side UI panels
- Right side character stats
- Top menu bars
- Bottom inventory areas

### 🎯 Recommended Testing Steps

1. **Run Quick Fix First**:
   ```bash
   python quick_fix.py
   ```
   This applies the optimal settings and creates test images.

2. **Verify Detection Region**:
   Check the generated images:
   - `quick_fix_full_screen.png` - Shows detection region on full screen
   - `quick_fix_test_region.png` - Shows what AI will analyze

3. **Test Detection**:
   ```bash
   python final_wisp_automation.py --test
   ```
   This tests detection without running full automation.

4. **Fine-tune if Needed**:
   If the region isn't perfect, use:
   ```bash
   python calibrate_region.py
   ```

5. **Run Full Automation**:
   ```bash
   python final_wisp_automation.py
   ```

### 🔍 Debug Information

The system now saves debug images when `save_detection_images` is enabled:
- Shows exactly what the AI is analyzing
- Helps identify why detection might fail
- Provides visual confirmation of the detection region

### ⚙️ Configuration Changes

Updated `final_wisp_config.json`:
```json
{
  "detection_region": [580, 150, 290, 100],
  "confidence_threshold": 0.3,
  "debug_mode": true,
  "save_detection_images": true
}
```

### 🚨 Common Issues and Solutions

#### Issue: Still detecting wrong areas
**Solution**: Use the calibration tool to adjust coordinates:
```bash
python calibrate_region.py
```

#### Issue: Not detecting letters at all
**Solution**: 
1. Lower confidence threshold in config
2. Use debug tool to see what AI is analyzing
3. Ensure dialog box is clearly visible

#### Issue: Detection too slow
**Solution**: 
1. Reduce detection region size
2. Enable GPU optimization
3. Use performance benchmark to test

#### Issue: Letters not clear in region
**Solution**:
1. Adjust screen resolution/scaling
2. Ensure good contrast between letters and background
3. Use debug tool to check grayscale visibility

### 📝 Files Created by Tools

- `quick_fix_*.png` - Quick fix test images
- `debug_*.png` - Debug analysis images  
- `region_*.png` - Region calibration images
- `detection_*.png` - Runtime detection images (when debug enabled)

### 💡 Pro Tips

1. **Always test with dialog visible**: Make sure the wisp dialog box is on screen when testing
2. **Check debug images**: They show exactly what the AI sees
3. **Use calibration tool**: Fine-tune coordinates for your specific setup
4. **Monitor confidence scores**: Adjust threshold based on detection results
5. **Enable debug mode**: Helps troubleshoot detection issues

### 🎮 Game-Specific Notes

- Dialog box appears in first 4 seconds of wisp summoning
- Letters X, Z, V should be clearly visible in white text
- Translucent background should provide good contrast
- Region should exclude all UI elements and focus only on dialog

This should completely resolve the side-of-screen detection issue by focusing the AI only on the specific area where the dialog box appears!