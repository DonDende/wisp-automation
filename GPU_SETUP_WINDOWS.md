# GPU Setup for Windows

## 🚀 Enable GPU Acceleration on Windows

### Prerequisites
- NVIDIA GPU (GTX 1060 or newer recommended)
- Windows 10/11

### Installation Steps

1. **Install NVIDIA Drivers**
   - Download latest drivers from [NVIDIA website](https://www.nvidia.com/drivers/)
   - Install and restart your computer

2. **Install CUDA Toolkit**
   ```bash
   # Download CUDA 11.8 or 12.x from NVIDIA
   # https://developer.nvidia.com/cuda-downloads
   ```

3. **Install GPU-enabled PyTorch**
   ```bash
   # Uninstall CPU version first
   pip uninstall torch torchvision torchaudio
   
   # Install GPU version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify GPU Detection**
   ```python
   import torch
   print("CUDA available:", torch.cuda.is_available())
   print("GPU name:", torch.cuda.get_device_name(0))
   ```

### Expected Performance
- **CPU only**: ~2-5 seconds per detection
- **With GPU**: ~0.5-1 second per detection

### Troubleshooting
- If GPU not detected, check CUDA installation
- Ensure PyTorch CUDA version matches your CUDA toolkit
- Restart after driver installation