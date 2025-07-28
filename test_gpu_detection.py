#!/usr/bin/env python3
"""
GPU Detection Test Script
Run this on your Windows system to test GPU detection
"""

import torch
import platform
from pathlib import Path

def test_gpu_detection():
    """Test GPU detection logic"""
    print("🔍 GPU Detection Test")
    print("=" * 50)
    
    print(f"Platform: {platform.system()}")
    print(f"Current directory: {Path.cwd()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Check if we're in development environment
    is_dev_env = platform.system() == 'Linux' and 'workspace' in str(Path.cwd())
    print(f"Development environment detected: {is_dev_env}")
    print()
    
    if is_dev_env:
        print("💻 Development environment - skipping GPU detection")
        device = torch.device('cpu')
    else:
        print("🔍 Checking for available hardware...")
        
        # CUDA detection
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA devices found: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            device = torch.device('cuda')
            print(f"🚀 Selected device: {device}")
        
        # MPS detection (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple Silicon GPU (MPS) detected")
            device = torch.device('mps')
        
        # CPU fallback
        else:
            print("💻 No compatible GPU found - using CPU")
            if platform.system() == 'Windows':
                print("   💡 To enable GPU acceleration on Windows:")
                print("   1. Install NVIDIA drivers")
                print("   2. Install CUDA toolkit")
                print("   3. Install GPU-enabled PyTorch:")
                print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            device = torch.device('cpu')
    
    print()
    print(f"Final selected device: {device}")
    
    # Test tensor creation
    try:
        test_tensor = torch.randn(10, 10).to(device)
        print(f"✅ Successfully created tensor on {device}")
        print(f"Tensor device: {test_tensor.device}")
    except Exception as e:
        print(f"❌ Failed to create tensor on {device}: {e}")

if __name__ == "__main__":
    test_gpu_detection()