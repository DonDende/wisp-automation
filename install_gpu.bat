@echo off
echo ========================================
echo   Wisp Automation GPU Setup for Windows
echo ========================================
echo.
echo This script will install GPU-enabled PyTorch and dependencies
echo.
pause

echo Installing GPU-enabled PyTorch...
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Testing GPU detection...
python test_gpu_detection.py

echo.
echo Setup complete! Press any key to exit.
pause