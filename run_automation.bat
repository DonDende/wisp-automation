@echo off
echo ============================================
echo 🤖 AI-POWERED WISP AUTOMATION SYSTEM
echo ============================================
echo.
echo AI-Powered Translucent Box Detection
echo GPU-Optimized with 80-100ms Keystroke Timing
echo.
echo SAFETY REMINDERS:
echo - Move mouse to top-left corner for emergency stop
echo - Press Ctrl+C to stop the script
echo - Make sure your game is ready
echo.
echo AI FEATURES:
echo - Hugging Face vision models for detection
echo - GPU acceleration (auto-detects hardware)
echo - Multi-method detection (AI + Template + CV)
echo - 80-100ms precise keystroke delays
echo - Real-time performance monitoring
echo - Session statistics and logging
echo.
echo Make sure:
echo - Your game is open and focused
echo - Translucent box with letters appears in first 4 seconds
echo - Letters X, Z, V are clearly visible
echo.
echo Choose your option:
echo [1] Run Full Automation
echo [2] Test Detection Only
echo [3] Performance Benchmark
echo [4] Calibrate Detection Region
echo [5] Debug Detection (Visual Analysis)
echo [6] Exit
echo.
:start
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto run_automation
if "%choice%"=="2" goto test_detection
if "%choice%"=="3" goto benchmark
if "%choice%"=="4" goto calibrate_region
if "%choice%"=="5" goto debug_detection
if "%choice%"=="6" goto exit
goto invalid_choice

:run_automation
echo.
echo Starting AI-powered automation in 3 seconds...
timeout /t 3 /nobreak >nul
python launcher.py
goto end

:test_detection
echo.
echo Running detection test...
python final_wisp_automation.py --test
goto end

:benchmark
echo.
echo Running performance benchmark...
python final_wisp_automation.py --benchmark
goto end

:calibrate_region
echo.
echo Running region calibration tool...
python calibrate_region.py
goto end

:debug_detection
echo.
echo Running detection debug tool...
python debug_detection.py
goto end

:invalid_choice
echo Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.
pause
goto start

:exit
echo Exiting...
goto end

:end
echo.
echo ============================================
echo AI Automation completed. Check output above.
echo ============================================
pause