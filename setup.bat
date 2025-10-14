@echo off
title Setup - Invisible Cloak Effect
color 0B

echo.
echo ===============================================
echo           INVISIBLE CLOAK SETUP 
echo ===============================================
echo    Created by: Vishnu Skandha (@vishnuskandha)
echo    GitHub: https://github.com/vishnuskandha
echo ===============================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo [INFO] This will set up everything needed for the Invisible Cloak Effect
echo.

REM Check if Python is installed
echo [STEP 1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo.
    echo [ACTION REQUIRED] Please:
    echo  1. Go to https://python.org/downloads/
    echo  2. Download Python 3.7 or higher
    echo  3. During installation, CHECK "Add Python to PATH"
    echo  4. Restart your computer after installation
    echo  5. Run this setup again
    echo.
    pause
    exit /b 1
)

echo [SUCCESS] Python is installed: 
python --version
echo.

echo [STEP 2/3] Installing required packages...
echo [INFO] This may take a few minutes depending on your internet speed...
echo.

REM Install all requirements
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Failed to install from requirements.txt
    echo [INFO] Trying individual installations...
    
    echo [INFO] Installing OpenCV...
    pip install opencv-python
    
    echo [INFO] Installing NumPy...
    pip install numpy
)

echo.
echo [STEP 3/3] Verifying installation...

REM Verify OpenCV
python -c "import cv2; print(f'OpenCV Version: {cv2.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERROR] OpenCV installation failed!
    pause
    exit /b 1
)

REM Verify NumPy  
python -c "import numpy; print(f'NumPy Version: {numpy.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERROR] NumPy installation failed!
    pause
    exit /b 1
)

echo.
echo ===============================================
echo               SETUP COMPLETE! 
echo ===============================================
echo.
echo [SUCCESS] All packages installed successfully!
echo.
echo [NEXT STEPS]
echo  1. Double-click 'run_invisible_cloak.bat' to start
echo  2. Or run: python invisible_cloak.py
echo.
echo [REQUIREMENTS FOR BEST RESULTS]
echo   Bright red cloth or shirt
echo   Good room lighting  
echo   Working webcam
echo   Plain background
echo.
echo ===============================================
echo   Ready to become invisible! 
echo ===============================================
echo.
pause