@echo off
title Magic Cloak Suite - @vishnuskandha
color 0A

:MAIN_MENU
cls
echo.
echo ===============================================
echo           MAGIC CLOAK SUITE 
echo ===============================================
echo    Created by: Vishnu Skandha (@vishnuskandha)
echo    GitHub: https://github.com/vishnuskandha
echo ===============================================
echo.
echo Choose your magical experience:
echo.
echo   [1]  Simple Invisible Cloak
echo       - Basic invisibility effect
echo       - Easy to use, no controls
echo       - Perfect for beginners
echo.
echo   [2]  Advanced Magic Cloak
echo       - Professional controls panel
echo       - Multiple color presets
echo       - Real-time adjustments
echo       - Color picker tool
echo.
echo   [3]  Setup/Install Dependencies
echo       - First-time setup
echo       - Install required packages
echo.
echo   [4]  Show System Information
echo       - Check Python & packages
echo.
echo   [0]  Exit
echo.
echo ===============================================
set /p choice="Enter your choice (0-4): "

if "%choice%"=="1" goto SIMPLE_CLOAK
if "%choice%"=="2" goto ADVANCED_CLOAK
if "%choice%"=="3" goto SETUP
if "%choice%"=="4" goto SYSTEM_INFO
if "%choice%"=="0" goto EXIT
echo [ERROR] Invalid choice! Please select 0-4.
timeout /t 2 /nobreak >nul
goto MAIN_MENU

:SIMPLE_CLOAK
cls
echo.
echo ===============================================
echo         SIMPLE INVISIBLE CLOAK
echo ===============================================
echo.

REM Change to script directory
cd /d "%~dp0"

call :CHECK_DEPENDENCIES
if errorlevel 1 goto MAIN_MENU

echo.
echo ===============================================
echo              IMPORTANT INSTRUCTIONS
echo ===============================================
echo  1. Make sure your webcam is connected
echo  2. Have a bright RED cloth/shirt ready
echo  3. Ensure good lighting in your room
echo  4. Press ESC to exit the program
echo ===============================================
echo.
echo [INFO] Starting Simple Invisible Cloak in 3 seconds...
timeout /t 3 /nobreak >nul

echo [INFO] Launching the magical experience...
echo.

REM Run the simple invisible cloak program
python invisible_cloak.py

echo.
echo ===============================================
echo [INFO] Program ended. Thanks for using!
echo ===============================================
echo.
pause
goto MAIN_MENU

:ADVANCED_CLOAK
cls
echo.
echo ===============================================
echo         ADVANCED MAGIC CLOAK
echo ===============================================
echo.

cd /d "%~dp0"

call :CHECK_DEPENDENCIES
if errorlevel 1 goto MAIN_MENU

echo.
echo ===============================================
echo            ADVANCED FEATURES GUIDE
echo ===============================================
echo  Controls Panel: Adjust colors in real-time
echo  Color Presets: 8 predefined colors (0-7)
echo  Color Picker: Click on objects to detect
echo  Background Options: Static/Adaptive modes
echo  Fine Tuning: HSV sliders for precision
echo.
echo  Keyboard Shortcuts:
echo  • q: Quit          • r: Reset background
echo  • c: Color picker  • m: Manual HSV mode
echo  • 0-7: Color presets
echo  • g: Grab clean background
echo ===============================================
echo.
echo [INFO] Starting Advanced Magic Cloak in 3 seconds...
timeout /t 3 /nobreak >nul

echo [INFO] Launching advanced controls...
echo.

REM Run the advanced magic cloak program
python magic.py

echo.
echo ===============================================
echo [INFO] Advanced session ended. Thanks for using!
echo ===============================================
echo.
pause
goto MAIN_MENU

:SETUP
cls
echo.
echo ===============================================
echo          SETUP & DEPENDENCIES
echo ===============================================
echo.

cd /d "%~dp0"

echo [INFO] This will install all required packages...
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
    goto MAIN_MENU
)

echo [SUCCESS] Python is installed: 
python --version
echo.

echo [STEP 2/3] Installing required packages...
echo [INFO] This may take a few minutes...
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
    goto MAIN_MENU
)

REM Verify NumPy  
python -c "import numpy; print(f'NumPy Version: {numpy.__version__}')" 2>nul
if errorlevel 1 (
    echo [ERROR] NumPy installation failed!
    pause
    goto MAIN_MENU
)

echo.
echo ===============================================
echo             SETUP COMPLETE!
echo ===============================================
echo.
echo [SUCCESS] All packages installed successfully!
echo [INFO] You can now use both cloak programs!
echo.
pause
goto MAIN_MENU

:SYSTEM_INFO
cls
echo.
echo ===============================================
echo          SYSTEM INFORMATION
echo ===============================================
echo.

cd /d "%~dp0"

echo [INFO] Python Installation:
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found!
) else (
    python -c "import sys; print(f'Python Path: {sys.executable}')"
)

echo.
echo [INFO] Package Status:

REM Check OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>nul
if errorlevel 1 (
    echo OpenCV: Not installed
)

REM Check NumPy
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>nul
if errorlevel 1 (
    echo NumPy: Not installed
)

echo.
echo [INFO] Available Files:
if exist "invisible_cloak.py" (
    echo Simple Cloak: invisible_cloak.py - Available
) else (
    echo Simple Cloak: File missing!
)

if exist "magic.py" (
    echo Advanced Cloak: magic.py - Available
) else (
    echo Advanced Cloak: File missing!
)

if exist "requirements.txt" (
    echo Requirements: requirements.txt - Available
) else (
    echo Requirements: File missing!
)

echo.
echo ===============================================
pause
goto MAIN_MENU

:CHECK_DEPENDENCIES
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo [INFO] Please install Python from https://python.org
    echo [INFO] Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [SUCCESS] Python is installed!
python --version

echo.
echo [INFO] Checking required packages...

REM Check if OpenCV is installed
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] OpenCV not found. Installing...
    echo [INFO] Installing opencv-python...
    pip install opencv-python
    if errorlevel 1 (
        echo [ERROR] Failed to install opencv-python!
        echo [INFO] Try running setup option or as administrator
        pause
        exit /b 1
    )
    echo [SUCCESS] OpenCV installed successfully!
) else (
    echo [SUCCESS] OpenCV is already installed!
)

REM Check if NumPy is installed
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NumPy not found. Installing...
    echo [INFO] Installing numpy...
    pip install numpy
    if errorlevel 1 (
        echo [ERROR] Failed to install numpy!
        echo [INFO] Try running setup option or as administrator
        pause
        exit /b 1
    )
    echo [SUCCESS] NumPy installed successfully!
) else (
    echo [SUCCESS] NumPy is already installed!
)

echo.
echo [SUCCESS] All dependencies are ready!
exit /b 0

:EXIT
cls
echo.
echo ===============================================
echo               GOODBYE!
echo ===============================================
echo.
echo Thanks for using Magic Cloak Suite!
echo.
echo Don't forget to:
echo   - Star the project on GitHub
echo   - Share with friends
echo   - Create amazing videos!
echo.
echo GitHub: https://github.com/vishnuskandha
echo.
echo ===============================================
timeout /t 3 /nobreak >nul
exit
