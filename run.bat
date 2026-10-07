@echo off
TITLE SmartAI - AI From Scratch
echo ============================================
echo   SmartAI - Artificial Intelligence
echo   Built From Scratch - No Pretrained Models
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)

pip install numpy --quiet 2>nul

if exist "checkpoints\best_model.npz" (
    echo Trained model found!
    echo.
    python main.py --dir checkpoints
) else (
    echo No local model found.
    echo.
    echo [1] Download latest trained model from GitHub
    echo [2] Train model locally
    echo [3] Chat using Knowledge Base only
    echo.
    set /p CHOICE="Select option (1/2/3, default 1): "
    if "%CHOICE%"=="2" (
        python train.py --epochs 30 --save_dir checkpoints
        python main.py --dir checkpoints
    ) else if "%CHOICE%"=="3" (
        python main.py --chat-only
    ) else (
        python main.py --download --dir checkpoints
    )
)
pause