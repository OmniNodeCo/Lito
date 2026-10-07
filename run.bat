@echo off
TITLE SmartAI - AI From Scratch
echo ============================================
echo   SmartAI - Artificial Intelligence
echo   Built From Scratch - No Pretrained Models
echo ============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Install dependencies
echo Installing dependencies...
pip install numpy --quiet 2>nul

:: Check if model exists
if exist "checkpoints\best_model.npz" (
    echo Trained model found!
    echo.
    set /p CHOICE="Start chatting? (Y/N, or T to retrain): "
    if /i "%CHOICE%"=="T" goto train
    if /i "%CHOICE%"=="N" goto end
    goto chat
) else (
    echo No trained model found.
    echo.
    set /p CHOICE="Train model now? This takes a few minutes. (Y/N): "
    if /i "%CHOICE%"=="N" (
        echo Starting chat with knowledge base only...
        goto chat
    )
    goto train
)

:train
echo.
echo ============================================
echo   Training AI Model...
echo   This may take several minutes.
echo ============================================
echo.
python train.py --epochs 30 --batch_size 4 --seq_len 64
if errorlevel 1 (
    echo Training failed!
    pause
    exit /b 1
)
echo.
echo Training complete!
echo.

:chat
echo.
echo Starting SmartAI Chat...
echo.
python main.py --dir checkpoints
goto end

:end
echo.
echo Thanks for using SmartAI!
pause