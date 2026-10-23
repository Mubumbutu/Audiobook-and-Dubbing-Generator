@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "VENV_NAME=venv_audio8"
if not defined APP_ROOT set "APP_ROOT=%~dp0..\"
set "VENV_DIR=%APP_ROOT%venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
echo ============================================================
echo Audio8 TTS Preview 0.6B - Installation
echo ============================================================
echo.
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Remember to check "Add Python to PATH"!
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%V in ('py -3 --version 2^>^&1') do set PYVER=%%V
for /f "tokens=1,2 delims=." %%A in ("!PYVER!") do (
    set PY_MAJOR=%%A
    set PY_MINOR=%%B
)
if !PY_MAJOR! LSS 3 (
    echo [ERROR] Python 3.10+ is required. Found: !PYVER!
    echo Download Python 3.10+ from https://www.python.org/downloads/
    pause & exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo [ERROR] Python 3.10+ is required. Found: !PYVER!
    echo Download Python 3.10+ from https://www.python.org/downloads/
    pause & exit /b 1
)
echo [OK] Python !PYVER!
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed.
    echo Download from: https://git-scm.com/download/win
    pause & exit /b 1
)
for /f "tokens=3" %%V in ('git --version 2^>^&1') do set GITVER=%%V
echo [OK] Git %GITVER%
echo.
echo Detecting NVIDIA GPU...
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
set GPU_FOUND=0
set CUDA_VER=0.0
set CUDA_MAJOR=0
set CUDA_MINOR=0
set GPU_NAME=no GPU
nvidia-smi >nul 2>&1
if errorlevel 1 goto :no_gpu
for /f "tokens=2,* delims=:" %%A in ('nvidia-smi -L 2^>nul') do (
    if "!GPU_NAME!"=="no GPU" (
        for /f "tokens=1 delims=(" %%N in ("%%A") do (
            set "GPU_NAME=%%N"
        )
        for /f "tokens=* delims= " %%T in ("!GPU_NAME!") do set "GPU_NAME=%%T"
    )
) 2>nul
for /f "tokens=*" %%L in ('nvidia-smi 2^>nul ^| findstr /i "CUDA Version"') do (
    for %%W in (%%L) do (
        echo %%W | findstr /r "^[0-9][0-9]*\.[0-9]" >nul 2>&1
        if not errorlevel 1 set CUDA_VER=%%W
    )
) 2>nul
set GPU_FOUND=1
echo [OK] GPU: !GPU_NAME!
echo [OK] CUDA: !CUDA_VER!
for /f "tokens=1,2 delims=." %%A in ("!CUDA_VER!") do (
    set CUDA_MAJOR=%%A
    set CUDA_MINOR=%%B
)
set TORCH_INDEX=https://download.pytorch.org/whl/cu121
if !CUDA_MAJOR! GEQ 13 set TORCH_INDEX=https://download.pytorch.org/whl/cu128
if !CUDA_MAJOR! EQU 12 (
    if !CUDA_MINOR! GEQ 8 set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    if !CUDA_MINOR! GEQ 4 if !CUDA_MINOR! LSS 8 set TORCH_INDEX=https://download.pytorch.org/whl/cu124
)
if !CUDA_MAJOR! EQU 11 set TORCH_INDEX=https://download.pytorch.org/whl/cu118
echo [OK] Selected Wheel Index: !TORCH_INDEX!
goto :start_install
:no_gpu
echo [INFO] No NVIDIA GPU found - installing CPU-only PyTorch.
:start_install
echo.
echo [1/12] Removing old venv (clean start)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)
echo.
echo [2/12] Creating new venv (%VENV_DIR%)...
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] venv created
echo.
echo [3/12] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded
echo.
echo [4/12] Installing requirements from audio8_requirements.txt (torch dependency will be overwritten in step 8)...
if not exist "%~dp0audio8_requirements.txt" (
    echo [ERROR] audio8_requirements.txt not found: %~dp0audio8_requirements.txt
    pause & exit /b 1
)
"%VENV_PY%" -m pip install -r "%~dp0audio8_requirements.txt" --quiet
if errorlevel 1 ( echo [ERROR] Requirements installation failed. & pause & exit /b 1 )
echo [OK] Requirements installed
echo.
echo [5/12] Installing pyannote.audio (requires HuggingFace token)...
"%VENV_PY%" -m pip install pyannote.audio --quiet
if errorlevel 1 ( echo [WARN] pyannote.audio failed - dubbing features may not work. )
echo [OK] pyannote.audio done
echo.
echo [6/12] Installing faster-whisper...
"%VENV_PY%" -m pip install faster-whisper --quiet
if errorlevel 1 ( echo [WARN] faster-whisper failed - auto transcription may not work. )
echo [OK] faster-whisper done
echo.
echo [7/12] Installing demucs...
"%VENV_PY%" -m pip install demucs --quiet
if errorlevel 1 ( echo [WARN] demucs failed - vocal isolation may not work. )
echo [OK] demucs done
echo.
echo [8/12] Force-installing correct PyTorch + torchaudio (!TORCH_INDEX!)...
"%VENV_PY%" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url !TORCH_INDEX! --force-reinstall --quiet
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )
echo [OK] PyTorch installed
echo.
echo [9/12] Removing incompatible torchcodec (pulled in by torchaudio)...
"%VENV_PY%" -m pip uninstall torchcodec -y --quiet 2>nul
echo [OK] torchcodec removed
echo.
echo [10/12] Verifying PyTorch...
"%VENV_PY%" -c "import torch; print('[OK] torch', torch.__version__)"
if errorlevel 1 ( echo [ERROR] PyTorch verification failed. & pause & exit /b 1 )
if !GPU_FOUND! EQU 1 (
    "%VENV_PY%" -c "import torch; print('[OK] CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
)
echo.
echo [11/12] Verifying transformers (Audio8 TTS requires trust_remote_code support)...
"%VENV_PY%" -c "import transformers; from transformers import AutoModel, AutoProcessor; print('[OK] transformers', transformers.__version__)"
if errorlevel 1 ( echo [ERROR] transformers verification failed. & pause & exit /b 1 )
echo.
echo [12/12] Final torchcodec check...
"%VENV_PY%" -c "import torchcodec" 2>nul && (
    echo [WARNING] torchcodec still present - run manually:
    echo           venvs\%VENV_NAME%\Scripts\pip uninstall torchcodec -y
) || echo [OK] torchcodec not present
echo.
echo ============================================================
echo Audio8 TTS Preview 0.6B is ready to use.
echo GPU: !GPU_NAME!
echo CUDA: !CUDA_VER!
echo Venv: %VENV_DIR%
echo Run start.bat to launch the application.
echo ============================================================
echo.
pause
