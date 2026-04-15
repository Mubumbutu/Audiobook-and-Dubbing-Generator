@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem Paths relative to the project root directory
set "ROOT=%~dp0.."
set "VENV_DIR=%ROOT%\venvs\venv_chatterbox"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ=%~dp0chatterbox_requirements.txt"

echo ============================================================
echo Chatterbox TTS - Installation
echo ============================================================
echo.

echo [1/8] Checking Python...
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('py -3 --version 2^>^&1') do set PY_VER=%%v
echo [OK] Found Python !PY_VER!

echo.
echo [2/8] Checking Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found in PATH.
    echo Git is required to install the resemble-perth dependency.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)
for /f "tokens=3" %%v in ('git --version 2^>^&1') do set GIT_VER=%%v
echo [OK] Git !GIT_VER! found.

echo.
echo [3/8] Detecting NVIDIA GPU...
set GPU_FOUND=0
set DRIVER_MAJOR=0
set DRIVER_FULL=unknown

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set GPU_FOUND=1
    for /f "tokens=*" %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set DRIVER_FULL=%%v
    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set DRIVER_MAJOR=%%a
    echo [OK] NVIDIA GPU detected.
    echo      Driver: !DRIVER_FULL! ^(major: !DRIVER_MAJOR!^)
) else (
    echo [INFO] No NVIDIA GPU detected.
)

echo.
echo Choose installation type:
echo.
echo [1] CPU only
echo [2] GPU ^(NVIDIA CUDA^)
echo.

:CHOICE
set USER_CHOICE=
set /p USER_CHOICE=Enter choice [1/2]: 
if "!USER_CHOICE!"=="1" goto CPU_MODE
if "!USER_CHOICE!"=="2" goto GPU_MODE
echo Invalid choice. Try again.
goto CHOICE

:CPU_MODE
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU only
goto CREATE_VENV

:GPU_MODE
if !GPU_FOUND! EQU 0 (
    echo [WARNING] No GPU detected - falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
    goto CREATE_VENV
)
if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
) else if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
) else if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
) else (
    echo [WARNING] Driver !DRIVER_MAJOR! not supported, falling back to CPU.
    set TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU fallback
)
goto CREATE_VENV

:CREATE_VENV
echo.
echo [4/8] Selected mode: !TORCH_VARIANT!
echo [4/8] Creating virtual environment in venvs\venv_chatterbox...
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed.
)
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.

echo.
echo [5/8] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)
echo [OK] pip upgraded.

echo.
echo [6/8] Installing PyTorch 2.6.0 ^(!TORCH_VARIANT!^)...
echo       ^(Downloading ~2.5 GB, please wait...^)
"%VENV_PY%" -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url !TORCH_INDEX_URL!
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)
echo [OK] PyTorch installed.

echo.
echo [7/8] Installing dependencies...
echo   [7a] numpy...
"%VENV_PY%" -m pip install "numpy>=1.24.0,<2.0.0"
if errorlevel 1 (
    echo [ERROR] Failed to install numpy.
    pause
    exit /b 1
)
echo   [OK] numpy installed.

echo   [7b] spacy-pkuseg 1.0.1 ^(binary wheel^)...
"%VENV_PY%" -m pip install "spacy-pkuseg==1.0.1" --only-binary :all:
if errorlevel 1 (
    echo [ERROR] Failed to install spacy-pkuseg.
    pause
    exit /b 1
)
echo   [OK] spacy-pkuseg installed.

echo   [7c] Remaining requirements ^(please wait...^)...
"%VENV_PY%" -m pip install -r "%REQ%"
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

echo.
echo [8/8] Verifying installation...
"%VENV_PY%" -c "from chatterbox.tts import ChatterboxTTS; print('[OK] ChatterboxTTS import OK')"
if errorlevel 1 (
    echo [ERROR] Import failed.
    pause
    exit /b 1
)
"%VENV_PY%" -c "import torch; print('[OK] torch', torch.__version__, '- CUDA:', torch.cuda.is_available())"
"%VENV_PY%" -c "import numpy; print('[OK] numpy', numpy.__version__)"
"%VENV_PY%" -c "import soundfile; print('[OK] soundfile OK')"
"%VENV_PY%" -c "import librosa; print('[OK] librosa', librosa.__version__)"

echo.
echo ============================================================
echo Installation complete! Mode: !TORCH_VARIANT!
echo Venv location: venvs\venv_chatterbox
echo Run start.bat to launch the application.
echo ============================================================
pause
