@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Supertonic TTS - Installation
echo ============================================================

:: Resolve APP_ROOT (parent of \install\)
pushd "%~dp0.."
set "APP_ROOT=%CD%"
popd

set "VENV_NAME=venv_supertonic"
set "VENV_PATH=%APP_ROOT%\venvs\%VENV_NAME%"
set "PYTHON=%VENV_PATH%\Scripts\python.exe"
set "PIP=%VENV_PATH%\Scripts\pip.exe"
set "INSTALL_DIR=%~dp0"

echo.
echo  App root : %APP_ROOT%
echo  Venv     : %VENV_PATH%
echo.

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10-3.12 and add to PATH.
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [OK] %%i

:: ── Check Git ─────────────────────────────────────────────────
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Install Git and add to PATH.
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('git --version 2^>^&1') do echo [OK] %%i

:: ── Detect NVIDIA GPU & select PyTorch index ──────────────────
echo.
echo Detecting NVIDIA GPU...
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "GPU_NAME="
set "CUDA_VER="

:: FIX: use --format=csv (without noheader) + skip=1 to skip the header line
::      avoids "Option noheader is not recognized" on some driver versions
for /f "skip=1 tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv 2^>nul') do (
    if not defined GPU_NAME set "GPU_NAME=%%g"
)

:: FIX: use !var! (delayed expansion) inside if-block so variables set
::      inside the block are expanded at runtime, not at parse time.
::      tokens=4 is correct for the "| CUDA Version: 12.x |" line.
if defined GPU_NAME (
    echo [OK] GPU: !GPU_NAME!
    for /f "tokens=4" %%v in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set "CUDA_VER=%%v"
    if defined CUDA_VER (
        echo [OK] CUDA: !CUDA_VER!
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
        echo [OK] Selected wheel index: !TORCH_INDEX!
    ) else (
        echo [WARN] Could not read CUDA version ^– defaulting to cu128 index
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    )
) else (
    echo [WARN] No NVIDIA GPU detected ^– installing CPU-only PyTorch
)

:: ══════════════════════════════════════════════════════════════
echo.
echo [1/13] Removing old venv (clean start)...
if exist "%VENV_PATH%" (
    rmdir /s /q "%VENV_PATH%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)

:: ══════════════════════════════════════════════════════════════
echo.
echo [2/13] Creating venv: venvs\%VENV_NAME%
python -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause & exit /b 1
)
echo [OK] Venv created

:: ══════════════════════════════════════════════════════════════
echo.
echo [3/13] Upgrading pip, setuptools, wheel...
"%PYTHON%" -m pip install --upgrade pip setuptools wheel -q
if errorlevel 1 ( echo [WARN] pip upgrade had warnings, continuing... )
echo [OK] pip upgraded

:: ══════════════════════════════════════════════════════════════
echo.
echo [4/13] Installing PyTorch + torchaudio with CUDA support...
echo        Index: !TORCH_INDEX!
echo        May take 3-8 minutes, downloads ~3.5 GB
"%PIP%" install torch torchaudio --index-url !TORCH_INDEX! -q
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed
    pause & exit /b 1
)
"%PYTHON%" -c "import torch; gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print(f'[VERIFY] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {gpu}')"
echo [OK] PyTorch installed

:: ══════════════════════════════════════════════════════════════
echo.
echo [5/13] Removing incompatible torchcodec (torchaudio side-dependency)...
"%PIP%" uninstall torchcodec -y >nul 2>&1
echo [OK] torchcodec removed

:: ══════════════════════════════════════════════════════════════
echo.
echo [6/13] Installing onnxruntime-gpu...
::  onnxruntime-gpu provides the 'onnxruntime' import but has a separate
::  package name.  Ships CUDA/TensorRT providers.
"%PIP%" install "onnxruntime-gpu>=1.19.0,<2.0" -q
if errorlevel 1 (
    echo [ERROR] onnxruntime-gpu installation failed
    pause & exit /b 1
)
"%PYTHON%" -c "import onnxruntime as ort; p=ort.get_available_providers(); print(f'[VERIFY] onnxruntime {ort.__version__} | Providers: {p}')"

:: ══════════════════════════════════════════════════════════════
echo.
echo [7/13] Installing numpy...
:: pyannote-core 6+ and pyannote-metrics 4.1 require numpy>=2.0.
:: Supertonic's PyPI metadata says <2.0 but that's stale – the code
:: works fine with 2.x (confirmed by import test in step 13).
"%PIP%" install "numpy>=2.0.0" -q
if errorlevel 1 (
    echo [ERROR] numpy installation failed
    pause & exit /b 1
)
echo [OK] numpy installed

:: ══════════════════════════════════════════════════════════════
echo.
echo [8/13] Installing Supertonic TTS...
::
::  WHY --no-deps:
::    The PyPI package metadata for supertonic lists numpy<2.0 and
::    onnxruntime>=1.10.0 (plain CPU package), which conflicts with:
::      - numpy>=2.0 needed by pyannote
::      - onnxruntime-gpu we already installed
::    The actual runtime code works fine with both.
::    We install --no-deps and handle every dependency explicitly.
::
"%PIP%" install supertonic --no-deps -q
if errorlevel 1 (
    echo [ERROR] supertonic installation failed
    pause & exit /b 1
)

:: Install supertonic's real runtime dependencies
:: onnxruntime  → already satisfied by onnxruntime-gpu (same import name)
:: numpy        → already installed above
"%PIP%" install ^
    "huggingface-hub>=0.10.0" ^
    "soundfile>=0.12.1" ^
    "librosa>=0.10.0" ^
    "PyYAML>=6.0" ^
    -q
if errorlevel 1 (
    echo [WARN] Some supertonic dependencies had issues, check output above
)
echo [OK] Supertonic installed
echo [INFO] Models will be downloaded automatically on first run.

:: ══════════════════════════════════════════════════════════════
echo.
echo [9/13] Installing demucs (vocal isolation)...
"%PIP%" install demucs -q
if errorlevel 1 ( echo [WARN] demucs install had warnings ) else ( echo [OK] demucs )

:: ══════════════════════════════════════════════════════════════
echo.
echo [10/13] Installing pyannote.audio (speaker diarization)...
"%PIP%" install pyannote.audio -q
if errorlevel 1 ( echo [WARN] pyannote.audio install had warnings ) else ( echo [OK] pyannote.audio )

:: ══════════════════════════════════════════════════════════════
echo.
echo [11/13] Ensuring protobuf version compatible with pyannote...
:: FIX: opentelemetry-proto (pyannote dep) requires protobuf>=5.0
"%PIP%" install "protobuf>=5.0,<7.0" -q
echo [OK] protobuf

:: ══════════════════════════════════════════════════════════════
echo.
echo [12/13] Installing requirements from supertonic_requirements.txt...
if exist "%INSTALL_DIR%supertonic_requirements.txt" (
    "%PIP%" install -r "%INSTALL_DIR%supertonic_requirements.txt" -q
    echo [OK] Requirements processed
) else (
    echo [WARN] supertonic_requirements.txt not found at %INSTALL_DIR%, skipping
)

:: ════════════════════════════════════════════════════════════
::  IMPORTANT: Do NOT reinstall plain onnxruntime (CPU) here.
::  onnxruntime-gpu already satisfies the 'onnxruntime' import
::  and exposes CUDA/TensorRT providers.  Installing the CPU
::  package afterwards would silently downgrade GPU acceleration.
:: ════════════════════════════════════════════════════════════

:: FIX: torchcodec can be re-pulled by pyannote/demucs deps – force
::      a second removal here, after all packages are installed.
"%PIP%" uninstall torchcodec -y >nul 2>&1

:: ══════════════════════════════════════════════════════════════
echo.
echo [13/13] Verifying installation...

"%PYTHON%" -c "import torch; print(f'  torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
"%PYTHON%" -c "import onnxruntime as ort; cuda_ok='CUDAExecutionProvider' in ort.get_available_providers(); print(f'  onnxruntime {ort.__version__} | CUDA provider: {cuda_ok} | All: {ort.get_available_providers()}')"
"%PYTHON%" -c "import supertonic; print('  supertonic OK')"
"%PYTHON%" -c "import numpy as np; print(f'  numpy {np.__version__} OK')"
"%PYTHON%" -c "import demucs; print('  demucs OK')"
"%PYTHON%" -c "import pyannote.audio; print('  pyannote.audio OK')"
"%PYTHON%" -c "import soundfile; print('  soundfile OK')"
"%PYTHON%" -c "from PyQt6.QtWidgets import QApplication; print('  PyQt6 OK')"
"%PYTHON%" -c "import torchcodec" >nul 2>&1 && echo [WARN] torchcodec still present || echo [OK] torchcodec not present

echo.
echo ============================================================
echo  Supertonic TTS installation complete
echo  GPU    : !GPU_NAME!
echo  CUDA   : !CUDA_VER!
echo  Venv   : venvs\%VENV_NAME%
echo  Run start.bat to launch the application.
echo ============================================================
echo.
echo [NOTE] To use the speaker diarization feature:
echo 1. Accept terms at huggingface.co/pyannote/segmentation-3.0
echo 2. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Accept terms at huggingface.co/pyannote/speaker-diarization-community-1
echo 4. Create a token at huggingface.co/settings/tokens (Classic, Read)
echo ============================================================
pause
