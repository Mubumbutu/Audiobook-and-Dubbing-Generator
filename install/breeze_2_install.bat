@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "VENV_NAME=venv_breeze_2"
if not defined APP_ROOT set "APP_ROOT=%~dp0..\"
set "VENV_DIR=%APP_ROOT%venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REPO_DIR=%APP_ROOT%repo\breeze-tts"
set "REPO_URL=https://github.com/breezeblue-ai/breeze-tts.git"
set "REPO_REQ=%REPO_DIR%\requirements.txt"
set "REPO_REQ_FILTERED=%TEMP%\breeze2_repo_requirements.txt"
set "CONSTRAINTS=%TEMP%\breeze2_constraints.txt"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
echo ============================================================
echo Breeze TTS 2 - Installation
echo ============================================================
echo.
echo [NOTE] Breeze TTS 2 is documented for Linux with an NVIDIA GPU only.
echo        This installer is a best-effort attempt for Windows.
echo        If the install or the import check fails, use WSL2 or Linux.
echo.
set "PY_CMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY_CMD (
        py -%%V --version >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -%%V"
    )
)
if not defined PY_CMD (
    echo [ERROR] Python 3.10, 3.11 or 3.12 is required for Breeze TTS 2.
    echo Download Python 3.12 from https://www.python.org/downloads/
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%V in ('!PY_CMD! --version 2^>^&1') do set PYVER=%%V
echo [OK] Python !PYVER! (!PY_CMD!)
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed. It is required to download the Breeze inference code.
    echo Download from: https://git-scm.com/download/win
    pause & exit /b 1
)
for /f "tokens=3" %%V in ('git --version 2^>^&1') do set GITVER=%%V
echo [OK] Git %GITVER%
echo.
echo Detecting NVIDIA GPU...
set CUDA_VER=0.0
set CUDA_MAJOR=0
set CUDA_MINOR=0
set GPU_NAME=no GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No NVIDIA GPU found. Breeze TTS 2 requires CUDA.
    pause & exit /b 1
)
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
echo [OK] GPU: !GPU_NAME!
echo [OK] CUDA: !CUDA_VER!
for /f "tokens=1,2 delims=." %%A in ("!CUDA_VER!") do (
    set CUDA_MAJOR=%%A
    set CUDA_MINOR=%%B
)
set CUDA_OK=0
if !CUDA_MAJOR! GEQ 13 set CUDA_OK=1
if !CUDA_MAJOR! EQU 12 if !CUDA_MINOR! GEQ 8 set CUDA_OK=1
if "!CUDA_OK!"=="0" (
    echo [WARNING] The NVIDIA driver supports CUDA !CUDA_VER! only. PyTorch cu128 needs a driver for CUDA 12.8 or newer.
    echo [WARNING] Update the NVIDIA driver, otherwise the model may not run.
)
echo.
echo [1/9] Removing old venv (clean start)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)
echo.
echo [2/9] Creating new venv (%VENV_DIR%)...
if not exist "%APP_ROOT%venvs" mkdir "%APP_ROOT%venvs"
!PY_CMD! -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] venv created
echo.
echo [3/9] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded
echo.
echo [4/9] Installing PyTorch (CUDA 12.8, downloads ~3.5 GB, may take several minutes)...
"%VENV_PY%" -m pip install torch torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )
echo [OK] PyTorch installed
"%VENV_PY%" -m pip freeze | findstr /b /i /c:"torch==" /c:"torchaudio==" > "%CONSTRAINTS%"
echo.
echo [5/9] Downloading Breeze TTS inference code...
if not exist "%APP_ROOT%repo" mkdir "%APP_ROOT%repo"
if exist "%REPO_DIR%\.git" (
    git -C "%REPO_DIR%" pull --ff-only
    if errorlevel 1 echo [WARNING] git pull failed - keeping the existing copy.
) else (
    if exist "%REPO_DIR%" rmdir /s /q "%REPO_DIR%"
    git clone "%REPO_URL%" "%REPO_DIR%"
    if errorlevel 1 ( echo [ERROR] git clone failed. & pause & exit /b 1 )
)
if not exist "%REPO_DIR%\breeze_infer" (
    echo [ERROR] breeze_infer package not found in %REPO_DIR%
    pause & exit /b 1
)
echo [OK] Inference code ready: %REPO_DIR%
echo.
echo [6/9] Installing Breeze TTS dependencies from the repository...
if not exist "%REPO_REQ%" (
    echo [WARNING] requirements.txt not found in the repository - skipping.
    goto :after_repo_req
)
findstr /v /i /b /c:"flash" "%REPO_REQ%" > "%REPO_REQ_FILTERED%"
"%VENV_PY%" -m pip install -r "%REPO_REQ_FILTERED%" -c "%CONSTRAINTS%"
if errorlevel 1 (
    echo [WARNING] Install with locked PyTorch failed. Retrying without constraints...
    "%VENV_PY%" -m pip install -r "%REPO_REQ_FILTERED%" --extra-index-url %TORCH_INDEX%
    if errorlevel 1 ( echo [ERROR] Repository requirements installation failed. & pause & exit /b 1 )
)
echo [OK] Repository requirements installed
echo [NOTE] flash-attn lines were skipped - it has no official Windows build.
:after_repo_req
"%VENV_PY%" -m pip freeze | findstr /b /i /c:"torch==" /c:"torchaudio==" /c:"numpy==" > "%CONSTRAINTS%"
type "%CONSTRAINTS%"
echo.
echo [7/9] Installing application requirements from breeze_2_requirements.txt...
if not exist "%~dp0breeze_2_requirements.txt" (
    echo [ERROR] breeze_2_requirements.txt not found: %~dp0breeze_2_requirements.txt
    pause & exit /b 1
)
"%VENV_PY%" -m pip install -r "%~dp0breeze_2_requirements.txt" -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [ERROR] Requirements installation failed. & pause & exit /b 1 )
echo [OK] Requirements installed
echo.
echo [8/9] Installing optional dubbing tools (pyannote.audio, faster-whisper, demucs)...
"%VENV_PY%" -m pip install pyannote.audio -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [WARN] pyannote.audio failed - speaker diarization will not work. ) else ( echo [OK] pyannote.audio installed )
"%VENV_PY%" -m pip install faster-whisper -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [WARN] faster-whisper failed - auto transcription will not work. ) else ( echo [OK] faster-whisper installed )
"%VENV_PY%" -m pip install demucs -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [WARN] demucs failed - vocal isolation will not work. ) else ( echo [OK] demucs installed )
"%VENV_PY%" -c "import torchcodec" >nul 2>&1 && (
    echo [OK] torchcodec works - keeping it
) || (
    "%VENV_PY%" -m pip uninstall torchcodec -y --quiet >nul 2>&1
    echo [OK] torchcodec absent or broken - removed
)
echo.
echo [9/9] Verifying installation...
"%VENV_PY%" -c "import torch; print('[OK] torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if errorlevel 1 ( echo [ERROR] PyTorch verification failed. & pause & exit /b 1 )
set "PYTHONPATH=%REPO_DIR%"
"%VENV_PY%" -c "import breeze_infer.api; print('[OK] breeze_infer.api import OK')"
if errorlevel 1 (
    echo [WARNING] breeze_infer.api cannot be imported. On Windows this is a known risk - use WSL2 or Linux.
)
"%VENV_PY%" -c "import requests; print('[OK] requests OK')" 2>nul || echo [WARNING] requests
"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; print('[OK] PyQt6 OK')" 2>nul || echo [WARNING] PyQt6
"%VENV_PY%" -c "import soundfile; print('[OK] soundfile OK')" 2>nul || echo [WARNING] soundfile
echo.
echo ============================================================
echo Breeze TTS 2 is ready to use.
echo GPU: !GPU_NAME!
echo CUDA: !CUDA_VER!
echo Venv: %VENV_DIR%
echo Inference code: %REPO_DIR%
echo The model checkpoint is downloaded by the "Download model" button.
echo Run start.bat to launch the application.
echo ============================================================
echo.
echo [NOTE] To use speaker diarization features:
echo 1. Accept terms at huggingface.co/pyannote/segmentation-3.0
echo 2. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Create a token at huggingface.co/settings/tokens (Classic, Read)
echo ============================================================
pause
