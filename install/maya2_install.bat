@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "VENV_NAME=venv_maya2"
if not defined APP_ROOT set "APP_ROOT=%~dp0..\"
set "VENV_DIR=%APP_ROOT%venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "CONSTRAINTS=%TEMP%\maya2_constraints.txt"
set "MAYA3_SPEC=maya3[server] @ git+https://github.com/MayaResearch/maya2-native"
set "TORCH_EXTRA_INDEX=https://download.pytorch.org/whl/cu128"
echo ============================================================
echo Maya 2 Native - Installation
echo ============================================================
echo.
echo [NOTE] Maya Research documents Maya 2 Native for Linux only.
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
    echo [ERROR] Python 3.10, 3.11 or 3.12 is required for Maya 2 Native.
    echo Download Python 3.12 from https://www.python.org/downloads/
    echo Newer versions such as 3.13 are not supported.
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%V in ('!PY_CMD! --version 2^>^&1') do set PYVER=%%V
echo [OK] Python !PYVER! (!PY_CMD!)
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed. It is required to install maya3 from GitHub.
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
    echo [ERROR] No NVIDIA GPU found. Maya 2 Native requires CUDA.
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
    echo [WARNING] The NVIDIA driver supports CUDA !CUDA_VER! only. Maya 2 Native needs a driver for CUDA 12.8 or newer.
    echo [WARNING] Update the NVIDIA driver, otherwise the model will not run.
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
!PY_CMD! -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] venv created
echo.
echo [3/9] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded
echo.
echo [4/9] Installing maya3 from GitHub and Triton for Windows (downloads PyTorch, may take several minutes)...
"%VENV_PY%" -m pip install "%MAYA3_SPEC%" --extra-index-url %TORCH_EXTRA_INDEX%
if errorlevel 1 ( echo [ERROR] maya3 installation failed. & pause & exit /b 1 )
echo [OK] maya3 installed
"%VENV_PY%" -m pip install -U triton-windows
if errorlevel 1 ( echo [ERROR] triton-windows installation failed. & pause & exit /b 1 )
echo [OK] triton-windows installed
echo.
echo [5/9] Locking the PyTorch and NumPy versions installed by maya3...
"%VENV_PY%" -m pip freeze | findstr /b /i /c:"torch==" /c:"torchaudio==" /c:"numpy==" > "%CONSTRAINTS%"
type "%CONSTRAINTS%"
echo [OK] Constraints saved
echo.
echo [6/9] Installing application requirements from maya2_requirements.txt...
if not exist "%~dp0maya2_requirements.txt" (
    echo [ERROR] maya2_requirements.txt not found: %~dp0maya2_requirements.txt
    pause & exit /b 1
)
"%VENV_PY%" -m pip install -r "%~dp0maya2_requirements.txt" -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [ERROR] Requirements installation failed. & pause & exit /b 1 )
echo [OK] Requirements installed
echo.
echo [7/9] Installing optional dubbing tools (pyannote.audio, faster-whisper, demucs)...
"%VENV_PY%" -m pip install pyannote.audio -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [WARN] pyannote.audio failed - speaker diarization will not work. ) else ( echo [OK] pyannote.audio installed )
"%VENV_PY%" -m pip install faster-whisper -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [WARN] faster-whisper failed - auto transcription will not work. ) else ( echo [OK] faster-whisper installed )
"%VENV_PY%" -m pip install demucs -c "%CONSTRAINTS%" --quiet
if errorlevel 1 ( echo [WARN] demucs failed - vocal isolation will not work. ) else ( echo [OK] demucs installed )
echo.
echo [8/9] Checking torchcodec...
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
"%VENV_PY%" -c "import triton; print('[OK] triton', triton.__version__)"
if errorlevel 1 (
    echo [ERROR] triton cannot be imported. Maya 2 Native cannot run without it.
    pause & exit /b 1
)
"%VENV_PY%" -c "from maya3 import Maya3TTS; print('[OK] maya3 import OK')"
if errorlevel 1 (
    echo [ERROR] maya3 cannot be imported. On Windows this is a known risk - use WSL2 or Linux.
    pause & exit /b 1
)
"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; print('[OK] PyQt6 OK')" 2>nul || echo [WARNING] PyQt6
"%VENV_PY%" -c "import soundfile; print('[OK] soundfile OK')" 2>nul || echo [WARNING] soundfile
echo.
echo ============================================================
echo Maya 2 Native is ready to use.
echo GPU: !GPU_NAME!
echo CUDA: !CUDA_VER!
echo Venv: %VENV_DIR%
echo The model, the Mimi codec and the ECAPA speaker encoder are
echo downloaded together by the "Download model" button.
echo Run start.bat to launch the application.
echo ============================================================
echo.
echo [NOTE] To use speaker diarization features:
echo 1. Accept terms at huggingface.co/pyannote/segmentation-3.0
echo 2. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Create a token at huggingface.co/settings/tokens (Classic, Read)
echo ============================================================
pause
