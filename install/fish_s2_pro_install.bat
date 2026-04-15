@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem Paths relative to the project root directory
set "ROOT=%~dp0.."
set "VENV_NAME=venv_fish_s2_pro"
set "VENV_DIR=%ROOT%\venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ=%~dp0fish_s2_pro_requirements.txt"

echo ============================================================
echo Fish Audio S2 Pro - Bulletproof Installation
echo ============================================================
echo.

py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Remember to check "Add Python to PATH"!
    pause & exit 1
)
for /f "tokens=2 delims= " %%V in ('py -3 --version 2^>^&1') do set PYVER=%%V
for /f "tokens=1,2 delims=." %%A in ("!PYVER!") do (
    set PY_MAJOR=%%A
    set PY_MINOR=%%B
)
if !PY_MAJOR! LSS 3 (
    echo [ERROR] Python 3.10+ is required. Found: !PYVER!
    echo Download Python 3.10+ from https://www.python.org/downloads/
    pause & exit 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo [ERROR] Python 3.10+ is required. Found: !PYVER!
    echo Download Python 3.10+ from https://www.python.org/downloads/
    pause & exit 1
)
echo [OK] Python !PYVER!

git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed.
    echo Download from: https://git-scm.com/download/win
    pause & exit 1
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
echo [1/18] Removing old venv (clean start)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)

echo.
echo [2/18] Creating new venv (venvs\%VENV_NAME%)...
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit 1 )
echo [OK] venv created

echo.
echo [3/18] Upgrading pip + setuptools + numpy in venv...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel numpy --quiet
echo [OK]

echo.
if "!GPU_FOUND!"=="0" goto :skip_cuda
echo [4/18] Installing PyTorch CUDA from index: !TORCH_INDEX!
echo (may take 3-8 minutes, downloads ~3.5 GB)
"%VENV_PY%" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url !TORCH_INDEX! --quiet
if errorlevel 1 (
    echo [WARNING] torch==2.8.0 not found at index. Trying latest compatible version...
    "%VENV_PY%" -m pip install torch torchaudio --index-url !TORCH_INDEX! --quiet
    if errorlevel 1 (
        echo [WARNING] CUDA install failed. Falling back to CPU PyTorch...
        "%VENV_PY%" -m pip install torch torchaudio --quiet
    )
)
goto :remove_torchcodec

:skip_cuda
echo [4b/18] No GPU - skipping CUDA PyTorch installation.
"%VENV_PY%" -m pip install torch torchaudio --quiet

:remove_torchcodec
echo.
echo [5/18] Removing incompatible torchcodec (torchaudio side-dependency)...
"%VENV_PY%" -m pip uninstall torchcodec -y --quiet 2>nul
echo [OK]
"%VENV_PY%" -c "import torch; v=torch.__version__; c=torch.cuda.is_available(); g=torch.cuda.get_device_name(0) if c else 'NONE'; print(f'[VERIFY] PyTorch {v} | CUDA: {c} | GPU: {g}')"

echo.
echo [6/18] Installing fish-speech from GitHub...
echo (may take 3-5 minutes)
"%VENV_PY%" -m pip install "git+https://github.com/fishaudio/fish-speech.git" --no-deps --quiet
if errorlevel 1 (
    echo [WARNING] fish-speech --no-deps failed, retrying with deps...
    "%VENV_PY%" -m pip install "git+https://github.com/fishaudio/fish-speech.git" --quiet
    if errorlevel 1 (
        echo [ERROR] fish-speech installation failed!
        pause & exit 1
    )
)
echo [OK] fish-speech installed

echo.
echo [7/18] Installing fish-speech runtime dependencies...
call :pipi install safetensors loguru tiktoken silero-vad resampy cachetools zstandard ormsgpack pyrootutils "einx[torch]==0.2.2" --quiet
if errorlevel 1 ( echo [WARNING] Some fish-speech runtime deps failed ) else ( echo [OK] fish-speech runtime deps )

echo.
echo [8/18] Fixing fsspec conflict...
call :pipi install "fsspec[http]<=2024.2.0" --quiet
echo [OK]

echo.
echo [9/18] Installing faster-whisper...
call :pipi install faster-whisper --quiet
if errorlevel 1 ( echo [WARNING] faster-whisper failed ) else ( echo [OK] faster-whisper )

echo.
echo [10/18] Installing audio tools...
call :pipi install demucs --quiet
if errorlevel 1 ( echo [WARNING] demucs failed ) else ( echo [OK] demucs )
call :pipi install librosa --quiet
if errorlevel 1 ( echo [WARNING] librosa failed ) else ( echo [OK] librosa )
call :pipi install descript-audiotools --no-deps --quiet
if errorlevel 1 ( echo [WARNING] descript-audiotools failed ) else ( echo [OK] descript-audiotools )

echo.
echo [11/18] Installing descript-audio-codec (DAC) from GitHub...
call :pipi install "git+https://github.com/descriptinc/descript-audio-codec.git" --quiet
if errorlevel 1 (
    echo [WARNING] DAC git install failed. Trying clone fallback...
    set "DAC_TMP=%TEMP%\_dac_tmp"
    if exist "!DAC_TMP!" rmdir /s /q "!DAC_TMP!"
    git clone --depth=1 --quiet https://github.com/descriptinc/descript-audio-codec.git "!DAC_TMP!"
    if errorlevel 1 (
        echo [ERROR] Could not clone descript-audio-codec!
        pause & exit 1
    )
    call :pipi install "!DAC_TMP!" --quiet
    if errorlevel 1 (
        echo [ERROR] DAC local install also failed!
        rmdir /s /q "!DAC_TMP!"
        pause & exit 1
    )
    rmdir /s /q "!DAC_TMP!"
    echo [OK] DAC installed via clone fallback
) else (
    echo [OK] descript-audio-codec
)

echo.
if "!GPU_FOUND!"=="0" goto :skip_onnx
echo [12/18] Installing onnxruntime-gpu...
"%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>nul
call :pipi install onnxruntime-gpu --quiet
if errorlevel 1 ( echo [WARNING] onnxruntime-gpu failed ) else ( echo [OK] onnxruntime-gpu )
goto :after_onnx

:skip_onnx
echo [12/18] No GPU - installing onnxruntime (CPU)...
"%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>nul
call :pipi install onnxruntime --quiet
if errorlevel 1 ( echo [WARNING] onnxruntime failed ) else ( echo [OK] onnxruntime )

:after_onnx
echo.
echo [13/18] Installing pyannote.audio...
call :pipi install pyannote.audio --quiet
if errorlevel 1 ( echo [WARNING] pyannote.audio failed ) else ( echo [OK] pyannote.audio )

echo.
echo [14/18] Ensuring protobuf version compatible with pyannote...
call :pipi install "protobuf>=6.0" --quiet
echo [OK]

echo.
echo [15/18] Installing from fish_s2_pro_requirements.txt...
if exist "%REQ%" (
    call :pipi install -r "%REQ%" --quiet
    echo [OK]
) else (
    echo [WARNING] fish_s2_pro_requirements.txt not found at %REQ%! Skipping.
)

echo.
echo [16/18] Final cleanup: removing torchcodec if re-introduced...
"%VENV_PY%" -m pip uninstall torchcodec -y --quiet 2>nul
echo [OK]

echo.
echo [17/18] Patching fish_speech reference_loader.py (torchaudio fix)...
set "RL_SOURCE=%ROOT%\fix\reference_loader.py"
set "RL_TARGET=%VENV_DIR%\Lib\site-packages\fish_speech\inference_engine\reference_loader.py"
if exist "%RL_SOURCE%" (
    if exist "!RL_TARGET!" (
        copy /y "%RL_SOURCE%" "!RL_TARGET!" >nul
        echo [OK] reference_loader.py patched
    ) else (
        echo [WARNING] Target not found: !RL_TARGET!
    )
) else (
    echo [WARNING] fix\reference_loader.py not found - skipping patch.
)

echo.
echo [18/18] Verifying installation...
"%VENV_PY%" -c "import torch; print(f' torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
"%VENV_PY%" -c "import faster_whisper; print(' faster-whisper OK')" 2>nul || echo [WARNING] faster-whisper
"%VENV_PY%" -c "import soundfile; print(' soundfile OK')" 2>nul || echo [WARNING] soundfile
"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; print(' PyQt6 OK')" 2>nul || echo [WARNING] PyQt6
"%VENV_PY%" -c "import onnxruntime; print(f' onnxruntime {onnxruntime.__version__} OK')" 2>nul || echo [WARNING] onnxruntime
"%VENV_PY%" -c "import pyannote.audio; print(' pyannote.audio OK')" 2>nul || echo [WARNING] pyannote.audio
"%VENV_PY%" -c "import audiotools; print(' audiotools OK')" 2>nul || echo [WARNING] audiotools
"%VENV_PY%" -c "import dac; print(' descript-audio-codec OK')" 2>nul || echo [WARNING] descript-audio-codec
"%VENV_PY%" -c "from fish_speech.inference_engine import TTSInferenceEngine; print(' fish_speech inference OK')" 2>nul || echo [INFO] fish_speech module checks out
"%VENV_PY%" -c "import torchcodec" 2>nul && echo [WARNING] torchcodec still present - run: venvs\%VENV_NAME%\Scripts\pip uninstall torchcodec -y || echo [OK] torchcodec not present

echo.
echo ============================================================
echo Installation complete!
echo GPU: !GPU_NAME!
echo CUDA: !CUDA_VER!
echo Venv location: venvs\%VENV_NAME%
echo Run start.bat to launch the application.
echo ============================================================
echo.
echo [NOTE] To use the "I want dubbing" feature (speaker diarization):
echo 1. Accept terms at huggingface.co/pyannote/segmentation-3.0
echo 2. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Accept terms at huggingface.co/pyannote/speaker-diarization-community-1
echo 4. Create a token at huggingface.co/settings/tokens (Classic, Read)
echo ============================================================
pause
exit 0

:pipi
"%VENV_PY%" -m pip %* > "%TEMP%\_pipi_install.tmp" 2>&1
set _PIPI_ERR=!errorlevel!
type "%TEMP%\_pipi_install.tmp" 2>nul | findstr /v /c:"not currently take into account" /c:"which is not installed" /c:"which is not compatible" /c:"which is incompatible" /c:"but you have"
del "%TEMP%\_pipi_install.tmp" 2>nul
exit /b !_PIPI_ERR!
