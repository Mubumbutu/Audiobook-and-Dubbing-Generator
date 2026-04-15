@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set "ROOT=%~dp0.."
set "VENV_NAME=venv_qwen3"
set "VENV_DIR=%ROOT%\venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ=%~dp0qwen3_requirements.txt"

echo ============================================================
echo Qwen3-TTS - Installation
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
    pause & exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo [ERROR] Python 3.10+ is required. Found: !PYVER!
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
        for /f "tokens=1 delims=(" %%N in ("%%A") do set "GPU_NAME=%%N"
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
echo [OK] Selected wheel index: !TORCH_INDEX!
goto :start_install

:no_gpu
echo [INFO] No NVIDIA GPU found - installing CPU-only PyTorch.

:start_install
echo.
echo [1/11] Removing old venv (clean start)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)

echo.
echo [2/11] Creating new venv (venvs\%VENV_NAME%)...
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] Venv created

echo.
echo [3/11] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded

echo.
if "!GPU_FOUND!"=="0" goto :torch_cpu
echo [4/11] Installing PyTorch (CUDA) from: !TORCH_INDEX!
echo (may take 3-8 minutes, downloads ~3.5 GB)
"%VENV_PY%" -m pip install torch torchaudio --index-url !TORCH_INDEX! --quiet
if errorlevel 1 (
    echo [WARNING] CUDA PyTorch failed - falling back to CPU...
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )
)
goto :torch_done

:torch_cpu
echo [4/11] Installing PyTorch (CPU only)...
"%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )

:torch_done
"%VENV_PY%" -c "import torch; c=torch.cuda.is_available(); g=torch.cuda.get_device_name(0) if c else 'N/A'; print(f'[OK] torch {torch.__version__} | CUDA: {c} | GPU: {g}')"

echo.
echo [5/11] Installing qwen-tts (--no-deps to prevent torch downgrade)...
"%VENV_PY%" -m pip install qwen-tts --no-deps --quiet
if errorlevel 1 (
    echo [WARNING] --no-deps failed, retrying with deps...
    "%VENV_PY%" -m pip install qwen-tts --quiet
    if errorlevel 1 ( echo [ERROR] qwen-tts installation failed. & pause & exit /b 1 )
)
echo [OK] qwen-tts installed

echo.
echo [6/11] Installing Qwen3-TTS core dependencies...
"%VENV_PY%" -m pip install "transformers==4.57.3" "accelerate==1.12.0" "einops>=0.6.0" "safetensors>=0.4.0" sox --quiet
if errorlevel 1 ( echo [WARNING] Some core deps may have failed - check above )
echo [OK] Core dependencies installed

echo.
echo [7/11] Installing onnxruntime...
"%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>nul
if "!GPU_FOUND!"=="0" goto :onnx_cpu
"%VENV_PY%" -m pip install onnxruntime-gpu --quiet
if errorlevel 1 goto :onnx_cpu
echo [OK] onnxruntime-gpu installed
goto :onnx_done

:onnx_cpu
echo [INFO] Installing onnxruntime (CPU)...
"%VENV_PY%" -m pip install onnxruntime --quiet
if errorlevel 1 ( echo [WARNING] onnxruntime installation failed ) else ( echo [OK] onnxruntime (CPU) installed )

:onnx_done
echo.
echo [8/11] Installing pyannote.audio...
"%VENV_PY%" -m pip install pyannote.audio --quiet
if errorlevel 1 ( echo [WARNING] pyannote.audio failed - dubbing features may not work. ) else ( echo [OK] pyannote.audio installed )

echo.
echo [9/11] Installing remaining requirements from qwen3_requirements.txt...
if not exist "%REQ%" (
    echo [WARNING] qwen3_requirements.txt not found at %REQ% - skipping.
    goto :req_done
)
"%VENV_PY%" -m pip install -r "%REQ%" --quiet
if errorlevel 1 ( echo [WARNING] Some requirements may have failed - check above )
echo [OK] Requirements processed

:req_done
echo.
echo [10/11] Installing FlashAttention 2 (optional, requires CUDA 12+)...
if "!GPU_FOUND!"=="0" goto :flash_skip
if !CUDA_MAJOR! LSS 12 goto :flash_skip
"%VENV_PY%" -m pip install flash-attn --extra-index-url !TORCH_INDEX! --quiet 2>nul
if errorlevel 1 (
    echo [INFO] FlashAttention 2 not installed - using standard attention fallback.
) else (
    echo [OK] FlashAttention 2 installed
)
goto :flash_done

:flash_skip
echo [INFO] FlashAttention 2 skipped (requires CUDA 12+ GPU).

:flash_done
echo.
echo [11/11] Verifying installation...
"%VENV_PY%" -c "import torch; print(f'[OK] torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
"%VENV_PY%" -c "from qwen_tts import Qwen3TTSModel; print('[OK] qwen_tts import OK')" 2>nul || echo [WARNING] qwen_tts import failed
"%VENV_PY%" -c "import transformers; print(f'[OK] transformers {transformers.__version__}')" 2>nul || echo [WARNING] transformers
"%VENV_PY%" -c "import accelerate; print(f'[OK] accelerate {accelerate.__version__}')" 2>nul || echo [WARNING] accelerate
"%VENV_PY%" -c "import soundfile; print('[OK] soundfile OK')" 2>nul || echo [WARNING] soundfile
"%VENV_PY%" -c "import librosa; print(f'[OK] librosa {librosa.__version__}')" 2>nul || echo [WARNING] librosa
"%VENV_PY%" -c "import pyannote.audio; print('[OK] pyannote.audio OK')" 2>nul || echo [WARNING] pyannote.audio
"%VENV_PY%" -c "import onnxruntime; print(f'[OK] onnxruntime {onnxruntime.__version__}')" 2>nul || echo [WARNING] onnxruntime
"%VENV_PY%" -c "import einops; print('[OK] einops OK')" 2>nul || echo [WARNING] einops

echo.
echo ============================================================
echo Qwen3-TTS installation complete!
echo GPU: !GPU_NAME!
echo CUDA: !CUDA_VER!
echo Venv location: venvs\%VENV_NAME%
echo Run start.bat to launch the application.
echo ============================================================
echo.
echo [NOTE] To use speaker diarization features:
echo 1. Accept terms at huggingface.co/pyannote/segmentation-3.0
echo 2. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Create a token at huggingface.co/settings/tokens (Classic, Read)
echo ============================================================
pause
exit /b 0
