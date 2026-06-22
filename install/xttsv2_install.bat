@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set "ROOT=%~dp0.."
set "VENV_NAME=venv_xttsv2"
set "VENV_DIR=%ROOT%\venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ=%~dp0xttsv2_requirements.txt"

echo ============================================================
echo XTTS v2 (coqui-tts) - Installation
echo ============================================================
echo.
echo NOTE: Uses the maintained coqui-tts fork (idiap).
echo       Original coqui-ai/TTS is abandoned and broken on Py 3.12.
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

echo.
echo Detecting NVIDIA GPU...
set GPU_FOUND=0
set CUDA_VER=0.0
set CUDA_MAJOR=0
set CUDA_MINOR=0
set GPU_NAME=no GPU
set TORCH_INDEX=https://download.pytorch.org/whl/cpu

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
echo [INFO] XTTS v2 on CPU is very slow. GPU is strongly recommended.

:start_install
echo.
echo [1/13] Removing old venv (clean start)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)

echo.
echo [2/13] Creating new venv (venvs\%VENV_NAME%)...
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] Venv created

echo.
echo [3/13] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded

echo.
if "!GPU_FOUND!"=="0" goto :torch_cpu
echo [4/13] Installing PyTorch (CUDA) from: !TORCH_INDEX!
echo (may take 3-8 minutes, downloads ~3.5 GB)
"%VENV_PY%" -m pip install torch torchaudio --index-url !TORCH_INDEX! --quiet
if errorlevel 1 (
    echo [WARNING] CUDA PyTorch failed - falling back to CPU...
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )
)
goto :torch_done

:torch_cpu
echo [4/13] Installing PyTorch (CPU only)...
"%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )

:torch_done
"%VENV_PY%" -c "import torch; c=torch.cuda.is_available(); g=torch.cuda.get_device_name(0) if c else 'N/A'; print(f'[OK] torch {torch.__version__} | CUDA: {c} | GPU: {g}')"

echo.
echo [5/13] Installing coqui-tts (maintained fork, Python 3.12 compatible)...
echo NOTE: Original coqui-ai/TTS is abandoned - using idiap fork via coqui-tts PyPI package.
"%VENV_PY%" -m pip install coqui-tts --quiet
if errorlevel 1 (
    echo [WARNING] coqui-tts quiet install reported errors, retrying verbose...
    "%VENV_PY%" -m pip install coqui-tts
    if errorlevel 1 ( echo [ERROR] coqui-tts installation failed. & pause & exit /b 1 )
)
echo [OK] coqui-tts installed

echo.
echo [5b/13] Installing XTTS v2 core dependencies...
echo         (coqpit-config, einops, encodec, unidic-lite - not bundled by coqui-tts)
echo         Removing legacy 'coqpit' package if present (conflicts with coqpit-config)...
"%VENV_PY%" -m pip uninstall coqpit -y --quiet 2>nul
"%VENV_PY%" -m pip install --no-cache-dir --force-reinstall "coqpit-config>=0.2.0" --quiet
"%VENV_PY%" -m pip install einops encodec unidic-lite --quiet
if errorlevel 1 (
    echo [WARNING] Some XTTS core deps failed - retrying verbose...
    "%VENV_PY%" -m pip install --no-cache-dir --force-reinstall "coqpit-config>=0.2.0"
    "%VENV_PY%" -m pip install einops encodec unidic-lite
    if errorlevel 1 ( echo [WARNING] Some XTTS core deps may be missing - import may fail. )
) else (
    echo [OK] XTTS core deps installed
)

echo.
echo [5c/13] Verifying PyTorch not downgraded by coqui-tts...
"%VENV_PY%" -c "import torch; c=torch.cuda.is_available(); g=torch.cuda.get_device_name(0) if c else 'N/A'; print(f'[OK] torch {torch.__version__} | CUDA: {c} | GPU: {g}')"
if errorlevel 1 (
    echo [WARNING] PyTorch check failed - reinstalling...
    if "!GPU_FOUND!"=="0" goto :retorch_cpu
    "%VENV_PY%" -m pip install torch torchaudio --index-url !TORCH_INDEX! --quiet
    goto :torch_reverify
    :retorch_cpu
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    :torch_reverify
    "%VENV_PY%" -c "import torch; print(f'[OK] torch {torch.__version__} after reinstall')"
)

echo.
echo [6/13] Installing onnxruntime...
"%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>nul
if "!GPU_FOUND!"=="0" goto :onnx_cpu
"%VENV_PY%" -m pip install onnxruntime-gpu --quiet
if errorlevel 1 goto :onnx_cpu
echo [OK] onnxruntime-gpu installed
goto :onnx_done

:onnx_cpu
echo [INFO] Installing onnxruntime (CPU)...
"%VENV_PY%" -m pip install onnxruntime --quiet
if errorlevel 1 ( echo [WARNING] onnxruntime failed ) else ( echo [OK] onnxruntime (CPU) installed )

:onnx_done
echo.
echo [7/13] Installing faster-whisper...
"%VENV_PY%" -m pip install faster-whisper --quiet
if errorlevel 1 ( echo [WARNING] faster-whisper failed. ) else ( echo [OK] faster-whisper installed )

echo.
echo [8/13] Installing demucs...
"%VENV_PY%" -m pip install demucs --quiet
if errorlevel 1 ( echo [WARNING] demucs failed. ) else ( echo [OK] demucs installed )

echo.
echo [9/13] Installing pyannote.audio...
"%VENV_PY%" -m pip install pyannote.audio --quiet
if errorlevel 1 ( echo [WARNING] pyannote.audio failed. ) else ( echo [OK] pyannote.audio installed )

echo.
echo [10/13] Installing remaining requirements from xttsv2_requirements.txt...
if not exist "%REQ%" (
    echo [WARNING] xttsv2_requirements.txt not found at %REQ% - skipping.
    goto :req_done
)
"%VENV_PY%" -m pip install -r "%REQ%" --quiet
if errorlevel 1 ( echo [WARNING] Some requirements may have failed. )
echo [OK] Requirements processed

:req_done
echo.
echo [10b/13] Re-checking for legacy 'coqpit' package after requirements install...
"%VENV_PY%" -m pip uninstall coqpit -y --quiet 2>nul

echo.
echo [10c/13] Pinning compatible 'transformers' version (XTTS/coqui-tts 0.27.x compatibility)...
echo         (coqui-tts 0.27.x requires transformers^>=4.57; transformers 5.x removed
echo          isin_mps_friendly, which coqui-tts still imports - known upstream bug,
echo          see idiap/coqui-ai-TTS issue #558. Pinning to a safe range.)
"%VENV_PY%" -m pip install --no-cache-dir --force-reinstall --upgrade "transformers>=4.57,<5.0" --quiet
if errorlevel 1 (
    echo [WARNING] transformers pin failed - retrying verbose...
    "%VENV_PY%" -m pip install --no-cache-dir --force-reinstall --upgrade "transformers>=4.57,<5.0"
    if errorlevel 1 ( echo [WARNING] transformers pin may have failed - XTTS import may break. )
) else (
    echo [OK] transformers pinned to compatible range
)

echo.
echo [11/13] Installing FlashAttention 2 (optional, requires CUDA 12+)...
if "!GPU_FOUND!"=="0" goto :flash_skip
if !CUDA_MAJOR! LSS 12 goto :flash_skip
"%VENV_PY%" -m pip install flash-attn --no-build-isolation --quiet 2>nul
if errorlevel 1 (
    echo [INFO] FlashAttention 2 not installed - standard attention will be used.
) else (
    echo [OK] FlashAttention 2 installed
)
goto :flash_done

:flash_skip
echo [INFO] FlashAttention 2 skipped (requires CUDA 12+ GPU).

:flash_done
echo.
echo [12/13] Verifying coqui-tts package version...
"%VENV_PY%" -m pip show coqui-tts 2>nul | findstr /i "Version:"
"%VENV_PY%" -m pip show coqpit 2>nul | findstr /i "Name:" && echo [WARNING] Legacy 'coqpit' package still present - this will break imports.
"%VENV_PY%" -c "import coqpit; print('[OK] coqpit module import OK')" 2>nul || echo [WARNING] coqpit module import failed
"%VENV_PY%" -c "import einops; print('[OK] einops OK')" 2>nul || echo [WARNING] einops not found
"%VENV_PY%" -c "import encodec; print('[OK] encodec OK')" 2>nul || echo [WARNING] encodec not found
"%VENV_PY%" -m pip show transformers 2>nul | findstr /i "Version:"
"%VENV_PY%" -c "from transformers.utils.import_utils import is_torchcodec_available; print('[OK] transformers is_torchcodec_available OK')" 2>nul || echo [WARNING] transformers too old - is_torchcodec_available missing (need ^>=4.57)
"%VENV_PY%" -c "from transformers.pytorch_utils import isin_mps_friendly; print('[OK] transformers isin_mps_friendly OK')" 2>nul || echo [WARNING] transformers too new - isin_mps_friendly missing (need ^<5.0, see idiap/coqui-ai-TTS#558)

echo.
echo [13/13] Verifying installation...
"%VENV_PY%" -c "import torch; print(f'[OK] torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
"%VENV_PY%" -c "from TTS.tts.configs.xtts_config import XttsConfig; from TTS.tts.models.xtts import Xtts; print('[OK] XttsConfig and Xtts import OK')"
if errorlevel 1 (
    echo [FAIL] XTTS imports failed. Running full diagnostic:
    echo ----
    "%VENV_PY%" -c "from TTS.tts.configs.xtts_config import XttsConfig"
    echo ----
    echo Manual fix attempt 1 - legacy coqpit conflict:
    echo   venvs\%VENV_NAME%\Scripts\pip uninstall coqpit -y
    echo   venvs\%VENV_NAME%\Scripts\pip install --no-cache-dir --force-reinstall coqpit-config
    echo.
    echo Manual fix attempt 2 - transformers version mismatch (coqui-tts 0.27.x needs
    echo a narrow range - ^>=4.57 for is_torchcodec_available, ^<5.0 because transformers 5.x
    echo removed isin_mps_friendly. See github.com/idiap/coqui-ai-TTS/issues/558
    echo   venvs\%VENV_NAME%\Scripts\pip install --no-cache-dir --force-reinstall --upgrade "transformers>=4.57,<5.0"
)
"%VENV_PY%" -c "import faster_whisper; print('[OK] faster-whisper OK')" 2>nul || echo [WARNING] faster-whisper
"%VENV_PY%" -c "import demucs; print('[OK] demucs OK')" 2>nul || echo [WARNING] demucs
"%VENV_PY%" -c "import pyannote.audio; print('[OK] pyannote.audio OK')" 2>nul || echo [WARNING] pyannote.audio
"%VENV_PY%" -c "import soundfile; print('[OK] soundfile OK')" 2>nul || echo [WARNING] soundfile
"%VENV_PY%" -c "import sounddevice; print('[OK] sounddevice OK')" 2>nul || echo [WARNING] sounddevice
"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; print('[OK] PyQt6 OK')" 2>nul || echo [WARNING] PyQt6
"%VENV_PY%" -c "import numpy as np; print(f'[OK] numpy {np.__version__}')" 2>nul || echo [WARNING] numpy
"%VENV_PY%" -c "import librosa; print(f'[OK] librosa {librosa.__version__}')" 2>nul || echo [WARNING] librosa

echo.
echo ============================================================
echo XTTS v2 installation complete!
echo GPU: !GPU_NAME!
echo CUDA: !CUDA_VER!
echo Venv location: venvs\%VENV_NAME%
echo.
echo Model (~1.8 GB) will be downloaded on first use via
echo the Download model button inside the application.
echo Repo: coqui/XTTS-v2 (HuggingFace)
echo.
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
