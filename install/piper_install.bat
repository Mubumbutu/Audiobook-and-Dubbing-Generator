@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set "ROOT=%~dp0.."
set "VENV_NAME=venv_piper"
set "VENV_DIR=%ROOT%\venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ=%~dp0piper_requirements.txt"

echo ============================================================
echo Piper TTS - Installation
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

echo.
echo Detecting NVIDIA GPU...
set GPU_FOUND=0
set CUDA_VER=0.0
set CUDA_MAJOR=0
set CUDA_MINOR=0
set GPU_NAME=no GPU
set ONNX_PKG=onnxruntime

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
set ONNX_PKG=onnxruntime-gpu
echo [OK] GPU: !GPU_NAME!
echo [OK] CUDA: !CUDA_VER!
goto :start_install

:no_gpu
echo [INFO] No NVIDIA GPU found - installing CPU onnxruntime.

:start_install
echo [OK] ONNX package: !ONNX_PKG!
echo.

echo [1/10] Removing old venv (clean start)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)

echo.
echo [2/10] Creating new venv (venvs\%VENV_NAME%)...
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] Venv created

echo.
echo [3/10] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded

echo.
echo [4/10] Installing !ONNX_PKG!...
"%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>nul
"%VENV_PY%" -m pip install "!ONNX_PKG!" --quiet
if errorlevel 1 (
    if "!ONNX_PKG!"=="onnxruntime-gpu" (
        echo [WARNING] onnxruntime-gpu failed - falling back to CPU onnxruntime...
        "%VENV_PY%" -m pip install onnxruntime --quiet
        if errorlevel 1 ( echo [ERROR] onnxruntime installation failed. & pause & exit /b 1 )
        set ONNX_PKG=onnxruntime
    ) else (
        echo [ERROR] onnxruntime installation failed. & pause & exit /b 1
    )
)
echo [OK] !ONNX_PKG! installed

echo.
echo [5/10] Installing piper-tts (--no-deps to prevent onnxruntime downgrade)...
"%VENV_PY%" -m pip install piper-tts --no-deps --quiet
if errorlevel 1 (
    echo [WARNING] --no-deps failed, retrying with deps...
    "%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>nul
    "%VENV_PY%" -m pip install piper-tts --quiet
    if errorlevel 1 ( echo [ERROR] piper-tts installation failed. & pause & exit /b 1 )
    "%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>nul
    "%VENV_PY%" -m pip install "!ONNX_PKG!" --quiet
)
echo [OK] piper-tts installed

echo.
echo [5b/10] Installing piper-tts runtime dependencies...
"%VENV_PY%" -m pip install "pathvalidate>=3,<4" --quiet
if errorlevel 1 ( echo [WARNING] pathvalidate install failed. ) else ( echo [OK] pathvalidate installed )

echo.
echo [6/10] Installing faster-whisper...
"%VENV_PY%" -m pip install faster-whisper --quiet
if errorlevel 1 ( echo [WARNING] faster-whisper failed - transcription features may not work. ) else ( echo [OK] faster-whisper installed )

echo.
echo [7/10] Installing demucs...
"%VENV_PY%" -m pip install demucs --quiet
if errorlevel 1 ( echo [WARNING] demucs failed - vocal isolation may not work. ) else ( echo [OK] demucs installed )

echo.
echo [8/10] Installing pyannote.audio...
"%VENV_PY%" -m pip install pyannote.audio --quiet
if errorlevel 1 ( echo [WARNING] pyannote.audio failed - diarization features may not work. ) else ( echo [OK] pyannote.audio installed )

echo.
echo [9/10] Installing remaining requirements from piper_requirements.txt...
if not exist "%REQ%" (
    echo [WARNING] piper_requirements.txt not found at %REQ% - skipping.
    goto :req_done
)
"%VENV_PY%" -m pip install -r "%REQ%" --quiet
if errorlevel 1 ( echo [WARNING] Some requirements may have failed - check above. )
echo [OK] Requirements processed

:req_done
echo.
echo [10/10] Verifying installation...
"%VENV_PY%" -c "import piper; print('[OK] piper import OK')" 2>nul || echo [WARNING] piper import failed
"%VENV_PY%" -c "import onnxruntime as ort; cuda_ok='CUDAExecutionProvider' in ort.get_available_providers(); print(f'[OK] onnxruntime {ort.__version__} | CUDA provider: {cuda_ok} | Providers: {ort.get_available_providers()}')" 2>nul || echo [WARNING] onnxruntime import failed
"%VENV_PY%" -c "import faster_whisper; print('[OK] faster-whisper OK')" 2>nul || echo [WARNING] faster-whisper
"%VENV_PY%" -c "import demucs; print('[OK] demucs OK')" 2>nul || echo [WARNING] demucs
"%VENV_PY%" -c "import pyannote.audio; print('[OK] pyannote.audio OK')" 2>nul || echo [WARNING] pyannote.audio
"%VENV_PY%" -c "import soundfile; print('[OK] soundfile OK')" 2>nul || echo [WARNING] soundfile
"%VENV_PY%" -c "import sounddevice; print('[OK] sounddevice OK')" 2>nul || echo [WARNING] sounddevice
"%VENV_PY%" -c "import librosa; print(f'[OK] librosa {librosa.__version__}')" 2>nul || echo [WARNING] librosa
"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; print('[OK] PyQt6 OK')" 2>nul || echo [WARNING] PyQt6
"%VENV_PY%" -c "import numpy as np; print(f'[OK] numpy {np.__version__}')" 2>nul || echo [WARNING] numpy
"%VENV_PY%" -c "import pathvalidate; print('[OK] pathvalidate OK')" 2>nul || echo [WARNING] pathvalidate

echo.
echo ============================================================
echo Piper TTS installation complete!
echo GPU: !GPU_NAME!
echo ONNX package: !ONNX_PKG!
echo Venv location: venvs\%VENV_NAME%
echo.
echo NOTE: Piper voice models (.onnx + .onnx.json) must be
echo downloaded separately. Use the Download model button in
echo the application or get them from:
echo https://huggingface.co/rhasspy/piper-voices
echo.
echo Run start.bat to launch the application.
echo ============================================================
echo.
pause
exit /b 0
