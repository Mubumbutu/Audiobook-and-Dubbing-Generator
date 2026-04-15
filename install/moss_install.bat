@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem ============================================================
rem  Resolve application root (one level above install\)
rem ============================================================
set "INSTALL_DIR=%~dp0"
pushd "%INSTALL_DIR%.."
set "APP_ROOT=%CD%"
popd

set "VENV_NAME=venv_moss"
set "VENV_DIR=%APP_ROOT%\venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REPO_DIR=%APP_ROOT%\repo\moss_tts_repo"

echo ============================================================
echo  MOSS-TTS - Installation Script
echo ============================================================
echo.
echo  App root : %APP_ROOT%
echo  Venv     : %VENV_DIR%
echo  Repo     : %REPO_DIR%
echo.

rem ============================================================
rem  [CHECK] Python 3.10+
rem ============================================================
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

rem ============================================================
rem  [CHECK] Git
rem ============================================================
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed.
    echo Download from: https://git-scm.com/download/win
    pause & exit /b 1
)
for /f "tokens=3" %%V in ('git --version 2^>^&1') do set GITVER=%%V
echo [OK] Git %GITVER%
echo.

rem ============================================================
rem  [GPU] Detect NVIDIA GPU and select PyTorch wheel index
rem ============================================================
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

rem ============================================================
rem  [1/12] Remove old venv
rem ============================================================
echo [1/12] Removing old venv (clean start)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
) else (
    echo [OK] No old venv found
)

rem ============================================================
rem  [2/12] Create venv inside venvs\
rem ============================================================
echo.
echo [2/12] Creating venv: venvs\%VENV_NAME%
if not exist "%APP_ROOT%\venvs" mkdir "%APP_ROOT%\venvs"
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] Venv created

rem ============================================================
rem  [3/12] Upgrade pip
rem ============================================================
echo.
echo [3/12] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded

rem ============================================================
rem  [4/12] PyTorch + torchaudio
rem ============================================================
echo.
echo [4/12] Installing PyTorch + torchaudio (!TORCH_INDEX!)...
if "!GPU_FOUND!"=="1" (
    "%VENV_PY%" -m pip install torch torchaudio --extra-index-url !TORCH_INDEX! --quiet
) else (
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
)
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )
echo [OK] PyTorch installed

rem ============================================================
rem  [5/12] torchcodec  <-- FIX: required by torchaudio.load()
rem ============================================================
echo.
echo [5/12] Installing torchcodec (required by torchaudio for audio loading)...
if "!GPU_FOUND!"=="1" goto :torchcodec_gpu
"%VENV_PY%" -m pip install torchcodec --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 goto :torchcodec_fail
echo [OK] torchcodec installed
goto :torchcodec_done

:torchcodec_gpu
"%VENV_PY%" -m pip install torchcodec --extra-index-url !TORCH_INDEX! --quiet
if errorlevel 1 goto :torchcodec_cpu_fallback
echo [OK] torchcodec installed
goto :torchcodec_done

:torchcodec_cpu_fallback
echo [INFO] torchcodec GPU wheel not found - trying CPU fallback...
"%VENV_PY%" -m pip install torchcodec --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 goto :torchcodec_fail
echo [OK] torchcodec installed (CPU fallback)
goto :torchcodec_done

:torchcodec_fail
echo [WARN] torchcodec installation failed - torchaudio.load() may not work at runtime.
echo [WARN] To fix manually, run inside the venv:
echo [WARN]   pip install torchcodec --extra-index-url !TORCH_INDEX!

:torchcodec_done

rem ============================================================
rem  [6/12] transformers 5.x
rem ============================================================
echo.
echo [6/12] Installing transformers 5.x...
"%VENV_PY%" -m pip install "transformers>=5.0.0" --quiet
if errorlevel 1 ( echo [ERROR] transformers installation failed. & pause & exit /b 1 )
echo [OK] transformers installed

rem ============================================================
rem  [7/12] accelerate
rem ============================================================
echo.
echo [7/12] Installing accelerate...
"%VENV_PY%" -m pip install accelerate --quiet
if errorlevel 1 ( echo [WARN] accelerate installation failed - continuing anyway )
echo [OK] accelerate installed

rem ============================================================
rem  [8/12] Clone / update MOSS-TTS repo  ->  repo\moss_tts_repo
rem ============================================================
echo.
echo [8/12] Cloning MOSS-TTS repository into repo\moss_tts_repo...
if not exist "%APP_ROOT%\repo" mkdir "%APP_ROOT%\repo"
if exist "%REPO_DIR%" (
    echo [INFO] Repo already exists - pulling latest...
    cd /d "%REPO_DIR%"
    git pull
    cd /d "%APP_ROOT%"
) else (
    git clone https://github.com/OpenMOSS/MOSS-TTS.git "%REPO_DIR%"
    if errorlevel 1 ( echo [ERROR] Failed to clone MOSS-TTS. & pause & exit /b 1 )
)
echo [OK] MOSS-TTS repository ready

rem ============================================================
rem  [9/12] Install MOSS-TTS package from local repo
rem ============================================================
echo.
echo [9/12] Installing MOSS-TTS package from local repo...
"%VENV_PY%" -m pip install --extra-index-url !TORCH_INDEX! -e "%REPO_DIR%" --quiet
if errorlevel 1 (
    echo [WARN] Install with extra index failed - retrying without it...
    "%VENV_PY%" -m pip install -e "%REPO_DIR%" --quiet
    if errorlevel 1 ( echo [WARN] MOSS-TTS package install failed - continuing anyway )
)
echo [OK] MOSS-TTS package installed

rem ============================================================
rem  [10/12] Common requirements  (install\moss_requirements.txt)
rem ============================================================
echo.
echo [10/12] Installing requirements from install\moss_requirements.txt...
if exist "%INSTALL_DIR%moss_requirements.txt" (
    "%VENV_PY%" -m pip install -r "%INSTALL_DIR%moss_requirements.txt" --quiet
    if errorlevel 1 ( echo [WARN] Some requirements may have failed - check above )
    echo [OK] Requirements processed
) else (
    echo [WARN] moss_requirements.txt not found in install\ - skipping
)

rem ============================================================
rem  [11/12] FlashAttention 2 (optional)
rem ============================================================
echo.
echo [11/12] Installing FlashAttention 2 (optional - requires CUDA 12+)...
if "!GPU_FOUND!"=="1" if !CUDA_MAJOR! GEQ 12 (
    "%VENV_PY%" -m pip install flash-attn --extra-index-url !TORCH_INDEX! --quiet 2>nul
    if errorlevel 1 (
        echo [INFO] FlashAttention 2 not installed - using SDPA fallback (still fast)
    ) else (
        echo [OK] FlashAttention 2 installed
    )
) else (
    echo [INFO] FlashAttention 2 skipped (requires CUDA 12+ GPU)
)

rem ============================================================
rem  [12/12] Verify
rem ============================================================
echo.
echo [12/12] Verifying installation...
"%VENV_PY%" -c "import torch; import torchaudio; import torchcodec; import transformers; print(f'torch {torch.__version__} | torchaudio {torchaudio.__version__} | transformers {transformers.__version__} | torchcodec OK')" 2>nul
if errorlevel 1 (
    echo [WARN] Full verification failed - checking core packages only...
    "%VENV_PY%" -c "import torch; import transformers; print(f'torch {torch.__version__} | transformers {transformers.__version__}')" 2>nul
    if errorlevel 1 (
        echo [WARN] Verification check failed - please review errors above
    ) else (
        echo [OK] Core packages verified (torchcodec may be missing - see warnings above)
    )
) else (
    echo [OK] All packages verified including torchcodec
)

echo.
echo ============================================================
echo  MOSS-TTS installation complete!
echo ============================================================
echo.
echo  Venv location : venvs\venv_moss
echo  Repo location : repo\moss_tts_repo
echo.
echo  IMPORTANT: Model weights must be downloaded separately.
echo   - MOSS-TTS Delay 8B  (~16 GB): OpenMOSS-Team/MossTTSDelay-8B
echo   - MOSS-TTS Local 1.7B (~3.5 GB): OpenMOSS-Team/MossTTSLocal-1.7B
echo  Use the 'Download model' button inside the application.
echo.
pause
