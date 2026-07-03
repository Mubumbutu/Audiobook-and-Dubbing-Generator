@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "VENV_NAME=venv_higgs_wsl"
set "WSL_DISTRO=Ubuntu"
set "DOCKER_IMAGE=lmsysorg/sglang-omni:dev"
set "MODEL_REPO=bosonai/higgs-tts-3-4b"
if not defined APP_ROOT set "APP_ROOT=%~dp0..\"
set "VENV_DIR=%APP_ROOT%venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo ============================================================
echo Higgs TTS 3 (SGLang-Omni / WSL2 / Docker) - Installation
echo ============================================================
echo.
echo This backend does NOT run the model on Windows directly.
echo It runs a Docker container inside WSL2 that serves Higgs TTS 3
echo over HTTP, and this app talks to it as a client.
echo.
echo The Windows-side venv created by this installer is a thin CLIENT
echo (HTTP calls to the container) that ALSO hosts local tools that
echo run directly on Windows: Demucs (vocal isolation), faster-whisper
echo (transcription) and pyannote.audio (speaker diarization for the
echo dubbing feature). Those local tools need their own PyTorch, so
echo this installer detects your GPU on Windows and installs a
echo matching CUDA build for them - exactly like the Fish S2 Pro
echo installer does.
echo.
echo Requirements:
echo   - Windows 10/11 with WSL2
echo   - An Ubuntu distribution registered in WSL2
echo   - Docker Desktop, with WSL2 integration enabled for that distro
echo   - An NVIDIA GPU with recent drivers (CUDA passthrough via WSL2)
echo.

rem ---------------------------------------------------------------
rem [1/12] Python check (for the thin Windows-side client venv)
rem ---------------------------------------------------------------
echo [1/12] Checking Python...
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Remember to check "Add Python to PATH"!
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%V in ('py -3 --version 2^>^&1') do set PYVER=%%V
echo [OK] Python !PYVER!
echo.

rem ---------------------------------------------------------------
rem [2/12] WSL2 check (auto-install if missing, requires Administrator)
rem ---------------------------------------------------------------
echo [2/12] Checking WSL2...
where wsl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 'wsl' command not found on this system.
    echo This usually means Windows itself is too old for WSL2.
    echo Update Windows, then run this installer again.
    pause & exit /b 1
)

wsl --status >nul 2>&1
if not errorlevel 1 goto :WSL_OK

echo [INFO] WSL2 is not installed yet on this machine.
echo This installer can set it up automatically, but Windows requires
echo Administrator rights to do so, and a RESTART afterwards.
echo.

rem --- Check for Administrator rights ---
net session >nul 2>&1
if errorlevel 1 (
    echo [INFO] This window is not running as Administrator.
    echo Relaunching this installer with Administrator rights...
    echo ^(Click "Yes" in the Windows prompt that appears^)
    echo.
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    echo.
    echo A new Administrator window was opened to continue the WSL2 setup.
    echo You can close this window.
    pause
    exit /b 0
)

echo [OK] Running as Administrator.
echo.
echo [INFO] Installing WSL2 + Ubuntu ^(this downloads files and can take
echo        several minutes^)...
echo.
wsl --install -d %WSL_DISTRO% --no-launch
set WSL_INSTALL_RC=%ERRORLEVEL%

echo.
if %WSL_INSTALL_RC% NEQ 0 (
    echo [ERROR] 'wsl --install' failed ^(exit code %WSL_INSTALL_RC%^).
    echo Try running manually in an Administrator PowerShell:
    echo     wsl --install -d %WSL_DISTRO%
    pause & exit /b 1
)

echo ============================================================
echo [OK] WSL2 + Ubuntu have been installed.
echo.
echo A RESTART of your computer is required before continuing.
echo.
echo After restarting:
echo   1. Re-run install.bat
echo   2. Select option [11] again ^(Higgs TTS 3^)
echo   3. The installer will detect WSL2 is ready and continue
echo      automatically from where it left off.
echo ============================================================
echo.
pause
exit /b 0

:WSL_OK
echo [OK] WSL2 is available
echo.

rem ---------------------------------------------------------------
rem [3/12] Check Ubuntu distro is registered
rem ---------------------------------------------------------------
echo [3/12] Checking for WSL distribution '%WSL_DISTRO%'...
rem NOTE: we deliberately do NOT parse 'wsl -l -q' with a batch for /f loop.
rem That command prints UTF-16LE text, which cmd.exe's for /f frequently
rem mis-decodes, causing false "distro not found" results even when it
rem exists. A direct launch test is reliable regardless of encoding.
wsl -d %WSL_DISTRO% -- true >nul 2>&1
if not errorlevel 1 goto :DISTRO_OK
echo [INFO] WSL distribution '%WSL_DISTRO%' does not seem reachable yet.
echo This installer can add it automatically ^(requires Administrator
echo rights^).
echo.

net session >nul 2>&1
if errorlevel 1 (
    echo [INFO] This window is not running as Administrator.
    echo Relaunching this installer with Administrator rights...
    echo ^(Click "Yes" in the Windows prompt that appears^)
    echo.
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    echo.
    echo A new Administrator window was opened to continue setup.
    echo You can close this window.
    pause
    exit /b 0
)

echo [OK] Running as Administrator.
echo [INFO] Installing distribution '%WSL_DISTRO%' ^(this can take a
echo        few minutes^)...
echo.
wsl --install -d %WSL_DISTRO% 2>"%TEMP%\wsl_distro_install_err.txt"
set DISTRO_INSTALL_RC=!ERRORLEVEL!

findstr /i "ALREADY_EXISTS" "%TEMP%\wsl_distro_install_err.txt" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Distribution '%WSL_DISTRO%' already exists - this is fine,
    echo continuing.
    del "%TEMP%\wsl_distro_install_err.txt" >nul 2>&1
    goto :DISTRO_RECHECK
)
del "%TEMP%\wsl_distro_install_err.txt" >nul 2>&1

if !DISTRO_INSTALL_RC! NEQ 0 (
    echo [ERROR] Failed to install distribution '%WSL_DISTRO%'.
    echo Available distributions you could use instead:
    wsl -l -o
    echo.
    echo Edit higgs_wsl_install.bat and higgs_wsl_backend.py, change
    echo WSL_DISTRO to one of the names above, then re-run.
    pause & exit /b 1
)

:DISTRO_RECHECK
echo.
wsl -d %WSL_DISTRO% -- true >nul 2>&1
if not errorlevel 1 goto :DISTRO_OK

echo [INFO] Distribution '%WSL_DISTRO%' was installed but is not fully
echo set up yet ^(it may need a one-time first launch to create a Linux
echo user account^).
echo.
echo Opening it now - please follow the prompts ^(choose a Linux
echo username and password^). When done, type 'exit' to leave it,
echo then re-run this installer to continue.
echo.
wsl -d %WSL_DISTRO%
echo.
echo Re-run this installer now to continue with setup.
pause
exit /b 0

:DISTRO_OK
echo [OK] Distribution '%WSL_DISTRO%' is registered
echo.

rem ---------------------------------------------------------------
rem [4/12] Check Docker is reachable from inside WSL2 (auto-start / auto-download if possible)
rem ---------------------------------------------------------------
echo [4/12] Checking Docker inside WSL2 (this requires Docker Desktop running)...
wsl -d %WSL_DISTRO% -- bash -lc "docker info >/dev/null 2>&1"
if not errorlevel 1 goto :DOCKER_OK

echo [INFO] Docker is not reachable yet from inside WSL2.

set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if exist "%DOCKER_DESKTOP_EXE%" (
    echo [INFO] Docker Desktop is installed but does not seem to be running.
    echo Starting Docker Desktop...
    start "" "%DOCKER_DESKTOP_EXE%"
    echo Waiting for Docker to become ready, up to 2 minutes...

    set DOCKER_WAIT_OK=0
    for /l %%T in (1,1,24) do (
        wsl -d %WSL_DISTRO% -- bash -lc "docker info >/dev/null 2>&1"
        if not errorlevel 1 (
            set DOCKER_WAIT_OK=1
        )
        if !DOCKER_WAIT_OK! EQU 0 (
            timeout /t 5 >nul
        )
    )

    if !DOCKER_WAIT_OK! EQU 1 (
        echo [OK] Docker is now running and reachable from WSL2.
        goto :DOCKER_OK
    )

    echo [ERROR] Docker Desktop was started but is still not reachable
    echo from WSL2 after waiting.
    echo.
    echo Open Docker Desktop manually and check:
    echo   Settings -^> Resources -^> WSL Integration
    echo   -^> enable integration for the '%WSL_DISTRO%' distribution
    echo   -^> Apply ^& Restart
    echo Then re-run this installer.
    pause & exit /b 1
)

echo [INFO] Docker Desktop does not appear to be installed.
echo This installer can download the official installer for you.
echo You will still need to click through its setup wizard once
echo ^(license agreement + WSL integration are not scriptable^).
echo.
set /p DL_DOCKER="Download Docker Desktop installer now? [Y/n]: "
if /i "!DL_DOCKER!"=="n" (
    echo.
    echo Install it manually from: https://www.docker.com/products/docker-desktop/
    echo Then re-run this installer.
    pause & exit /b 1
)

set "DOCKER_INSTALLER=%TEMP%\DockerDesktopInstaller.exe"
echo Downloading Docker Desktop installer...
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://desktop.docker.com/win/main/amd64/Docker Desktop Installer.exe' -OutFile '%DOCKER_INSTALLER%'"
if not exist "%DOCKER_INSTALLER%" (
    echo [ERROR] Download failed. Install manually from:
    echo https://www.docker.com/products/docker-desktop/
    pause & exit /b 1
)

echo Launching Docker Desktop installer - please follow the on-screen
echo steps ^(accept the license, keep default options^), then let it
echo finish and start Docker Desktop at least once.
start "" /wait "%DOCKER_INSTALLER%"

echo.
echo After Docker Desktop has finished installing and is running:
echo   1. Open it -^> Settings -^> Resources -^> WSL Integration
echo   2. Enable integration for the '%WSL_DISTRO%' distribution
echo   3. Apply ^& Restart
echo   4. Re-run this installer to continue
pause
exit /b 0

:DOCKER_OK
echo [OK] Docker is reachable from WSL2
echo.

rem ---------------------------------------------------------------
rem [5/12] Check NVIDIA GPU passthrough inside WSL2 (for the Docker
rem         container that actually serves Higgs TTS 3)
rem ---------------------------------------------------------------
echo [5/12] Checking NVIDIA GPU passthrough inside WSL2...
wsl -d %WSL_DISTRO% -- bash -lc "nvidia-smi >/dev/null 2>&1"
if errorlevel 1 (
    echo [WARN] 'nvidia-smi' failed inside WSL2.
    echo Higgs TTS 3 needs a working NVIDIA GPU. Make sure:
    echo   - You have an NVIDIA GPU with up-to-date Windows drivers
    echo     ^(WSL2 CUDA support comes from the Windows driver, no
    echo      separate Linux NVIDIA driver should be installed in WSL2^)
    echo   - WSL2 kernel is up to date:  wsl --update
    echo Continuing anyway, but model loading will likely fail without a GPU.
) else (
    echo [OK] GPU detected inside WSL2
    wsl -d %WSL_DISTRO% -- bash -lc "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
)
echo.

rem ---------------------------------------------------------------
rem [6/12] Detect NVIDIA GPU on the Windows HOST itself. This is
rem         separate from the WSL2 passthrough check above: it picks
rem         the right PyTorch/CUDA wheel for the LOCAL client-side
rem         tools (Demucs, faster-whisper, pyannote.audio) that run
rem         directly on Windows, not inside the Docker container.
rem         Same detection logic as fish_s2_pro_install.bat.
rem ---------------------------------------------------------------
echo [6/12] Detecting NVIDIA GPU on Windows (for local Demucs / faster-whisper / pyannote.audio)...
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
set GPU_FOUND=0
set CUDA_VER=0.0
set CUDA_MAJOR=0
set CUDA_MINOR=0
set GPU_NAME=no GPU

nvidia-smi >nul 2>&1
if errorlevel 1 goto :no_gpu_client

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
echo [OK] Selected wheel index: !TORCH_INDEX!
goto :client_gpu_done

:no_gpu_client
echo [INFO] No NVIDIA GPU found on the Windows host - local tools
echo        ^(Demucs / faster-whisper / pyannote.audio^) will use
echo        CPU-only PyTorch. The Higgs TTS 3 model itself still runs
echo        on GPU inside the Docker container ^(see step 5 above^).

:client_gpu_done
echo.

rem ---------------------------------------------------------------
rem [7/12] Create thin Windows-side venv (client + local tools)
rem ---------------------------------------------------------------
echo [7/12] Creating Windows-side client venv (%VENV_DIR%)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
)
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] venv created
echo.

echo [7b/12] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded
echo.

rem ---------------------------------------------------------------
rem [8/12] Install PyTorch + torchaudio for the local tools, using
rem         the wheel index selected in step 6 (GPU) or CPU fallback.
rem ---------------------------------------------------------------
if "!GPU_FOUND!"=="0" goto :torch_cpu_client
echo [8/12] Installing PyTorch (CUDA) for local tools from: !TORCH_INDEX!
echo (may take a few minutes, downloads ~2-3.5 GB)
"%VENV_PY%" -m pip install torch torchaudio --index-url !TORCH_INDEX! --quiet
if errorlevel 1 (
    echo [WARNING] CUDA PyTorch failed - falling back to CPU...
    "%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
    if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )
)
goto :torch_client_done

:torch_cpu_client
echo [8/12] Installing PyTorch (CPU only) for local tools...
"%VENV_PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )

:torch_client_done
"%VENV_PY%" -c "import torch; c=torch.cuda.is_available(); g=torch.cuda.get_device_name(0) if c else 'N/A'; print(f'[OK] torch {torch.__version__} | CUDA: {c} | GPU: {g}')"
echo [OK] PyTorch installed for local client tools
echo.

rem ---------------------------------------------------------------
rem [9/12] Install remaining Windows-side client requirements
rem         (numpy, soundfile, Demucs, faster-whisper, pyannote.audio, ...)
rem ---------------------------------------------------------------
echo [9/12] Installing Windows-side client requirements...
if not exist "%~dp0higgs_wsl_requirements.txt" (
    echo [ERROR] higgs_wsl_requirements.txt not found: %~dp0higgs_wsl_requirements.txt
    pause & exit /b 1
)
"%VENV_PY%" -m pip install -r "%~dp0higgs_wsl_requirements.txt" --quiet
if errorlevel 1 ( echo [ERROR] Requirements installation failed. & pause & exit /b 1 )
echo [OK] Client requirements installed
echo.

rem ---------------------------------------------------------------
rem [10/12] Pull the Docker image (heavy: ~12 GB, prebuilt sglang-omni + CUDA)
rem ---------------------------------------------------------------
echo [10/12] Pulling Docker image %DOCKER_IMAGE% (this can take a while, ~12 GB)...
wsl -d %WSL_DISTRO% -- bash -lc "docker pull %DOCKER_IMAGE%"
if errorlevel 1 (
    echo [ERROR] Failed to pull Docker image %DOCKER_IMAGE%.
    echo Check your internet connection and Docker Desktop status, then retry.
    pause & exit /b 1
)
echo [OK] Docker image pulled
echo.

rem ---------------------------------------------------------------
rem [11/12] Download model weights (public model, no token required)
rem ---------------------------------------------------------------
echo [11/12] Downloading model weights %MODEL_REPO%...
echo This is a public model - no HuggingFace token required.
echo Download size is approximately 9 GB, please wait...
echo.
wsl -d %WSL_DISTRO% -- bash -lc "docker run --rm --gpus all -v sglang_omni_hf_cache:/root/.cache/huggingface %DOCKER_IMAGE% bash -lc 'hf download %MODEL_REPO%'"
if errorlevel 1 (
    echo.
    echo [ERROR] Model download failed.
    echo Check your internet connection and try re-running this installer.
    pause & exit /b 1
)
echo [OK] Model weights downloaded

echo [12/12] Setting up the sgl-omni serving environment...
echo This installs the sglang-omni package inside the container so the
echo 'sgl-omni' command becomes available (one-time, a few minutes)...
echo.
wsl -d %WSL_DISTRO% -- bash -lc "docker run --rm --gpus all -e UV_LINK_MODE=symlink -v sglang_omni_workspace:/workspace -v sglang_omni_uv_cache:/root/.cache/uv %DOCKER_IMAGE% bash -lc 'mkdir -p /workspace && cd /workspace && (test -d sglang-omni || git clone https://github.com/sgl-project/sglang-omni.git) && cd sglang-omni && (test -d .venv || uv venv .venv -p 3.12 --system-site-packages) && source .venv/bin/activate && (sgl-omni --help >/dev/null 2>&1 </dev/null || uv pip install -v -e .)'"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to set up the sgl-omni serving environment.
    echo Check your internet connection and try re-running this installer.
    pause & exit /b 1
)
echo [OK] Serving environment ready
echo.

if not exist "%APP_ROOT%models" mkdir "%APP_ROOT%models"
type nul > "%APP_ROOT%models\.higgs_tts3_wsl_ok"
echo.

echo ============================================================
echo Verifying local client-side packages...
echo ============================================================
"%VENV_PY%" -c "import torch; print(f' torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" 2>nul || echo [WARNING] torch
"%VENV_PY%" -c "import soundfile; print(' soundfile OK')" 2>nul || echo [WARNING] soundfile
"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; print(' PyQt6 OK')" 2>nul || echo [WARNING] PyQt6
"%VENV_PY%" -c "import faster_whisper; print(' faster-whisper OK')" 2>nul || echo [WARNING] faster-whisper
"%VENV_PY%" -c "import demucs; print(' demucs OK')" 2>nul || echo [WARNING] demucs
"%VENV_PY%" -c "import pyannote.audio; print(' pyannote.audio OK')" 2>nul || echo [WARNING] pyannote.audio

echo.
echo ============================================================
echo Higgs TTS 3 (WSL2 / Docker) setup finished.
echo.
echo Summary:
echo   WSL distro           : %WSL_DISTRO%
echo   Docker image         : %DOCKER_IMAGE%
echo   Model repo           : %MODEL_REPO%
echo   Client venv          : %VENV_DIR%
echo   Local tools GPU      : !GPU_NAME!
echo   Local tools CUDA     : !CUDA_VER!
echo   Local tools PyTorch  : !TORCH_INDEX!
echo.
echo Run start.bat and select 'Higgs TTS 3 (WSL2)' to launch the app.
echo Clicking 'Load model' in the app will start the container and
echo load the model into VRAM automatically on first use.
echo ============================================================
echo.
echo [NOTE] To use the "I want dubbing" feature (speaker diarization):
echo 1. Accept terms at huggingface.co/pyannote/segmentation-3.0
echo 2. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Accept terms at huggingface.co/pyannote/speaker-diarization-community-1
echo 4. Create a token at huggingface.co/settings/tokens (Classic, Read)
echo ============================================================
pause
