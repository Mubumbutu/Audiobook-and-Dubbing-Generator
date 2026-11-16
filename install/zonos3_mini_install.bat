@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "VENV_NAME=venv_zonos3_mini"
set "WSL_DISTRO=Ubuntu"
set "DOCKER_IMAGE=lmsysorg/sglang-omni:dev"
set "MODEL_REPO=Zyphra/ZONOS3-Mini"
set "CONTAINER_NAME=sglang_omni_zonos3"
if not defined APP_ROOT set "APP_ROOT=%~dp0..\"
set "VENV_DIR=%APP_ROOT%venvs\%VENV_NAME%"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo ============================================================
echo ZONOS3-Mini (SGLang-Omni / WSL2 / Docker) - Installation
echo ============================================================
echo.
echo This backend does NOT run the model on Windows directly.
echo It runs a Docker container inside WSL2 that serves ZONOS3-Mini
echo over HTTP, and this app talks to it as a client.
echo.
echo The Windows-side venv created by this installer is a thin CLIENT
echo (HTTP calls to the container) that ALSO hosts local tools that
echo run directly on Windows: Demucs (vocal isolation), faster-whisper
echo (transcription) and pyannote.audio (speaker diarization for the
echo dubbing feature).
echo Those local tools need their own PyTorch, so
echo this installer detects your GPU on Windows and installs a
echo matching CUDA build for them.
echo.
echo The Docker image and the model weights are NOT downloaded here.
echo Use the 'Download model' button in the app after installation.
echo.
echo Requirements:
echo   - Windows 10/11 with WSL2
echo   - An Ubuntu distribution registered in WSL2
echo   - Docker Desktop, with WSL2 integration enabled for that distro
echo   - An NVIDIA GPU with recent drivers (CUDA passthrough via WSL2)
echo.

rem ---------------------------------------------------------------
rem [1/10] Python check
rem ---------------------------------------------------------------
echo [1/10] Checking Python...
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
rem [2/10] WSL2 check (auto-install if missing, requires Administrator)
rem ---------------------------------------------------------------
echo [2/10] Checking WSL2...
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
echo   2. Select option [17] again ^(ZONOS3-Mini^)
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
rem [3/10] Check Ubuntu distro is registered
rem ---------------------------------------------------------------
echo [3/10] Checking for WSL distribution '%WSL_DISTRO%'...
wsl -d %WSL_DISTRO% -- true >nul 2>&1
if not errorlevel 1 goto :DISTRO_OK
echo [INFO] WSL distribution '%WSL_DISTRO%' does not seem reachable yet.
echo This installer can add it automatically ^(requires Administrator rights^).
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
echo [INFO] Installing distribution '%WSL_DISTRO%' ^(this can take a few minutes^)...
echo.
wsl --install -d %WSL_DISTRO% 2>"%TEMP%\wsl_distro_install_err.txt"
set DISTRO_INSTALL_RC=!ERRORLEVEL!

findstr /i "ALREADY_EXISTS" "%TEMP%\wsl_distro_install_err.txt" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Distribution '%WSL_DISTRO%' already exists - this is fine, continuing.
    del "%TEMP%\wsl_distro_install_err.txt" >nul 2>&1
    goto :DISTRO_RECHECK
)
del "%TEMP%\wsl_distro_install_err.txt" >nul 2>&1

if !DISTRO_INSTALL_RC! NEQ 0 (
    echo [ERROR] Failed to install distribution '%WSL_DISTRO%'.
    echo Available distributions you could use instead:
    wsl -l -o
    echo.
    echo Edit zonos3_mini_install.bat and zonos3_mini_backend.py, change
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
echo username and password^).
echo When done, type 'exit' to leave it, then re-run this installer to continue.
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
rem [4/10] Check Docker is reachable from inside WSL2
rem ---------------------------------------------------------------
echo [4/10] Checking Docker inside WSL2 (this requires Docker Desktop running)...
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
        if !DOCKER_WAIT_OK! EQU 0 (
            wsl -d %WSL_DISTRO% -- bash -lc "docker info >/dev/null 2>&1"
            if not errorlevel 1 (
                set DOCKER_WAIT_OK=1
            ) else (
                timeout /t 5 >nul
            )
        )
    )

    if !DOCKER_WAIT_OK! EQU 1 (
        echo [OK] Docker is now running and reachable from WSL2.
        goto :DOCKER_OK
    )

    echo [ERROR] Docker Desktop was started but is still not reachable from WSL2 after waiting.
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
rem [5/10] Check NVIDIA GPU passthrough inside WSL2
rem ---------------------------------------------------------------
echo [5/10] Checking NVIDIA GPU passthrough inside WSL2...
wsl -d %WSL_DISTRO% -- bash -lc "nvidia-smi >/dev/null 2>&1"
if errorlevel 1 (
    echo [WARN] 'nvidia-smi' failed inside WSL2.
    echo ZONOS3-Mini needs a working NVIDIA GPU. Make sure:
    echo   - You have an NVIDIA GPU with up-to-date Windows drivers
    echo     ^(WSL2 CUDA support comes from the Windows driver, no separate Linux NVIDIA driver should be installed in WSL2^)
    echo   - WSL2 kernel is up to date:  wsl --update
    echo Continuing anyway, but model loading will likely fail without a GPU.
) else (
    echo [OK] GPU detected inside WSL2
    wsl -d %WSL_DISTRO% -- bash -lc "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
)
echo.

rem ---------------------------------------------------------------
rem [6/10] Detect NVIDIA GPU on the Windows HOST (for local tools)
rem ---------------------------------------------------------------
echo [6/10] Detecting NVIDIA GPU on Windows (for local Demucs / faster-whisper / pyannote.audio)...
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

set TORCH_INDEX=https://download.pytorch.org/whl/cpu
set TORCH_CUDA_OK=0

if !CUDA_MAJOR! GEQ 13 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu129
    set TORCH_CUDA_OK=1
) else if !CUDA_MAJOR! EQU 12 (
    if !CUDA_MINOR! GEQ 9 (
        set TORCH_INDEX=https://download.pytorch.org/whl/cu129
        set TORCH_CUDA_OK=1
    ) else if !CUDA_MINOR! GEQ 8 (
        set TORCH_INDEX=https://download.pytorch.org/whl/cu128
        set TORCH_CUDA_OK=1
    ) else if !CUDA_MINOR! GEQ 6 (
        set TORCH_INDEX=https://download.pytorch.org/whl/cu126
        set TORCH_CUDA_OK=1
    )
)

if !TORCH_CUDA_OK! EQU 0 (
    echo [WARNING] Your driver reports CUDA !CUDA_VER!, which is older than
    echo           the CUDA 12.6+ needed by the pinned PyTorch 2.8.0 wheels.
    echo           Local tools ^(Demucs / faster-whisper / pyannote.audio^)
    echo           will install as CPU-only. Update your NVIDIA driver and
    echo           re-run this installer for GPU acceleration.
)
echo [OK] Selected wheel index: !TORCH_INDEX!
goto :client_gpu_done

:no_gpu_client
echo [INFO] No NVIDIA GPU found on the Windows host - local tools
echo        ^(Demucs / faster-whisper / pyannote.audio^) will use
echo        CPU-only PyTorch.
echo        The ZONOS3-Mini model itself still runs
echo        on GPU inside the Docker container ^(see step 5 above^).

:client_gpu_done
echo.

rem ---------------------------------------------------------------
rem [7/10] Create thin Windows-side venv
rem ---------------------------------------------------------------
echo [7/10] Creating Windows-side client venv (%VENV_DIR%)...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed
)
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & pause & exit /b 1 )
echo [OK] venv created
echo.

echo [7b/10] Upgrading pip, setuptools, wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo [ERROR] pip upgrade failed. & pause & exit /b 1 )
echo [OK] pip upgraded
echo.

rem ---------------------------------------------------------------
rem [8/10] Install PyTorch + torchaudio + torchcodec (pinned)
rem ---------------------------------------------------------------
set "TORCH_PIN=torch==2.8.0"
set "TORCHAUDIO_PIN=torchaudio==2.8.0"
set "TORCHCODEC_PIN=torchcodec==0.7.0"

if "!GPU_FOUND!"=="0" goto :torch_cpu_client
echo [8/10] Installing PyTorch (CUDA) for local tools from: !TORCH_INDEX!
echo (may take a few minutes, downloads ~2-3.5 GB)
"%VENV_PY%" -m pip install %TORCH_PIN% %TORCHAUDIO_PIN% --index-url !TORCH_INDEX! --quiet
if errorlevel 1 (
    echo [WARNING] CUDA PyTorch failed - falling back to CPU...
    "%VENV_PY%" -m pip install %TORCH_PIN% %TORCHAUDIO_PIN% --index-url https://download.pytorch.org/whl/cpu --quiet
    if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )
)
goto :torch_client_done

:torch_cpu_client
echo [8/10] Installing PyTorch (CPU only) for local tools...
"%VENV_PY%" -m pip install %TORCH_PIN% %TORCHAUDIO_PIN% --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & pause & exit /b 1 )

:torch_client_done
echo [OK] PyTorch installed for local client tools
echo.

echo [8b/10] Installing torchcodec for local tools (from PyPI, CPU build)...
"%VENV_PY%" -m pip install %TORCHCODEC_PIN% --quiet
if errorlevel 1 ( echo [ERROR] torchcodec installation failed. & pause & exit /b 1 )
echo [OK] torchcodec installed
echo.

rem ---------------------------------------------------------------
rem [9/10] FFmpeg 7.1 "shared" build for torchcodec.
rem   Search order: local tools\ffmpeg -> winget package folders ->
rem   winget install -> direct download from BtbN GitHub releases.
rem ---------------------------------------------------------------
echo [9/10] Setting up FFmpeg (shared build) for torchcodec...
set "FFMPEG_BIN="
set "FFMPEG_LOCAL=%APP_ROOT%tools\ffmpeg"

if exist "%FFMPEG_LOCAL%" (
    for /f "delims=" %%D in ('dir /b /s /a-d "%FFMPEG_LOCAL%\avformat-*.dll" 2^>nul') do (
        if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpD"
    )
)
if defined FFMPEG_BIN goto :ffmpeg_found

call :FIND_WINGET_FFMPEG
if defined FFMPEG_BIN goto :ffmpeg_found

where winget >nul 2>&1
if errorlevel 1 (
    echo [INFO] 'winget' not found - will download FFmpeg directly.
    goto :ffmpeg_download
)

echo Installing BtbN.FFmpeg.GPL.Shared.7.1 via winget...
winget install --id=BtbN.FFmpeg.GPL.Shared.7.1 -e --accept-package-agreements --accept-source-agreements
call :FIND_WINGET_FFMPEG
if defined FFMPEG_BIN goto :ffmpeg_found

:ffmpeg_download
echo [INFO] Downloading FFmpeg 7.1 shared build directly from GitHub...
if not exist "%FFMPEG_LOCAL%" mkdir "%FFMPEG_LOCAL%"
powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; try { Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip' -OutFile '%TEMP%\ffmpeg_shared.zip' -ErrorAction Stop; Expand-Archive -Path '%TEMP%\ffmpeg_shared.zip' -DestinationPath '%FFMPEG_LOCAL%' -Force -ErrorAction Stop; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
del "%TEMP%\ffmpeg_shared.zip" >nul 2>&1

for /f "delims=" %%D in ('dir /b /s /a-d "%FFMPEG_LOCAL%\avformat-*.dll" 2^>nul') do (
    if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpD"
)
if defined FFMPEG_BIN goto :ffmpeg_found

echo [WARNING] Could not set up FFmpeg shared DLLs automatically.
echo Demucs / pyannote.audio audio encode/decode may fail until FFmpeg
echo is available. Download "ffmpeg-n7.1-latest-win64-gpl-shared-7.1.zip" from
echo https://github.com/BtbN/FFmpeg-Builds/releases and extract it into:
echo   %FFMPEG_LOCAL%
echo then re-run this installer.
goto :ffmpeg_done

:ffmpeg_found
if "!FFMPEG_BIN:~-1!"=="\" set "FFMPEG_BIN=!FFMPEG_BIN:~0,-1!"
echo [OK] Found FFmpeg shared build: !FFMPEG_BIN!
set "SITECUSTOMIZE=%VENV_DIR%\Lib\site-packages\sitecustomize.py"
(
    echo import os
    echo(
    echo _FFMPEG_BIN = r"!FFMPEG_BIN!"
    echo(
    echo if _FFMPEG_BIN and os.path.isdir^(_FFMPEG_BIN^):
    echo     try:
    echo         os.add_dll_directory^(_FFMPEG_BIN^)
    echo     except ^(AttributeError, OSError^):
    echo         pass
    echo     os.environ["PATH"] = _FFMPEG_BIN + os.pathsep + os.environ.get^("PATH", ""^)
) > "!SITECUSTOMIZE!"
echo [OK] Wired FFmpeg into the venv via sitecustomize.py

:ffmpeg_done
echo.

rem ---------------------------------------------------------------
rem [10/10] Install remaining Windows-side client requirements
rem ---------------------------------------------------------------
echo [10/10] Installing Windows-side client requirements...
if not exist "%~dp0zonos3_mini_requirements.txt" (
    echo [ERROR] zonos3_mini_requirements.txt not found: %~dp0zonos3_mini_requirements.txt
    pause & exit /b 1
)
"%VENV_PY%" -m pip install -r "%~dp0zonos3_mini_requirements.txt" --quiet
if errorlevel 1 ( echo [ERROR] Requirements installation failed. & pause & exit /b 1 )
echo [OK] Client requirements installed
echo.

echo ============================================================
echo Verifying local client-side packages...
echo ============================================================
"%VENV_PY%" -c "import torch; print(f' torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')" 2>nul || echo [WARNING] torch
"%VENV_PY%" -c "import torchaudio; print(f' torchaudio {torchaudio.__version__}')" 2>nul || echo [WARNING] torchaudio
"%VENV_PY%" -c "import torchcodec; from torchcodec.encoders import AudioEncoder; print(f' torchcodec {torchcodec.__version__}')" 2>nul || echo [WARNING] torchcodec
"%VENV_PY%" -c "import soundfile; print(' soundfile OK')" 2>nul || echo [WARNING] soundfile
"%VENV_PY%" -c "from PyQt6.QtWidgets import QApplication; print(' PyQt6 OK')" 2>nul || echo [WARNING] PyQt6
"%VENV_PY%" -c "import faster_whisper; print(' faster-whisper OK')" 2>nul || echo [WARNING] faster-whisper
"%VENV_PY%" -c "import demucs; print(' demucs OK')" 2>nul || echo [WARNING] demucs
"%VENV_PY%" -c "import pyannote.audio; print(' pyannote.audio OK')" 2>nul || echo [WARNING] pyannote.audio
echo.

echo Testing the exact save path Demucs uses (torchaudio.save -^> torchcodec)...
"%VENV_PY%" -c "import torch, torchaudio, tempfile, os; p=os.path.join(tempfile.gettempdir(), '_zonos3_mini_save_test.wav'); torchaudio.save(p, torch.zeros(1, 8000), 8000); os.remove(p); print(' Demucs-style audio save OK (torchcodec + FFmpeg working)')" 2>nul || echo [WARNING] torchaudio.save failed - Demucs will likely still crash. Check that FFmpeg was set up correctly in step 9.

echo.
echo ============================================================
echo ZONOS3-Mini (WSL2 / Docker) client setup finished.
echo.
echo Summary:
echo   Model                : ZONOS3-Mini
echo   WSL distro           : %WSL_DISTRO%
echo   Docker image         : %DOCKER_IMAGE%
echo   Docker container     : %CONTAINER_NAME% ^(port 8001^)
echo   Model repo           : %MODEL_REPO%
echo   Client venv          : %VENV_DIR%
echo   Local tools GPU      : !GPU_NAME!
echo   Local tools CUDA     : !CUDA_VER!
echo   Local tools PyTorch  : !TORCH_INDEX! ^(torch/torchaudio 2.8.0, torchcodec 0.7.0 - pinned for pyannote.audio compatibility^)
echo   FFmpeg for torchcodec: !FFMPEG_BIN!
echo.
echo Run start.bat and select 'ZONOS3-Mini (WSL2)' to launch the app.
echo In the app click 'Download model' ^(Docker image + weights + serving
echo environment^), then 'Load model'.
echo ============================================================
echo.
echo [NOTE] To use the "I want dubbing" feature (speaker diarization):
echo 1. Accept terms at huggingface.co/pyannote/segmentation-3.0
echo 2. Accept terms at huggingface.co/pyannote/speaker-diarization-3.1
echo 3. Accept terms at huggingface.co/pyannote/speaker-diarization-community-1
echo 4. Create a token at huggingface.co/settings/tokens (Classic, Read)
echo ============================================================
pause
exit /b 0

:FIND_WINGET_FFMPEG
for %%R in ("%LOCALAPPDATA%\Microsoft\WinGet\Packages" "%ProgramFiles%\WinGet\Packages") do (
    if exist "%%~R" (
        for /d %%P in ("%%~R\BtbN.FFmpeg.GPL.Shared.7.1_*") do (
            for /f "delims=" %%D in ('dir /b /s /a-d "%%~P\avformat-*.dll" 2^>nul') do (
                if not defined FFMPEG_BIN set "FFMPEG_BIN=%%~dpD"
            )
        )
    )
)
exit /b 0