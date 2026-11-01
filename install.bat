@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "APP_ROOT=%~dp0"

echo ============================================================
echo  TTS Suite - Model Installer
echo ============================================================
echo.
echo  Select the TTS model you want to install:
echo.
echo  [1]   Audio8 TTS Preview and Realistic 0.6B
echo  [2]   Chatterbox TTS
echo  [3]   Fish Audio Speech-2 Pro
echo  [4]   Higgs TTS 3  (SGLang-Omni via WSL2 / Docker, requires NVIDIA GPU)
echo  [5]   KittenTTS 2
echo  [6]   Maya 1 + Veena TTS
echo  [7]   Maya 2 Native
echo  [8]   Ming-3.0-tts 0.5B
echo  [9]   MOSS TTS
echo  [10]  OmniVoice
echo  [11]  Piper TTS
echo  [12]  Qwen3 TTS
echo  [13]  Supertonic 3
echo  [14]  TADA TTS  (requires HuggingFace token + Meta license approval)
echo  [15]  VoxCPM 2
echo  [16]  XTTS v2  (Coqui TTS)
echo.
echo  [0]   Exit
echo.

:PICK
set CHOICE=
set /p CHOICE=Enter choice [0-16]: 

if "!CHOICE!"=="0"  goto :EXIT
if "!CHOICE!"=="1"  goto :AUDIO8
if "!CHOICE!"=="2"  goto :CHATTERBOX
if "!CHOICE!"=="3"  goto :FISH
if "!CHOICE!"=="4"  goto :HIGGSWSL
if "!CHOICE!"=="5"  goto :KITTEN2
if "!CHOICE!"=="6"  goto :MAYA1
if "!CHOICE!"=="7"  goto :MAYA2
if "!CHOICE!"=="8"  goto :MING3
if "!CHOICE!"=="9"  goto :MOSS
if "!CHOICE!"=="10" goto :OMNIVOICE
if "!CHOICE!"=="11" goto :PIPER
if "!CHOICE!"=="12" goto :QWEN3
if "!CHOICE!"=="13" goto :SUPERTONIC
if "!CHOICE!"=="14" goto :TADA
if "!CHOICE!"=="15" goto :VOXCPM2
if "!CHOICE!"=="16" goto :XTTSV2

echo  Invalid choice. Please enter a number from 0 to 16.
goto :PICK

:AUDIO8
set "SCRIPT=%APP_ROOT%install\audio8_install.bat"
set "MODEL=Audio8 TTS Preview 0.6B"
goto :RUN

:CHATTERBOX
set "SCRIPT=%APP_ROOT%install\chatterbox_install.bat"
set "MODEL=Chatterbox TTS"
goto :RUN

:FISH
set "SCRIPT=%APP_ROOT%install\fish_s2_pro_install.bat"
set "MODEL=Fish Audio Speech-2 Pro"
goto :RUN

:HIGGSWSL
set "SCRIPT=%APP_ROOT%install\higgs_wsl_install.bat"
set "MODEL=Higgs TTS 3 (SGLang-Omni / WSL2 / Docker)"
goto :RUN

:KITTEN2
set "SCRIPT=%APP_ROOT%install\kitten2_install.bat"
set "MODEL=KittenTTS 2"
goto :RUN

:MAYA1
set "SCRIPT=%APP_ROOT%install\maya_veena_install.bat"
set "MODEL=Maya 1 + Veena TTS"
goto :RUN

:MAYA2
set "SCRIPT=%APP_ROOT%install\maya2_install.bat"
set "MODEL=Maya 2 Native"
goto :RUN

:MING3
set "SCRIPT=%APP_ROOT%install\ming3_install.bat"
set "MODEL=Ming-3.0-tts 0.5B"
goto :RUN

:MOSS
set "SCRIPT=%APP_ROOT%install\moss_install.bat"
set "MODEL=MOSS TTS"
goto :RUN

:OMNIVOICE
set "SCRIPT=%APP_ROOT%install\omnivoice_install.bat"
set "MODEL=OmniVoice"
goto :RUN

:PIPER
set "SCRIPT=%APP_ROOT%install\piper_install.bat"
set "MODEL=Piper TTS"
goto :RUN

:QWEN3
set "SCRIPT=%APP_ROOT%install\qwen3_install.bat"
set "MODEL=Qwen3 TTS"
goto :RUN

:SUPERTONIC
set "SCRIPT=%APP_ROOT%install\supertonic_install.bat"
set "MODEL=Supertonic 3"
goto :RUN

:TADA
set "SCRIPT=%APP_ROOT%install\tada_install.bat"
set "MODEL=TADA TTS"
goto :RUN

:VOXCPM2
set "SCRIPT=%APP_ROOT%install\voxcpm2_install.bat"
set "MODEL=VoxCPM 2"
goto :RUN

:XTTSV2
set "SCRIPT=%APP_ROOT%install\xttsv2_install.bat"
set "MODEL=XTTS v2 (Coqui TTS)"
goto :RUN

:RUN
echo.
if not exist "!SCRIPT!" (
    echo  [ERROR] Installer not found: !SCRIPT!
    echo  Make sure the 'install\' folder is present next to this file.
    echo.
    pause
    exit /b 1
)

echo  Starting installation: !MODEL!
echo  Script: !SCRIPT!
echo ============================================================
echo.

call "!SCRIPT!"

echo.
echo ============================================================
echo  !MODEL! installer finished.
echo  Run start.bat to launch the application.
echo ============================================================
echo.
pause
exit /b 0

:EXIT
echo.
echo  Exiting installer. No changes were made.
echo.
exit /b 0
