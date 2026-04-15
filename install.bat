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
echo  [1]  Chatterbox TTS
echo  [2]  Fish Audio Speech-2 Pro
echo  [3]  MOSS TTS
echo  [4]  OmniVoice
echo  [5]  Qwen3 TTS
echo  [6]  TADA TTS  (requires HuggingFace token + Meta license approval)
echo  [7]  VoxCPM 2
echo.
echo  [0]  Exit
echo.

:PICK
set CHOICE=
set /p CHOICE=Enter choice [0-7]: 

if "!CHOICE!"=="0" goto :EXIT
if "!CHOICE!"=="1" goto :CHATTERBOX
if "!CHOICE!"=="2" goto :FISH
if "!CHOICE!"=="3" goto :MOSS
if "!CHOICE!"=="4" goto :OMNIVOICE
if "!CHOICE!"=="5" goto :QWEN3
if "!CHOICE!"=="6" goto :TADA
if "!CHOICE!"=="7" goto :VOXCPM2

echo  Invalid choice. Please enter a number from 0 to 7.
goto :PICK

:CHATTERBOX
set "SCRIPT=%APP_ROOT%install\chatterbox_install.bat"
set "MODEL=Chatterbox TTS"
goto :RUN

:FISH
set "SCRIPT=%APP_ROOT%install\fish_s2_pro_install.bat"
set "MODEL=Fish Audio Speech-2 Pro"
goto :RUN

:MOSS
set "SCRIPT=%APP_ROOT%install\moss_install.bat"
set "MODEL=MOSS TTS"
goto :RUN

:OMNIVOICE
set "SCRIPT=%APP_ROOT%install\omnivoice_install.bat"
set "MODEL=OmniVoice"
goto :RUN

:QWEN3
set "SCRIPT=%APP_ROOT%install\qwen3_install.bat"
set "MODEL=Qwen3 TTS"
goto :RUN

:TADA
set "SCRIPT=%APP_ROOT%install\tada_install.bat"
set "MODEL=TADA TTS"
goto :RUN

:VOXCPM2
set "SCRIPT=%APP_ROOT%install\voxcpm2_install.bat"
set "MODEL=VoxCPM 2"
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
