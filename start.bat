@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================================
echo  TTS Suite - Launcher
echo ============================================================
echo.

if not exist "venvs" (
    echo [ERROR] Folder 'venvs' not found.
    echo Please run install.bat and install at least one model first.
    echo.
    pause
    exit /b 1
)

if not exist "app\main.py" (
    echo [ERROR] app\main.py not found.
    echo Make sure the application code is located in the 'app\' folder.
    echo.
    pause
    exit /b 1
)

set "LABEL_venv_chatterbox=Chatterbox TTS"
set "LABEL_venv_fish_s2_pro=Fish Audio Speech-2 Pro"
set "LABEL_venv_moss=MOSS TTS"
set "LABEL_venv_omnivoice=OmniVoice"
set "LABEL_venv_qwen3=Qwen3 TTS"
set "LABEL_venv_tada=TADA TTS"
set "LABEL_venv_voxcpm2=VoxCPM 2"
set "LABEL_venv_supertonic=Supertonic 3"
set "LABEL_venv_piper=Piper TTS"
set "LABEL_venv_xttsv2=XTTS v2"

set COUNT=0
for /d %%D in ("venvs\*") do (
    if exist "venvs\%%~nxD\Scripts\activate.bat" (
        set /a COUNT+=1
        set "VENV_!COUNT!=%%~nxD"
    )
)

if !COUNT! EQU 0 (
    echo [ERROR] No environments found in the 'venvs\' folder.
    echo Please run install.bat and install at least one model first.
    echo.
    echo Available install scripts:
    for %%F in ("install\*_install.bat") do echo   %%~nxF
    echo.
    pause
    exit /b 1
)

echo  Available TTS environments:
echo.
for /l %%I in (1,1,!COUNT!) do (
    set "VNAME=!VENV_%%I!"
    set "DISP=!LABEL_%VNAME%!"
    if not defined DISP set "DISP=!VNAME!"
    echo  [%%I]  !DISP!  ^(!VNAME!^)
)
echo.

:PICK
set "CHOICE="
set /p CHOICE= Select environment [1-!COUNT!]: 
if "!CHOICE!"=="" goto :PICK

set /a CHOICE_NUM=!CHOICE! 2>nul
if !CHOICE_NUM! LSS 1 (
    echo  Invalid choice. Please enter a number between 1 and !COUNT!.
    goto :PICK
)
if !CHOICE_NUM! GTR !COUNT! (
    echo  Invalid choice. Please enter a number between 1 and !COUNT!.
    goto :PICK
)

set "SELECTED=!VENV_%CHOICE_NUM%!"
set "ACTIVATE=venvs\!SELECTED!\Scripts\activate.bat"

echo.
echo [OK] Selected : !SELECTED!
echo [OK] Activating environment...
call "!ACTIVATE!"

echo [OK] Launching app\main.py...
echo ============================================================
echo.

python app\main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo [ERROR] The application exited with an error ^(code: %ERRORLEVEL%^).
    echo Please copy the logs above and report this issue on GitHub.
    echo.
    pause
) else (
    echo.
    echo ============================================================
    echo  Application closed successfully.
    echo.
)
