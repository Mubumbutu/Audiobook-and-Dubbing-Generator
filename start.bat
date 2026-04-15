@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul

echo ============================================================
echo TTS Launcher
echo ============================================================
echo.

if not exist "venvs" (
    echo [ERROR] Folder 'venvs' not found.
    echo Please run one of the install scripts from the 'install\' folder first.
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

set COUNT=0
for /d %%D in ("venvs\*") do (
    if exist "venvs\%%~nxD\Scripts\activate.bat" (
        set /a COUNT+=1
        set "VENV_!COUNT!=%%~nxD"
    )
)

if !COUNT! EQU 0 (
    echo [ERROR] No environments found in the 'venvs\' folder.
    echo Please run one of the install scripts from the 'install\' folder first.
    echo.
    echo Available install scripts:
    for %%F in ("install\*_install.bat") do echo %%~nxF
    echo.
    pause
    exit /b 1
)

echo Available TTS environments:
echo.
for /l %%I in (1,1,!COUNT!) do (
    echo [%%I] !VENV_%%I!
)
echo.

:PICK
set CHOICE=
set /p CHOICE=Select environment [1-!COUNT!]:
if "!CHOICE!"=="" goto PICK

set /a CHOICE_NUM=!CHOICE! 2>nul
if !CHOICE_NUM! LSS 1 (
    echo Invalid choice. Please try again.
    goto PICK
)
if !CHOICE_NUM! GTR !COUNT! (
    echo Invalid choice. Please try again.
    goto PICK
)

set "SELECTED=!VENV_%CHOICE_NUM%!"
set "ACTIVATE=venvs\!SELECTED!\Scripts\activate.bat"

echo.
echo [OK] Selected: !SELECTED!
echo [OK] Activating environment...
call "!ACTIVATE!"

echo [OK] Launching app\main.py...
echo ============================================================
echo.

python app\main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo [ERROR] The application exited with an error (code: %ERRORLEVEL%).
    echo Please copy the logs above and report this issue on GitHub.
    echo.
    pause
) else (
    echo.
    echo ============================================================
    echo Application closed successfully.
)