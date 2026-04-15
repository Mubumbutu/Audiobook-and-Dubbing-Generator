@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem ============================================================
rem Paths relative to project root
rem ============================================================
set "ROOT=%~dp0.."
set "VENV_DIR=%ROOT%\venvs\venv_tada"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ=%~dp0tada_requirements.txt"

echo ============================================================
echo TADA TTS - Installation
echo ============================================================
echo.

if not exist "%REQ%" (
    echo [ERROR] tada_requirements.txt not found in %~dp0
    goto :ERROR
)

echo Checking Python...
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not found in PATH.
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    goto :ERROR
)
for /f "tokens=2 delims= " %%V in ('py -3 --version 2^>^&1') do set PYVER=%%V
echo [OK] Python %PYVER%

echo.
echo Detecting NVIDIA GPU...
set GPU_FOUND=0
set DRIVER_MAJOR=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set GPU_FOUND=1
    for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu=driver_version --format=csv 2^>nul ^| findstr /R "^[0-9]"') do (
        set DRIVER_MAJOR=%%a
        goto DRIVER_DONE
    )
)
:DRIVER_DONE
if !GPU_FOUND! EQU 1 (
    echo [OK] NVIDIA GPU detected. Driver major version: !DRIVER_MAJOR!
) else (
    echo [INFO] No NVIDIA GPU detected.
)

echo.
echo Choose installation type:
echo.
echo [1] CPU only
echo [2] GPU (NVIDIA CUDA)
echo.
:CHOICE
set USER_CHOICE=
set /p USER_CHOICE=Enter choice [1/2]: 
if "%USER_CHOICE%"=="1" goto CPU_MODE
if "%USER_CHOICE%"=="2" goto GPU_MODE
echo Invalid choice. Try again.
goto CHOICE

:CPU_MODE
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
set ONNX_PKG=onnxruntime
set INSTALL_MODE=cpu
goto MODE_DONE

:GPU_MODE
if !GPU_FOUND! EQU 0 (
    echo [WARNING] No GPU detected - falling back to CPU.
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    set ONNX_PKG=onnxruntime
    set INSTALL_MODE=cpu
    goto MODE_DONE
)
if !DRIVER_MAJOR! GEQ 560 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu126
    set ONNX_PKG=onnxruntime-gpu
) else if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    set ONNX_PKG=onnxruntime-gpu
) else if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    set ONNX_PKG=onnxruntime-gpu
) else (
    echo [WARNING] Driver too old ^(minimum 450^) - falling back to CPU.
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    set ONNX_PKG=onnxruntime
    set INSTALL_MODE=cpu
    goto MODE_DONE
)
set INSTALL_MODE=gpu

:MODE_DONE
echo [OK] Mode: !INSTALL_MODE!
echo [OK] PyTorch index: !TORCH_INDEX!

rem ============================================================
rem HuggingFace token GUI
rem Token is saved in project root: ROOT\.hf_token
rem ============================================================
set "TOKEN_PATH=%ROOT%\.hf_token"
set "PS1FILE=%TEMP%\tada_setup.ps1"

if exist "%PS1FILE%" del "%PS1FILE%"

>> "%PS1FILE%" echo Add-Type -AssemblyName System.Windows.Forms
>> "%PS1FILE%" echo Add-Type -AssemblyName System.Drawing
>> "%PS1FILE%" echo $tp = '%TOKEN_PATH%'
>> "%PS1FILE%" echo $form = New-Object System.Windows.Forms.Form
>> "%PS1FILE%" echo $form.Text = 'TADA TTS - Configuration'
>> "%PS1FILE%" echo $form.Size = New-Object System.Drawing.Size(540, 370)
>> "%PS1FILE%" echo $form.StartPosition = 'CenterScreen'
>> "%PS1FILE%" echo $form.FormBorderStyle = 'FixedDialog'
>> "%PS1FILE%" echo $form.MaximizeBox = $false
>> "%PS1FILE%" echo $form.MinimizeBox = $false
>> "%PS1FILE%" echo $l1 = New-Object System.Windows.Forms.Label
>> "%PS1FILE%" echo $l1.Text = 'Hugging Face Token:'
>> "%PS1FILE%" echo $l1.Location = New-Object System.Drawing.Point(20, 20)
>> "%PS1FILE%" echo $l1.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $form.Controls.Add($l1)
>> "%PS1FILE%" echo $txt = New-Object System.Windows.Forms.TextBox
>> "%PS1FILE%" echo $txt.Location = New-Object System.Drawing.Point(20, 42)
>> "%PS1FILE%" echo $txt.Size = New-Object System.Drawing.Size(480, 24)
>> "%PS1FILE%" echo $form.Controls.Add($txt)
>> "%PS1FILE%" echo $lNote = New-Object System.Windows.Forms.Label
>> "%PS1FILE%" echo $lNote.Text = 'When creating your token, enable the permission:'
>> "%PS1FILE%" echo $lNote.Location = New-Object System.Drawing.Point(20, 74)
>> "%PS1FILE%" echo $lNote.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $form.Controls.Add($lNote)
>> "%PS1FILE%" echo $lPerm = New-Object System.Windows.Forms.Label
>> "%PS1FILE%" echo $lPerm.Text = '"Read access to contents of all public gated repos you can access"'
>> "%PS1FILE%" echo $lPerm.Location = New-Object System.Drawing.Point(20, 94)
>> "%PS1FILE%" echo $lPerm.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $lPerm.ForeColor = [System.Drawing.Color]::DarkRed
>> "%PS1FILE%" echo $lPerm.Font = New-Object System.Drawing.Font($lPerm.Font, [System.Drawing.FontStyle]::Bold)
>> "%PS1FILE%" echo $form.Controls.Add($lPerm)
>> "%PS1FILE%" echo $l2 = New-Object System.Windows.Forms.Label
>> "%PS1FILE%" echo $l2.Text = 'Accept the model license on HuggingFace (click the link below):'
>> "%PS1FILE%" echo $l2.Location = New-Object System.Drawing.Point(20, 126)
>> "%PS1FILE%" echo $l2.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $form.Controls.Add($l2)
>> "%PS1FILE%" echo $lk1 = New-Object System.Windows.Forms.LinkLabel
>> "%PS1FILE%" echo $lk1.Text = 'Llama 3.2 3B - click to accept license (grants access to all models)'
>> "%PS1FILE%" echo $lk1.Location = New-Object System.Drawing.Point(20, 150)
>> "%PS1FILE%" echo $lk1.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $lk1.Add_LinkClicked({Start-Process 'https://huggingface.co/meta-llama/Llama-3.2-3B'})
>> "%PS1FILE%" echo $form.Controls.Add($lk1)
>> "%PS1FILE%" echo $lWarn = New-Object System.Windows.Forms.Label
>> "%PS1FILE%" echo $lWarn.Text = 'IMPORTANT: After submitting the license, you must wait for'
>> "%PS1FILE%" echo $lWarn.Location = New-Object System.Drawing.Point(20, 180)
>> "%PS1FILE%" echo $lWarn.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $lWarn.ForeColor = [System.Drawing.Color]::DarkRed
>> "%PS1FILE%" echo $lWarn.Font = New-Object System.Drawing.Font($lWarn.Font, [System.Drawing.FontStyle]::Bold)
>> "%PS1FILE%" echo $form.Controls.Add($lWarn)
>> "%PS1FILE%" echo $lWarn2 = New-Object System.Windows.Forms.Label
>> "%PS1FILE%" echo $lWarn2.Text = 'Meta approval (typically 10-20 minutes). Do NOT continue until'
>> "%PS1FILE%" echo $lWarn2.Location = New-Object System.Drawing.Point(20, 200)
>> "%PS1FILE%" echo $lWarn2.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $lWarn2.ForeColor = [System.Drawing.Color]::DarkRed
>> "%PS1FILE%" echo $lWarn2.Font = New-Object System.Drawing.Font($lWarn2.Font, [System.Drawing.FontStyle]::Bold)
>> "%PS1FILE%" echo $form.Controls.Add($lWarn2)
>> "%PS1FILE%" echo $lWarn3 = New-Object System.Windows.Forms.Label
>> "%PS1FILE%" echo $lWarn3.Text = 'you receive a confirmation email. Installation will fail without it.'
>> "%PS1FILE%" echo $lWarn3.Location = New-Object System.Drawing.Point(20, 220)
>> "%PS1FILE%" echo $lWarn3.Size = New-Object System.Drawing.Size(480, 18)
>> "%PS1FILE%" echo $lWarn3.ForeColor = [System.Drawing.Color]::DarkRed
>> "%PS1FILE%" echo $lWarn3.Font = New-Object System.Drawing.Font($lWarn3.Font, [System.Drawing.FontStyle]::Bold)
>> "%PS1FILE%" echo $form.Controls.Add($lWarn3)
>> "%PS1FILE%" echo $btn = New-Object System.Windows.Forms.Button
>> "%PS1FILE%" echo $btn.Text = 'Save and continue'
>> "%PS1FILE%" echo $btn.Location = New-Object System.Drawing.Point(175, 265)
>> "%PS1FILE%" echo $btn.Size = New-Object System.Drawing.Size(170, 34)
>> "%PS1FILE%" echo $btn.Add_Click({
>> "%PS1FILE%" echo   if ($txt.Text.Trim() -eq '') {
>> "%PS1FILE%" echo     [System.Windows.Forms.MessageBox]::Show('HuggingFace token is required.','TADA')
>> "%PS1FILE%" echo     return
>> "%PS1FILE%" echo   }
>> "%PS1FILE%" echo   [System.IO.File]::WriteAllText($tp, $txt.Text.Trim())
>> "%PS1FILE%" echo   $form.Close()
>> "%PS1FILE%" echo })
>> "%PS1FILE%" echo $form.Controls.Add($btn)
>> "%PS1FILE%" echo $form.AcceptButton = $btn
>> "%PS1FILE%" echo $form.ShowDialog()

powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%PS1FILE%"
del "%PS1FILE%" 2>nul

if not exist "%TOKEN_PATH%" (
    echo [ERROR] Token was not saved. Installation aborted.
    goto :ERROR
)
echo [OK] HF token saved.

rem ============================================================
rem [1/8] Remove old venv
rem ============================================================
echo.
echo [1/8] Removing old venv_tada...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    echo [OK] Old venv removed.
) else (
    echo [OK] No old venv found.
)

rem ============================================================
rem [2/8] Create venv in venvs\venv_tada
rem ============================================================
echo.
echo [2/8] Creating venvs\venv_tada...
if not exist "%ROOT%\venvs" mkdir "%ROOT%\venvs"
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create venv. & goto :ERROR )
echo [OK] venv created.

rem ============================================================
rem [3/8] Upgrade pip, setuptools, wheel
rem ============================================================
echo.
echo [3/8] Upgrading pip, setuptools and wheel...
"%VENV_PY%" -m pip install --upgrade pip 2>&1
"%VENV_PY%" -m pip install --upgrade "setuptools>=70.1" wheel 2>&1
if errorlevel 1 ( echo [ERROR] Failed to upgrade pip/setuptools/wheel. & goto :ERROR )
echo [OK] pip, setuptools and wheel upgraded.

rem ============================================================
rem [4/8] Install PyTorch
rem ============================================================
echo.
echo [4/8] Installing PyTorch 2.7.0...
"%VENV_PY%" -m pip install "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" ^
    --index-url !TORCH_INDEX! 2>&1
if errorlevel 1 ( echo [ERROR] PyTorch installation failed. & goto :ERROR )
echo [OK] PyTorch installed.
"%VENV_PY%" -c "import torch; print(f'[VERIFY] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

rem ============================================================
rem [5/8] Write constraint pins + pre-install HTTP stack
rem       Temp pins file stored in %TEMP%, not in project dir
rem ============================================================
echo.
echo [5/8] Writing constraint pins...
set "PINS=%TEMP%\tada_pins.txt"

(
    rem --- PyTorch stack ---
    echo torch==2.7.0
    echo torchvision==0.22.0
    echo torchaudio==2.7.0
    rem --- Core numeric ---
    echo numpy==2.4.3
    echo scipy==1.17.1
    rem --- Audio ---
    echo librosa==0.11.0
    echo lingua-language-detector==2.2.0
    echo sounddevice==0.5.5
    echo soundfile==0.13.1
    echo pyyaml==6.0.3
    echo cffi==2.0.0
    echo ctranslate2==4.7.1
    echo tqdm==4.67.3
    echo demucs==4.0.1
    rem --- UI ---
    echo PyQt6==6.10.2
    echo PyQt6-sip==13.11.1
    rem --- HuggingFace ---
    echo huggingface-hub==1.7.1
    echo hf_transfer==0.1.9
    echo hf_xet==1.4.2
    echo tokenizers==0.22.2
    echo safetensors==0.7.0
    echo transformers==5.3.0
    rem --- Diarization ---
    echo omegaconf==2.3.0
    rem --- HTTP stack (stops resolver backtracking) ---
    echo certifi==2026.2.25
    echo idna==3.11
    echo h11==0.16.0
    echo anyio==4.12.1
    echo httpcore==1.0.9
    echo httpx==0.28.1
    rem --- Other transitive deps from uv.lock ---
    echo click==8.3.1
    echo colorama==0.4.6
    echo exceptiongroup==1.3.1
    echo filelock==3.25.2
    echo fsspec==2026.2.0
    echo jinja2==3.1.6
    echo markdown-it-py==4.0.0
    echo markupsafe==3.0.3
    echo mdurl==0.1.2
    echo mpmath==1.3.0
    echo networkx==3.6.1
    echo packaging==26.0
    echo pycparser==3.0
    echo pygments==2.19.2
    echo regex==2026.2.28
    echo rich==14.3.3
    echo setuptools==82.0.1
    echo shellingham==1.5.4
    echo sympy==1.14.0
    echo typer==0.24.1
    echo typing-extensions==4.15.0
) > "%PINS%"
echo [OK] Constraint pins written.

echo.
echo [5b/8] Pre-installing HTTP stack and key packages...
"%VENV_PY%" -m pip install ^
    "certifi==2026.2.25" ^
    "idna==3.11" ^
    "h11==0.16.0" ^
    "anyio==4.12.1" ^
    "httpcore==1.0.9" ^
    "httpx==0.28.1" ^
    "huggingface-hub==1.7.1" ^
    "omegaconf==2.3.0" ^
    "rich==14.3.3" ^
    "click==8.3.1" ^
    "typer==0.24.1" ^
    --prefer-binary 2>&1
if errorlevel 1 ( echo [ERROR] Pre-install step failed. & goto :ERROR )
echo [OK] HTTP stack and key packages pre-installed.

rem ============================================================
rem [6/8] Install onnxruntime
rem ============================================================
echo.
"%VENV_PY%" -m pip uninstall onnxruntime onnxruntime-gpu -y 2>&1
echo [6/8] Installing onnxruntime ^(!ONNX_PKG! 1.24.4^)...
"%VENV_PY%" -m pip install "!ONNX_PKG!==1.24.4" --prefer-binary 2>&1
if errorlevel 1 ( echo [ERROR] onnxruntime installation failed. & goto :ERROR )
echo [OK] onnxruntime installed.

rem ============================================================
rem [7/8] Install packages that require --no-deps
rem ============================================================
echo.
echo [7/8] Installing packages with --no-deps ^(stale/conflicting PyPI metadata^)...

echo   [7a] faster-whisper 1.2.1...
"%VENV_PY%" -m pip install "faster-whisper==1.2.1" --no-deps 2>&1
if errorlevel 1 ( echo [ERROR] faster-whisper failed. & goto :ERROR )

echo   [7b] descript-audiotools 0.7.2...
"%VENV_PY%" -m pip install "descript-audiotools==0.7.2" --no-deps 2>&1
if errorlevel 1 ( echo [ERROR] descript-audiotools failed. & goto :ERROR )

echo   [7c] descript-audiotools runtime deps ^(all except protobuf^)...
"%VENV_PY%" -m pip install ^
    "argbind>=0.3.2" ^
    "ffmpy" ^
    "flatten-dict" ^
    "importlib-resources" ^
    "ipython" ^
    "markdown2" ^
    "pyloudnorm" ^
    "pystoi" ^
    "randomname" ^
    "tensorboard" ^
    --prefer-binary 2>&1
if errorlevel 1 ( echo [ERROR] descript-audiotools runtime deps failed. & goto :ERROR )
"%VENV_PY%" -m pip install "torch-stoi" --no-deps 2>&1
echo [OK] descript-audiotools and its runtime deps installed.

echo   [7d] descript-audio-codec...
"%VENV_PY%" -m pip install "descript-audio-codec>=1.0.0" --no-deps 2>&1
if errorlevel 1 ( echo [ERROR] descript-audio-codec failed. & goto :ERROR )

echo   [7e] hume-tada 0.1.8...
"%VENV_PY%" -m pip install "hume-tada==0.1.8" --no-deps 2>&1
if errorlevel 1 ( echo [ERROR] hume-tada failed. & goto :ERROR )

echo [OK] All --no-deps packages installed.

rem ============================================================
rem [8/8] Install remaining dependencies from tada_requirements.txt
rem ============================================================
echo.
echo [8/8] Installing remaining dependencies from tada_requirements.txt...
"%VENV_PY%" -m pip install ^
    -r "%REQ%" ^
    --prefer-binary ^
    --extra-index-url !TORCH_INDEX! ^
    --constraint "%PINS%" ^
    2>&1
if errorlevel 1 (
    del "%PINS%" 2>nul
    echo [ERROR] Dependency installation failed.
    goto :ERROR
)
del "%PINS%" 2>nul
echo [OK] Dependencies installed.

rem ============================================================
rem Verification
rem ============================================================
echo.
echo ============================================================
echo Verifying installation...
echo ============================================================
"%VENV_PY%" -c "import torch; print(f'  torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
"%VENV_PY%" -c "import torchaudio; print('  torchaudio OK')"
"%VENV_PY%" -c "import faster_whisper; print('  faster_whisper OK')"
"%VENV_PY%" -c "import transformers; print(f'  transformers {transformers.__version__}')"
"%VENV_PY%" -c "import huggingface_hub; print(f'  huggingface_hub {huggingface_hub.__version__}')"
"%VENV_PY%" -c "import audiotools; print('  descript-audiotools OK')" 2>&1
"%VENV_PY%" -c "import dac; print('  descript-audio-codec OK')" 2>&1

rem TADA imports run from %TEMP% to prevent local tada/ folder shadowing
echo   --- TADA module ---
pushd "%TEMP%"
"%VENV_PY%" -c "import tada; print('  tada OK')" 2>&1
"%VENV_PY%" -c "from tada.modules.tada import TadaForCausalLM; print('  TadaForCausalLM OK')" 2>&1
popd

echo.
echo ============================================================
echo Installation complete!
echo Mode: !INSTALL_MODE! ^| PyTorch index: !TORCH_INDEX!
echo Venv location: venvs\venv_tada
echo.
echo NOTE: pip will show warnings about protobuf, transformers
echo and onnxruntime conflicts. These are stale PyPI metadata
echo issues and can be safely ignored - the environment works.
echo.
echo Run start.bat to launch the application.
echo ============================================================
echo.
pause
exit /b 0

:ERROR
echo.
echo ============================================================
echo Installation failed. See errors above.
echo ============================================================
pause
exit /b 1