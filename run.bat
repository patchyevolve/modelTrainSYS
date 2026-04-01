@echo off
setlocal EnableDelayedExpansion

chcp 65001 >nul 2>&1
title ML Training System

:: ═══════════════════════════════════════════════════════════════════════════════
:: CONFIG
:: ═══════════════════════════════════════════════════════════════════════════════
set "SCRIPT_DIR=%~dp0"
set "LOG_FILE=%SCRIPT_DIR%mlsys.log"
set "PYTHON="
set "VENV_DIR="
set "PYVER="

:: ═══════════════════════════════════════════════════════════════════════════════
:: ANSI COLORS
:: ═══════════════════════════════════════════════════════════════════════════════
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "RESET=%ESC%[0m"
set "BOLD=%ESC%[1m"
set "DIM=%ESC%[2m"
set "CYAN=%ESC%[96m"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "RED=%ESC%[91m"
set "MAGENTA=%ESC%[95m"
set "WHITE=%ESC%[97m"

:: ═══════════════════════════════════════════════════════════════════════════════
:: INIT LOG
:: ═══════════════════════════════════════════════════════════════════════════════
call :log "ML Training System started"

:: ═══════════════════════════════════════════════════════════════════════════════
:: FIND PYTHON
:: ═══════════════════════════════════════════════════════════════════════════════
call :find_python
if "!PYTHON!"=="" (
    call :error "Python not found. Install from https://python.org"
    call :wait
    exit /b 1
)

:: ═══════════════════════════════════════════════════════════════════════════════
:: PARSE ARGUMENTS
:: ═══════════════════════════════════════════════════════════════════════════════
if not "%~1"=="" (
    call :parse_arg %1 %2 %3 %4 %5
    exit /b %ERRORLEVEL%
)

:: ═══════════════════════════════════════════════════════════════════════════════
:: MAIN MENU
:: ═══════════════════════════════════════════════════════════════════════════════
:menu
cls
call :header "ML TRAINING SYSTEM"
echo.
echo  %DIM%Python: !PYVER!  ^|  Venv: !VENV_DIR!  ^|  GPU: !GPU_NAME!%RESET%
if defined GROQ_API_KEY (
    echo  %GREEN%  Groq API: configured%RESET%
) else (
    echo  %YELLOW%  Groq API: not set%RESET%
)
echo.
echo  %WHITE%%BOLD%  ═══ TRAINING ═══%RESET%
echo  %GREEN%  [1]%RESET%  Training GUI         Interactive drag-drop training
echo  %GREEN%  [2]%RESET%  Quick Train Text     Text generation (.txt/.jsonl)
echo  %GREEN%  [3]%RESET%  Quick Train Image   Image classification
echo  %GREEN%  [4]%RESET%  Quick Train Cyber   Network intrusion detection
echo.
echo  %WHITE%%BOLD%  ═══ INFERENCE ═══%RESET%
echo  %CYAN%  [5]%RESET%  Chat with Model      Interactive model chat
echo  %CYAN%  [6]%RESET%  Batch Inference      Classify/generate on data file
echo  %CYAN%  [7]%RESET%  List Models          Show trained checkpoints
echo  %CYAN%  [8]%RESET%  Compare Models       Compare two model outputs
echo.
echo  %WHITE%%BOLD%  ═══ SYSTEM ═══%RESET%
echo  %MAGENTA%  [9]%RESET%  Auto-Upgrade        LLM-powered architecture upgrade
echo  %MAGENTA%  [A]%RESET%  GPU Info             Check device/driver status
echo  %MAGENTA%  [B]%RESET%  Health Check         Verify all dependencies
echo  %MAGENTA%  [C]%RESET%  Setup Wizard         First-time setup (venv, deps)
echo.
echo  %YELLOW%  [S]%RESET%  Settings            Configure API keys, paths
echo  %YELLOW%  [L]%RESET%  View Logs           Show recent log entries
echo  %YELLOW%  [X]%RESET%  Exit
echo.
set /p "CHOICE=%BOLD%  > %RESET%"

if "%CHOICE%"=="1" goto :do_gui
if "%CHOICE%"=="2" goto :do_train_text
if "%CHOICE%"=="3" goto :do_train_image
if "%CHOICE%"=="4" goto :do_train_cyber
if "%CHOICE%"=="5" goto :do_chat
if "%CHOICE%"=="6" goto :do_inference
if "%CHOICE%"=="7" goto :do_list
if "%CHOICE%"=="8" goto :do_compare
if "%CHOICE%"=="9" goto :do_upgrade
if /i "%CHOICE%"=="A" goto :do_gpu
if /i "%CHOICE%"=="B" goto :do_check
if /i "%CHOICE%"=="C" goto :do_setup
if /i "%CHOICE%"=="S" goto :do_settings
if /i "%CHOICE%"=="L" goto :do_logs
if /i "%CHOICE%"=="X" goto :do_exit

echo.  %RED%Invalid choice%RESET%
timeout /t 1 >nul
goto :menu

:: ═══════════════════════════════════════════════════════════════════════════════
:: TRAINING ACTIONS
:: ═══════════════════════════════════════════════════════════════════════════════

:do_gui
call :header "TRAINING GUI"
call :log "Launching Training GUI"
!PYTHON! start.py --ui 2>>"%LOG_FILE%"
call :wait
goto :menu

:do_train_text
call :header "QUICK TRAIN - TEXT"
echo  %DIM%Train text generation model%RESET%
echo.
set /p "DATA=  %BOLD%Data file (.txt/.jsonl): %RESET%"
if "!DATA!"=="" set "DATA=randomDATA\big.txt"
set /p "EPOCHS=  %BOLD%Epochs [5]: %RESET%"
if "!EPOCHS!"=="" set "EPOCHS=5"
echo.
call :log "Training text: !DATA!, epochs=!EPOCHS!"
!PYTHON! trainer.py --data "!DATA!" --type text --epochs !EPOCHS! 2>>"%LOG_FILE%"
if errorlevel 1 (
    echo.  %RED%Training failed%RESET%
) else (
    echo.  %GREEN%Training complete!%RESET%
)
call :wait
goto :menu

:do_train_image
call :header "QUICK TRAIN - IMAGE"
echo  %DIM%Train image classification model%RESET%
echo.
set /p "DATA=  %BOLD%Data folder: %RESET%"
if "!DATA!"=="" (
    echo  %YELLOW%No folder specified%RESET%
    call :wait
    goto :menu
)
set /p "EPOCHS=  %BOLD%Epochs [10]: %RESET%"
if "!EPOCHS!"=="" set "EPOCHS=10"
echo.
call :log "Training image: !DATA!, epochs=!EPOCHS!"
!PYTHON! trainer.py --data "!DATA!" --type image --epochs !EPOCHS! 2>>"%LOG_FILE%"
if errorlevel 1 (
    echo.  %RED%Training failed%RESET%
) else (
    echo.  %GREEN%Training complete!%RESET%
)
call :wait
goto :menu

:do_train_cyber
call :header "QUICK TRAIN - CYBERSECURITY"
call :log "Training cybersecurity model"
!PYTHON! trainer.py --data randomDATA\cybersecurity_intrusion_data.csv --type cybersecurity --epochs 50 2>>"%LOG_FILE%"
if errorlevel 1 (
    echo.  %RED%Training failed%RESET%
) else (
    echo.  %GREEN%Training complete!%RESET%
)
call :wait
goto :menu

:: ═══════════════════════════════════════════════════════════════════════════════
:: INFERENCE ACTIONS
:: ═══════════════════════════════════════════════════════════════════════════════

:do_chat
call :header "CHAT WITH MODEL"
call :log "Opening chat"
!PYTHON! start.py --list 2>>"%LOG_FILE%"
echo.
set /p "MODEL=  %BOLD%Model (Enter = latest): %RESET%"
echo.
call :log "Chat with: !MODEL!"
if "!MODEL!"=="" (
    !PYTHON! start.py --chat 2>>"%LOG_FILE%"
) else (
    !PYTHON! start.py --chat !MODEL! 2>>"%LOG_FILE%"
)
call :wait
goto :menu

:do_inference
call :header "BATCH INFERENCE"
!PYTHON! start.py --list 2>>"%LOG_FILE%"
echo.
set /p "MODEL=  %BOLD%Model: %RESET%"
set /p "DATA=   %BOLD%Data file: %RESET%"
set /p "OUTPUT= %BOLD%Output file [predictions.csv]: %RESET%"
if "!OUTPUT!"=="" set "OUTPUT=predictions.csv"
echo.
call :log "Inference: !MODEL!, data=!DATA!"
!PYTHON! inference.py --model "trained_models\!MODEL!.pt" --data "!DATA!" --save --output "!OUTPUT!" 2>>"%LOG_FILE%"
echo.  %GREEN%Results saved to !OUTPUT!%RESET%
call :wait
goto :menu

:do_list
call :header "TRAINED MODELS"
!PYTHON! start.py --list 2>>"%LOG_FILE%"
echo.
echo  %WHITE%  [D]%RESET% Delete model
echo  %WHITE%  [E]%RESET% Export model
echo  %WHITE%  [B]%RESET% Backup all models
echo.
set /p "CMD=%BOLD%  > %RESET%"
if /i "!CMD!"=="D" goto :delete_model
if /i "!CMD!"=="E" goto :export_model
if /i "!CMD!"=="B" goto :backup_models
goto :menu

:delete_model
set /p "NAME=  %BOLD%Model name to delete: %RESET%"
if exist "trained_models\!NAME!.pt" (
    del "trained_models\!NAME!.pt"
    del "trained_models\!NAME!.json"
    echo  %GREEN%Deleted !NAME!%RESET%
    call :log "Deleted model: !NAME!"
) else (
    echo  %RED%Model not found%RESET%
)
call :wait
goto :menu

:export_model
set /p "NAME=  %BOLD%Model name to export: %RESET%"
set /p "PATH=  %BOLD%Export path: %RESET%"
if exist "trained_models\!NAME!.pt" (
    copy "trained_models\!NAME!.pt" "!PATH!"
    copy "trained_models\!NAME!.json" "!PATH!" 2>nul
    echo  %GREEN%Exported to !PATH!%RESET%
    call :log "Exported model: !NAME! to !PATH!"
) else (
    echo  %RED%Model not found%RESET%
)
call :wait
goto :menu

:backup_models
set "BACKUP_DIR=backups\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
mkdir "!BACKUP_DIR!" 2>nul
xcopy /E /I /Y "trained_models" "!BACKUP_DIR!" >nul
echo  %GREEN%Backed up to !BACKUP_DIR!%RESET%
call :log "Backup created: !BACKUP_DIR!"
call :wait
goto :menu

:do_compare
call :header "COMPARE MODELS"
!PYTHON! start.py --list 2>>"%LOG_FILE%"
echo.
set /p "MODEL1=  %BOLD%Model 1: %RESET%"
set /p "MODEL2=  %BOLD%Model 2: %RESET%"
set /p "INPUT=   %BOLD%Test input: %RESET%"
echo.
!PYTHON! -c "from inference import compare_models; compare_models('!MODEL1!', '!MODEL2!', '!INPUT!')" 2>>"%LOG_FILE%"
call :wait
goto :menu

:: ═══════════════════════════════════════════════════════════════════════════════
:: SYSTEM ACTIONS
:: ═══════════════════════════════════════════════════════════════════════════════

:do_upgrade
call :header "AUTO-UPGRADE"
if not defined GROQ_API_KEY (
    echo  %RED%[ERROR]%RESET% GROQ_API_KEY not set
    echo  Run: setx GROQ_API_KEY "your-key"
    call :wait
    goto :menu
)
echo  %WHITE%  [1]%RESET% GUI Upgrade         Visual interface
echo  %WHITE%  [2]%RESET% Smart Upgrade      Full project analysis with LLM
echo.
set /p "OPT=%BOLD%  > %RESET%"
if "!OPT!"=="1" (
    call :log "Opening Auto-Upgrade GUI"
    !PYTHON! start.py --upgrade 2>>"%LOG_FILE%"
) else if "!OPT!"=="2" (
    call :log "Running Smart Upgrade"
    !PYTHON! start.py --smart 2>>"%LOG_FILE%"
)
call :wait
goto :menu

:do_gpu
call :header "GPU / DEVICE INFO"
!PYTHON! -c "
from device_manager import device_info, get_best_device
print(device_info())
print()
for params, batch in [(1000000, 32), (10000000, 64)]:
    dev, name = get_best_device(params, batch)
    print(f'  {params//1000000}M params, batch={batch}: {name}')
" 2>>"%LOG_FILE%"
call :wait
goto :menu

:do_check
call :header "HEALTH CHECK"
echo.
call :check_python
call :check_torch
call :check_deps
call :check_gpu
call :check_models
echo.
echo  %WHITE%  Summary:%RESET%
if "!STATUS!"=="OK" (
    echo  %GREEN%  All systems operational%RESET%
) else (
    echo  %YELLOW%  !STATUS!%RESET%
)
call :log "Health check complete: !STATUS!"
call :wait
goto :menu

:do_setup
call :header "SETUP WIZARD"
echo.
echo  %WHITE%Step 1: Create virtual environment? [Y/n]%RESET%
set /p "YN=%BOLD%  > %RESET%"
if /i "!YN!" neq "N" (
    call :create_venv
)
echo.
echo  %WHITE%Step 2: Install dependencies? [Y/n]%RESET%
set /p "YN=%BOLD%  > %RESET%"
if /i "!YN!" neq "N" (
    call :install_deps
)
echo.
echo  %WHITE%Step 3: Set GROQ_API_KEY? [y/N]%RESET%
set /p "YN=%BOLD%  > %RESET%"
if /i "!YN!"=="Y" (
    set /p "KEY=  %BOLD%API Key: %RESET%"
    setx GROQ_API_KEY "!KEY!"
    set "GROQ_API_KEY=!KEY!"
)
echo.
call :log "Setup complete"
echo  %GREEN%Setup complete! Restart run.bat for changes to take effect.%RESET%
call :wait
goto :menu

:do_settings
call :header "SETTINGS"
echo.
echo  %WHITE%  Current settings:%RESET%
echo  %DIM%  GROQ_API_KEY: !GROQ_API_KEY:~0,10!...%RESET%
echo  %DIM%  Python: !PYTHON!%RESET%
echo  %DIM%  Venv: !VENV_DIR!%RESET%
echo.
echo  %WHITE%  [1]%RESET% Set GROQ_API_KEY
echo  %WHITE%  [2]%RESET% Set custom Python path
echo  %WHITE%  [3]%RESET% Toggle verbose logging
echo  %WHITE%  [B]%RESET% Back
echo.
set /p "CMD=%BOLD%  > %RESET%"
if "!CMD!"=="1" (
    set /p "KEY=  %BOLD%GROQ_API_KEY: %RESET%"
    setx GROQ_API_KEY "!KEY!"
    set "GROQ_API_KEY=!KEY!"
)
if "!CMD!"=="2" (
    set /p "PY=  %BOLD%Python path: %RESET%"
    set "PYTHON=!PY!"
)
goto :do_settings

:do_logs
call :header "LOG FILE"
if exist "%LOG_FILE%" (
    type "%LOG_FILE%"
) else (
    echo  %DIM%No log file yet%RESET%
)
call :wait
goto :menu

:do_exit
call :log "Exited"
echo.  %DIM%Goodbye!%RESET%
echo.
exit /b 0

:: ═══════════════════════════════════════════════════════════════════════════════
:: ARGUMENT PARSER
:: ═══════════════════════════════════════════════════════════════════════════════
:parse_arg
set "ARG=%~1"
if /i "%ARG%"=="gui"       goto :do_gui
if /i "%ARG%"=="chat"      goto :do_chat
if /i "%ARG%"=="infer"     goto :do_inference
if /i "%ARG%"=="train"    goto :do_train_text
if /i "%ARG%"=="list"      goto :do_list
if /i "%ARG%"=="upgrade"   goto :do_upgrade
if /i "%ARG%"=="smart"     goto :do_smart_upgrade
if /i "%ARG%"=="check"     goto :do_check
if /i "%ARG%"=="setup"     goto :do_setup
if /i "%ARG%"=="gpu"       goto :do_gpu
if /i "%ARG%"=="help"      goto :do_help
echo  %YELLOW%Unknown: %ARG%%RESET%
echo  Valid: gui chat infer train list upgrade check setup gpu help
exit /b 1

:do_help
echo  Usage: run.bat [command]
echo.
echo  Commands:
echo    gui       - Training GUI
echo    chat      - Chat with model
echo    infer     - Batch inference
echo    train     - Quick text training
echo    list      - List models
echo    upgrade   - Auto-upgrade
echo    check     - Health check
echo    setup     - Setup wizard
echo    gpu       - GPU info
exit /b 0

:: ═══════════════════════════════════════════════════════════════════════════════
:: HELPER FUNCTIONS
:: ═══════════════════════════════════════════════════════════════════════════════

:find_python
if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"
    set "VENV_DIR=venv"
) else if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
    set "VENV_DIR=.venv"
) else (
    for %%P in (python python3 py) do (
        if "!PYTHON!"=="" (
            %%P --version >nul 2>&1 && set "PYTHON=%%P"
        )
    )
)
for /f "tokens=2" %%V in ('!PYTHON! --version 2^>^&1') do set "PYVER=%%V"
call :detect_gpu
exit /b 0

:detect_gpu
!PYTHON! -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>nul >"%TEMP%\gpu.txt"
set /p GPU_NAME=<"%TEMP%\gpu.txt"
if "!GPU_NAME!"=="" set "GPU_NAME=Detecting..."
exit /b 0

:create_venv
echo  Creating venv...
!PYTHON! -m venv venv
if exist "venv\Scripts\activate.bat" (
    echo  %GREEN%  venv created. Run 'venv\Scripts\activate' to activate%RESET%
    set "PYTHON=venv\Scripts\python.exe"
    set "VENV_DIR=venv"
) else (
    echo  %RED%  Failed to create venv%RESET%
)
exit /b 0

:install_deps
echo  Installing dependencies...
!PYTHON! -m pip install --upgrade pip -q
!PYTHON! -m pip install torch torchvision numpy pandas Pillow -q
!PYTHON! -m pip install tkinterdnd2 torch-directml -q
echo  %GREEN%  Done%RESET%
exit /b 0

:check_python
!PYTHON! -c "import sys; sys.exit(0)" 2>nul
if errorlevel 1 (
    echo  %RED%  [FAIL] Python module error%RESET%
    set "STATUS=Python error"
) else (
    echo  %GREEN%  [OK]   Python !PYVER!%RESET%
)
exit /b 0

:check_torch
!PYTHON! -c "import torch; print(torch.__version__)" 2>nul >"%TEMP%\tv.txt"
set /p TORCH_VER=<"%TEMP%\tv.txt"
if defined TORCH_VER (
    echo  %GREEN%  [OK]   PyTorch !TORCH_VER!%RESET%
) else (
    echo  %RED%  [FAIL] PyTorch not installed%RESET%
    set "STATUS=Missing PyTorch"
)
exit /b 0

:check_deps
!PYTHON! -c "import numpy; import pandas; import PIL" 2>nul
if errorlevel 1 (
    echo  %RED%  [FAIL] Missing numpy/pandas/Pillow%RESET%
    set "STATUS=Missing deps"
) else (
    echo  %GREEN%  [OK]   numpy, pandas, Pillow%RESET%
)
exit /b 0

:check_gpu
!PYTHON! -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>nul >"%TEMP%\gpu_check.txt"
set /p GPU_CHECK=<"%TEMP%\gpu_check.txt"
if "!GPU_CHECK!"=="CUDA" (
    echo  %GREEN%  [OK]   GPU: CUDA%RESET%
) else (
    echo  %YELLOW%  [WARN] GPU: !GPU_CHECK!%RESET%
)
exit /b 0

:check_models
if exist "trained_models\*.pt" (
    dir /B "trained_models\*.pt" 2>nul >"%TEMP%\models.txt"
    find /c /v "" "%TEMP%\models.txt" >"%TEMP%\count.txt"
    set /p MODEL_COUNT=<"%TEMP%\count.txt"
    echo  %GREEN%  [OK]   !MODEL_COUNT! trained model(s)%RESET%
) else (
    echo  %YELLOW%  [WARN] No trained models%RESET%
    set "STATUS=No models"
)
exit /b 0

:header
cls
echo.
echo  %CYAN%%BOLD%═══════════════════════════════════════════════════════════════%RESET%
echo  %CYAN%%BOLD%  %~1%RESET%
echo  %CYAN%%BOLD%═══════════════════════════════════════════════════════════════%RESET%
exit /b 0

:log
echo [%date% %time:~0,8%] %~1 >> "%LOG_FILE%"
exit /b 0

:wait
echo.
set /p "DUMMY=%DIM%Press Enter...%RESET%"
exit /b 0

:error
echo.
echo  %RED%[ERROR]%RESET% %~1
call :log "ERROR: %~1"
exit /b 0
