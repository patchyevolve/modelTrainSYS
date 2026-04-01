@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title ML System — Hierarchical Mamba + Transformer

:: ════════════════════════════════════════════════════════════════════════════
:: COLOUR CODES  (Windows 10+ ANSI via PowerShell echo trick)
:: ════════════════════════════════════════════════════════════════════════════
set "ESC="
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
set "BGBLUE=%ESC%[44m"

:: ════════════════════════════════════════════════════════════════════════════
:: FIND PYTHON
:: ════════════════════════════════════════════════════════════════════════════
set PYTHON=
for %%P in (python python3 py) do (
    if "!PYTHON!"=="" (
        %%P --version >nul 2>&1 && set PYTHON=%%P
    )
)
if "!PYTHON!"=="" (
    echo.
    echo  %RED%%BOLD%[ERROR]%RESET% Python not found in PATH.
    echo  %YELLOW%Install Python 3.8+ from https://python.org and tick "Add to PATH"%RESET%
    echo.
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%V in ('!PYTHON! --version 2^>^&1') do set PYVER=%%V

:: ════════════════════════════════════════════════════════════════════════════
:: DIRECT ARGUMENT MODE  (run.bat ui / run.bat chat / etc.)
:: ════════════════════════════════════════════════════════════════════════════
if not "%~1"=="" (
    set ARG=%~1
    if /i "!ARG!"=="ui"        goto :do_gui
    if /i "!ARG!"=="chat"      goto :do_chat
    if /i "!ARG!"=="infer"     goto :do_inference
    if /i "!ARG!"=="inference" goto :do_inference
    if /i "!ARG!"=="list"      goto :do_list
    if /i "!ARG!"=="models"    goto :do_list
    if /i "!ARG!"=="upgrade"   goto :do_upgrade
    if /i "!ARG!"=="check"     goto :do_check
    if /i "!ARG!"=="install"   goto :do_install
    if /i "!ARG!"=="gpu"       goto :do_gpu
    if /i "!ARG!"=="help"      goto :do_help
    echo  %YELLOW%Unknown argument: !ARG!%RESET%
    echo  Valid: ui  chat  inference  list  upgrade  check  install  gpu  help
    pause & exit /b 1
)

:: ════════════════════════════════════════════════════════════════════════════
:: MAIN MENU
:: ════════════════════════════════════════════════════════════════════════════
:menu
cls
echo.
echo  %CYAN%%BOLD%╔══════════════════════════════════════════════════════════════╗%RESET%
echo  %CYAN%%BOLD%║                                                              ║%RESET%
echo  %CYAN%%BOLD%║   ⬡  ML TRAINING SYSTEM                                     ║%RESET%
echo  %CYAN%%BOLD%║      Hierarchical Mamba + Transformer  │  Groq LLM           ║%RESET%
echo  %CYAN%%BOLD%║      AMD Radeon 680M  │  16-core CPU  │  DirectML            ║%RESET%
echo  %CYAN%%BOLD%║                                                              ║%RESET%
echo  %CYAN%%BOLD%╚══════════════════════════════════════════════════════════════╝%RESET%
echo.
echo  %DIM%  Python %PYVER%   │   %CD%%RESET%
echo.
echo  %WHITE%%BOLD%  TRAINING%RESET%
echo  %GREEN%  [1]%RESET%  Training GUI         %DIM%drag-and-drop data, live charts, model manager%RESET%
echo  %GREEN%  [2]%RESET%  Text Generation      %DIM%train on .txt / .jsonl files%RESET%
echo  %GREEN%  [3]%RESET%  Image Classification %DIM%train on image folders (class/img.jpg)%RESET%
echo  %GREEN%  [4]%RESET%  Cybersecurity        %DIM%train on network intrusion CSV data%RESET%
echo.
echo  %WHITE%%BOLD%  INFERENCE ^& CHAT%RESET%
echo  %CYAN%  [5]%RESET%  Chat with model      %DIM%talk to any trained model%RESET%
echo  %CYAN%  [6]%RESET%  Run inference        %DIM%classify / generate on a data file%RESET%
echo  %CYAN%  [7]%RESET%  List trained models  %DIM%see all saved .pt checkpoints%RESET%
echo.
echo  %WHITE%%BOLD%  SYSTEM%RESET%
echo  %MAGENTA%  [8]%RESET%  Auto-Upgrade         %DIM%LLM-powered architecture upgrader (Groq)%RESET%
echo  %MAGENTA%  [9]%RESET%  GPU / Device info    %DIM%check CPU, AMD iGPU, DirectML status%RESET%
echo  %YELLOW%  [0]%RESET%  Health check         %DIM%verify all files and dependencies%RESET%
echo  %YELLOW%  [I]%RESET%  Install dependencies %DIM%pip install all required packages%RESET%
echo  %RED%  [X]%RESET%  Exit
echo.
set /p "CHOICE=  %BOLD%Choice: %RESET%"

if "%CHOICE%"=="1" goto :do_gui
if "%CHOICE%"=="2" goto :do_textgen
if "%CHOICE%"=="3" goto :do_imgcls
if "%CHOICE%"=="4" goto :do_cybersec
if "%CHOICE%"=="5" goto :do_chat
if "%CHOICE%"=="6" goto :do_inference
if "%CHOICE%"=="7" goto :do_list
if "%CHOICE%"=="8" goto :do_upgrade
if "%CHOICE%"=="9" goto :do_gpu
if "%CHOICE%"=="0" goto :do_check
if /i "%CHOICE%"=="I" goto :do_install
if /i "%CHOICE%"=="X" goto :do_exit
if /i "%CHOICE%"=="H" goto :do_help

echo  %YELLOW%  Invalid choice — try again.%RESET%
timeout /t 1 >nul
goto :menu

:: ════════════════════════════════════════════════════════════════════════════
:: ACTIONS
:: ════════════════════════════════════════════════════════════════════════════

:do_gui
cls
call :header "TRAINING GUI"
echo  %DIM%  Launching visual training window...%RESET%
echo  %DIM%  Drag data files in, pick model type, hit Start.%RESET%
echo.
!PYTHON! start.py --ui
goto :back

:do_textgen
cls
call :header "TEXT GENERATION TRAINING"
echo  %DIM%  Opens the Training GUI with Text Generation pre-selected.%RESET%
echo  %DIM%  Drop in .txt / .jsonl / .json files and train.%RESET%
echo.
echo  %YELLOW%  Tip: big.txt (6 MB) = ~2000 steps/epoch, ~9 min/epoch on CPU%RESET%
echo  %YELLOW%  Tip: use AdamW optimizer + CosineAnnealing for best results%RESET%
echo.
!PYTHON! start.py --ui
goto :back

:do_imgcls
cls
call :header "IMAGE CLASSIFICATION TRAINING"
echo  %DIM%  Opens the Training GUI with Image Classification pre-selected.%RESET%
echo.
echo  %WHITE%  Expected folder structure:%RESET%
echo  %DIM%    your_data/%RESET%
echo  %DIM%      cats/   img1.jpg  img2.jpg  ...%RESET%
echo  %DIM%      dogs/   img3.jpg  img4.jpg  ...%RESET%
echo  %DIM%      birds/  img5.jpg  ...%RESET%
echo.
echo  %YELLOW%  Tip: 50+ images per class recommended%RESET%
echo  %YELLOW%  Tip: use AdamW + CosineAnnealing, batch=32, hidden=256%RESET%
echo.
pause
!PYTHON! start.py --ui
goto :back

:do_cybersec
cls
call :header "CYBERSECURITY TRAINING"
echo  %DIM%  Opens the Training GUI with Cybersecurity pre-selected.%RESET%
echo.
echo  %WHITE%  Available data:%RESET%
echo  %DIM%    randomDATA\cybersecurity_intrusion_data.csv%RESET%
echo  %DIM%    9,537 rows  │  16 features  │  binary (attack/benign)%RESET%
echo.
echo  %YELLOW%  Tip: SGD + CosineAnnealing, batch=32, hidden=512, epochs=50%RESET%
echo  %YELLOW%  Tip: achieved 87%% accuracy in previous training run%RESET%
echo.
pause
!PYTHON! start.py --ui
goto :back

:do_chat
cls
call :header "CHAT WITH MODEL"
echo.
!PYTHON! start.py --list
echo.
set /p "MODEL=  %BOLD%Model name (Enter = latest): %RESET%"
echo.
if "!MODEL!"=="" (
    !PYTHON! start.py --chat
) else (
    !PYTHON! start.py --chat !MODEL!
)
goto :back

:do_inference
cls
call :header "RUN INFERENCE"
echo.
!PYTHON! start.py --list
echo.
set /p "MODEL=  %BOLD%Model name (Enter = latest): %RESET%"
set /p "DATAFILE=  %BOLD%Data file  (Enter = randomDATA/): %RESET%"
echo.
if "!MODEL!"=="" (
    if "!DATAFILE!"=="" (
        !PYTHON! inference.py --save
    ) else (
        !PYTHON! inference.py --data "!DATAFILE!" --save
    )
) else (
    if "!DATAFILE!"=="" (
        !PYTHON! inference.py --model "trained_models\!MODEL!.pt" --save
    ) else (
        !PYTHON! inference.py --model "trained_models\!MODEL!.pt" --data "!DATAFILE!" --save
    )
)
goto :back

:do_list
cls
call :header "TRAINED MODELS"
echo.
!PYTHON! start.py --list
echo.
echo  %DIM%  Checkpoints are in: trained_models\%RESET%
echo  %DIM%  Each model has a .pt (weights) and .json (metadata) file.%RESET%
goto :back

:do_upgrade
cls
call :header "AUTO-UPGRADE SYSTEM"
echo  %DIM%  Launching upgrade window...%RESET%
echo  %DIM%  Uses Groq llama-3.3-70b to suggest and apply architecture changes.%RESET%
echo  %DIM%  All changes are recorded in upgrade_system.db%RESET%
echo.
!PYTHON! start.py --upgrade
goto :back

:do_gpu
cls
call :header "GPU / DEVICE INFO"
echo.
!PYTHON! -c "
from device_manager import device_info, get_best_device
print(device_info())
print()
dev, name = get_best_device(model_params=1_000_000, batch_size=32)
print(f'Auto-select (1M params, batch=32): {name}')
dev, name = get_best_device(model_params=10_000_000, batch_size=64)
print(f'Auto-select (10M params, batch=64): {name}')
print()
print('Force GPU: set force=dml in device_manager.get_best_device()')
"
goto :back

:do_check
cls
call :header "HEALTH CHECK"
echo.
!PYTHON! start.py --check
goto :back

:do_install
cls
call :header "INSTALL DEPENDENCIES"
echo.
echo  %WHITE%  Core packages:%RESET%
!PYTHON! -m pip install torch torchvision numpy pandas Pillow
echo.
echo  %WHITE%  UI packages:%RESET%
!PYTHON! -m pip install tkinterdnd2
echo.
echo  %WHITE%  AMD iGPU acceleration (DirectML):%RESET%
!PYTHON! -m pip install torch-directml
echo.
echo  %GREEN%  Done.%RESET%
echo.
echo  %DIM%  For NVIDIA GPU (CUDA 12.1):%RESET%
echo  %DIM%  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121%RESET%
goto :back

:do_help
cls
call :header "USAGE"
echo.
echo  %WHITE%  Double-click run.bat%RESET%  %DIM%— interactive menu%RESET%
echo.
echo  %WHITE%  Command-line shortcuts:%RESET%
echo  %DIM%    run.bat ui          — Training GUI%RESET%
echo  %DIM%    run.bat chat        — Chat with latest model%RESET%
echo  %DIM%    run.bat inference   — Run inference%RESET%
echo  %DIM%    run.bat list        — List trained models%RESET%
echo  %DIM%    run.bat upgrade     — Auto-Upgrade window%RESET%
echo  %DIM%    run.bat check       — Health check%RESET%
echo  %DIM%    run.bat install     — Install dependencies%RESET%
echo  %DIM%    run.bat gpu         — GPU / device info%RESET%
echo  %DIM%    run.bat help        — This screen%RESET%
echo.
echo  %WHITE%  Python CLI (same options):%RESET%
echo  %DIM%    python start.py --ui%RESET%
echo  %DIM%    python start.py --chat [model_name]%RESET%
echo  %DIM%    python start.py --list%RESET%
echo  %DIM%    python inference.py --model name.pt --data file.csv --save%RESET%
goto :back

:do_exit
echo.
echo  %DIM%  Goodbye.%RESET%
echo.
exit /b 0

:: ════════════════════════════════════════════════════════════════════════════
:: HELPERS
:: ════════════════════════════════════════════════════════════════════════════

:header
echo  %CYAN%%BOLD%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
echo  %CYAN%%BOLD%  %~1%RESET%
echo  %CYAN%%BOLD%━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%RESET%
exit /b 0

:back
echo.
set /p "DUMMY=  %DIM%Press Enter to return to menu...%RESET%"
goto :menu
