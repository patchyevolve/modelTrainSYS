@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul 2>&1
title ML Training System

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PYTHON="
set "PYVER=unknown"

for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "RESET=%ESC%[0m"
set "BOLD=%ESC%[1m"
set "CYAN=%ESC%[96m"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "RED=%ESC%[91m"
set "WHITE=%ESC%[97m"
set "DIM=%ESC%[2m"

call :find_python
if not defined PYTHON (
    echo.
    echo  %RED%[ERROR] Python not found.%RESET%
    echo  Install Python 3.10+ from https://python.org and re-run.
    pause
    exit /b 1
)

for /f "tokens=2" %%V in ('%PYTHON% --version 2^>^&1') do set "PYVER=%%V"

if not "%~1"=="" goto :parse_args

:menu
cls
echo.
echo  %CYAN%%BOLD%===============================================%RESET%
echo  %CYAN%%BOLD%  ML TRAINING SYSTEM - WINDOWS LAUNCHER       %RESET%
echo  %CYAN%%BOLD%===============================================%RESET%
echo.
echo  %DIM%Project: %ROOT%%RESET%
echo  %DIM%Python : %PYTHON%  (%PYVER%)%RESET%
echo.
echo  %WHITE%[1]%RESET%  Training GUI
echo  %WHITE%[2]%RESET%  Chat with model
echo  %WHITE%[3]%RESET%  Run inference (interactive)
echo  %WHITE%[4]%RESET%  List trained models
echo  %WHITE%[5]%RESET%  Health check
echo  %WHITE%[6]%RESET%  Auto-upgrade window
echo  %WHITE%[7]%RESET%  Install dependencies
echo  %WHITE%[8]%RESET%  CSV Train (quick)
echo  %WHITE%[9]%RESET%  CSV Predict (quick)
echo  %WHITE%[C]%RESET%  CI local smoke (test_models + production_smoke)
echo  %WHITE%[X]%RESET%  Exit
echo.
set /p "CHOICE=%BOLD%> %RESET%"

if /i "%CHOICE%"=="1" call :run_start --ui & goto :menu
if /i "%CHOICE%"=="2" call :run_start --chat & goto :menu
if /i "%CHOICE%"=="3" call :run_start --inference & goto :menu
if /i "%CHOICE%"=="4" call :run_start --list & goto :menu
if /i "%CHOICE%"=="5" call :run_start --check & goto :menu
if /i "%CHOICE%"=="6" call :run_start --upgrade & goto :menu
if /i "%CHOICE%"=="7" call :install_deps & goto :menu
if /i "%CHOICE%"=="8" call :menu_csv_train & goto :menu
if /i "%CHOICE%"=="9" call :menu_csv_predict & goto :menu
if /i "%CHOICE%"=="C" call :ci_local & goto :menu
if /i "%CHOICE%"=="X" exit /b 0
goto :menu

:parse_args
set "ORIG_CMD=%~1"
set "CMD=%~1"
REM Same flags as: python start.py --ui  (users often copy that style)
if /i "%CMD%"=="--ui"         set "CMD=ui"
if /i "%CMD%"=="-ui"          set "CMD=ui"
if /i "%CMD%"=="gui"          set "CMD=ui"
if /i "%CMD%"=="--chat"       set "CMD=chat"
if /i "%CMD%"=="-chat"        set "CMD=chat"
if /i "%CMD%"=="--inference"  set "CMD=infer"
if /i "%CMD%"=="-inference"   set "CMD=infer"
if /i "%CMD%"=="--infer"      set "CMD=infer"
if /i "%CMD%"=="--list"       set "CMD=list"
if /i "%CMD%"=="-list"        set "CMD=list"
if /i "%CMD%"=="--models"      set "CMD=list"
if /i "%CMD%"=="--check"      set "CMD=check"
if /i "%CMD%"=="-check"       set "CMD=check"
if /i "%CMD%"=="--health"     set "CMD=check"
if /i "%CMD%"=="--upgrade"    set "CMD=upgrade"
if /i "%CMD%"=="--upg"        set "CMD=upgrade"
if /i "%CMD%"=="--install"    set "CMD=install"
if /i "%CMD%"=="-install"     set "CMD=install"
if /i "%CMD%"=="--csv-train"  set "CMD=csv-train"
if /i "%CMD%"=="--csv-predict" set "CMD=csv-predict"
if /i "%CMD%"=="--ci-local"   set "CMD=ci-local"
if /i "%CMD%"=="--help"       goto :help
if /i "%CMD%"=="-h"           goto :help
if /i "%CMD%"=="/?"           goto :help

shift
if /i "%CMD%"=="ui"          call :run_start --ui %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="chat"        call :run_start --chat %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="infer"       call :run_start --inference %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="inference"   call :run_start --inference %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="list"        call :run_start --list %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="check"       call :run_start --check %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="upgrade"     call :run_start --upgrade %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="install"     call :install_deps & exit /b %ERRORLEVEL%
if /i "%CMD%"=="csv-train"   call :csv_train %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="csv-predict" call :csv_predict %* & exit /b %ERRORLEVEL%
if /i "%CMD%"=="ci-local"    call :ci_local & exit /b %ERRORLEVEL%
if /i "%CMD%"=="help"        goto :help
echo %RED%Unknown command: %ORIG_CMD%%RESET%
echo %DIM%Tip: use  run.bat ui   or   run.bat --ui   same as  python start.py --ui%RESET%
goto :help

:run_start
%PYTHON% start.py %*
exit /b %ERRORLEVEL%

:csv_train
if "%~1"=="" (
    echo %YELLOW%Usage: run.bat csv-train ^<data.csv^>%RESET%
    exit /b 1
)
%PYTHON% start.py --csv-train "%~1"
exit /b %ERRORLEVEL%

:csv_predict
if "%~2"=="" (
    echo %YELLOW%Usage: run.bat csv-predict ^<model.pt^> ^<data.csv^>%RESET%
    exit /b 1
)
%PYTHON% start.py --csv-predict "%~1" "%~2"
exit /b %ERRORLEVEL%

:menu_csv_train
set "CSV_FILE="
echo.
set /p "CSV_FILE=Path to CSV file: "
if not defined CSV_FILE exit /b 0
call :csv_train "%CSV_FILE%"
exit /b 0

:menu_csv_predict
set "MODEL_FILE="
set "CSV_FILE="
echo.
set /p "MODEL_FILE=Path to model (.pt): "
if not defined MODEL_FILE exit /b 0
set /p "CSV_FILE=Path to CSV file: "
if not defined CSV_FILE exit /b 0
call :csv_predict "%MODEL_FILE%" "%CSV_FILE%"
exit /b 0

:ci_local
echo.
echo %CYAN%Running local CI smoke...%RESET%
%PYTHON% test_models.py
if errorlevel 1 exit /b 1
%PYTHON% production_smoke.py
exit /b %ERRORLEVEL%

:install_deps
echo.
echo %CYAN%Installing dependencies...%RESET%
%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install torch torchvision numpy pandas Pillow tkinterdnd2
exit /b %ERRORLEVEL%

:find_python
if exist "%ROOT%venv\Scripts\python.exe" set "PYTHON=%ROOT%venv\Scripts\python.exe"
if not defined PYTHON if exist "%ROOT%.venv\Scripts\python.exe" set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not defined PYTHON (
    for %%P in (python python3 py) do (
        if not defined PYTHON (
            %%P --version >nul 2>&1 && set "PYTHON=%%P"
        )
    )
)
exit /b 0

:help
echo.
echo Usage: run.bat [command] [args]
echo   run.bat ui  and  run.bat --ui  both open the GUI (same as python start.py --ui^)
echo.
echo Commands (also accept --name where shown):
echo   ui, --ui                  Open training UI
echo   chat, --chat              Chat with model
echo   infer, --infer            Interactive inference
echo   list, --list              List models
echo   check, --check            Health check
echo   upgrade, --upgrade        Open upgrade window
echo   install, --install        Install dependencies
echo   csv-train ^<data.csv^>     Train classifier from CSV
echo   csv-predict ^<model.pt^> ^<data.csv^>  Predict from CSV
echo   ci-local                  Run local smoke checks
echo   help, -h, /?              Show this help
echo.
exit /b 0
