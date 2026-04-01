@echo off
setlocal EnableDelayedExpansion

chcp 65001 >nul 2>&1
title ML Training System

set "SCRIPT_DIR=%~dp0"
set "PYTHON="
set "PYVER="

for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "RESET=%ESC%[0m"
set "BOLD=%ESC%[1m"
set "CYAN=%ESC%[96m"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "RED=%ESC%[91m"
set "WHITE=%ESC%[97m"
set "DIM=%ESC%[2m"

:: Find Python
if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    for %%P in (python python3 py) do (
        if "!PYTHON!"=="" (
            %%P --version >nul 2>&1 && set "PYTHON=%%P"
        )
    )
)

if "!PYTHON!"=="" (
    echo.
    echo  %RED%[ERROR] Python not found%RESET%
    echo  Install from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%V in ('!PYTHON! --version 2^>^&1') do set "PYVER=%%V"

:: Parse arguments
if not "%~1"=="" goto :parse_arg

:menu
cls
echo.
echo  %CYAN%%BOLD%========================================%RESET%
echo  %CYAN%%BOLD%     ML TRAINING SYSTEM                 %RESET%
echo  %CYAN%%BOLD%========================================%RESET%
echo.
echo  %DIM%  Python: !PYVER!%RESET%
echo.
echo  %WHITE%  [1]%RESET%  Training GUI
echo  %WHITE%  [2]%RESET%  Chat with Model
echo  %WHITE%  [3]%RESET%  Run Inference
echo  %WHITE%  [4]%RESET%  List Models
echo  %WHITE%  [5]%RESET%  Auto-Upgrade
echo  %WHITE%  [6]%RESET%  Health Check
echo  %WHITE%  [7]%RESET%  Install Dependencies
echo  %WHITE%  [X]%RESET%  Exit
echo.
set /p "CHOICE=%BOLD%  > %RESET%"

if "%CHOICE%"=="1" goto :do_gui
if "%CHOICE%"=="2" goto :do_chat
if "%CHOICE%"=="3" goto :do_inference
if "%CHOICE%"=="4" goto :do_list
if "%CHOICE%"=="5" goto :do_upgrade
if "%CHOICE%"=="6" goto :do_check
if "%CHOICE%"=="7" goto :do_install
if /i "%CHOICE%"=="X" exit /b 0

goto :menu

:do_gui
!PYTHON! start.py --ui
pause
goto :menu

:do_chat
!PYTHON! start.py --chat
pause
goto :menu

:do_inference
!PYTHON! start.py --inference
pause
goto :menu

:do_list
!PYTHON! start.py --list
pause
goto :menu

:do_upgrade
!PYTHON! start.py --upgrade
pause
goto :menu

:do_check
!PYTHON! start.py --check
pause
goto :menu

:do_install
!PYTHON! -m pip install torch torchvision numpy pandas Pillow
pause
goto :menu

:parse_arg
set "ARG=%~1"
if /i "%ARG%"=="gui"       goto :do_gui
if /i "%ARG%"=="chat"      goto :do_chat
if /i "%ARG%"=="infer"     goto :do_inference
if /i "%ARG%"=="list"      goto :do_list
if /i "%ARG%"=="upgrade"   goto :do_upgrade
if /i "%ARG%"=="smart"     goto :do_smart
if /i "%ARG%"=="check"     goto :do_check
if /i "%ARG%"=="help"      goto :do_help
echo  Unknown: %ARG%
exit /b 1

:do_smart
!PYTHON! start.py --smart
pause
goto :menu

:do_help
echo  Usage: run.bat [command]
echo  Commands: gui chat infer list upgrade smart check help
exit /b 0
