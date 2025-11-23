@echo off
REM Drone AI Path Finder - Windows Installer
REM
REM Usage:
REM   install.bat           Basic install
REM   install.bat full      Install with AI training support
REM   install.bat viz       Install with visualization
REM   install.bat help      Show help

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo          Drone AI Path Finder - Installer
echo ============================================================
echo.

REM Parse arguments
set INSTALL_TYPE=basic
if "%1"=="full" set INSTALL_TYPE=full
if "%1"=="viz" set INSTALL_TYPE=viz
if "%1"=="help" goto :show_help
if "%1"=="-h" goto :show_help
if "%1"=="--help" goto :show_help

REM Check Python
echo [*] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found! Please install Python 3.8+
    echo     Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip -q

REM Install package
echo [*] Installing drone-ai-pathfinder...

if "%INSTALL_TYPE%"=="basic" (
    python -m pip install -e . -q
    echo [OK] Core package installed
)

if "%INSTALL_TYPE%"=="viz" (
    python -m pip install -e ".[viz]" -q
    echo [OK] Package installed with visualization
)

if "%INSTALL_TYPE%"=="full" (
    python -m pip install -e ".[all]" -q
    echo [*] Installing stable-baselines3 for AI training...
    python -m pip install stable-baselines3 -q
    echo [OK] Full package installed with AI training support
)

REM Verify installation
echo [*] Verifying installation...
python -c "from drone_ai import DroneEnv, PathPlanner; print('[OK] Installation verified!')"
if errorlevel 1 (
    echo [X] Installation verification failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo                   Installation complete!
echo ============================================================
echo.
echo Quick start commands:
echo.
echo   # Run demo (math-based controller)
echo   python -m drone_ai.demo --task delivery_route
echo.
echo   # Run with visualization
echo   python -m drone_ai.demo --task delivery_route --render
echo.

if "%INSTALL_TYPE%"=="full" (
echo   # Train AI with reinforcement learning
echo   python -m drone_ai.train_rl --algo ppo --task hover
echo.
)

echo   # Run curriculum learning
echo   python -m drone_ai.learning_sequence
echo.
echo For more info, see README.md
echo.
pause
exit /b 0

:show_help
echo Drone AI Path Finder - Windows Installer
echo.
echo Usage: install.bat [OPTION]
echo.
echo Options:
echo   (none)    Install core package only
echo   viz       Install with visualization (matplotlib, pygame)
echo   full      Install everything including AI training
echo   help      Show this help message
echo.
echo Examples:
echo   install.bat           Quick install
echo   install.bat full      Full install with AI training
echo.
pause
exit /b 0
