@echo off
REM Drone AI Path Finder - Windows Installer with Menu
REM
REM Double-click to run or use: install.bat
REM

setlocal enabledelayedexpansion
title Drone AI Path Finder - Installer

:main_menu
cls
echo.
echo  ╔════════════════════════════════════════════════════════════╗
echo  ║                                                            ║
echo  ║           DRONE AI PATH FINDER                             ║
echo  ║           Interactive Installer                            ║
echo  ║                                                            ║
echo  ╠════════════════════════════════════════════════════════════╣
echo  ║                                                            ║
echo  ║   [1] Install - Basic (core package only)                  ║
echo  ║   [2] Install - With Visualization (matplotlib, pygame)    ║
echo  ║   [3] Install - Full (includes AI training)                ║
echo  ║                                                            ║
echo  ║   [4] Run Demo (after install)                             ║
echo  ║   [5] Run Demo with 3D Visualization                       ║
echo  ║   [6] Run AI Training                                      ║
echo  ║                                                            ║
echo  ║   [7] Check Installation Status                            ║
echo  ║   [8] View 3D Movement Map                                 ║
echo  ║                                                            ║
echo  ║   [0] Exit                                                 ║
echo  ║                                                            ║
echo  ╚════════════════════════════════════════════════════════════╝
echo.
set /p choice="  Enter your choice [0-8]: "

if "%choice%"=="1" goto install_basic
if "%choice%"=="2" goto install_viz
if "%choice%"=="3" goto install_full
if "%choice%"=="4" goto run_demo
if "%choice%"=="5" goto run_demo_viz
if "%choice%"=="6" goto run_training
if "%choice%"=="7" goto check_status
if "%choice%"=="8" goto view_3d_map
if "%choice%"=="0" goto exit_installer

echo.
echo  Invalid choice. Press any key to try again...
pause >nul
goto main_menu

:install_basic
cls
echo.
echo  ============================================================
echo   Installing Basic Package...
echo  ============================================================
echo.
call :check_python
if errorlevel 1 goto main_menu

echo  [*] Upgrading pip...
python -m pip install --upgrade pip -q

echo  [*] Installing drone-ai-pathfinder...
python -m pip install -e . -q

echo.
echo  [OK] Basic installation complete!
echo.
pause
goto main_menu

:install_viz
cls
echo.
echo  ============================================================
echo   Installing Package with Visualization...
echo  ============================================================
echo.
call :check_python
if errorlevel 1 goto main_menu

echo  [*] Upgrading pip...
python -m pip install --upgrade pip -q

echo  [*] Installing drone-ai-pathfinder with visualization...
python -m pip install -e ".[viz]" -q

echo.
echo  [OK] Installation with visualization complete!
echo.
pause
goto main_menu

:install_full
cls
echo.
echo  ============================================================
echo   Installing Full Package with AI Training...
echo  ============================================================
echo.
call :check_python
if errorlevel 1 goto main_menu

echo  [*] Upgrading pip...
python -m pip install --upgrade pip -q

echo  [*] Installing drone-ai-pathfinder (full)...
python -m pip install -e ".[all]" -q

echo  [*] Installing stable-baselines3 for AI training...
python -m pip install stable-baselines3 -q

echo.
echo  [OK] Full installation complete! AI training is now available.
echo.
pause
goto main_menu

:run_demo
cls
echo.
echo  ============================================================
echo   Running Demo (Math-based controller)
echo  ============================================================
echo.
echo  Task: delivery_route
echo  Difficulty: 0.5
echo  Press Ctrl+C to stop
echo.
python -m drone_ai.demo --task delivery_route --difficulty 0.5 --episodes 1
echo.
pause
goto main_menu

:run_demo_viz
cls
echo.
echo  ============================================================
echo   Running Demo with 3D Visualization
echo  ============================================================
echo.
echo  This will open a 3D view of the drone navigating waypoints.
echo  Close the window or press Ctrl+C to stop.
echo.
python -m drone_ai.demo --task delivery_route --difficulty 0.5 --render --episodes 1
echo.
pause
goto main_menu

:run_training
cls
echo.
echo  ============================================================
echo   AI Training Menu
echo  ============================================================
echo.
echo   [1] Train on Hover task (easiest, ~2 min)
echo   [2] Train on Delivery task (medium, ~5 min)
echo   [3] Train on Delivery Route (hardest, ~10 min)
echo   [4] Run Curriculum Learning (all stages)
echo   [0] Back to main menu
echo.
set /p train_choice="  Enter choice [0-4]: "

if "%train_choice%"=="1" (
    echo.
    echo  Training PPO on Hover task...
    python -m drone_ai.train_rl --algo ppo --task hover --timesteps 50000
)
if "%train_choice%"=="2" (
    echo.
    echo  Training PPO on Delivery task...
    python -m drone_ai.train_rl --algo ppo --task delivery --timesteps 100000
)
if "%train_choice%"=="3" (
    echo.
    echo  Training PPO on Delivery Route task...
    python -m drone_ai.train_rl --algo ppo --task delivery_route --timesteps 200000
)
if "%train_choice%"=="4" (
    echo.
    echo  Running Curriculum Learning...
    python -m drone_ai.learning_sequence --episodes 500
)
if "%train_choice%"=="0" goto main_menu

echo.
pause
goto main_menu

:check_status
cls
echo.
echo  ============================================================
echo   Installation Status Check
echo  ============================================================
echo.
call :check_python
echo.

echo  [*] Checking drone_ai package...
python -c "from drone_ai import DroneEnv, PathPlanner; print('  [OK] drone_ai installed')" 2>nul || echo   [X] drone_ai NOT installed

echo.
echo  [*] Checking visualization libraries...
python -c "import matplotlib; print('  [OK] matplotlib installed')" 2>nul || echo   [X] matplotlib NOT installed
python -c "import pygame; print('  [OK] pygame installed')" 2>nul || echo   [-] pygame not installed (optional)

echo.
echo  [*] Checking AI training library...
python -c "import stable_baselines3; print('  [OK] stable-baselines3 installed')" 2>nul || echo   [-] stable-baselines3 not installed (use option 3 to install)

echo.
pause
goto main_menu

:view_3d_map
cls
echo.
echo  ============================================================
echo   3D Movement Map Viewer
echo  ============================================================
echo.
echo  This will show a real-time 3D visualization of drone movement.
echo  - 3D view with rotating camera
echo  - Top-down view (XY plane)
echo  - Side view (altitude profile)
echo  - Live drone position and trail
echo.
echo  Close the window to return to menu.
echo.
python -m drone_ai.map_viewer --task delivery_route --difficulty 0.5
echo.
pause
goto main_menu

:check_python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [X] Python not found!
    echo      Please install Python 3.8+ from https://www.python.org/downloads/
    echo      Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo  [OK] Python %PYTHON_VERSION% found
exit /b 0

:exit_installer
cls
echo.
echo  ============================================================
echo   Thank you for using Drone AI Path Finder!
echo  ============================================================
echo.
echo  Quick commands you can run anytime:
echo.
echo    python -m drone_ai.demo --render
echo    python -m drone_ai.learning_sequence
echo    python -m drone_ai.train_rl --algo ppo
echo.
echo  See README.md for more information.
echo.
timeout /t 3 >nul
exit /b 0
