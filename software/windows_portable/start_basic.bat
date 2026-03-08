@echo off
setlocal
cd /d %~dp0
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH.
  pause
  exit /b 1
)
python GPR_GUI\app.py
