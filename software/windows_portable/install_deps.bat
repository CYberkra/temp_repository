@echo off
setlocal
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH.
  pause
  exit /b 1
)
python -m pip install --upgrade pip
python -m pip install -r GPR_GUI\dependencies_qt.txt
if exist GPR_GUI\requirements.txt python -m pip install -r GPR_GUI\requirements.txt
echo Dependencies installed.
pause
