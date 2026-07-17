@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

echo [1/3] Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 (
  echo.
  echo Failed to create .venv. Please install Python 3.11 or newer and try again.
  pause
  exit /b 1
)

echo [2/3] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo.
  echo Failed to upgrade pip.
  pause
  exit /b 1
)

echo [3/3] Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency installation failed. Check the network connection and retry.
  pause
  exit /b 1
)

echo.
echo Environment is ready.
echo You can now drag a video onto run.bat, or double-click run.bat and paste a video path.
pause
