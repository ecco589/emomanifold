@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','User') + ';' + [Environment]::GetEnvironmentVariable('Path','Machine')"`) do set "PATH=%%I;%PATH%"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Run setup_env.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "%~dp0check_install.py"

echo.
pause
