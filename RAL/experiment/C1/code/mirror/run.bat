@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','User') + ';' + [Environment]::GetEnvironmentVariable('Path','Machine')"`) do set "PATH=%%I;%PATH%"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Running setup_env.bat first...
  call "%~dp0setup_env.bat"
  if errorlevel 1 exit /b 1
)

if "%~1"=="" (
  echo Paste the video path, then press Enter.
  echo Example: D:\videos\robot_test.mp4
  set /p VIDEO_PATH=Video path: 
) else (
  set "VIDEO_PATH=%~1"
)

if "%VIDEO_PATH%"=="" (
  echo No video path was provided.
  pause
  exit /b 1
)

echo.
echo Optional: paste original VA CSV path for comparison, or just press Enter to skip.
set /p SOURCE_VA=Original VA CSV: 

if "%SOURCE_VA%"=="" (
  ".venv\Scripts\python.exe" "%~dp0video_va_pipeline.py" --video "%VIDEO_PATH%" --out-dir "%~dp0results"
) else (
  ".venv\Scripts\python.exe" "%~dp0video_va_pipeline.py" --video "%VIDEO_PATH%" --source-va "%SOURCE_VA%" --out-dir "%~dp0results"
)

if errorlevel 1 (
  echo.
  echo The run failed. No valid result files were created.
  echo Read the error message above, then run diagnose.bat.
  pause
  exit /b 1
)

echo.
echo Done. Check the results folder.
pause
