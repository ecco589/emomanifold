@echo off
setlocal
chcp 65001 >nul

echo This installs the shared FFmpeg build needed by TorchCodec on Windows.
echo It uses winget and may need a few minutes.
echo.
echo Command:
echo winget install --accept-source-agreements --accept-package-agreements Gyan.FFmpeg.Shared
echo.
set /p RUN_INSTALL=Type Y and press Enter to run it now: 

if /I not "%RUN_INSTALL%"=="Y" (
  echo Skipped.
  pause
  exit /b 0
)

winget install --accept-source-agreements --accept-package-agreements Gyan.FFmpeg.Shared
if errorlevel 1 (
  echo.
  echo FFmpeg installation failed. You can also install "FFmpeg Shared" manually.
  pause
  exit /b 1
)

echo.
echo FFmpeg installation finished.
echo Close this window, open a new one, then run diagnose.bat or run.bat again.
pause
