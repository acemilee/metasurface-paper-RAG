@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_services.ps1"
if errorlevel 1 (
  echo.
  echo Startup failed. Review the error above.
  pause
  exit /b 1
)
