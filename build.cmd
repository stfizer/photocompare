@echo off
rem Rebuild the exe. Keeps one previous build (dist\*_previous.exe).
rem Requires the .deps folder (pip --target) and base python.
cd /d %~dp0
set PYTHON=C:\Users\stfizer\.workbuddy\binaries\python\versions\3.13.12\python.exe
set PYTHONPATH=%~dp0.deps
"%PYTHON%" prebuild_backup.py
if errorlevel 1 exit /b 1
for %%f in (*.spec) do "%PYTHON%" -m PyInstaller --noconfirm --clean "%%f"
echo.
for %%f in (dist\*.exe) do echo Done. EXE at: %%~ff
pause
