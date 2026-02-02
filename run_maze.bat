@echo off
cd /d "%~dp0"

if not exist venv (
    echo Virtual environment 'venv' not found.
    echo Please create it first or ensure you are in the correct directory.
    pause
    exit /b
)

call venv\Scripts\activate.bat
python gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo The application exited with an error.
    pause
)
