@echo off
setlocal

set "VENV_DIR=.venv"
set "ACTIVATE_PATH=%VENV_DIR%\Scripts\activate.bat"

if not exist "%ACTIVATE_PATH%" (
    echo Virtual environment not found. Creating at %VENV_DIR%...
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

call "%ACTIVATE_PATH%"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

if exist requirements.txt (
    python -m pip install --upgrade pip
    if errorlevel 1 goto pip_fail
    python -m pip install -r requirements.txt
    if errorlevel 1 goto pip_fail
)

set "TFBOT_MODE=gacha"
python bot.py
exit /b 0

:pip_fail
echo Failed to install Python dependencies.
exit /b 1
