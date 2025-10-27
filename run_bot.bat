@echo off
setlocal

set "VENV_DIR=.venv"
set "ACTIVATE_PATH=%VENV_DIR%\Scripts\activate.bat"

if not exist "%ACTIVATE_PATH%" (
    echo Virtual environment not found at %ACTIVATE_PATH%.
    echo Create one with: python -m venv %VENV_DIR%
    exit /b 1
)

call "%ACTIVATE_PATH%"
python bot.py
