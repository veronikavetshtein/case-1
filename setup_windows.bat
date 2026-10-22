@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (echo Ошибка установки Python-библиотек & pause & exit /b 1)
python download_model.py
pause
