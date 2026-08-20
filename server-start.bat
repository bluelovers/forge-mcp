@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" server.py
    exit /b %errorlevel%
)

uv run server.py
exit /b %errorlevel%