@echo off
rem Starts the CSV Data Explorer with the project's own Python environment.
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
    echo Creating the virtual environment and installing requirements...
    python -m venv .venv || exit /b 1
    ".venv\Scripts\python" -m pip install -r requirements.txt || exit /b 1
)
".venv\Scripts\streamlit" run app.py
