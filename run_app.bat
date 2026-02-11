@echo off
REM Unmount T: if it exists to avoid errors
subst T: /D >nul 2>&1

REM Mount current directory as T:
subst T: "%CD%"

REM Switch to T: drive
T:

REM Run the app
echo Starting Medical AI App...
.\env\Scripts\python app.py

REM Pause so the window doesn't close on error
pause
