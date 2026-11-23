@echo off
cd /d "%~dp0"
if not exist release\win-unpacked\ReAGroundClient.exe (
    echo ReAGround Client has not been built. Run build.bat first.
    pause
    exit /b 1
)
start "" release\win-unpacked\ReAGroundClient.exe
