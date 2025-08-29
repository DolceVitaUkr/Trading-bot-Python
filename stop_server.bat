@echo off
echo Stopping all processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    echo Killing process %%a
    taskkill /PID %%a /F 2>nul
)
echo Done!