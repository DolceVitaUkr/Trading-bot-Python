@echo off
echo Stopping all Python processes on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing process %%a
    taskkill /PID %%a /F 2>nul
)
echo Done! You can now start the server again.