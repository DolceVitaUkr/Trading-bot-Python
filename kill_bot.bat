@echo off
echo Killing trading bot processes...

:: Kill process on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing process %%a
    taskkill /PID %%a /F 2>nul
)

:: Also try to kill by process name
taskkill /F /IM python.exe 2>nul
taskkill /F /IM python3.exe 2>nul
taskkill /F /IM python3.11.exe 2>nul

echo Done!
pause