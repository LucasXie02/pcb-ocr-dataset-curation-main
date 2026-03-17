@echo off
title Stop PCB OCR Review Server
echo Stopping PCB OCR Review Server on port 5001...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5001"') do (
    echo Found PID: %%a
    taskkill /PID %%a /F
    echo Server stopped.
    goto done
)
echo No service found on port 5001.
:done
pause
