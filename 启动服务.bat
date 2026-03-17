@echo off
title PCB OCR Review Server
cd /d "%~dp0"
echo Starting PCB OCR Review Server...
echo LAN access: http://10.0.0.100:5001
echo Press Ctrl+C to stop.
echo.
C:\ProgramData\miniconda3\python.exe ocr_review_app.py --port 5001
pause
