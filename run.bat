@echo off
title AI Visual Guidance System
chcp 65001 >nul
cd /d "%~dp0"

echo ======================================================================
echo   HE THONG DAN DUONG THI GIAC AI CHO NGUOI KHIEM THI (AI GUIDANCE)
echo ======================================================================
echo Dang khoi dong server va chuan bi mo trinh duyet web...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u main.py
) else (
    python -u main.py
)

pause
