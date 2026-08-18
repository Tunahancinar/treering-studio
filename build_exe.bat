@echo off
setlocal
cd /d "%~dp0"

echo TreeRingAI Studio icin EXE hazirlaniyor...
echo.

py -m pip install -r requirements.txt
py -m pip install pyinstaller

pyinstaller --noconfirm --clean "TreeRingAI Studio.spec"

echo.
echo Bitti.
echo EXE: %cd%\dist\TreeRingAI Studio\TreeRingAI Studio.exe
pause
