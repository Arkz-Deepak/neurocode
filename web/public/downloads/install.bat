@echo off
setlocal
echo ========================================================
echo   NeuroCode AI - 1-Click VS Code Extension Installer
echo ========================================================
echo.

where code >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Visual Studio Code was not found in your PATH.
    echo Please make sure VS Code is installed from https://code.visualstudio.com/
    pause
    exit /b 1
)

set "TEMP_VSIX=%TEMP%\neurocode-ai-1.3.0.vsix"
echo Downloading NeuroCode AI v1.3.0...
powershell -Command "Invoke-WebRequest -Uri 'https://neurocode.deepak-arkz.me/downloads/neurocode-ai-1.3.0.vsix' -OutFile '%TEMP_VSIX%'"

if not exist "%TEMP_VSIX%" (
    echo [ERROR] Failed to download extension package.
    pause
    exit /b 1
)

echo Installing extension into Visual Studio Code...
call code --install-extension "%TEMP_VSIX%" --force
del "%TEMP_VSIX%" 2>nul

echo.
echo [SUCCESS] NeuroCode AI Extension v1.3.0 installed successfully!
echo Open VS Code to start using NeuroCode.
echo.
pause
