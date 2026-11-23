@echo off
cd /d "%~dp0"
set ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/
call pnpm install --frozen-lockfile
if errorlevel 1 exit /b 1
call pnpm run build
if errorlevel 1 exit /b 1
pushd sidecar
python -m PyInstaller --noconfirm --distpath ..\dist --workpath ..\build\sidecar reaground-sidecar.spec
if errorlevel 1 exit /b 1
popd
call pnpm exec electron-builder --dir
if errorlevel 1 exit /b 1
echo.
echo Built: release\win-unpacked\ReAGroundClient.exe
