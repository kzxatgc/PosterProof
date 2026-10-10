@echo off
setlocal
cd /d "%~dp0"

echo PosterProof runner (Windows)
echo ---------------------------

rem Prefer the Python Launcher, then try PATH commands, then known install folders.
set "PY_CMD="
set "_STORE_STUB=0"

rem Candidate 1: py launcher
where py >nul 2>nul
if not errorlevel 1 call :probe py -3
if defined PY_CMD goto :found

rem Candidate 2: python3 on PATH
where python3 >nul 2>nul
if not errorlevel 1 call :probe python3
if defined PY_CMD goto :found
where python3 >nul 2>nul
if not errorlevel 1 (
    set "_STORE_STUB=1"
)

rem Candidate 3: python on PATH
where python >nul 2>nul
if not errorlevel 1 call :probe python
if defined PY_CMD goto :found
where python >nul 2>nul
if not errorlevel 1 (
    set "_STORE_STUB=1"
)

rem PATH lookup failed. Scan standard install folders.
set "_BEST_PY="

if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "_BEST_PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto :scanned
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "_BEST_PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :scanned
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "_BEST_PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :scanned
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "_BEST_PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :scanned
)

if exist "%APPDATA%\Python\Python313\python.exe" (
    set "_BEST_PY=%APPDATA%\Python\Python313\python.exe"
    goto :scanned
)
if exist "%APPDATA%\Python\Python312\python.exe" (
    set "_BEST_PY=%APPDATA%\Python\Python312\python.exe"
    goto :scanned
)
if exist "%APPDATA%\Python\Python311\python.exe" (
    set "_BEST_PY=%APPDATA%\Python\Python311\python.exe"
    goto :scanned
)
if exist "%APPDATA%\Python\Python310\python.exe" (
    set "_BEST_PY=%APPDATA%\Python\Python310\python.exe"
    goto :scanned
)

if exist "C:\Python313\python.exe" (
    set "_BEST_PY=C:\Python313\python.exe"
    goto :scanned
)
if exist "C:\Python312\python.exe" (
    set "_BEST_PY=C:\Python312\python.exe"
    goto :scanned
)
if exist "C:\Python311\python.exe" (
    set "_BEST_PY=C:\Python311\python.exe"
    goto :scanned
)
if exist "C:\Python310\python.exe" (
    set "_BEST_PY=C:\Python310\python.exe"
    goto :scanned
)

if exist "%ProgramFiles%\Python313\python.exe" (
    set "_BEST_PY=%ProgramFiles%\Python313\python.exe"
    goto :scanned
)
if exist "%ProgramFiles%\Python312\python.exe" (
    set "_BEST_PY=%ProgramFiles%\Python312\python.exe"
    goto :scanned
)
if exist "%ProgramFiles%\Python311\python.exe" (
    set "_BEST_PY=%ProgramFiles%\Python311\python.exe"
    goto :scanned
)
if exist "%ProgramFiles%\Python310\python.exe" (
    set "_BEST_PY=%ProgramFiles%\Python310\python.exe"
    goto :scanned
)

if exist "%ProgramFiles(x86)%\Python313\python.exe" (
    set "_BEST_PY=%ProgramFiles(x86)%\Python313\python.exe"
    goto :scanned
)
if exist "%ProgramFiles(x86)%\Python312\python.exe" (
    set "_BEST_PY=%ProgramFiles(x86)%\Python312\python.exe"
    goto :scanned
)
if exist "%ProgramFiles(x86)%\Python311\python.exe" (
    set "_BEST_PY=%ProgramFiles(x86)%\Python311\python.exe"
    goto :scanned
)
if exist "%ProgramFiles(x86)%\Python310\python.exe" (
    set "_BEST_PY=%ProgramFiles(x86)%\Python310\python.exe"
    goto :scanned
)

:scanned
if NOT "%_BEST_PY%"=="" (
    echo (Python was found outside PATH: %_BEST_PY%)
    set "PY_CMD="%_BEST_PY%""
    goto :found
)

echo.
if "%_STORE_STUB%"=="1" (
    echo ERROR: The `python`/`python3` commands on PATH are Store stubs or Python is too old.
    echo Scanned standard install folders but no real Python ^>= 3.10 was found.
) else (
    echo ERROR: No usable Python ^>= 3.10 was found on this machine.
)
echo.
pause
exit /b 1

:found
echo Using Python command: %PY_CMD%
%PY_CMD% -c "import sys; v=sys.version_info; print('Python',v.major,'.',v.minor,'.',v.micro); assert (v.major,v.minor)>=(3,10) or sys.exit('Python >= 3.10 required')"
if errorlevel 1 (
    echo Python ^>= 3.10 required
    pause
    exit /b 1
)

set "_VENV_OK="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
    if not errorlevel 1 set "_VENV_OK=1"
)
if not defined _VENV_OK (
    echo Creating Windows virtual environment at .venv ...
    %PY_CMD% -m venv --clear .venv
    if errorlevel 1 (
        echo Failed to create venv.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>nul
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo pip install failed.
    pause
    exit /b 1
)

if exist requirements-dev.txt (
    echo Installing dev dependencies ...
    python -m pip install -r requirements-dev.txt
    if errorlevel 1 (
        echo Failed to install development dependencies.
        pause
        exit /b 1
    )
)

echo.
echo Running test suite ...
set "PYTHONPATH=%cd%;%PYTHONPATH%"
python -m pytest -v
if errorlevel 1 (
    echo Tests did not all pass. Fix above; server not started.
    pause
    exit /b 1
)

echo.
echo Starting PosterProof on http://127.0.0.1:8765/ ...
echo Press Ctrl+C to stop.
python app.py
pause
endlocal
exit /b

:probe
set "_probe="
for /f "delims=" %%i in ('%* -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2^>nul') do set "_probe=%%i"
if "%_probe%"=="1" set "PY_CMD=%*"
exit /b
