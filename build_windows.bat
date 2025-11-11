@echo off
REM Simple build script for OCTproEngine with Python bindings on Windows

echo ============================================
echo OCTproEngine Build Script (Windows)
echo ============================================
echo.

REM Check if we're in the right directory
if not exist "CMakeLists.txt" (
	echo ERROR: CMakeLists.txt not found!
	echo Please run this script from the octproengine root directory.
	pause
	exit /b 1
)

REM Find Python
REM Priority: 1) User-set PYTHON_EXE, 2) python in PATH, 3) Common install locations
if not "%PYTHON_EXE%"=="" (
	echo Using user-specified Python: %PYTHON_EXE%
	goto :python_found
)

REM Try python in PATH (most common)
where python.exe >nul 2>&1
if %errorlevel% equ 0 (
	for /f "delims=" %%i in ('where python.exe') do set "PYTHON_EXE=%%i"
	goto :python_found
)

REM Check common installation locations
if exist "C:\Users\%USERNAME%\anaconda3\python.exe" (
	set "PYTHON_EXE=C:\Users\%USERNAME%\anaconda3\python.exe"
	set "PATH=C:\Users\%USERNAME%\anaconda3;C:\Users\%USERNAME%\anaconda3\Scripts;%PATH%"
	goto :python_found
)
if exist "C:\Python312\python.exe" (
	set "PYTHON_EXE=C:\Python312\python.exe"
	goto :python_found
)
if exist "C:\Python311\python.exe" (
	set "PYTHON_EXE=C:\Python311\python.exe"
	goto :python_found
)

echo ERROR: Python not found!
echo.
echo Please either:
echo   1. Add Python to your PATH, or
echo   2. Set PYTHON_EXE environment variable before running this script
echo.
echo Example: set PYTHON_EXE=C:\Path\To\python.exe
pause
exit /b 1

:python_found

echo Using Python: %PYTHON_EXE%
echo.

REM Install pybind11 if needed
echo Installing Python dependencies...
"%PYTHON_EXE%" -m pip install pybind11 numpy
echo.

REM Create build directory
if not exist "build" mkdir build
cd build

REM Check if FFTW already exists
if exist "..\thirdparty\fftw\libfftw3f-3.dll" (
	echo FFTW3 found in thirdparty folder
	set FFTW_DOWNLOAD_FLAG=
	goto :configure
)

REM Ask user if they want to download FFTW
echo.
echo ========================================
echo FFTW3 not found
echo ========================================
echo.
echo FFTW3 is required to build the CPU backend build this project.
echo.
echo Option 1: Auto-download FFTW3 from https://fftw.org
echo           Download size: ~2 MB
echo           Location: thirdparty/fftw/
echo.
echo Option 2: Download manually later and place in thirdparty/fftw/
echo.
set /p "DOWNLOAD_CHOICE=Do you want to auto-download FFTW3 now? (Y/N): "

if /i "%DOWNLOAD_CHOICE%"=="Y" (
	set FFTW_DOWNLOAD_FLAG=-DFFTW3_AUTO_DOWNLOAD=ON
	echo.
	echo Auto-download enabled
	echo.
) else (
	set FFTW_DOWNLOAD_FLAG=-DFFTW3_AUTO_DOWNLOAD=OFF
	echo.
	echo Auto-download disabled. Please download FFTW3 manually.
	echo.
)

:configure

REM Configure (FFTW will auto-download to thirdparty/fftw if user said yes)
echo Configuring CMake...
cmake .. %FFTW_DOWNLOAD_FLAG% -DBUILD_PYTHON=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
	echo.
	echo ERROR: CMake configuration failed!
	cd ..
	pause
	exit /b 1
)
echo.

REM Build
echo Building...
cmake --build . --config Release
if errorlevel 1 (
	echo.
	echo ERROR: Build failed!
	cd ..
	pause
	exit /b 1
)
echo.

echo ============================================
echo Build completed successfully!
echo ============================================
echo.
echo Python module location:
dir /b python\Release\octproengine*.pyd 2>nul
echo.
echo To quickly test:
echo   cd build\python\Release
echo   "%PYTHON_EXE%" -c "import octproengine; print('SUCCESS!')"
echo.
echo To use the Python module, set PYTHONPATH:
echo   set PYTHONPATH=%CD%\python\Release;%%PYTHONPATH%%
echo.
echo Then you can import from anywhere:
echo   python -c "import octproengine; print('SUCCESS!')"
echo.
echo Or run tests:
echo   cd python\tests (from project root)
echo   python run_all_tests.py
echo.

cd ..
pause