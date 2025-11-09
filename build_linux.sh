# Simple build script for OCTproEngine with Python bindings on Linux/Jetson

set -e  # Exit on error

echo "============================================"
echo "OCTproEngine Build Script (Linux/Jetson)"
echo "============================================"
echo

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "ERROR: CMakeLists.txt not found!"
    echo "Please run this script from the octproengine root directory."
    exit 1
fi

# Find Python
if command -v python3 &> /dev/null; then
    PYTHON_EXE=$(which python3)
else
    echo "ERROR: python3 not found!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo "Python version: $($PYTHON_EXE --version)"
echo

# Install dependencies
echo "Installing Python dependencies..."
$PYTHON_EXE -m pip install --user pybind11 numpy
echo

# Check for CUDA (optional)
if command -v nvcc &> /dev/null; then
    BUILD_CUDA=ON
    echo "CUDA found: $(nvcc --version | grep release)"
else
    BUILD_CUDA=OFF
    echo "CUDA not found - building CPU-only version"
fi
echo

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring CMake..."
cmake .. \
    -DBUILD_PYTHON=ON \
    -DBUILD_CUDA=$BUILD_CUDA \
    -DCMAKE_BUILD_TYPE=Release

echo

# Build
echo "Building..."
cmake --build . --config Release -j$(nproc)

echo
echo "============================================"
echo "Build completed successfully!"
echo "============================================"
echo
echo "Python module location:"
find python -name "octproengine*.so" 2>/dev/null || echo "  (not found)"
echo
echo "To test:"
echo "  cd python"
echo "  $PYTHON_EXE -c 'import octproengine; print(\"SUCCESS!\")'"
echo
