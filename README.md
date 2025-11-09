# OCTproEngine

High-performance Optical Coherence Tomography (OCT) processing library with GPU acceleration.


## Requirements

- **CUDA Toolkit** 11.0 or higher
- **CMake** 3.18 or higher
- **FFTW3** (single precision)
  - Windows: Included in `src/thirdparty/fftw/`
  - Linux: `sudo apt-get install libfftw3-dev` (Debian/Ubuntu)
- **Python** 3.8+ (optional, for Python bindings)
- **C++ Compiler**
  - Windows: Visual Studio 2019+ with C++14 support
  - Linux: GCC or Clang with C++14 support

## Build

### Windows

Open **Developer Command Prompt for VS 2022** (or your Visual Studio version), navigate to the project root directory:

```bash
cd path\to\octproengine  # Replace with your actual path
```
 then run:

```bash
build_windows.bat
```
The bat script builds the C++ library as well as the Python bindings.


**Alternative - C++ library only (skip Python bindings):**
```bash
cd path\to\octproengine  # Replace with your actual path
mkdir build
cd build
cmake .. -DBUILD_PYTHON=OFF -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Linux

not tested yet

## Running Tests

### C++ Tests

```bash
cd build/tests/Release

test_backend_output_comparison
test_performance_benchmark
test_curve_functionality
```

### Python Tests

After building, the Python module is in the build directory but not yet on your Python path. 

**Set PYTHONPATH (temporary, per-session):**

The build script outputs the exact command you need. After `build_windows.bat` completes, it will show:
```
set PYTHONPATH=C:\your\actual\path\octproengine\build\python\Release;%PYTHONPATH%
```
Copy and paste that exact command into your command prompt.

Then you can run Python tests:
```bash
cd python/tests
python run_all_tests.py
```

## Quick Start - C++

```cpp
#include "processor.h"
#include <iostream>

int main() {
    // Create processor
    ope::Processor processor(ope::Backend::CUDA);
    
    // Configure
    processor.setInputParameters(2048, 512, 1, ope::DataType::UINT16);
    processor.enableResampling(true);
    processor.enableWindowing(true);
    processor.enableLogScaling(true);
    
    // Initialize
    processor.initialize();
    
    // Set callback
    processor.setOutputCallback([](const ope::IOBuffer& output) {
        std::cout << "Processed " << output.getSizeInBytes() << " bytes" << std::endl;
    });
    
    // Get buffer, fill with data, process
    ope::IOBuffer& buffer = processor.getNextAvailableInputBuffer();
    // ... fill buffer with your OCT data ...
    processor.process(buffer);
    
    return 0;
}
```


## Quick Start - Python


```python
import octproengine as ope
import numpy as np

# Create processor
proc = ope.Processor(ope.Backend.CUDA)

# Configure
proc.set_input_parameters(2048, 512, 1, ope.DataType.UINT16)
proc.enable_resampling(True)
proc.enable_windowing(True)
proc.enable_log_scaling(True)
proc.initialize()

# Set callback
def on_output(output_array):
    print(f"Processed: {output_array.shape}, dtype={output_array.dtype}")

proc.set_callback(on_output)

# Process data
data = np.random.randint(0, 65535, size=2048*512, dtype=np.uint16)
buffer = proc.get_next_available_buffer()
buffer[:] = data
proc.process(buffer)
```