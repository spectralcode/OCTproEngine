# OCTproEngine

High-performance Optical Coherence Tomography (OCT) processing library with GPU acceleration.

## Performance
Preliminary results: [Performance Benchmark](documentation/developer/PERFORMANCE.md)

## Requirements

- **CMake** ≥ 3.18
- **CUDA Toolkit** ≥ 11.0 (optional, for CUDA backend)
- **FFTW3** (optional, for CPU backend)
- **OpenCL** (optional, for OpenCL backend)
	- VkFFT
- **Python** ≥ 3.8 (optional, for Python bindings)
	- pybind11, NumPy
- **C++ Compiler**
	- Win: VS 2019+ (C++14, must be compatible with your cuda version)  

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


**Alternative: build manually**  
For default build without python bindings, run:
```bash
cd path\to\octproengine  # Replace with your actual path
mkdir build
cd build
cmake .. 
cmake --build . --config Release
```
or you can enable python bindings with:  
```bash
cmake .. -DBUILD_PYTHON=ON
cmake --build . --config Release
```
Here is a list of all available build options:  

**Build Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_CUDA` | `ON` | Build with CUDA backend support |
| `BUILD_CPU` | `ON` | Build with CPU backend (requires FFTW3) |
| `BUILD_OPENCL` | `ON` | Build with OpenCL backend (requires VkFFT) |
| `BUILD_PYTHON` | `OFF` | Build Python bindings (requires pybind11, NumPy) |
| `BUILD_TESTS` | `ON` | Build test suite |
| `BUILD_EXAMPLES` | `ON` | Build example applications |
| `BUILD_TOOLS` | `ON` | Build optional ProcessorTools (Recorder, etc.) |
| `BUILD_OCT_VIEWER` | `OFF` | Build interactive OCTproViewer with ImGui (auto-downloads GLFW & ImGui) |
| `FFTW3_AUTO_DOWNLOAD` | `ON` | Auto-download FFTW3 if not found (Windows only) |
| `VKFFT_AUTO_DOWNLOAD` | `ON` | Auto-download VkFFT if not found |

**Note:** At least one backend (`BUILD_CUDA`, `BUILD_CPU`, or `BUILD_OPENCL`) must be enabled.

### Linux / Jetson

See [Jetson Build Instructions](documentation/BUILDING_JETSON.md) for NVIDIA Jetson platforms.

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
def on_output(output_array, buffer_id):
    print(f"Processed buffer {buffer_id}: {output_array.shape}, dtype={output_array.dtype}")

proc.add_output_callback(on_output)

# Process data
data = np.random.randint(0, 65535, size=2048*512, dtype=np.uint16)
buffer = proc.get_next_available_buffer()
buffer[:] = data
proc.process(buffer)
```