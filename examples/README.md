# OCTproEngine Examples

## Files

- **example_basic.cpp** - Minimal example showing library usage
- **example_recorder.cpp** - Recording raw and processed data to disk with the Recorder tool
- **example_cuda_device_selection.cpp** - Enumerating CUDA devices and selecting one via backend configuration
- **octproviewer.cpp** - OCTproViewer - Interactive GUI for OCTproEngine.
- **octproviewer_benchmark.cpp** - Throughput benchmark variant of OCTproViewer

Python examples live in [python/examples/](../python/examples/).

## Building

```bash
# Configure with examples enabled and FFTW auto-download enabled
cmake .. -DBUILD_EXAMPLES=ON -DBUILD_OCT_VIEWER=ON -DCMAKE_BUILD_TYPE=Release -DFFTW3_AUTO_DOWNLOAD=ON

# Build all
cmake --build . --config Release

# Or build specific target
cmake --build . --config Release --target octproviewer
```

## Requirements

- OCTproViewer requires GLFW and Dear ImGui (automatically downloaded to `thirdparty/` folder during build process)