# OCTproEngine Examples

## Files

- **example_basic.cpp** - Minimal example showing library usage
- **octproviewer.cpp** - OCTproViewer - Interactive GUI for OCTproEngine.

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