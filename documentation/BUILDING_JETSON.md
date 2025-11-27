# Building OCTproEngine on NVIDIA Jetson devices 


```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y libglfw3-dev libgl1-mesa-dev

# 2. Clone repository
git clone https://github.com/yourusername/OCTproEngine.git
cd OCTproEngine

# 3. Setup ImGui dependency (optional, but required for OCTproViewer)
mkdir -p thirdparty
cd thirdparty
git clone --depth 1 --branch v1.91.5 https://github.com/ocornut/imgui.git
cd ..

# 4. Configure
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DBUILD_OCT_VIEWER=ON

# 5. Build
cmake --build . 

# 6. Run examples
./examples/example_basic
./examples/octproviewer

# 7. Run tests
./tests/test_performance_benchmark
```


## Build Steps for Python Bindings

```bash
# 1. Install Python dependencies
pip3 install pybind11

# 2. Configure with Python bindings
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON

# 3. Build
cmake --build .

# 4. set up environment variables
cd ..
export LD_LIBRARY_PATH=$(pwd)/build/src:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd)/build/python/Release:$PYTHONPATH

# 5. Run Python examples
python3 python/examples/example_basic.py

# 6. Run specific Python test
python3 python/tests/test_performance_benchmark.py

# 7. or run all Python tests
python3 python/tests/run_all_tests.py
```
