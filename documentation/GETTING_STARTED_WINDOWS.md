# Getting Started on Windows

This is the shortest path to build OCTproEngine, open the ImGui app, and run the Python examples.

## 1. Install what you need

- Git: <https://git-scm.com/install/windows.html>
- MSVC build tools / Visual Studio 2022:
  <https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#fixed-version-bootstrappers>
- CMake: <https://cmake.org/download/>
- Python 3: <https://www.python.org/downloads/windows/>

Optional:
- CUDA Toolkit if you want the CUDA backend:
  <https://developer.nvidia.com/cuda-downloads>
- Vulkan SDK if you want the Vulkan backend:
  <https://vulkan.lunarg.com/sdk/home>

Notes:
- On the Microsoft page above, use the `Build Tools` bootstrapper, or install Visual Studio 2022 with `Desktop development with C++`.
- FFTW3 is needed for the CPU backend, but on Windows the build script can download it for you.
- OpenCL is often already available through your GPU driver, so no separate OpenCL install is needed in many cases.
- If CUDA, OpenCL, or Vulkan are missing, CMake will disable those backends automatically.

## 2. Open Developer Command Prompt
Open the Windows Start menu, type `Developer Command Prompt for VS 2022`, and open it.

## 3. Clone the repo
Navigate to a folder that already exists. For example, your Documents folder:

```bat
cd %USERPROFILE%\Documents
```

Then clone the repo with:

```bat
git clone https://github.com/spectralcode/OCTproEngine.git
```

Then navigate into the project folder:

```bat
cd OCTproEngine
```

## 4. Build the project

Then run:

```bat
build_windows.bat
```

Recommended answers:
- `Build Python bindings?` -> `Y` if you want to run the Python example
- `Auto-download FFTW3?` -> `Y` if asked
- `Build OCTproViewer?` -> `Y` if you want the ImGui app

When it works, the script prints `Build completed successfully!`.


## 5. Open the ImGui app

If you built `OCTproViewer`, run:

```bat
build\examples\Release\octproviewer.exe
```

## 6. Optional: Run a simple Python example

If you built the Python bindings, you can run the Python examples and tests.

First, set `PYTHONPATH` so Python can find the built `octproengine` module:

```bat
set PYTHONPATH=%CD%\build\python\Release;%PYTHONPATH%
```

This command assumes you are still in the `OCTproEngine` folder.

Then run the simple synthetic-data example:

```bat
py -3 python\examples\example_basic.py
```

If `py` is not available on your machine, use `python` instead.

## 7. Optional: Run the raw-file Python example

This example does not include sample OCT data. It needs your own raw OCT file.

Open `python/examples/example_load_raw_file.py` in a text editor or in your Python IDE and set:

```python
RAW_FILE_PATH = Path(r"C:\path\to\your\data.raw")
```

Also modify the processing settings and input data parameters in that file if needed.

Save the modified file and then run:

```bat
py -3 -m pip install matplotlib
py -3 python\examples\example_load_raw_file.py
```

`matplotlib` is only needed for this example so it can display the processed image.
