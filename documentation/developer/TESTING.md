# Running tests

The default CTest suite runs short functional tests with **CUDA**. On a fresh
configuration without CUDA support, the default is the first enabled backend in
this order: CPU, OpenCL, Vulkan. CMake prints the selection.

From the repository root:

```sh
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Two CMake settings control the suite:

| Setting | Default | Purpose |
|---|---|---|
| `OPE_TEST_BACKENDS` | CUDA, or the fallback above | One backend or a quoted list, e.g. `"CUDA;CPU"`. Names: `CUDA`, `CPU`, `OPENCL`, `VULKAN`; each must be enabled in the build. |
| `OPE_TEST_EXTENDED` | `OFF` | `ON` adds long stress tests, comparisons between backends, and benchmarks. |

Backend selection applies to tests registered with a backend suffix, such as
`test_output_buffer_ownership_CUDA`. Shared tests run once; tests for specific
backends and comparisons retain their own backend requirements. Unavailable
hardware is reported as **Skipped**.

Full functional coverage (enable only backends compiled into your build):

```sh
cmake -S . -B build "-DOPE_TEST_BACKENDS=CUDA;CPU;OPENCL;VULKAN" -DOPE_TEST_EXTENDED=ON
ctest --test-dir build -C Release -LE perf --output-on-failure
```

Run benchmarks separately with `ctest --test-dir build -C Release -L perf -V`.
Run one registered case with `ctest --test-dir build -C Release -R "^test_buffer_ordering_content_CUDA$" --output-on-failure`.
Test executables are built even when excluded from CTest. To run one directly,
pass its backend as the argument, e.g. `CUDA` for `test_buffer_ordering_content`;
this also works without enabling extended mode.

Settings persist in the build directory. Return to everyday testing with:

```sh
cmake -S . -B build -DOPE_TEST_BACKENDS=CUDA -DOPE_TEST_EXTENDED=OFF
```
