# Performance Tests

All performance tests carry the ctest label `perf` (which excludes them from the
functional test run, `ctest --test-dir build -C Release -LE perf`). They print their
results to the console, so run them directly from `build/tests/Release/`:

- **test_performance_benchmark** - cross-backend throughput over a signal-length/A-scan grid, also saves the results as CSV
- **test_consumer_performance_impact** - impact of input/output consumers and drop policies on the processing rate
- **test_zerocopy_performance** - CUDA zero-copy vs pinned memory transfer (mainly relevant on Jetson)

---

# Performance Benchmark 
2025-11-30  

OS: Windows 10  
RAM: 32 GB  
GPU: NVIDIA GeForce GTX 1080  
CPU: AMD Ryzen 5 5600X  

This benchmark was performed using [test_performance_benchmark.cpp](../../tests/test_performance_benchmark.cpp) from the `tests` folder.

Configuration:  
  Processing: Resampling(CUBIC) + Windowing + Dispersion + BG-Removal + Log-Scale  
  Iterations per test: 200  
  Backends: CPU CUDA OpenCL Vulkan  
  Bitdepth: 16-bit unsigned integer  

_Signal_ is the number of samples per raw A-scan  
_Ascans_ is the number of A-scans per B-scan  
_Bscans_ is the number of B-scans per buffer  
_A buffer_ is the amount of data processed in one go on GPU  

| Signal  | AScans | BScans | Backend | Time(ms) | BScans/s | AScans/s | MB/s | Speedup |
|--------|--------|--------|---------|----------|----------|----------|------|---------|
| 512 | 256 | 1 | CPU | 1.904 | 525 | 134,481 | 131.33 | - |
| 512 | 256 | 1 | CUDA | 0.189 | 5,289 | 1,353,888 | 1322.16 | 10.07x |
| 512 | 256 | 1 | OpenCL | 0.287 | 3,483 | 891,598 | 870.70 | 6.63x |
| 512 | 256 | 1 | Vulkan | 0.102 | 9,809 | 2,511,035 | 2452.18 | 18.67x |
| | | | | | | | | |
| 512 | 512 | 1 | CPU | 3.735 | 268 | 137,081 | 133.87 | - |
| 512 | 512 | 1 | CUDA | 0.173 | 5,775 | 2,956,632 | 2887.34 | 21.57x |
| 512 | 512 | 1 | OpenCL | 0.363 | 2,754 | 1,410,138 | 1377.09 | 10.29x |
| 512 | 512 | 1 | Vulkan | 0.173 | 5,782 | 2,960,222 | 2890.84 | 21.59x |
| | | | | | | | | |
| 512 | 1024 | 1 | CPU | 7.464 | 134 | 137,195 | 133.98 | - |
| 512 | 1024 | 1 | CUDA | 0.336 | 2,976 | 3,047,664 | 2976.23 | 22.21x |
| 512 | 1024 | 1 | OpenCL | 0.684 | 1,462 | 1,496,802 | 1461.72 | 10.91x |
| 512 | 1024 | 1 | Vulkan | 0.310 | 3,229 | 3,306,212 | 3228.72 | 24.10x |
| | | | | | | | | |
| 1024 | 256 | 1 | CPU | 3.639 | 275 | 70,340 | 137.38 | - |
| 1024 | 256 | 1 | CUDA | 0.240 | 4,170 | 1,067,490 | 2084.94 | 15.18x |
| 1024 | 256 | 1 | OpenCL | 0.389 | 2,569 | 657,692 | 1284.55 | 9.35x |
| 1024 | 256 | 1 | Vulkan | 0.174 | 5,737 | 1,468,732 | 2868.62 | 20.88x |
| | | | | | | | | |
| 1024 | 512 | 1 | CPU | 7.321 | 137 | 69,940 | 136.60 | - |
| 1024 | 512 | 1 | CUDA | 0.343 | 2,914 | 1,491,928 | 2913.92 | 21.33x |
| 1024 | 512 | 1 | OpenCL | 0.679 | 1,472 | 753,734 | 1472.14 | 10.78x |
| 1024 | 512 | 1 | Vulkan | 0.313 | 3,193 | 1,634,634 | 3192.64 | 23.37x |
| | | | | | | | | |
| 1024 | 1024 | 1 | CPU | 14.916 | 67 | 68,649 | 134.08 | - |
| 1024 | 1024 | 1 | CUDA | 0.502 | 1,992 | 2,039,556 | 3983.51 | 29.71x |
| 1024 | 1024 | 1 | OpenCL | 1.108 | 902 | 923,971 | 1804.63 | 13.46x |
| 1024 | 1024 | 1 | Vulkan | 0.581 | 1,722 | 1,763,708 | 3444.74 | 25.69x |
| | | | | | | | | |
| 2048 | 256 | 1 | CPU | 7.463 | 134 | 34,301 | 133.99 | - |
| 2048 | 256 | 1 | CUDA | 0.308 | 3,243 | 830,279 | 3243.28 | 24.21x |
| 2048 | 256 | 1 | OpenCL | 0.686 | 1,457 | 373,020 | 1457.11 | 10.87x |
| 2048 | 256 | 1 | Vulkan | 0.354 | 2,824 | 723,011 | 2824.26 | 21.08x |
| | | | | | | | | |
| 2048 | 512 | 1 | CPU | 14.970 | 67 | 34,201 | 133.60 | - |
| 2048 | 512 | 1 | CUDA | 0.475 | 2,104 | 1,077,067 | 4207.29 | 31.49x |
| 2048 | 512 | 1 | OpenCL | 1.146 | 872 | 446,580 | 1744.45 | 13.06x |
| 2048 | 512 | 1 | Vulkan | 0.693 | 1,443 | 739,041 | 2886.88 | 21.61x |
| | | | | | | | | |
| 2048 | 1024 | 1 | CPU | 30.994 | 32 | 33,038 | 129.06 | - |
| 2048 | 1024 | 1 | CUDA | 0.798 | 1,252 | 1,282,413 | 5009.42 | 38.82x |
| 2048 | 1024 | 1 | OpenCL | 2.131 | 469 | 480,565 | 1877.21 | 14.55x |
| 2048 | 1024 | 1 | Vulkan | 1.345 | 743 | 761,140 | 2973.20 | 23.04x |

---

# Performance Benchmark Jetson Orin Nano 
2025-11-30  

Device: NVIDIA Jetson Orin Nano 8GB  
GPU: NVIDIA Ampere architecture (1024 CUDA cores, Compute Capability 8.7)  
RAM: 8 GB LPDDR5  
CPU: 6-core ARM Cortex-A78AE  

This benchmark was performed using [test_performance_benchmark.cpp](../../tests/test_performance_benchmark.cpp) from the `tests` folder.

Configuration:  
  Processing: Resampling(CUBIC) + Windowing + Dispersion + DC-Removal + Log-Scale  
  Iterations per test: 20000  
  Backends: CUDA  
  Bitdepth: 16-bit unsigned integer  

_Signal_ is the number of samples per raw A-scan  
_Ascans_ is the number of A-scans per B-scan  
_Bscans_ is the number of B-scans per buffer  
_A buffer_ is the amount of data processed in one go on GPU  


| Signal  | AScans | BScans | Backend | Time(ms) | BScans/s | AScans/s | MB/s | Speedup |
|--------|--------|--------|---------|----------|----------|----------|------|---------|
| 512 | 256 | 1 | CUDA | 0.448 | 2,232 | 571,503 | 558.11 | - |
| | | | | | | | | |
| 512 | 512 | 1 | CUDA | 0.493 | 2,029 | 1,038,764 | 1014.42 | - |
| | | | | | | | | |
| 512 | 1024 | 1 | CUDA | 0.686 | 1,458 | 1,492,818 | 1457.83 | - |
| | | | | | | | | |
| 1024 | 256 | 1 | CUDA | 0.488 | 2,048 | 524,194 | 1023.82 | - |
| | | | | | | | | |
| 1024 | 512 | 1 | CUDA | 0.697 | 1,436 | 735,040 | 1435.62 | - |
| | | | | | | | | |
| 1024 | 1024 | 1 | CUDA | 1.462 | 684 | 700,247 | 1367.67 | - |
| | | | | | | | | |
| 2048 | 256 | 1 | CUDA | 0.703 | 1,423 | 364,359 | 1423.28 | - |
| | | | | | | | | |
| 2048 | 512 | 1 | CUDA | 1.425 | 702 | 359,256 | 1403.34 | - |
| | | | | | | | | |
| 2048 | 1024 | 1 | CUDA | 2.802 | 357 | 365,451 | 1427.54 | - |