#!/usr/bin/env python3
"""
OCT Processing Performance Benchmark

This test mirrors the C++ test_performance_benchmark.cpp
It benchmarks processing performance across different buffer configurations
and optionally compares CPU vs CUDA performance.
"""

import sys
import numpy as np
import time
import threading

try:
    import octproengine as ope
except ImportError:
    print("ERROR: octproengine module not found. Please build and install the Python bindings first.")
    sys.exit(1)

# ============================================
# CONFIGURE BENCHMARK HERE (matches C++ test)
# ============================================

# Which backends to benchmark
BENCHMARK_CPU = False
BENCHMARK_CUDA = True

# Buffer sizes to test (will test all combinations)
SIGNAL_LENGTHS = [512, 1024, 2048, 4096]
ASCANS_PER_BSCAN = [256, 512, 1024, 2048]
BSCANS_PER_BUFFER = [1]

# Number of iterations per test
ITERATIONS = 100

# Processing configuration
ENABLE_RESAMPLING = True
ENABLE_WINDOWING = True
ENABLE_DISPERSION = True
ENABLE_BACKGROUND_REMOVAL = True
ENABLE_LOG_SCALING = True
ENABLE_BSCAN_FLIP = False

# Resampling configuration
INTERPOLATION_METHOD = ope.InterpolationMethod.CUBIC
RESAMPLING_COEFFS = [0.5, 2048.0, -100.0, 0.0]

# Windowing configuration
WINDOW_TYPE = ope.WindowType.HANN
WINDOW_CENTER = 0.5
WINDOW_FILL_FACTOR = 0.95

# Dispersion configuration
DISPERSION_COEFFS = [0.0, 0.0, 1.0, 2.0]
DISPERSION_FACTOR = 1.0

# Post-processing
GRAYSCALE_MIN = 30.0
GRAYSCALE_MAX = 100.0

# Output options
SAVE_CSV = True
CSV_FILENAME = "benchmark_results.csv"

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_synthetic_ascan(signal_length, ascan_index):
    """Generate synthetic A-scan data - vectorized"""
    peak1_depth = signal_length * 0.2
    peak2_depth = signal_length * 0.5
    peak3_depth = signal_length * 0.7
    
    peak1_width = 50.0
    peak2_width = 30.0
    peak3_width = 40.0
    
    peak1_amp = 8000.0
    peak2_amp = 5000.0
    peak3_amp = 3000.0
    
    lateral_phase = ascan_index * 0.1
    
    # Vectorized computation - all samples at once
    i = np.arange(signal_length, dtype=np.float64)
    
    value = np.full(signal_length, 2000.0, dtype=np.float64)
    
    # Add Gaussian peaks
    value += peak1_amp * np.exp(-((i - peak1_depth) / peak1_width) ** 2)
    value += peak2_amp * np.exp(-((i - peak2_depth) / peak2_width) ** 2)
    value += peak3_amp * np.exp(-((i - peak3_depth) / peak3_width) ** 2)
    
    # Add sinusoidal components
    value += 1000.0 * np.sin(i * 0.3 + lateral_phase)
    value += 500.0 * np.sin(i * 0.15 + lateral_phase * 0.5)
    
    # Add noise
    value += np.random.randint(-200, 201, size=signal_length)
    
    # Clip and convert to uint16
    ascan = np.clip(value, 0, 65535).astype(np.uint16)
    
    return ascan


def generate_test_data(signal_length, ascans_per_bscan, bscans_per_buffer):
    """Generate test data (optimized - reuse same B-scan)"""
    samples_per_bscan = signal_length * ascans_per_bscan
    
    # Generate one B-scan
    single_bscan = np.zeros(samples_per_bscan, dtype=np.uint16)
    idx = 0
    for a in range(ascans_per_bscan):
        ascan = generate_synthetic_ascan(signal_length, a)
        single_bscan[idx:idx+signal_length] = ascan
        idx += signal_length
    
    # Copy the same B-scan multiple times (fast!)
    data = np.tile(single_bscan, bscans_per_buffer)
    
    return data


def configure_processor(processor, signal_length, ascans_per_bscan, bscans_per_buffer):
    """Configure processor with test parameters"""
    processor.set_input_parameters(
        signal_length,
        ascans_per_bscan,
        bscans_per_buffer,
        ope.DataType.UINT16
    )
    
    processor.enable_resampling(ENABLE_RESAMPLING)
    if ENABLE_RESAMPLING:
        processor.set_interpolation_method(INTERPOLATION_METHOD)
        processor.set_resampling_coefficients(RESAMPLING_COEFFS)
    
    processor.enable_windowing(ENABLE_WINDOWING)
    if ENABLE_WINDOWING:
        processor.set_window_parameters(
            WINDOW_TYPE,
            WINDOW_CENTER,
            WINDOW_FILL_FACTOR
        )
    
    processor.enable_dispersion_compensation(ENABLE_DISPERSION)
    if ENABLE_DISPERSION:
        processor.set_dispersion_coefficients(
            DISPERSION_COEFFS,
            DISPERSION_FACTOR
        )
    
    processor.enable_background_removal(ENABLE_BACKGROUND_REMOVAL)
    processor.enable_log_scaling(ENABLE_LOG_SCALING)
    processor.set_grayscale_range(GRAYSCALE_MIN, GRAYSCALE_MAX)
    processor.enable_bscan_flip(ENABLE_BSCAN_FLIP)


class BenchmarkResult:
    """Container for benchmark results"""
    def __init__(self):
        self.signal_length = 0
        self.ascans_per_bscan = 0
        self.bscans_per_buffer = 0
        self.backend = ""
        self.iterations = 0
        self.total_time_ms = 0.0
        self.avg_time_ms = 0.0
        self.bscans_per_sec = 0.0
        self.ascans_per_sec = 0.0
        self.mb_per_sec = 0.0
        self.speedup = 1.0


def run_benchmark(backend, signal_length, ascans_per_bscan, bscans_per_buffer, test_data):
    """Run benchmark for specified configuration"""
    result = BenchmarkResult()
    result.signal_length = signal_length
    result.ascans_per_bscan = ascans_per_bscan
    result.bscans_per_buffer = bscans_per_buffer
    result.backend = "CPU" if backend == ope.Backend.CPU else "CUDA"
    result.iterations = ITERATIONS
    
    try:
        processor = ope.Processor(backend)
    except Exception as e:
        print(f"    ERROR: Failed to create processor: {e}")
        return None
    
    configure_processor(processor, signal_length, ascans_per_bscan, bscans_per_buffer)
    processor.initialize()
    
    completed_iterations = {'count': 0}
    lock = threading.Lock()
    
    def on_output(output_array):
        with lock:
            completed_iterations['count'] += 1
    
    processor.set_callback(on_output)
    
    start_time = time.time()
    
    for iter in range(ITERATIONS):
        # Get next available buffer (blocks if all buffers are busy)
        buffer = processor.get_next_available_buffer()
        
        # Copy data to buffer
        buffer[:] = test_data
        
        # Submit for processing (returns immediately, processing happens async)
        processor.process(buffer)
    
    # Wait for all iterations to complete
    while completed_iterations['count'] < ITERATIONS:
        time.sleep(0.0001)
    
    end_time = time.time()
    
    # Calculate statistics
    result.total_time_ms = (end_time - start_time) * 1000.0
    result.avg_time_ms = result.total_time_ms / ITERATIONS
    
    # Calculate throughput
    result.bscans_per_sec = 1000.0 * bscans_per_buffer / result.avg_time_ms
    result.ascans_per_sec = result.bscans_per_sec * ascans_per_bscan
    
    # Calculate bandwidth (input data size)
    bytes_per_iteration = signal_length * ascans_per_bscan * bscans_per_buffer * 2  # uint16 = 2 bytes
    result.mb_per_sec = (bytes_per_iteration * 1000.0 / result.avg_time_ms) / (1024.0 * 1024.0)
    
    processor.stop()
    return result


def format_number(value):
    """Format large numbers with thousand separators"""
    return f"{value:,.0f}"


def print_results_table(results):
    """Print results in a formatted table"""
    print()
    print("+" + "-" * 110 + "+")
    print(f"| {'Signal':<6} | {'AScans':<6} | {'BScans':<6} | {'Backend':<7} | {'Time(ms)':>9} | "
          f"{'BScans/s':>12} | {'AScans/s':>14} | {'MB/s':>10} | {'Speedup':>8} |")
    print("+" + "-" * 110 + "+")
    
    # Group by buffer configuration
    last_signal_length = -1
    last_ascans = -1
    last_bscans = -1
    
    for i, r in enumerate(results):
        # Add separator between different configurations
        if (r.signal_length != last_signal_length or 
            r.ascans_per_bscan != last_ascans or 
            r.bscans_per_buffer != last_bscans):
            if i > 0:
                print("+" + "-" * 110 + "+")
            last_signal_length = r.signal_length
            last_ascans = r.ascans_per_bscan
            last_bscans = r.bscans_per_buffer
        
        # Format speedup string
        speedup_str = "-" if r.speedup <= 1.0 else f"{r.speedup:.2f}x"
        
        print(f"| {r.signal_length:<6} | {r.ascans_per_bscan:<6} | {r.bscans_per_buffer:<6} | "
              f"{r.backend:<7} | {r.avg_time_ms:>9.3f} | "
              f"{format_number(r.bscans_per_sec):>12} | "
              f"{format_number(r.ascans_per_sec):>14} | "
              f"{r.mb_per_sec:>10.2f} | {speedup_str:>8} |")
    
    print("+" + "-" * 110 + "+")
    print()


def save_results_csv(results, filename):
    """Save results to CSV file"""
    try:
        with open(filename, 'w') as f:
            f.write("SignalLength,AScansPerBScan,BScansPerBuffer,Backend,Iterations,"
                   "TotalTime_ms,AvgTime_ms,BScansPerSec,AScansPerSec,MB_per_sec,Speedup\n")
            
            for r in results:
                f.write(f"{r.signal_length},{r.ascans_per_bscan},{r.bscans_per_buffer},"
                       f"{r.backend},{r.iterations},"
                       f"{r.total_time_ms:.6f},{r.avg_time_ms:.6f},"
                       f"{r.bscans_per_sec:.2f},{r.ascans_per_sec:.2f},"
                       f"{r.mb_per_sec:.2f},{r.speedup:.2f}\n")
        
        print(f"Results saved to: {filename}")
        return True
    except Exception as e:
        print(f"Failed to save CSV file: {e}")
        return False


def print_configuration():
    """Print test configuration"""
    print("Configuration:")
    print("  Processing: ", end="")
    
    enabled = []
    if ENABLE_RESAMPLING:
        method_name = {
            ope.InterpolationMethod.LINEAR: "LINEAR",
            ope.InterpolationMethod.CUBIC: "CUBIC",
            ope.InterpolationMethod.LANCZOS: "LANCZOS"
        }.get(INTERPOLATION_METHOD, "UNKNOWN")
        enabled.append(f"Resampling({method_name})")
    if ENABLE_WINDOWING:
        enabled.append("Windowing")
    if ENABLE_DISPERSION:
        enabled.append("Dispersion")
    if ENABLE_BACKGROUND_REMOVAL:
        enabled.append("BG-Removal")
    if ENABLE_LOG_SCALING:
        enabled.append("Log-Scale")
    
    print(" + ".join(enabled))
    print(f"  Iterations per test: {ITERATIONS}")
    print("  Backends: ", end="")
    if BENCHMARK_CPU:
        print("CPU ", end="")
    if BENCHMARK_CUDA:
        print("CUDA ", end="")
    print()
    print()

# ============================================
# MAIN BENCHMARK
# ============================================

def main():
    try:
        print("=" * 40)
        print("OCT Processing Performance Benchmark")
        print("=" * 40)
        print()
        
        print_configuration()
        
        print("Running benchmarks...")
        print()
        
        all_results = []
        
        # Calculate total number of tests
        num_backends = (1 if BENCHMARK_CPU else 0) + (1 if BENCHMARK_CUDA else 0)
        total_tests = len(SIGNAL_LENGTHS) * len(ASCANS_PER_BSCAN) * len(BSCANS_PER_BUFFER) * num_backends
        current_test = 0
        
        # Iterate through all combinations
        for signal_length in SIGNAL_LENGTHS:
            for ascans_per_bscan in ASCANS_PER_BSCAN:
                for bscans_per_buffer in BSCANS_PER_BUFFER:
                    
                    # Generate test data once for this size
                    test_data = generate_test_data(signal_length, ascans_per_bscan, bscans_per_buffer)
                    
                    cpu_result = None
                    cuda_result = None
                    
                    # Benchmark CPU
                    if BENCHMARK_CPU:
                        current_test += 1
                        print(f"[{current_test}/{total_tests}] Testing CPU: "
                              f"{signal_length}x{ascans_per_bscan}x{bscans_per_buffer} ... ", end="", flush=True)
                        
                        cpu_result = run_benchmark(ope.Backend.CPU, signal_length, 
                                                  ascans_per_bscan, bscans_per_buffer, test_data)
                        
                        if cpu_result:
                            all_results.append(cpu_result)
                            print(f"{cpu_result.avg_time_ms:.3f} ms")
                        else:
                            print("FAILED")
                    
                    # Benchmark CUDA
                    if BENCHMARK_CUDA:
                        current_test += 1
                        print(f"[{current_test}/{total_tests}] Testing CUDA: "
                              f"{signal_length}x{ascans_per_bscan}x{bscans_per_buffer} ... ", end="", flush=True)
                        
                        cuda_result = run_benchmark(ope.Backend.CUDA, signal_length, 
                                                   ascans_per_bscan, bscans_per_buffer, test_data)
                        
                        if cuda_result:
                            # Calculate speedup if we have CPU result
                            if cpu_result:
                                cuda_result.speedup = cpu_result.avg_time_ms / cuda_result.avg_time_ms
                            
                            all_results.append(cuda_result)
                            print(f"{cuda_result.avg_time_ms:.3f} ms", end="")
                            if cpu_result:
                                print(f" (speedup: {cuda_result.speedup:.2f}x)")
                            else:
                                print()
                        else:
                            print("FAILED")
        
        # Print results table
        print()
        print("=" * 40)
        print("Results")
        print("=" * 40)
        
        print_results_table(all_results)
        
        # Save CSV if requested
        if SAVE_CSV:
            save_results_csv(all_results, CSV_FILENAME)
        
        print("Benchmark complete!")
        
        return 0
    
    except Exception as e:
        print()
        print("=" * 40)
        print("EXCEPTION OCCURRED:")
        print("=" * 40)
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
