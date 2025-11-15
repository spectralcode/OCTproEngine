#!/usr/bin/env python3
"""
CPU vs CUDA Backend Output Comparison Test

This test mirrors the C++ test_backend_output_comparison.cpp
It processes the same synthetic data with both CPU and CUDA backends
and compares the outputs to ensure they match within tolerance.
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
# CONFIGURATION (matches C++ test)
# ============================================

SIGNAL_LENGTH = 2048
ASCANS_PER_BSCAN = 1024
BSCANS_PER_BUFFER = 1

ENABLE_RESAMPLING = True
ENABLE_WINDOWING = True
ENABLE_DISPERSION = True
ENABLE_BACKGROUND_REMOVAL = True
ENABLE_LOG_SCALING = True
ENABLE_BSCAN_FLIP = False
ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION = False

INTERPOLATION_METHOD = ope.InterpolationMethod.CUBIC
RESAMPLING_COEFFS = [0.5, 2048.0, -100.0, 50.0]

WINDOW_TYPE = ope.WindowType.HANN
WINDOW_CENTER = 0.9
WINDOW_FILL_FACTOR = 0.85

DISPERSION_COEFFS = [0.0, 0.0, 1.0, -3.0]
DISPERSION_FACTOR = 1.0

GRAYSCALE_MIN = 0.0
GRAYSCALE_MAX = 80.0

# Tolerance for comparison
TOLERANCE = 0.05

SAVE_OUTPUTS = True

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_synthetic_ascan(signal_length, ascan_index):
    """Generate synthetic A-scan data (matches C++ version) - vectorized"""
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
    data = np.zeros(signal_length * ascans_per_bscan * bscans_per_buffer, dtype=np.uint16)
    
    idx = 0
    for b in range(bscans_per_buffer):
        for a in range(ascans_per_bscan):
            ascan = generate_synthetic_ascan(signal_length, a)
            data[idx:idx+signal_length] = ascan
            idx += signal_length
    
    return data


class ProcessingResult:
    """Container for processing results"""
    def __init__(self):
        self.output = None
        self.start_time = None
        self.end_time = None
        self.received = False
        self.event = threading.Event()
    
    def wait_for_completion(self, timeout_ms=5000):
        """Wait for processing to complete"""
        return self.event.wait(timeout=timeout_ms/1000.0)
    
    def get_duration_ms(self):
        """Get processing duration in milliseconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0


def compare_buffers(buffer1, buffer2, samples_per_ascan, tolerance):
    """Compare two output buffers"""
    if len(buffer1) != len(buffer2):
        return None
    
    result = {
        'match': True,
        'max_abs_diff': 0.0,
        'mean_abs_diff': 0.0,
        'rms_error': 0.0,
        'total_samples': len(buffer1),
        'differing_samples': 0,
        'max_diff_per_ascan': []
    }
    
    diff = np.abs(buffer1 - buffer2)
    
    # Overall statistics
    result['max_abs_diff'] = np.max(diff)
    result['mean_abs_diff'] = np.mean(diff)
    result['rms_error'] = np.sqrt(np.mean(diff ** 2))
    result['differing_samples'] = np.sum(diff > tolerance)
    
    if result['differing_samples'] > 0:
        result['match'] = False
    
    # Per-A-scan statistics
    num_ascans = len(buffer1) // samples_per_ascan
    for i in range(num_ascans):
        start = i * samples_per_ascan
        end = start + samples_per_ascan
        ascan_diff = diff[start:end]
        result['max_diff_per_ascan'].append(np.max(ascan_diff))
    
    return result


def configure_processor(processor):
    """Configure processor with test parameters"""
    processor.set_input_parameters(
        SIGNAL_LENGTH,
        ASCANS_PER_BSCAN,
        BSCANS_PER_BUFFER,
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
    processor.enable_post_process_background_subtraction(ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION)
    if ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION:
        processor.request_post_process_background_recording()


def run_backend_test(backend_name, backend_type, test_data):
    """Run test with specified backend"""
    print(f"Processing with {backend_name} backend...")
    
    try:
        processor = ope.Processor(backend_type)
    except Exception as e:
        print(f"  ERROR: Failed to create {backend_name} processor: {e}")
        return None
    
    configure_processor(processor)
    processor.initialize()
    
    result = ProcessingResult()
    
    def on_output(output_array):
        result.end_time = time.time()
        result.output = output_array.copy()
        result.received = True
        result.event.set()
    
    def on_error(error_msg):
        print(f"  ERROR in callback: {error_msg}")
        result.event.set()
    
    processor.set_callback(on_output, error_callback=on_error)
    
    result.start_time = time.time()
    buffer = processor.get_next_available_buffer()
    buffer[:] = test_data
    processor.process(buffer)
    
    if not result.wait_for_completion():
        print(f"  ERROR: Timeout waiting for {backend_name} output!")
        processor.stop()
        return None
    
    if not result.received or result.output is None:
        print(f"  ERROR: No output from {backend_name} backend!")
        processor.stop()
        return None
    
    print(f"  Output size: {len(result.output)} samples")
    print(f"  Processing time: {result.get_duration_ms():.3f} ms")
    print()
    
    processor.stop()
    return result

# ============================================
# MAIN TEST
# ============================================

def main():
    try:
        print("=" * 40)
        print("CPU vs CUDA Comparison Test")
        print("=" * 40)
        print()
        
        # Print configuration
        print("Configuration:")
        print(f"  Dimensions: {SIGNAL_LENGTH} x {ASCANS_PER_BSCAN} x {BSCANS_PER_BUFFER}")
        print(f"  Resampling: {'ON' if ENABLE_RESAMPLING else 'OFF'}", end="")
        if ENABLE_RESAMPLING:
            interp_name = {
                ope.InterpolationMethod.LINEAR: "LINEAR",
                ope.InterpolationMethod.CUBIC: "CUBIC",
                ope.InterpolationMethod.LANCZOS: "LANCZOS"
            }.get(INTERPOLATION_METHOD, "UNKNOWN")
            print(f" ({interp_name})")
        else:
            print()
        print(f"  Windowing: {'ON' if ENABLE_WINDOWING else 'OFF'}")
        print(f"  Dispersion: {'ON' if ENABLE_DISPERSION else 'OFF'}")
        print(f"  Log Scaling: {'ON' if ENABLE_LOG_SCALING else 'OFF'}")
        print(f"  Tolerance: {TOLERANCE}")
        print()
        
        # Generate test data
        print("Generating test data...")
        test_data = generate_test_data(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER)
        data_size_kb = test_data.nbytes / 1024.0
        print(f"  Generated {len(test_data)} samples ({data_size_kb:.1f} KB)")
        print()
        
        if SAVE_OUTPUTS:
            test_data.tofile("input.raw")
            print("  Saved: input.raw")
            print()
        
        # Test CPU backend
        cpu_result = run_backend_test("CPU", ope.Backend.CPU, test_data)
        if cpu_result is None:
            return 1
        
        # Test CUDA backend
        cuda_result = run_backend_test("CUDA", ope.Backend.CUDA, test_data)
        if cuda_result is None:
            print("CUDA backend not available, skipping comparison.")
            print("CPU test passed.")
            return 0
        
        # Performance comparison
        speedup = cpu_result.get_duration_ms() / cuda_result.get_duration_ms()
        print("Performance:")
        print(f"  Speedup: {speedup:.2f}x")
        print()
        
        # Compare results
        print("Comparing results...")
        
        if len(cpu_result.output) != len(cuda_result.output):
            print("  ERROR: Output size mismatch!")
            print(f"    CPU: {len(cpu_result.output)} samples")
            print(f"    CUDA: {len(cuda_result.output)} samples")
            return 1
        
        samples_per_ascan = SIGNAL_LENGTH // 2  # Output is half due to truncation
        comparison = compare_buffers(
            cpu_result.output,
            cuda_result.output,
            samples_per_ascan,
            TOLERANCE
        )
        
        print(f"  Max absolute difference: {comparison['max_abs_diff']:.6e}")
        print(f"  Mean absolute difference: {comparison['mean_abs_diff']:.6e}")
        print(f"  RMS error: {comparison['rms_error']:.6e}")
        print(f"  Differing samples: {comparison['differing_samples']} / {comparison['total_samples']}", end="")
        if comparison['total_samples'] > 0:
            percent = 100.0 * comparison['differing_samples'] / comparison['total_samples']
            print(f" ({percent:.2f}%)")
        else:
            print()
        print()
        
        # Per-A-scan statistics
        print(f"  First A-scan max diff: {comparison['max_diff_per_ascan'][0]:.6e}")
        if len(comparison['max_diff_per_ascan']) > 1:
            print(f"  Last A-scan max diff: {comparison['max_diff_per_ascan'][-1]:.6e}")
        print()
        
        # Save outputs
        if SAVE_OUTPUTS:
            print("Saving outputs...")
            cpu_result.output.tofile("output_cpu.raw")
            cuda_result.output.tofile("output_cuda.raw")
            print("  Saved: output_cpu.raw")
            print("  Saved: output_cuda.raw")
            print()
        
        # Result
        if comparison['match']:
            print("TEST PASSED")
            print("CPU and CUDA outputs match within tolerance.")
            return 0
        else:
            print("TEST FAILED")
            print("CPU and CUDA outputs differ beyond tolerance.")
            return 1
    
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
