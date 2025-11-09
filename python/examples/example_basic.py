"""
Basic OCT processing example using octproengine

This example demonstrates:
1. Creating a processor
2. Configuring parameters
3. Setting up async callback
4. Processing synthetic data
"""

import numpy as np
import octproengine as ope
import time
import threading

# Configuration parameters
SIGNAL_LENGTH = 2048
ASCANS_PER_BSCAN = 512
BSCANS_PER_BUFFER = 1

def generate_synthetic_ascan(signal_length, ascan_index):
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
	
	i = np.arange(signal_length, dtype=np.float64)
	
	value = np.full(signal_length, 2000.0, dtype=np.float64)
	
	value += peak1_amp * np.exp(-((i - peak1_depth) / peak1_width) ** 2)
	value += peak2_amp * np.exp(-((i - peak2_depth) / peak2_width) ** 2)
	value += peak3_amp * np.exp(-((i - peak3_depth) / peak3_width) ** 2)
	
	value += 1000.0 * np.sin(i * 0.3 + lateral_phase)
	value += 500.0 * np.sin(i * 0.15 + lateral_phase * 0.5)
	
	value += np.random.randint(-200, 201, size=signal_length)
	
	ascan = np.clip(value, 0, 65535).astype(np.uint16)
	
	return ascan

def generate_test_data(signal_length, ascans_per_bscan, bscans_per_buffer):
	"""Generate test data"""
	data = np.zeros(signal_length * ascans_per_bscan * bscans_per_buffer, dtype=np.uint16)
	
	idx = 0
	for b in range(bscans_per_buffer):
		for a in range(ascans_per_bscan):
			ascan = generate_synthetic_ascan(signal_length, a)
			data[idx:idx+signal_length] = ascan
			idx += signal_length
	
	return data

def main():
	print("========================================")
	print("OCTproEngine Python Example")
	print("========================================\n")
	
	# Try CUDA first, fall back to CPU if not available
	try:
		print("Attempting to create CUDA processor...")
		processor = ope.Processor(backend=ope.Backend.CUDA)
		print("CUDA processor created successfully\n")
	except ope.BackendError as e:
		print(f"CUDA not available: {e}")
		print("Falling back to CPU processor...\n")
		processor = ope.Processor(backend=ope.Backend.CPU)
	
	# Configure processor
	print("Configuring processor...")
	processor.set_input_parameters(
		signal_length=SIGNAL_LENGTH,
		ascans_per_bscan=ASCANS_PER_BSCAN,
		bscans_per_buffer=BSCANS_PER_BUFFER,
		data_type=ope.DataType.UINT16
	)
	
	# Enable processing features
	processor.enable_resampling(True)
	processor.set_interpolation_method(ope.InterpolationMethod.CUBIC)
	processor.set_resampling_coefficients([0.5, 2048.0, -100.0, 50.0])
	
	processor.enable_windowing(True)
	processor.set_window_parameters(
		window_type=ope.WindowType.HANN,
		center_position=0.9,
		fill_factor=0.85
	)
	
	processor.enable_dispersion_compensation(True)
	processor.set_dispersion_coefficients([0.0, 0.0, 1.0, -3.0], factor=1.0)
	
	processor.enable_log_scaling(True)
	processor.set_grayscale_range(min=0.0, max=80.0)
	
	print("Configuration complete\n")
	
	# Initialize processor
	print("Initializing processor...")
	processor.initialize()
	print("Processor initialized\n")
	
	# Set up callback to receive results
	result_ready = threading.Event()
	output_data = {'array': None}
	
	def on_result(output_array):
		"""Callback function that receives processed output"""
		print(f"Received processed output: shape={output_array.shape}, dtype={output_array.dtype}")
		
		# Save reference to output data
		output_data['array'] = output_array.copy()
		
		# Signal that result is ready
		result_ready.set()
	
	def on_error(error_msg):
		"""Error callback"""
		print(f"Error in callback: {error_msg}")
		result_ready.set()
	
	processor.set_callback(on_result, error_callback=on_error)
	print("Callback configured\n")
	
	# Generate test data
	print(f"Generating test data ({SIGNAL_LENGTH}x{ASCANS_PER_BSCAN})...")
	test_data = generate_test_data(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER)
	print(f"Generated test data: shape={test_data.shape}, dtype={test_data.dtype}\n")
	
	# Get input buffer and fill with data
	print("Getting input buffer...")
	buffer = processor.get_next_available_buffer()
	print(f"Got buffer: shape={buffer.shape}, dtype={buffer.dtype}\n")
	
	# Fill buffer with test data (flatten to 1D)
	print("Filling buffer with test data...")
	buffer[:] = test_data.ravel()
	print("Buffer filled\n")
	
	# Process asynchronously
	print("Processing data (async)...")
	start_time = time.time()
	processor.process(buffer)
	
	# Wait for callback (with timeout)
	print("Waiting for result...")
	if result_ready.wait(timeout=5.0):
		elapsed = time.time() - start_time
		print(f"Processing complete in {elapsed*1000:.2f} ms\n")
		
		# Display the first portion of the result
		if output_data['array'] is not None:
			output = output_data['array']
			print("========================================")
			print("FIRST VALUES OF OUTPUT")
			print("========================================")
			first_values = output[:10]
			print(first_values)
			
			# Optionally save output
			output_file = "output_bscan.raw"
			output.tofile(output_file)
			print(f"  Saved output to {output_file}")
			print(f"  You can visualize this with ImageJ or similar tools")
			print(f"  Width: {SIGNAL_LENGTH/2}, Height: {ASCANS_PER_BSCAN}, Bit depth: 32-bit float")
	else:
		print("Timeout waiting for result")
	
	# Clean up
	print("\nCleaning up...")
	processor.stop()
	print("Processor stopped")
	
	print("\n========================================")
	print("Example complete!")
	print("========================================")

if __name__ == "__main__":
	main()
