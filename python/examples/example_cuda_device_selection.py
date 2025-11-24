"""
Minimal example demonstrating CUDA device enumeration and selection
using the unified backend configuration API
"""

import sys
import octproengine as ope


def main():
	print("=" * 40)
	print("CUDA Device Selection Example")
	print("=" * 40)
	print()

	# Step 1: Check if CUDA is available
	if not ope.BackendUtils.is_cuda_available():
		print("CUDA is not available on this system.")
		return 0

	# Step 2: Query available CUDA devices
	print("Querying CUDA devices...")
	print()

	devices = ope.BackendUtils.get_cuda_devices()

	if not devices:
		print("No CUDA devices found.")
		return 0

	print(f"Found {len(devices)} CUDA device(s):")
	print()

	# Display information about each device
	for device in devices:
		print(f"Device {device.id}: {device.name}")
		print(f"  Total Memory: {device.total_memory / (1024*1024):.0f} MB")
		print(f"  Available Memory: {device.available_memory / (1024*1024):.0f} MB")
		print(f"  Compute Capability: {device.compute_capability_major}.{device.compute_capability_minor}")
		print()

	# Step 3: Select a specific device
	selected_device = 0  # Select first device

	if len(devices) > 1:
		print(f"Multiple devices available. Selecting device {selected_device}.")

	print(f"Creating processor with CUDA device {selected_device}...")

	# Step 4: Create processor with specific CUDA device
	try:
		# Create processor (can start with any backend)
		processor = ope.Processor(ope.Backend.CPU)

		# Create CUDA configuration
		cuda_config = ope.CudaConfig()
		cuda_config.device_id = selected_device
		cuda_config.enable_zero_copy = False  # Optional: configure zero-copy mode, only for Jetson devices

		# Apply configuration (this will switch to CUDA backend)
		processor.set_backend_config(cuda_config)

		# Verify the configuration
		current_config = processor.get_backend_config()
		if current_config:
			print(f"Processor configured with: {current_config.to_string()}")

		# Step 5: Initialize and use the processor
		processor.set_input_parameters(
			2048,  # samples_per_raw_ascan
			512,   # ascans_per_bscan
			1,     # bscans_per_buffer
			ope.DataType.UINT16
		)

		processor.initialize()
		print(f"Processor initialized successfully on CUDA device {selected_device}")

		# Processor is now ready to use with the selected CUDA device
		# ... your processing code here ...

		# Cleanup (if method is available)
		if hasattr(processor, 'cleanup'):
			processor.cleanup()

	except Exception as e:
		print(f"Error: {e}")
		return 1

	print()
	print("=" * 40)
	print("Example completed successfully!")
	print("=" * 40)

	return 0


if __name__ == "__main__":
	sys.exit(main())