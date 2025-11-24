"""
Recorder example using octproengine

This example demonstrates:
1. Creating and configuring a processor
2. Attaching a Recorder to the processor
3. Configuring recording settings
4. Recording raw and processed data to disk
"""

import numpy as np
import octproengine as ope

# Configuration parameters
SIGNAL_LENGTH = 1024
ASCANS_PER_BSCAN = 512
BSCANS_PER_BUFFER = 1

def example_basic_recording():
	print("=== Example 1: Basic Recording ===")

	# Create and configure processor
	try:
		processor = ope.Processor(backend=ope.Backend.CUDA)
		print("Using CUDA backend")
	except ope.BackendError:
		processor = ope.Processor(backend=ope.Backend.CPU)
		print("Using CPU backend")

	processor.set_input_parameters(
		signal_length=SIGNAL_LENGTH,
		ascans_per_bscan=ASCANS_PER_BSCAN,
		bscans_per_buffer=BSCANS_PER_BUFFER,
		data_type=ope.DataType.UINT16
	)
	processor.initialize()

	# Create recorder and attach to processor
	recorder = ope.Recorder()
	recorder.attach_to_processor(processor)

	# Configure recording
	recorder.set_mode(ope.RecorderMode.BOTH)  # BOTH will record raw and processed data
	recorder.set_buffer_count(20)  # set how many buffers to record
	recorder.set_output_base_name("example_basic")  # base name for output files. you can use set_output_directory() to change directory if needed
	recorder.set_use_timestamp(False)

	# Optional: Pre-allocate buffers to avoid allocation overhead during start_recording()
	recorder.set_manual_allocation(True)
	recorder.allocate_buffers()

	# Start recording
	recorder.start_recording()
	print("Recording started...")
	print(f"Buffers to record: {recorder.get_buffer_count()}")

	# Start Processing
	for i in range(30):
		input_buffer = processor.get_next_available_buffer()

		# Fill buffer with data (in real application, this comes from your OCT hardware)
		# Here we just fill with dummy data
		input_buffer[:] = i

		processor.process(input_buffer)
		print(f"  Processed buffer {i + 1}/30")

	# Wait for recording to complete (auto-completes at 20 buffers)
	print("Waiting for recording to complete...")
	success = recorder.wait_for_completion(10000)

	if success:
		print("Recording complete!")
		summary = recorder.get_last_recording_summary()
		print(f"  Raw buffers recorded: {summary.raw_recorded}")
		print(f"  Processed buffers recorded: {summary.processed_recorded}")
	else:
		print("Recording failed or timed out!")
		print(f"  Error: {recorder.get_last_error()}")

		# Print buffer ids collected before failure
		summary = recorder.get_last_recording_summary()
		print(f"  Raw buffers recorded: {summary.raw_recorded}")
		print(f"  Processed buffers recorded: {summary.processed_recorded}")

		# You can also print buffer IDs here if needed
		raw_buffer_ids = summary.raw_buffer_ids
		processed_buffer_ids = summary.processed_buffer_ids
		print(f"  Raw Buffer IDs: {raw_buffer_ids}")
		print(f"  Processed Buffer IDs: {processed_buffer_ids}")

	# Cleanup
	recorder.free_buffers()
	processor.stop()
	print()

def main():
	print("========================================")
	print("OCTproEngine Recorder Python Example")
	print("========================================\n")

	example_basic_recording()

	print("========================================")
	print("Example complete!")
	print("========================================")

if __name__ == "__main__":
	main()
