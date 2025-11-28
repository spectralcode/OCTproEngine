import sys
import os

# Add the build directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'build', 'python', 'Release')
sys.path.insert(0, build_dir)

import octproengine as ope
import numpy as np
import time

def test_recorder_enums():
	"""Test that all recorder enums are accessible"""
	print("Testing Recorder enums...")

	# Test RecorderMode
	assert hasattr(ope, 'RecorderMode')
	assert hasattr(ope.RecorderMode, 'RAW_ONLY')
	assert hasattr(ope.RecorderMode, 'PROCESSED_ONLY')
	assert hasattr(ope.RecorderMode, 'BOTH')
	print("  [OK] RecorderMode enum available")

	# Test RecorderFormat
	assert hasattr(ope, 'RecorderFormat')
	assert hasattr(ope.RecorderFormat, 'RAW_BINARY')
	print("  [OK] RecorderFormat enum available")

	# Test RecorderStatus
	assert hasattr(ope, 'RecorderStatus')
	assert hasattr(ope.RecorderStatus, 'IDLE')
	assert hasattr(ope.RecorderStatus, 'RECORDING')
	assert hasattr(ope.RecorderStatus, 'WRITING')
	assert hasattr(ope.RecorderStatus, 'COMPLETE')
	assert hasattr(ope.RecorderStatus, 'ERROR_STATUS')
	print("  [OK] RecorderStatus enum available")

def test_recorder_basic():
	"""Test basic recorder functionality"""
	print("\nTesting Recorder basic functionality...")

	# Create processor
	proc = ope.Processor(ope.Backend.CUDA)

	# Set cuda memory config:
	cuda_config = ope.CudaConfig()
	cuda_config.enable_zero_copy = False
	proc.set_backend_config(cuda_config)

	proc.set_input_parameters(1024, 512, 1, ope.DataType.UINT16)

	proc.initialize()
	print("  [OK] Processor created and initialized")

	# Create recorder
	recorder = ope.Recorder()
	print("  [OK] Recorder created")
	print(f"  Recorder repr: {recorder}")

	# Attach to processor
	recorder.attach_to_processor(proc)
	print("  [OK] Attached to processor")

	# Configure recorder
	buffers_to_record = 128
	recorder.set_mode(ope.RecorderMode.BOTH)
	recorder.set_buffer_count(buffers_to_record)
	recorder.set_output_base_name("test_python_recording")
	recorder.set_use_timestamp(False)
	recorder.set_manual_allocation(True)
	print("  [OK] Recorder configured")

	# Check configuration
	assert recorder.get_buffer_count() == buffers_to_record
	print(f"  [OK] Buffer count verified: {buffers_to_record}")

	# Allocate buffers
	recorder.allocate_buffers()
	assert recorder.is_allocated()
	print("  [OK] Buffers allocated")

	# Start recording
	recorder.start_recording()
	assert recorder.is_recording()
	print("  [OK] Recording started")

	# Process some data
	for i in range(buffers_to_record+7):
		buf = proc.get_next_available_buffer()
		buf[:] = i  # Fill with test data
		time.sleep(0.002)  # wait to simulate acquisition time
		proc.process(buf)

	print(f"  [OK] Processed {buffers_to_record+7} buffers")

	# Wait for recording to complete
	success = recorder.wait_for_completion(100000)
	assert success, f"Recording failed: {recorder.get_last_error()}"
	print("  [OK] Recording completed")

	# Get summary
	summary = recorder.get_last_recording_summary()
	print(f"  Summary: {summary}")
	print(f"    Expected: {summary.expected_buffers}")
	print(f"    Raw recorded: {summary.raw_recorded}")
	print(f"    Processed recorded: {summary.processed_recorded}")
	print(f"    Complete: {summary.complete}")

	assert summary.expected_buffers == buffers_to_record
	assert summary.raw_recorded == buffers_to_record
	assert summary.processed_recorded == buffers_to_record
	assert summary.complete == True
	print("  [OK] Recording summary verified")

	# Cleanup
	proc.stop()
	print("  [OK] Cleanup complete")

def test_start_recording_while_processing():
	"""Test starting recording while processing is already running (replicates C++ testStartRecordingWhileProcessing)"""
	print("\nTesting start recording while processing is already running...")

	# Create and initialize processor
	proc = ope.Processor(ope.Backend.CUDA)
	proc.set_input_parameters(1024, 512, 1, ope.DataType.UINT16)
	proc.initialize()
	print("  [OK] Processor initialized")

	# Create and configure recorder
	recorder = ope.Recorder()
	recorder.attach_to_processor(proc)
	recorder.set_mode(ope.RecorderMode.BOTH)
	recorder.set_manual_allocation(True)

	print("  Processing buffers...")
	buffers_before_recording = 63
	buffers_to_record = 512

	recorder.set_buffer_count(buffers_to_record)
	recorder.allocate_buffers()
	recorder.set_output_base_name("test_python_midstream")
	recorder.set_use_timestamp(False)
	total_buffers = buffers_before_recording + buffers_to_record

	# Process buffers, starting recording mid-stream
	for i in range(total_buffers):
		if i == buffers_before_recording:
			print(f"  Starting recording at buffer {i}...")
			recorder.start_recording()

		buf = proc.get_next_available_buffer()
		buf[:] = (i * 100) % 255  # Fill with test data pattern
		time.sleep(0.002)  # wait to simulate acquisition time
		proc.process(buf)


	print("  [OK] Processed all buffers. Waiting for recording to complete...")

	# Wait for recording to complete
	success = recorder.wait_for_completion(50000)

	# Get and verify summary
	summary = recorder.get_last_recording_summary()
	raw_ids = summary.raw_buffer_ids
	processed_ids = summary.processed_buffer_ids
	if not success:
		print(f"  Recording failed. Recorded buffer IDs:")
		print(f"  Raw buffer IDs: {raw_ids}")
		print(f"  Processed buffer IDs: {processed_ids}")

	assert success, f"Recording failed. Error: {recorder.get_last_error()}"
	print("  [OK] Recording completed successfully")

	assert summary.raw_recorded == buffers_to_record, \
		f"Raw buffers should be {buffers_to_record}, got {summary.raw_recorded}"
	assert summary.processed_recorded == buffers_to_record, \
		f"Processed buffers should be {buffers_to_record}, got {summary.processed_recorded}"
	print(f"  [OK] Buffer counts verified: Processed buffers should be {buffers_to_record}, got {summary.processed_recorded}")


	# Verify IDs match
	assert len(raw_ids) == len(processed_ids), \
		"Raw and processed ID counts should match"
	for i in range(len(raw_ids)):
		assert raw_ids[i] == processed_ids[i], \
			f"Buffer IDs should match at index {i}"
	print("  [OK] Buffer IDs match between raw and processed")

	# Verify expected range
	assert raw_ids[0] == buffers_before_recording, \
		f"First buffer ID should be {buffers_before_recording}, got {raw_ids[0]}"
	assert raw_ids[-1] == buffers_before_recording + buffers_to_record - 1, \
		f"Last buffer ID should be {buffers_before_recording + buffers_to_record - 1}, got {raw_ids[-1]}"
	print("  [OK] Buffer ID range verified")

	# Cleanup
	proc.stop()
	print("  [OK] Recording started successfully during ongoing processing")

if __name__ == "__main__":
	try:
		print("="*60)
		print("Testing OCTproEngine Recorder Python Bindings")
		print("="*60)

		test_recorder_enums()
		test_recorder_basic()
		test_start_recording_while_processing()

		print("\n" + "="*60)
		print("All tests passed!")
		print("="*60)

	except Exception as e:
		print(f"\n[FAIL] Test failed: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
