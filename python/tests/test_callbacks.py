"""
Callback tests for octproengine

Covers:
- Single input/output callbacks
- Multiple buffers and buffer IDs
- Cleanup / stop without crash
"""

import sys
import time
import numpy as np

try:
	import octproengine as ope
	print(f"[OK] Successfully imported octproengine version {ope.__version__}")
except ImportError as e:
	print(f"[FAIL] Failed to import octproengine: {e}")
	sys.exit(1)


# Common parameters for all tests
WIDTH = 512
HEIGHT = 512
BSCAN = 1
DTYPE = ope.DataType.UINT16


def make_processor():
	"""Create and initialize a cuda processor with common parameters."""
	processor = ope.Processor(ope.Backend.CUDA)
	processor.set_input_parameters(WIDTH, HEIGHT, BSCAN, DTYPE)
	processor.initialize()
	return processor


def test_single_input_output():
	"""Test that both input and output callbacks are called once."""
	print("TEST 1: Single input and output callbacks")

	processor = make_processor()

	calls = {"input": 0, "output": 0}

	def input_callback(data, buffer_id):
		calls["input"] += 1
		print(f"  Input callback: buffer_id={buffer_id}, shape={data.shape}")

	def output_callback(data, buffer_id):
		calls["output"] += 1
		print(f"  Output callback: buffer_id={buffer_id}, shape={data.shape}")

	processor.add_input_callback(input_callback)
	processor.add_output_callback(output_callback)

	buffer = processor.get_next_available_buffer()
	buffer.fill(123)
	processor.process(buffer)

	time.sleep(0.2)

	assert calls["input"] == 1, f"Expected 1 input callback, got {calls['input']}"
	assert calls["output"] == 1, f"Expected 1 output callback, got {calls['output']}"
	print("  [OK] Both input and output callbacks called exactly once")

	processor.clear_input_callbacks()
	processor.clear_output_callbacks()
	processor.stop()
	time.sleep(0.1)

	print("  PASSED\n")
	return True


def test_multiple_output_buffers():
	"""Test that output callback sees all buffer IDs in order."""
	print("TEST 2: Multiple buffers and buffer IDs")

	processor = make_processor()

	buffer_ids = []
	nbuffers = 10

	def output_callback(data, buffer_id):
		buffer_ids.append(buffer_id)
		print(f"  Output callback: buffer_id={buffer_id}")

	processor.add_output_callback(output_callback)

	for i in range(nbuffers):
		buffer = processor.get_next_available_buffer()
		buffer.fill(i)
		processor.process(buffer)
		print(f"  Processed buffer {i + 1}/{nbuffers}")

	time.sleep(0.5)

	assert len(buffer_ids) == nbuffers, f"Expected {nbuffers} callbacks, got {len(buffer_ids)}"
	expected_ids = list(range(nbuffers))
	assert buffer_ids == expected_ids, f"Expected IDs {expected_ids}, got {buffer_ids}"
	print(f"  [OK] Received all {nbuffers} buffer IDs in order")

	processor.clear_output_callbacks()
	processor.stop()
	time.sleep(0.1)

	print("  PASSED\n")
	return True


def test_cleanup_no_crash():
	"""Test that clearing callbacks and stopping does not crash."""
	print("TEST 3: Cleanup behavior")

	processor = make_processor()

	def callback(data, buffer_id):
		print(f"  Callback: buffer_id={buffer_id}")

	processor.add_output_callback(callback)

	buffer = processor.get_next_available_buffer()
	buffer.fill(999)
	processor.process(buffer)

	time.sleep(0.2)

	print("  Clearing callbacks...")
	processor.clear_output_callbacks()
	time.sleep(0.1)

	print("  Stopping processor...")
	processor.stop()
	time.sleep(0.1)

	print("  [OK] Cleanup completed without crash")
	print("  PASSED\n")
	return True


def main():
	print("=" * 60)
	print("Callback Python Tests")
	print("=" * 60)
	print()

	tests = [
		test_single_input_output,
		test_multiple_output_buffers,
		test_cleanup_no_crash,
	]

	passed = 0
	total = len(tests)

	for test in tests:
		try:
			if test():
				passed += 1
		except Exception as e:
			print(f"  [FAIL] {test.__name__}: {e}")
			import traceback
			traceback.print_exc()
			print()

	print("=" * 60)
	print(f"RESULTS: {passed}/{total} tests passed")
	if passed == total:
		print("[OK] ALL TESTS PASSED!")
	else:
		print("[FAIL] SOME TESTS FAILED!")
	print("=" * 60)

	return 0 if passed == total else 1


if __name__ == "__main__":
	sys.exit(main())
