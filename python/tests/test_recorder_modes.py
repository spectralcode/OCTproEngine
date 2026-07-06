"""
Recorder mode and lifecycle tests:
 - RAW_ONLY / PROCESSED_ONLY / BOTH recordings (counts, ID sequences, files)
 - repeated recordings on one attached recorder (start ID fence hygiene)
 - abort followed by a clean full recording

Deliberately small frames and counts: total disk output is ~160 MB so the
test stays fast on slow drives.
"""
import sys
import os

# Add the build directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(script_dir, '..', '..', 'build', 'python', 'Release')
sys.path.insert(0, build_dir)

import octproengine as ope
import numpy as np

SIGNAL = 512
ASCANS = 256
RAW_FRAME_BYTES = SIGNAL * ASCANS * 2  # uint16
DELETE_TEST_FILES = True


def delete_test_files(*names):
	if not DELETE_TEST_FILES:
		return
	for name in names:
		if os.path.exists(name):
			os.remove(name)


def make_processor():
	proc = ope.Processor(ope.Backend.CUDA)
	proc.set_input_parameters(SIGNAL, ASCANS, 1, ope.DataType.UINT16)
	proc.initialize()
	return proc


def pump(proc, n):
	for i in range(n):
		buf = proc.get_next_available_buffer()
		buf[:] = i % 251
		proc.process(buf)


def file_size(path):
	return os.path.getsize(path) if os.path.exists(path) else -1


def assert_sequential_pairs(summary, expected_count, both_mode):
	raw_ids = summary.raw_buffer_ids
	processed_ids = summary.processed_buffer_ids
	if both_mode:
		assert len(raw_ids) == len(processed_ids), "raw/processed ID counts differ"
		for i in range(len(raw_ids)):
			assert raw_ids[i] == processed_ids[i], f"ID mismatch at {i}"
	ids = raw_ids if len(raw_ids) > 0 else processed_ids
	assert len(ids) == expected_count, f"expected {expected_count} IDs, got {len(ids)}"
	for i in range(1, len(ids)):
		assert ids[i] == ids[i - 1] + 1, f"ID gap at {i}: {ids[i-1]} -> {ids[i]}"


def test_single_modes():
	"""RAW_ONLY, PROCESSED_ONLY and BOTH each record the exact frame range"""
	print("Testing RAW_ONLY / PROCESSED_ONLY / BOTH recordings...")

	frames = 64
	# processed bytes per frame is derived from the recordings themselves and
	# must be consistent between PROCESSED_ONLY and BOTH (guards output geometry)
	processed_frame_bytes = []

	for mode, name, expect_raw, expect_processed in (
		(ope.RecorderMode.RAW_ONLY, "modes_raw_only", True, False),
		(ope.RecorderMode.PROCESSED_ONLY, "modes_processed_only", False, True),
		(ope.RecorderMode.BOTH, "modes_both", True, True),
	):
		proc = make_processor()
		try:
			rec = ope.Recorder()
			rec.attach_to_processor(proc)
			rec.set_mode(mode)
			rec.set_buffer_count(frames)
			rec.set_output_base_name(name)
			rec.set_use_timestamp(False)
			rec.start_recording()
			pump(proc, frames + 4)
			assert rec.wait_for_completion(15000), f"{name}: {rec.get_last_error()}"

			summary = rec.get_last_recording_summary()
			if expect_raw:
				assert summary.raw_recorded == frames, f"{name}: raw {summary.raw_recorded}"
			else:
				assert summary.raw_recorded == 0, f"{name}: raw should be 0"
			if expect_processed:
				assert summary.processed_recorded == frames, f"{name}: processed {summary.processed_recorded}"
			else:
				assert summary.processed_recorded == 0, f"{name}: processed should be 0"
			assert_sequential_pairs(summary, frames, both_mode=(expect_raw and expect_processed))

			raw_size = file_size(f"{name}_raw.raw")
			processed_size = file_size(f"{name}.raw")
			if expect_raw:
				assert raw_size == frames * RAW_FRAME_BYTES, f"{name}: raw file size {raw_size}"
			else:
				assert raw_size == -1, f"{name}: unexpected raw file"
			if expect_processed:
				assert processed_size > 0 and processed_size % frames == 0, \
					f"{name}: processed file size {processed_size}"
				processed_frame_bytes.append(processed_size // frames)
			else:
				assert processed_size == -1, f"{name}: unexpected processed file"
		finally:
			proc.stop()
			delete_test_files(f"{name}_raw.raw", f"{name}.raw")
		print(f"  [OK] {name}")

	assert processed_frame_bytes[0] == processed_frame_bytes[1], \
		"processed bytes per frame differ between PROCESSED_ONLY and BOTH"
	print("  [OK] processed frame size consistent across modes")


def test_restart_cycles():
	"""10 recordings on one attached recorder: each starts strictly after the previous"""
	print("\nTesting 10 sequential recordings on one attached recorder...")

	frames = 16
	proc = make_processor()
	try:
		rec = ope.Recorder()
		rec.attach_to_processor(proc)
		rec.set_mode(ope.RecorderMode.BOTH)
		rec.set_buffer_count(frames)
		rec.set_use_timestamp(False)

		first_ids = []
		for cycle in range(10):
			name = f"modes_cycle{cycle}"
			rec.set_output_base_name(name)
			try:
				rec.start_recording()
				pump(proc, frames + 2)
				assert rec.wait_for_completion(15000), f"cycle {cycle}: {rec.get_last_error()}"
				summary = rec.get_last_recording_summary()
				assert summary.raw_recorded == frames and summary.processed_recorded == frames, \
					f"cycle {cycle}: raw {summary.raw_recorded} processed {summary.processed_recorded}"
				assert_sequential_pairs(summary, frames, both_mode=True)
				first_ids.append(summary.raw_buffer_ids[0])
			finally:
				delete_test_files(f"{name}_raw.raw", f"{name}.raw")

		for i in range(1, len(first_ids)):
			assert first_ids[i] > first_ids[i - 1], f"start IDs not increasing: {first_ids}"
	finally:
		proc.stop()
	print("  [OK] 10 cycles recorded, start IDs strictly increasing")


def test_abort_then_record():
	"""abortRecording() discards cleanly and the next recording is complete"""
	print("\nTesting abort followed by a clean recording...")

	frames = 32
	proc = make_processor()
	try:
		rec = ope.Recorder()
		rec.attach_to_processor(proc)
		rec.set_mode(ope.RecorderMode.BOTH)
		rec.set_buffer_count(frames)
		rec.set_use_timestamp(False)
		rec.set_output_base_name("modes_abort")

		try:
			rec.start_recording()
			pump(proc, 8)  # far fewer than requested
			rec.abort_recording()
			assert not rec.is_recording(), "still recording after abort"

			rec.set_output_base_name("modes_after_abort")
			rec.start_recording()
			pump(proc, frames + 2)
			assert rec.wait_for_completion(15000), f"after abort: {rec.get_last_error()}"
			summary = rec.get_last_recording_summary()
			assert summary.raw_recorded == frames and summary.processed_recorded == frames, \
				f"after abort: raw {summary.raw_recorded} processed {summary.processed_recorded}"
			assert_sequential_pairs(summary, frames, both_mode=True)
		finally:
			delete_test_files("modes_abort_raw.raw", "modes_abort.raw",
			                  "modes_after_abort_raw.raw", "modes_after_abort.raw")
	finally:
		proc.stop()
	print("  [OK] recording after abort is complete and paired")


if __name__ == "__main__":
	try:
		print("=" * 60)
		print("Testing OCTproEngine Recorder Modes and Lifecycle")
		print("=" * 60)

		test_single_modes()
		test_restart_cycles()
		test_abort_then_record()

		print("\n" + "=" * 60)
		print("All tests passed!")
		print("=" * 60)

	except Exception as e:
		print(f"\n[FAIL] Test failed: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
