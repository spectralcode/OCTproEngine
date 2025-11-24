2025-11-25
recorder crashes for large number of buffers in python
set buffers_to_record to 2048 in test_start_recording_while_processing in test_recorder.py and you will get as output: AssertionError: Recording failed. Error: Processed data validation failed: Buffer ID gaps detected in processed data at indices: 521 524 
However, using the same number of buffers in the cpp test does not result in any issues. 
using the CPU backend in python --> no error
todo: investigate 