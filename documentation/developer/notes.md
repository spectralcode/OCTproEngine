2025-11-24
recorder crashes for large number of buffers in python
set buffers_to_record to 2048 in test_start_recording_while_processing in test_recorder.py and you will get as output: AssertionError: Recording failed. Error: Processed data validation failed: Buffer ID gaps detected in processed data at indices: 521 524 
However, using the same number of buffers in the cpp test does not result in any issues. 
using the CPU backend in python --> no error
todo: investigate 

2025-11-30
vulkans dc-removal shader seems to be much slower than cudas equivalent kernel. with dc-removal disalbed, vulkan and cuda performance is comparable. with dc-removal enabled, vulkan is much slower.

2025-12-01
Profiling on Jetson Orin Nano:
```bash
nsys-ui
```
select target: ubuntu  
Target application, command line with arguments: test_performance_benchmark   
Working directory: /home/orinoct/projects/OCTproEngine/build/tests  
Evironment variables, add: LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

2026-07-06
The 2025-11-24 python recorder bug (buffer ID gaps at 2048 buffers) is resolved.
Re-tested after the consumer-queue rework with buffers_to_record = 2048: passes
repeatedly (3x python, 3x C++ equivalent, with strict per-index ID sequence checks;
peak RAM 4.13 GB as expected for the preallocated arrays). The gaps were caused by
the raw/processed pairing race fixed on the feature-consumer-queues branch: a
processed buffer could overtake the raw callback (asynchronous since the callback
rework, previously ordered by chance) and was wrongly skipped by the
firstRawBufferId synchronization; collectProcessedBuffer() now waits for the raw
side instead. This also explains why the CPU backend never showed the bug - its
processing path never let processed buffers overtake raw ones.

