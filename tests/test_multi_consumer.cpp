// Simple test for multi-consumer callback functionality
// Tests that multiple callbacks can be registered and all receive data

#include "../include/processor.h"
#include "../include/processorconfiguration.h"
#include "../include/types.h"
#include "../include/iobuffer.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>
#include "test_utils.h"

// Test configuration
const int SIGNAL_LENGTH = 1024;
const int ASCANS_PER_BSCAN = 256;
const int BSCANS_PER_BUFFER = 1;

// Test configuration for performance test
const int NUM_ITERATIONS = 1000;
const int PERF_SIGNAL_LENGTH = 1024;
const int PERF_ASCANS_PER_BSCAN = 512;
const int PERF_BSCANS_PER_BUFFER = 1;

const bool RESAMPLING_ENABLED = true;
const bool WINDOWING_ENABLED = true;
const bool DISPERSION_ENABLED = true;
const bool DC_REMOVAL_ENABLED = false;
const bool INTENSITY_LOG_SCALE = true;

// Test configuration for slow consumer test
const int NUM_ITERATIONS_SLOW_CONSUMER = 1000;
const int SIGNAL_LENGTH_SLOW_CONSUMER = 1024;
const int ASCANS_PER_BSCAN_SLOW_CONSUMER = 512;
const int BSCANS_PER_BUFFER_SLOW_CONSUMER = 32;

// Backend for all tests
const ope::Backend TEST_BACKEND = ope::Backend::CUDA;


// Generate simple test data
std::vector<uint16_t> generateTestData() {
	size_t totalSamples = SIGNAL_LENGTH * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;
	std::vector<uint16_t> data(totalSamples);
	
	// Simple pattern: ramp from 0 to 65535
	for (size_t i = 0; i < totalSamples; ++i) {
		data[i] = static_cast<uint16_t>((i * 65535) / totalSamples);
	}
	
	return data;
}

// Configure processor with basic settings
void configureProcessor(ope::Processor& processor) {
	auto config = processor.getConfig();
	
	// Data parameters
	config.dataParams.signalLength = SIGNAL_LENGTH;
	config.dataParams.ascansPerBscan = ASCANS_PER_BSCAN;
	config.dataParams.bscansPerBuffer = BSCANS_PER_BUFFER;
	config.dataParams.inputDataType = ope::DataType::UINT16;
	
	// Disable all processing for simple test
	config.processingParams.resampling.enabled = false;
	config.processingParams.windowing.enabled = false;
	config.processingParams.dispersion.enabled = false;
	config.processingParams.dcRemoval.enabled = false;
	config.processingParams.intensity.logScale = false;
	
	processor.setConfig(config);
}

// ============================================
// TEST 1: Basic Multi-Consumer
// ============================================
bool test_basic_multi_consumer() {
	std::cout << "TEST 1: Basic Multi-Consumer" << std::endl;
	std::cout << "  Testing that 3 callbacks all receive data..." << std::endl;
	
	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();
	
	// Counters for each callback
	std::atomic<int> count1{0}, count2{0}, count3{0};
	
	// Add 3 callbacks
	auto id1 = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		count1++;
	});
	
	auto id2 = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		count2++;
	});
	
	auto id3 = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		count3++;
	});
	
	// Verify callback count
	TEST_ASSERT(processor.getOutputCallbackCount() == 3, "Expected 3 callbacks registered");
	std::cout << "  [OK] Registered 3 callbacks" << std::endl;
	
	// Process one frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	// Wait for callbacks to complete
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	// Verify all callbacks were called
	if (count1 != 1 || count2 != 1 || count3 != 1) {
		std::cerr << "  [FAIL] Callbacks not all called!" << std::endl;
		std::cerr << "    count1=" << count1 << ", count2=" << count2 << ", count3=" << count3 << std::endl;
		return false;
	}
	
	std::cout << "  [OK] All 3 callbacks received data" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 2: Remove Callback
// ============================================
bool test_remove_callback() {
	std::cout << "TEST 2: Remove Callback" << std::endl;
	std::cout << "  Testing callback removal..." << std::endl;
	
	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();
	
	std::atomic<int> count1{0}, count2{0}, count3{0};
	
	// Add 3 callbacks
	auto id1 = processor.addOutputCallback([&](const ope::IOBuffer& buf) { count1++; });
	auto id2 = processor.addOutputCallback([&](const ope::IOBuffer& buf) { count2++; });
	auto id3 = processor.addOutputCallback([&](const ope::IOBuffer& buf) { count3++; });
	
	TEST_ASSERT(processor.getOutputCallbackCount() == 3, "Expected 3 callbacks registered");
	std::cout << "  [OK] Added 3 callbacks" << std::endl;
	
	// Remove middle callback
	bool removed = processor.removeOutputCallback(id2);
	if (!removed) {
		std::cerr << "  [FAIL] removeOutputCallback returned false!" << std::endl;
		return false;
	}
	
	TEST_ASSERT(processor.getOutputCallbackCount() == 2, "Expected 2 callbacks registered");
	std::cout << "  [OK] Removed callback 2" << std::endl;
	
	// Process frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	// Verify only callbacks 1 and 3 were called
	if (count1 != 1 || count2 != 0 || count3 != 1) {
		std::cerr << "  [FAIL] Incorrect callback counts!" << std::endl;
		std::cerr << "    count1=" << count1 << " (expected 1)" << std::endl;
		std::cerr << "    count2=" << count2 << " (expected 0)" << std::endl;
		std::cerr << "    count3=" << count3 << " (expected 1)" << std::endl;
		return false;
	}
	
	std::cout << "  [OK] Callbacks 1 and 3 called, callback 2 not called" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 3: Clear All Callbacks
// ============================================
bool test_clear_callbacks() {
	std::cout << "TEST 3: Clear All Callbacks" << std::endl;
	std::cout << "  Testing clearOutputCallbacks()..." << std::endl;
	
	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();
	
	std::atomic<int> count1{0}, count2{0};
	
	// Add callbacks
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count1++; });
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count2++; });
	
	TEST_ASSERT(processor.getOutputCallbackCount() == 2, "Expected 2 callbacks registered");
	std::cout << "  [OK] Added 2 callbacks" << std::endl;
	
	// Clear all
	processor.clearOutputCallbacks();
	
	TEST_ASSERT(processor.getOutputCallbackCount() == 0, "Expected 0 callbacks after clear");
	std::cout << "  [OK] Cleared all callbacks" << std::endl;
	
	// Process frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	// Verify no callbacks were called
	if (count1 != 0 || count2 != 0) {
		std::cerr << "  [FAIL] Callbacks were called after clear!" << std::endl;
		std::cerr << "    count1=" << count1 << ", count2=" << count2 << std::endl;
		return false;
	}
	
	std::cout << "  [OK] No callbacks called after clear" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 4: Data Integrity
// ============================================
bool test_data_integrity() {
	std::cout << "TEST 4: Data Integrity" << std::endl;
	std::cout << "  Testing that all callbacks receive same data..." << std::endl;
	
	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();
	
	// Storage for data received by each callback
	std::vector<float> data1, data2, data3;
	std::atomic<bool> done1{false}, done2{false}, done3{false};
	
	// Callbacks that copy data
	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		size_t numFloats = buf.getSizeInBytes() / sizeof(float);
		data1.resize(numFloats);
		std::memcpy(data1.data(), buf.getDataPointer(), buf.getSizeInBytes());
		done1 = true;
	});
	
	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		size_t numFloats = buf.getSizeInBytes() / sizeof(float);
		data2.resize(numFloats);
		std::memcpy(data2.data(), buf.getDataPointer(), buf.getSizeInBytes());
		done2 = true;
	});
	
	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		size_t numFloats = buf.getSizeInBytes() / sizeof(float);
		data3.resize(numFloats);
		std::memcpy(data3.data(), buf.getDataPointer(), buf.getSizeInBytes());
		done3 = true;
	});
	
	// Process frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	// Wait for all callbacks
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	if (!done1 || !done2 || !done3) {
		std::cerr << "  [FAIL] Not all callbacks completed!" << std::endl;
		return false;
	}
	
	// Verify all received same size data
	if (data1.size() != data2.size() || data2.size() != data3.size()) {
		std::cerr << "  [FAIL] Data sizes don't match!" << std::endl;
		std::cerr << "    data1.size=" << data1.size() << std::endl;
		std::cerr << "    data2.size=" << data2.size() << std::endl;
		std::cerr << "    data3.size=" << data3.size() << std::endl;
		return false;
	}
	
	std::cout << "  [OK] All callbacks received " << data1.size() << " floats" << std::endl;
	
	// Verify data is identical
	for (size_t i = 0; i < data1.size(); ++i) {
		if (data1[i] != data2[i] || data2[i] != data3[i]) {
			std::cerr << "  [FAIL] Data mismatch at index " << i << "!" << std::endl;
			std::cerr << "    data1[" << i << "]=" << data1[i] << std::endl;
			std::cerr << "    data2[" << i << "]=" << data2[i] << std::endl;
			std::cerr << "    data3[" << i << "]=" << data3[i] << std::endl;
			return false;
		}
	}
	
	std::cout << "  [OK] All callbacks received identical data" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 5: Multiple Frames
// ============================================
bool test_multiple_frames() {
	std::cout << "TEST 5: Multiple Frames" << std::endl;
	std::cout << "  Testing multiple frames with multiple callbacks..." << std::endl;
	
	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();
	
	const int NUM_FRAMES = 100;
	std::atomic<int> count1{0}, count2{0};
	
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count1++; });
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count2++; });
	
	auto testData = generateTestData();
	
	// Process multiple frames
	for (int i = 0; i < NUM_FRAMES; ++i) {
		auto& inputBuf = processor.getNextAvailableInputBuffer();
		std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
		processor.process(inputBuf);
	}
	
	// Wait for all callbacks
	std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	
	if (count1 != NUM_FRAMES || count2 != NUM_FRAMES) {
		std::cerr << "  [FAIL] Incorrect callback counts!" << std::endl;
		std::cerr << "    count1=" << count1 << " (expected " << NUM_FRAMES << ")" << std::endl;
		std::cerr << "    count2=" << count2 << " (expected " << NUM_FRAMES << ")" << std::endl;
		return false;
	}
	
	std::cout << "  [OK] Both callbacks called " << NUM_FRAMES << " times" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 6: Multi-Consumer Performance
// ============================================
bool test_multi_consumer_performance() {
	std::cout << "TEST 6: Multi-Consumer Performance" << std::endl;
	std::cout << "  Measuring performance impact of multiple consumers..." << std::endl;

	// Generate test data
	size_t inputSize = PERF_SIGNAL_LENGTH * PERF_ASCANS_PER_BSCAN * PERF_BSCANS_PER_BUFFER;
	std::vector<uint16_t> testData(inputSize);
	for (size_t i = 0; i < inputSize; ++i) {
		testData[i] = static_cast<uint16_t>(i % 65536);
	}

	double baselineAvgTime = 0.0;

	// Test with a increasing number of consumers
	for (int numConsumers = 0; numConsumers <= 8; ++numConsumers) {
		ope::Processor processor(TEST_BACKEND);
		auto config = processor.getConfig();
		config.dataParams.signalLength = PERF_SIGNAL_LENGTH;
		config.dataParams.ascansPerBscan = PERF_ASCANS_PER_BSCAN;
		config.dataParams.bscansPerBuffer = PERF_BSCANS_PER_BUFFER;
		config.dataParams.inputDataType = ope::DataType::UINT16;
		config.processingParams.resampling.enabled = true;
		config.processingParams.windowing.enabled = true;
		config.processingParams.dispersion.enabled = true;
		config.processingParams.dcRemoval.enabled = false;
		config.processingParams.intensity.logScale = true;
		processor.setConfig(config);
		processor.initialize();

		auto& dataParams = processor.getConfig().dataParams;
		size_t outputBufferSize = dataParams.outputSignalLength() * dataParams.ascansPerBscan * dataParams.bscansPerBuffer * dataParams.getOutputBytesPerSample();

		std::atomic<int> completedIterations{0};

		// Create temp buffers for each consumer
		std::vector<std::vector<uint8_t>> tempBuffers(numConsumers);
		for (int c = 0; c < numConsumers; ++c) {
			tempBuffers[c].resize(outputBufferSize);
		}

		// Add callbacks
		if (numConsumers == 0) {
			// For 0 consumers: add lightweight counting callback only
			processor.addOutputCallback([&](const ope::IOBuffer& buf) {
				completedIterations++;
			});
		} else {
			for (int c = 0; c < numConsumers; ++c) {
				bool isLast = (c == numConsumers - 1);
				processor.addOutputCallback([&, c, isLast](const ope::IOBuffer& buf) {
					// copy output data to temp buffer to simulate actual use of data
					memcpy(tempBuffers[c].data(), buf.getDataPointer(), outputBufferSize);

					// Simulate some processing that takes time. without memcpy
					/*volatile float sink = 0;
					const float* data = static_cast<const float*>(buf.getDataPointer());
					size_t numFloats = outputBufferSize / sizeof(float);
					for (size_t i = 0; i < numFloats; i += 16) {
						sink = data[i];
					}*/

					if (isLast) {
						completedIterations++;
					}
				});
			}
		}	

		auto startTime = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < NUM_ITERATIONS; ++i) {
			auto& inputBuf = processor.getNextAvailableInputBuffer();
			std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
			processor.process(inputBuf);
		}
		while (completedIterations < NUM_ITERATIONS) {
			std::this_thread::sleep_for(std::chrono::microseconds(100));
		}
		auto endTime = std::chrono::high_resolution_clock::now();

		double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
		double avgTime = duration / NUM_ITERATIONS;  // ms per buffer
		double buffersPerSec = 1000.0 / avgTime;
		double bscansPerSec = buffersPerSec * config.dataParams.bscansPerBuffer;
		double ascansPerSec = bscansPerSec * PERF_ASCANS_PER_BSCAN;
		double mbPerSec = buffersPerSec * outputBufferSize / (1024.0 * 1024.0);

		if (numConsumers == 0) {
			baselineAvgTime = avgTime;
		}

		double ratio = ((baselineAvgTime/avgTime));
		std::cout << "  " << numConsumers << " consumers: " << avgTime << " ms/buffer, " << ascansPerSec << " ascans/s, " << bscansPerSec << " bscans/s, " << mbPerSec << " MB/s (" << ratio << "x)" << std::endl;
		
	}

	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 7: Slow Consumer Throughput
// Compares fast vs slow consumer to measure async callback overhead
// ============================================
bool test_slow_consumer_throughput() {
	std::cout << "TEST 7: Slow Consumer Throughput" << std::endl;
	std::cout << "  Comparing fast vs slow consumer completion times..." << std::endl;

	int NUM_BUFFERS = NUM_ITERATIONS_SLOW_CONSUMER;

	// Generate test data
	size_t inputSize = SIGNAL_LENGTH_SLOW_CONSUMER * ASCANS_PER_BSCAN_SLOW_CONSUMER * BSCANS_PER_BUFFER_SLOW_CONSUMER;
	std::vector<uint16_t> testData(inputSize);
	for (size_t i = 0; i < inputSize; ++i) {
		testData[i] = static_cast<uint16_t>(i % 65536);
	}

	double fastCompletionMs = 0;
	double slowCompletionMs = 0;
	double slowCallbackWorkMs = 0;

	// ---- RUN 1: FAST CONSUMER (no delay) ----
	{
		ope::Processor processor(TEST_BACKEND);
		auto config = processor.getConfig();
		config.dataParams.signalLength = SIGNAL_LENGTH_SLOW_CONSUMER;
		config.dataParams.ascansPerBscan = ASCANS_PER_BSCAN_SLOW_CONSUMER;
		config.dataParams.bscansPerBuffer = BSCANS_PER_BUFFER_SLOW_CONSUMER;
		config.dataParams.inputDataType = ope::DataType::UINT16;
		config.processingParams.resampling.enabled = false;
		config.processingParams.windowing.enabled = WINDOWING_ENABLED;
		config.processingParams.dispersion.enabled = DISPERSION_ENABLED;
		config.processingParams.dcRemoval.enabled = DC_REMOVAL_ENABLED;
		config.processingParams.intensity.logScale = INTENSITY_LOG_SCALE;
		processor.setConfig(config);
		processor.initialize();

		std::atomic<int> callbacksCompleted{0};
		std::chrono::high_resolution_clock::time_point callbackEndTime;

		processor.addOutputCallback([&](const ope::IOBuffer& buf) {
			int count = ++callbacksCompleted;
			if (count == NUM_BUFFERS) {
				callbackEndTime = std::chrono::high_resolution_clock::now();
			}
		});

		auto startTime = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < NUM_BUFFERS; ++i) {
			auto& inputBuf = processor.getNextAvailableInputBuffer();
			std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
			processor.process(inputBuf);
		}

		while (callbacksCompleted < NUM_BUFFERS) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		fastCompletionMs = std::chrono::duration<double, std::milli>(callbackEndTime - startTime).count();
	}

	// Calculate slow consumer delay from fast consumer's average buffer time
	double avgBufferTimeMs = fastCompletionMs / NUM_BUFFERS;
	int slowCallbackDelayUs = static_cast<int>(avgBufferTimeMs * 1000.0);  // Convert ms to us

	std::cout << "  Processing speed for fast consumer: " << fastCompletionMs << " ms (" << avgBufferTimeMs << " ms/buffer)" << std::endl;
	std::cout << "  this gives us max allowable callback delay for our slow consumer: " << slowCallbackDelayUs << " us (= avg buffer time)" << std::endl;

	// ---- RUN 2: SLOW CONSUMER (busy-wait delay = avg buffer time) ----
	{
		ope::Processor processor(TEST_BACKEND);
		auto config = processor.getConfig();
		config.dataParams.signalLength = SIGNAL_LENGTH_SLOW_CONSUMER;
		config.dataParams.ascansPerBscan = ASCANS_PER_BSCAN_SLOW_CONSUMER;
		config.dataParams.bscansPerBuffer = BSCANS_PER_BUFFER_SLOW_CONSUMER;
		config.dataParams.inputDataType = ope::DataType::UINT16;
		config.processingParams.resampling.enabled = RESAMPLING_ENABLED;
		config.processingParams.windowing.enabled = WINDOWING_ENABLED;
		config.processingParams.dispersion.enabled = DISPERSION_ENABLED;
		config.processingParams.dcRemoval.enabled = DC_REMOVAL_ENABLED;
		config.processingParams.intensity.logScale = INTENSITY_LOG_SCALE;
		processor.setConfig(config);
		processor.initialize();

		std::atomic<int> callbacksCompleted{0};
		std::atomic<double> totalBusyWaitTimeUs{0};
		std::chrono::high_resolution_clock::time_point callbackEndTime;

		processor.addOutputCallback([&, slowCallbackDelayUs](const ope::IOBuffer& buf) {
			// Busy-wait spin loop with accurate timing
			auto busyStart = std::chrono::high_resolution_clock::now();
			auto targetEnd = busyStart + std::chrono::microseconds(slowCallbackDelayUs);
			volatile int dummy = 0;
			while (std::chrono::high_resolution_clock::now() < targetEnd) {
				// Spin
				//std::atomic_signal_fence(std::memory_order_seq_cst);
				dummy++;
			}
			auto busyEnd = std::chrono::high_resolution_clock::now();

			// Accumulate actual busy-wait time
			double actualUs = std::chrono::duration<double, std::micro>(busyEnd - busyStart).count();
			double oldVal = totalBusyWaitTimeUs.load();
			while (!totalBusyWaitTimeUs.compare_exchange_weak(oldVal, oldVal + actualUs)) {}

			int count = ++callbacksCompleted;
			if (count == NUM_BUFFERS) {
				callbackEndTime = std::chrono::high_resolution_clock::now();
			}
		});

		auto startTime = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < NUM_BUFFERS; ++i) {
			auto& inputBuf = processor.getNextAvailableInputBuffer();
			std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
			processor.process(inputBuf);
		}

		while (callbacksCompleted < NUM_BUFFERS) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		slowCompletionMs = std::chrono::duration<double, std::milli>(callbackEndTime - startTime).count();
		slowCallbackWorkMs = totalBusyWaitTimeUs.load() / 1000.0;
	}

	// ---- RUN 3: FAST + EXTREMELY SLOW CONSUMER  ----
	// This tests if a slow consumer blocks the processing and decreases performance for all other consumers
	double run3FastCompletionMs = 0;
	double run3SlowCompletionMs = 0;
	{
		ope::Processor processor(TEST_BACKEND);
		auto config = processor.getConfig();
		config.dataParams.signalLength = SIGNAL_LENGTH_SLOW_CONSUMER;
		config.dataParams.ascansPerBscan = ASCANS_PER_BSCAN_SLOW_CONSUMER;
		config.dataParams.bscansPerBuffer = BSCANS_PER_BUFFER_SLOW_CONSUMER;
		config.dataParams.inputDataType = ope::DataType::UINT16;
		config.processingParams.resampling.enabled = RESAMPLING_ENABLED;
		config.processingParams.windowing.enabled = WINDOWING_ENABLED;
		config.processingParams.dispersion.enabled = DISPERSION_ENABLED;
		config.processingParams.dcRemoval.enabled = DC_REMOVAL_ENABLED;
		config.processingParams.intensity.logScale = INTENSITY_LOG_SCALE;
		processor.setConfig(config);
		processor.initialize();

		std::atomic<int> fastConsumerCount{0};
		std::atomic<int> slowConsumerCount{0};
		std::chrono::high_resolution_clock::time_point fastConsumerEndTime;
		std::chrono::high_resolution_clock::time_point slowConsumerEndTime;
		std::chrono::high_resolution_clock::time_point startTime;

		// Fast consumer
		processor.addOutputCallback([&](const ope::IOBuffer& buf) {
			int count = ++fastConsumerCount;
			if (count == NUM_BUFFERS) {
				fastConsumerEndTime = std::chrono::high_resolution_clock::now();
			}
		});

		// Slow consumer 
		processor.addOutputCallback([&, slowCallbackDelayUs](const ope::IOBuffer& buf) {
			// Busy-wait spin loop
			auto busyStart = std::chrono::high_resolution_clock::now();
			auto targetEnd = busyStart + std::chrono::microseconds(slowCallbackDelayUs);
			volatile int dummy = 0;
			while (std::chrono::high_resolution_clock::now() < targetEnd) {
				dummy++;
			}

			int count = ++slowConsumerCount;
			if (count == NUM_BUFFERS) {
				slowConsumerEndTime = std::chrono::high_resolution_clock::now();
			}
		});

		startTime = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < NUM_BUFFERS; ++i) {
			auto& inputBuf = processor.getNextAvailableInputBuffer();
			std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
			processor.process(inputBuf);
		}

		// Wait for both consumers to finish
		while (fastConsumerCount < NUM_BUFFERS || slowConsumerCount < NUM_BUFFERS) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		run3FastCompletionMs = std::chrono::duration<double, std::milli>(fastConsumerEndTime - startTime).count();
		run3SlowCompletionMs = std::chrono::duration<double, std::milli>(slowConsumerEndTime - startTime).count();
	}

	// ---- RESULTS ----
	double callbackDelayMs = 2 * slowCallbackDelayUs / 1000.0; // to get extremly slow consumer time is multiplied by 2
	double overheadMs = slowCompletionMs - fastCompletionMs;
	double expectedSlowTimeMs = fastCompletionMs + (callbackDelayMs * NUM_BUFFERS);

	std::cout << "  Processing speed for slow consumer: " << slowCompletionMs << " ms" << std::endl;
	std::cout << "  Overhead: " << overheadMs << " ms" << std::endl;
	std::cout << "  Expected processing duration if slow consumer would block processing: " << expectedSlowTimeMs << " ms" << std::endl;

	// Since callback delay = avg buffer time, we expect backpressure
	// The slow consumer should take roughly 2x the fast consumer time
	double ratio = slowCompletionMs / fastCompletionMs;
	std::cout << "  Ratio (slow/fast): " << ratio << "x" << std::endl;

	if (ratio >= 1.5) {
		std::cout << "  [OK] Backpressure detected: slow consumer took " << ratio << "x longer" << std::endl;
	} else {
		std::cout << "  [INFO] Minimal backpressure impact (ratio " << ratio << "x)" << std::endl;
	}

	// Test impact of extremly slow consumer on other consumers
	std::cout << std::endl;
	std::cout << "  Fast + Extremly slow consumer simultaneous:" << std::endl;
	std::cout << "    Fast consumer finished: " << run3FastCompletionMs << " ms" << std::endl;
	std::cout << "    Slow consumer finished: " << run3SlowCompletionMs << " ms" << std::endl;
	double consumerTimeDiff = run3SlowCompletionMs - run3FastCompletionMs;
	double expectedDiff = (callbackDelayMs * NUM_BUFFERS) - fastCompletionMs;  // slow consumer time - fast consumer time
	std::cout << "    Time difference: " << consumerTimeDiff << " ms" << std::endl;
	std::cout << "    Expected diff (if slow consumer does not block): ~" << expectedDiff << " ms (total slow consumer work)" << std::endl;

	if (consumerTimeDiff > expectedDiff * 0.5) {
		std::cout << "    [ASYNC] Fast consumer finished significantly earlier! Slow consumers do not block processing." << std::endl;
	} else {
		std::cout << "    [BLOCKING] Both consumers finished at similar time. Slow consumer blocks processing." << std::endl;
	}

	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 8: Queue Depth (In-Flight Buffers)
// Tests that multiple buffers can be "in flight" simultaneously
// With async callbacks: maxInFlight > 1
// With blocking callbacks: maxInFlight == 1 always
// ============================================
bool test_queue_depth() {
	std::cout << "TEST 8: Queue Depth (In-Flight Buffers)" << std::endl;
	std::cout << "  Testing that multiple buffers can be in-flight..." << std::endl;

	const int NUM_BUFFERS = 100;
	const int CALLBACK_DELAY_US = 500;  // Small delay to allow queue buildup

	ope::Processor processor(TEST_BACKEND);
	auto config = processor.getConfig();
	config.dataParams.signalLength = PERF_SIGNAL_LENGTH;
	config.dataParams.ascansPerBscan = PERF_ASCANS_PER_BSCAN;
	config.dataParams.bscansPerBuffer = PERF_BSCANS_PER_BUFFER;
	config.dataParams.inputDataType = ope::DataType::UINT16;
	config.processingParams.resampling.enabled = RESAMPLING_ENABLED;
	config.processingParams.windowing.enabled = WINDOWING_ENABLED;
	config.processingParams.dispersion.enabled = DISPERSION_ENABLED;
	config.processingParams.dcRemoval.enabled = DC_REMOVAL_ENABLED;
	config.processingParams.intensity.logScale = INTENSITY_LOG_SCALE;
	processor.setConfig(config);
	processor.initialize();

	// Generate test data
	size_t inputSize = PERF_SIGNAL_LENGTH * PERF_ASCANS_PER_BSCAN * PERF_BSCANS_PER_BUFFER;
	std::vector<uint16_t> testData(inputSize);
	for (size_t i = 0; i < inputSize; ++i) {
		testData[i] = static_cast<uint16_t>(i % 65536);
	}

	std::atomic<int> inFlight{0};
	std::atomic<int> maxInFlight{0};
	std::atomic<int> callbacksCompleted{0};

	// Callback that tracks in-flight count
	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		// Small delay to simulate work and allow queue to build up
		std::this_thread::sleep_for(std::chrono::microseconds(CALLBACK_DELAY_US));

		int currentInFlight = --inFlight;
		callbacksCompleted++;
	});

	// Submit buffers, tracking in-flight count
	for (int i = 0; i < NUM_BUFFERS; ++i) {
		auto& inputBuf = processor.getNextAvailableInputBuffer();
		std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));

		int currentInFlight = ++inFlight;

		// Update max in-flight
		int expected = maxInFlight.load();
		while (currentInFlight > expected && !maxInFlight.compare_exchange_weak(expected, currentInFlight)) {
			// Loop until we successfully update or someone else set a higher value
		}

		processor.process(inputBuf);
	}

	// Wait for all callbacks to complete
	while (callbacksCompleted < NUM_BUFFERS) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	std::cout << "  Max buffers in-flight: " << maxInFlight.load() << std::endl;

	// With async callbacks, we should see multiple buffers in-flight
	if (maxInFlight.load() > 1) {
		std::cout << "  [OK] Pipelining working - multiple buffers processed concurrently" << std::endl;
	} else {
		std::cout << "  [WARN] Only 1 buffer in-flight - may indicate blocking behavior" << std::endl;
	}

	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// Main
// ============================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Multi-Consumer Callback Tests" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;
	
	int passed = 0;
	int total = 0;
	
	#define RUN_TEST(test_func) \
		total++; \
		if (test_func()) { \
			passed++; \
		}
	
	RUN_TEST(test_basic_multi_consumer);
	RUN_TEST(test_remove_callback);
	RUN_TEST(test_clear_callbacks);
	RUN_TEST(test_data_integrity);
	RUN_TEST(test_multiple_frames);
	RUN_TEST(test_multi_consumer_performance);
	RUN_TEST(test_slow_consumer_throughput);
	RUN_TEST(test_queue_depth);
	
	std::cout << "========================================" << std::endl;
	std::cout << "RESULTS: " << passed << "/" << total << " tests passed" << std::endl;
	
	if (passed == total) {
		std::cout << "[OK] ALL TESTS PASSED!" << std::endl;
	} else {
		std::cout << "[FAIL] SOME TESTS FAILED!" << std::endl;
	}
	std::cout << "========================================" << std::endl;
	
	return (passed == total) ? 0 : 1;
}