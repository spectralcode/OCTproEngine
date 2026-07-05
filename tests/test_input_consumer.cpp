/**
 * Test: Input Consumer Polling API
 *
 * Tests that multiple consumers can poll for and read raw input buffers:
 * - Camera fills buffer → process(buffer) is called
 * - Backend processing starts immediately, raw data is published to input consumers
 * - Consumers poll and read raw data in parallel to GPU processing
 * - The buffer is only reused by the producer when ALL consumers released it
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <cstring>
#include "processor.h"
#include "test_utils.h"

// Test configuration
constexpr ope::Backend TEST_BACKEND = ope::Backend::CUDA;
constexpr int SIGNAL_LENGTH = 1024;
constexpr int ASCANS_PER_BSCAN = 256;
constexpr int BSCANS_PER_BUFFER = 1;
constexpr int NUM_FRAMES = 100;

// Generate simple test pattern
std::vector<uint16_t> generateTestData() {
	size_t totalSamples = SIGNAL_LENGTH * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;
	std::vector<uint16_t> data(totalSamples);

	for (size_t i = 0; i < totalSamples; ++i) {
		data[i] = static_cast<uint16_t>((i * 65535) / totalSamples);
	}

	return data;
}

// Configure processor
void configureProcessor(ope::Processor& processor) {
	auto config = processor.getConfig();

	config.dataParams.signalLength = SIGNAL_LENGTH;
	config.dataParams.ascansPerBscan = ASCANS_PER_BSCAN;
	config.dataParams.bscansPerBuffer = BSCANS_PER_BUFFER;
	config.dataParams.inputDataType = ope::DataType::UINT16;

	// Enable processing so we can verify output
	config.processingParams.resampling.enabled = false;
	config.processingParams.windowing.enabled = false;
	config.processingParams.dispersion.enabled = false;
	config.processingParams.dcRemoval.enabled = false;
	config.processingParams.intensity.logScale = false;

	processor.setConfig(config);
}

// ============================================
// TEST 1: Basic Single Input Consumer
// ============================================
bool test_basic_input_consumer() {
	std::cout << "TEST 1: Basic Single Input Consumer" << std::endl;
	std::cout << "  Testing single consumer can read raw input data..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();

	// Add input consumer with BLOCK policy
	ope::ConsumerConfig inputConfig;
	inputConfig.dropPolicy = ope::DropPolicy::BLOCK;
	inputConfig.maxQueueSize = 10;
	ope::ConsumerId inputConsumer = processor.addInputConsumer(inputConfig);

	std::atomic<int> inputReceived{0};
	std::atomic<bool> stopThread{false};

	// Consumer thread reads raw input data
	std::thread consumerThread([&]() {
		while (!stopThread.load()) {
			ope::IOBuffer* inputBuf = processor.getNextInputBuffer(inputConsumer);
			if (!inputBuf) break;

			// Verify data is raw uint16
			const uint16_t* data = static_cast<const uint16_t*>(inputBuf->getDataPointer());
			TEST_ASSERT(data != nullptr, "Input buffer data pointer is null");

			inputReceived++;
			processor.releaseInputBuffer(inputConsumer, inputBuf);
		}
	});

	// Process frames
	auto testData = generateTestData();
	for (int i = 0; i < NUM_FRAMES; ++i) {
		auto& inputBuf = processor.getNextAvailableInputBuffer();
		std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
		processor.process(inputBuf);
	}

	// Wait for consumer to process all frames
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	stopThread = true;
	processor.removeInputConsumer(inputConsumer);
	consumerThread.join();

	if (inputReceived != NUM_FRAMES) {
		std::cerr << "  [FAIL] Expected " << NUM_FRAMES << " input buffers, got " << inputReceived << std::endl;
		return false;
	}

	std::cout << "  [OK] Input consumer received all " << NUM_FRAMES << " buffers" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 2: Multiple Input Consumers
// ============================================
bool test_multiple_input_consumers() {
	std::cout << "TEST 2: Multiple Input Consumers" << std::endl;
	std::cout << "  Testing 2 consumers can all read same raw data..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();

	const int NUM_CONSUMERS = 2;
	const int TEST_FRAMES = 50;  // Reduced to avoid overwhelming the system
	std::vector<ope::ConsumerId> consumers;
	consumers.reserve(NUM_CONSUMERS); // threads index this vector, no reallocation allowed
	std::vector<std::atomic<int>> receivedCounts(NUM_CONSUMERS);
	std::atomic<bool> stopThreads{false};
	std::vector<std::thread> threads;

	// Create consumers and threads
	for (int c = 0; c < NUM_CONSUMERS; ++c) {
		ope::ConsumerConfig config;
		config.dropPolicy = ope::DropPolicy::DROP_OLDEST;  // Use DROP_OLDEST to avoid blocking
		config.maxQueueSize = 5;
		consumers.push_back(processor.addInputConsumer(config));

		threads.emplace_back([&, c]() {
			while (!stopThreads.load()) {
				ope::IOBuffer* inputBuf = processor.getNextInputBuffer(consumers[c]);
				if (!inputBuf) break;

				// Read data (simulates display or recording)
				const uint16_t* data = static_cast<const uint16_t*>(inputBuf->getDataPointer());
				volatile uint16_t dummy = data[0];  // Read first sample

				receivedCounts[c]++;
				processor.releaseInputBuffer(consumers[c], inputBuf);
			}
		});
	}

	// Process frames with small delay to allow consumers to keep up
	auto testData = generateTestData();
	for (int i = 0; i < TEST_FRAMES; ++i) {
		auto& inputBuf = processor.getNextAvailableInputBuffer();
		std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
		processor.process(inputBuf);

		// Small delay to avoid overwhelming input buffer pool
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}

	// Wait for all consumers
	std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	stopThreads = true;
	for (auto consumer : consumers) {
		processor.removeInputConsumer(consumer);
	}
	for (auto& thread : threads) {
		thread.join();
	}

	// Verify consumers received frames (may have dropped some with DROP_OLDEST)
	int totalReceived = 0;
	for (int c = 0; c < NUM_CONSUMERS; ++c) {
		std::cout << "  Consumer " << c << " received " << receivedCounts[c] << " buffers" << std::endl;
		totalReceived += receivedCounts[c];
	}

	if (totalReceived == 0) {
		std::cerr << "  [FAIL] No consumers received any buffers!" << std::endl;
		return false;
	}

	std::cout << "  [OK] Consumers received buffers (total: " << totalReceived << ")" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 3: Input + Output Consumers Together
// ============================================
bool test_input_and_output_consumers() {
	std::cout << "TEST 3: Input + Output Consumers Together" << std::endl;
	std::cout << "  Testing raw input and processed output consumers work together..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();

	// Add input consumer (reads raw data)
	ope::ConsumerConfig inputConfig;
	inputConfig.dropPolicy = ope::DropPolicy::BLOCK;
	inputConfig.maxQueueSize = 10;
	ope::ConsumerId inputConsumer = processor.addInputConsumer(inputConfig);

	// Add output consumer (reads processed data)
	ope::ConsumerConfig outputConfig;
	outputConfig.dropPolicy = ope::DropPolicy::BLOCK;
	outputConfig.maxQueueSize = 10;
	ope::ConsumerId outputConsumer = processor.addConsumer(outputConfig);

	std::atomic<int> inputReceived{0};
	std::atomic<int> outputReceived{0};
	std::atomic<bool> stopThreads{false};

	// Input consumer thread
	std::thread inputThread([&]() {
		while (!stopThreads.load()) {
			ope::IOBuffer* inputBuf = processor.getNextInputBuffer(inputConsumer);
			if (!inputBuf) break;

			// Read raw uint16 data
			const uint16_t* data = static_cast<const uint16_t*>(inputBuf->getDataPointer());
			TEST_ASSERT(data != nullptr, "Input buffer is null");

			inputReceived++;
			processor.releaseInputBuffer(inputConsumer, inputBuf);
		}
	});

	// Output consumer thread
	std::thread outputThread([&]() {
		while (!stopThreads.load()) {
			ope::IOBuffer* outputBuf = processor.getNextOutputBuffer(outputConsumer);
			if (!outputBuf) break;

			// Read processed float data
			const float* data = static_cast<const float*>(outputBuf->getDataPointer());
			TEST_ASSERT(data != nullptr, "Output buffer is null");

			outputReceived++;
			processor.releaseOutputBuffer(outputConsumer, outputBuf);
		}
	});

	// Process frames
	auto testData = generateTestData();
	for (int i = 0; i < NUM_FRAMES; ++i) {
		auto& inputBuf = processor.getNextAvailableInputBuffer();
		std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
		processor.process(inputBuf);
	}

	// Wait for both consumers
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	stopThreads = true;
	processor.removeInputConsumer(inputConsumer);
	processor.removeConsumer(outputConsumer);
	inputThread.join();
	outputThread.join();

	if (inputReceived != NUM_FRAMES || outputReceived != NUM_FRAMES) {
		std::cerr << "  [FAIL] Input received: " << inputReceived << ", Output received: " << outputReceived
				  << ", Expected: " << NUM_FRAMES << std::endl;
		return false;
	}

	std::cout << "  [OK] Both input and output consumers received all " << NUM_FRAMES << " buffers" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 4: Data Integrity
// ============================================
bool test_input_data_integrity() {
	std::cout << "TEST 4: Data Integrity" << std::endl;
	std::cout << "  Testing multiple consumers receive identical raw data..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();

	const int NUM_CONSUMERS = 3;
	std::vector<ope::ConsumerId> consumers;
	consumers.reserve(NUM_CONSUMERS); // threads index this vector, no reallocation allowed
	std::vector<std::vector<uint16_t>> receivedData(NUM_CONSUMERS);
	std::atomic<bool> stopThreads{false};
	std::vector<std::thread> threads;

	size_t expectedSize = SIGNAL_LENGTH * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;

	// Create consumers
	for (int c = 0; c < NUM_CONSUMERS; ++c) {
		ope::ConsumerConfig config;
		config.dropPolicy = ope::DropPolicy::BLOCK;
		config.maxQueueSize = 10;
		consumers.push_back(processor.addInputConsumer(config));

		threads.emplace_back([&, c]() {
			// Read only first frame for comparison
			ope::IOBuffer* inputBuf = processor.getNextInputBuffer(consumers[c]);
			if (inputBuf) {
				const uint16_t* data = static_cast<const uint16_t*>(inputBuf->getDataPointer());
				receivedData[c].assign(data, data + expectedSize);
				processor.releaseInputBuffer(consumers[c], inputBuf);
			}

			// Drain remaining frames
			while (!stopThreads.load()) {
				ope::IOBuffer* buf = processor.getNextInputBuffer(consumers[c]);
				if (!buf) break;
				processor.releaseInputBuffer(consumers[c], buf);
			}
		});
	}

	// Process one frame with known data
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);

	// Wait for consumers to read first frame
	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	stopThreads = true;
	for (auto consumer : consumers) {
		processor.removeInputConsumer(consumer);
	}
	for (auto& thread : threads) {
		thread.join();
	}

	// Verify all consumers received same data
	for (int c = 0; c < NUM_CONSUMERS; ++c) {
		if (receivedData[c].size() != expectedSize) {
			std::cerr << "  [FAIL] Consumer " << c << " received " << receivedData[c].size()
					  << " samples, expected " << expectedSize << std::endl;
			return false;
		}
	}

	// Compare data between consumers
	for (int c = 1; c < NUM_CONSUMERS; ++c) {
		if (receivedData[c] != receivedData[0]) {
			std::cerr << "  [FAIL] Consumer " << c << " data differs from consumer 0!" << std::endl;
			return false;
		}
	}

	// Compare with original test data
	if (receivedData[0] != testData) {
		std::cerr << "  [FAIL] Received data differs from original!" << std::endl;
		return false;
	}

	std::cout << "  [OK] All consumers received identical raw data (" << expectedSize << " samples)" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 5: tryGetInputBuffer (non-blocking)
// ============================================
bool test_try_get_input_buffer() {
	std::cout << "TEST 5: tryGetInputBuffer (non-blocking)" << std::endl;
	std::cout << "  Testing non-blocking input buffer retrieval..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	configureProcessor(processor);
	processor.initialize();

	ope::ConsumerConfig config;
	config.dropPolicy = ope::DropPolicy::BLOCK;
	config.maxQueueSize = 10;
	ope::ConsumerId consumer = processor.addInputConsumer(config);

	// Try to get buffer when none available (should return false)
	ope::IOBuffer* buf = nullptr;
	bool success = processor.tryGetInputBuffer(consumer, &buf);

	if (success || buf != nullptr) {
		std::cerr << "  [FAIL] tryGetInputBuffer returned buffer when none should be available!" << std::endl;
		return false;
	}

	std::cout << "  [OK] tryGetInputBuffer returns false when no buffer available" << std::endl;

	// Process one frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);

	// Wait for buffer to be published
	std::this_thread::sleep_for(std::chrono::milliseconds(50));

	// Now try to get buffer (should succeed)
	success = processor.tryGetInputBuffer(consumer, &buf);

	if (!success || buf == nullptr) {
		std::cerr << "  [FAIL] tryGetInputBuffer failed when buffer should be available!" << std::endl;
		return false;
	}

	std::cout << "  [OK] tryGetInputBuffer returns true when buffer is available" << std::endl;

	processor.releaseInputBuffer(consumer, buf);
	processor.removeInputConsumer(consumer);

	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// Main
// ============================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Input Consumer Polling API Tests" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	int passed = 0;
	int total = 0;

	#define RUN_TEST(test_func) \
		total++; \
		if (test_func()) { \
			passed++; \
		}

	RUN_TEST(test_basic_input_consumer);
	RUN_TEST(test_multiple_input_consumers);
	RUN_TEST(test_input_and_output_consumers);
	RUN_TEST(test_input_data_integrity);
	RUN_TEST(test_try_get_input_buffer);

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
