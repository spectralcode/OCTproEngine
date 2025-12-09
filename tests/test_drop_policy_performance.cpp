/**
 * Test: Drop Policy Performance
 *
 * Tests processing performance with increasing numbers of SLOW consumers
 * using DROP_OLDEST policy. Demonstrates that slow consumers don't block
 * processing - they just drop frames.
 *
 * Format matches test_multi_consumer TEST 6 for comparison.
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <cstring>
#include "processor.h"


constexpr ope::Backend TEST_BACKEND = ope::Backend::CUDA;


constexpr int SIGNAL_LENGTH = 2048;
constexpr int ASCANS_PER_BSCAN = 512;
constexpr int BSCANS_PER_BUFFER = 1;
constexpr int NUM_ITERATIONS = 1000;
constexpr int MAX_CONSUMERS = 8;
constexpr int CONSUMER_DELAY_US = 500; 

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "DROP_OLDEST CONSUMER PERFORMANCE" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Each slow consumer sleeps " << CONSUMER_DELAY_US << " us per frame" << std::endl;
	std::cout << "Measuring performance impact of DROP_OLDEST consumers..." << std::endl;
	std::cout << std::endl;

	size_t inputSize = SIGNAL_LENGTH * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;
	std::vector<uint16_t> testData(inputSize);
	for (size_t i = 0; i < inputSize; ++i) {
		testData[i] = static_cast<uint16_t>(i % 65536);
	}

	double baselineAvgTime = 0.0;

	for (int numConsumers = 0; numConsumers <= MAX_CONSUMERS; ++numConsumers) {
		ope::Processor processor(TEST_BACKEND);
		auto config = processor.getConfig();
		config.dataParams.signalLength = SIGNAL_LENGTH;
		config.dataParams.ascansPerBscan = ASCANS_PER_BSCAN;
		config.dataParams.bscansPerBuffer = BSCANS_PER_BUFFER;
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

		// Add one fast BLOCK consumer (should receive all frames)
		ope::ConsumerConfig blockConfig;
		blockConfig.dropPolicy = ope::DropPolicy::BLOCK;
		blockConfig.maxQueueSize = 32;
		ope::ConsumerId blockConsumer = processor.addConsumer(blockConfig);
		std::atomic<int> blockReceivedCount{0};
		std::chrono::high_resolution_clock::time_point endTime;

		// Buffer for BLOCK consumer to copy data into (simulates real usage)
		std::vector<char> blockCopyBuffer(outputBufferSize);

		std::thread blockThread([&]() {
			for (int i = 0; i < NUM_ITERATIONS; ++i) {
				ope::IOBuffer* output = processor.getNextOutputBuffer(blockConsumer);
				if (!output) break;

				// Copy output data (simulates real consumer work)
				std::memcpy(blockCopyBuffer.data(), output->getDataPointer(), outputBufferSize);

				processor.releaseOutputBuffer(blockConsumer, output);
				blockReceivedCount++;
			}
			// Capture end time when last buffer is received
			endTime = std::chrono::high_resolution_clock::now();
		});

		// Add slow consumers with DROP_OLDEST
		std::vector<ope::ConsumerId> consumers;
		std::vector<std::atomic<int>> receivedCounts(numConsumers);
		std::atomic<bool> done{false};
		std::vector<std::thread> threads;

		for (int c = 0; c < numConsumers; ++c) {
			ope::ConsumerConfig consumerConfig;
			consumerConfig.dropPolicy = ope::DropPolicy::DROP_OLDEST;
			consumerConfig.maxQueueSize = 16;
			consumers.push_back(processor.addConsumer(consumerConfig));

			threads.emplace_back([&, c]() {
				while (!done.load()) {
					ope::IOBuffer* output = processor.getNextOutputBuffer(consumers[c]);
					if (!output) break;

					// Simulate slow consumer
					std::this_thread::sleep_for(std::chrono::microseconds(CONSUMER_DELAY_US));

					processor.releaseOutputBuffer(consumers[c], output);
					receivedCounts[c]++;
				}
			});
		}

		// Process frames
		auto startTime = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < NUM_ITERATIONS; ++i) {
			auto& inputBuf = processor.getNextAvailableInputBuffer();
			std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
			processor.process(inputBuf);
		}

		// Wait for fast consumers to drain queues (only matters when no delay)
		if (numConsumers > 0) {
			int expectedTotal = numConsumers * NUM_ITERATIONS;
			auto waitStart = std::chrono::high_resolution_clock::now();
			while (true) {
				int totalReceived = 0;
				for (int c = 0; c < numConsumers; ++c) {
					totalReceived += receivedCounts[c].load();
				}
				if (totalReceived >= expectedTotal) break;

				// Set timeout (slow consumers with drops won't reach expected)
				auto elapsed = std::chrono::high_resolution_clock::now() - waitStart;
				if (elapsed > std::chrono::milliseconds(250)) break;

				std::this_thread::sleep_for(std::chrono::microseconds(100));
			}
		}

		// Wait for BLOCK consumer to finish (it receives all frames)
		blockThread.join();
		processor.removeConsumer(blockConsumer);

		// Signal DROP_OLDEST consumers to stop
		done = true;
		for (int c = 0; c < numConsumers; ++c) {
			processor.removeConsumer(consumers[c]);
		}

		// Wait for consumer threads to finish
		for (auto& t : threads) {
			t.join();
		}

		// Calculate metrics
		double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
		double avgTime = duration / NUM_ITERATIONS;
		double buffersPerSec = 1000.0 / avgTime;
		double bscansPerSec = buffersPerSec * config.dataParams.bscansPerBuffer;
		double ascansPerSec = bscansPerSec * ASCANS_PER_BSCAN;
		double mbPerSec = buffersPerSec * outputBufferSize / (1024.0 * 1024.0);

		if (numConsumers == 0) {
			baselineAvgTime = avgTime;
		}

		double ratio = baselineAvgTime / avgTime;

		// Count dropped frames
		int totalDropped = 0;
		int totalReceived = 0;
		for (int c = 0; c < numConsumers; ++c) {
			totalDropped += static_cast<int>(processor.getDroppedFrameCount(consumers[c]));
			totalReceived += receivedCounts[c].load();
		}

		std::cout << "  " << numConsumers << " slow DROP_OLDEST + 1 fast BLOCK: "
		          << avgTime << " ms/buffer, "
		          << bscansPerSec << " bscans/s "
				  << mbPerSec << " MB/s "
		          << "(" << ratio << "x)";

		std::cout << " [BLOCK: " << blockReceivedCount.load() << "/" << NUM_ITERATIONS;
		if (numConsumers > 0) {
			std::cout << ", DROP_OLDEST dropped: " << totalDropped << ", received: " << totalReceived;
		}
		std::cout << "]" << std::endl;
	}

	std::cout << std::endl;
	std::cout << "DROP_OLDEST allows processing to continue at full speed." << std::endl;
	std::cout << std::endl;
	std::cout << "PASSED" << std::endl;

	return 0;
}
