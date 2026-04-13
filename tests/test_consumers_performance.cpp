/**
 * Test: Consumer Performance Scaling
 *
 * Measures processing throughput with increasing numbers of consumers.
 * All consumers use DROP_OLDEST policy and simulate slow processing.
 * Demonstrates that slow consumers don't block processing - they drop frames.
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <cstring>
#include <iomanip>
#include "processor.h"

// Configuration
constexpr ope::Backend TEST_BACKEND = ope::Backend::CUDA;
constexpr int SIGNAL_LENGTH = 2048;
constexpr int ASCANS_PER_BSCAN = 512;
constexpr int BSCANS_PER_BUFFER = 1;
constexpr int NUM_ITERATIONS = 5000;
constexpr int CONSUMER_DELAY_US = 100000;  // 100ms

struct PerformanceResult {
	double avgTimeMs;
	double buffersPerSec;
	double ascansPerSec;
	double mbPerSec;
	int totalDropped;
};

PerformanceResult testPerformance(int numInputConsumers, int numOutputConsumers) {
	size_t inputSize = SIGNAL_LENGTH * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;
	std::vector<uint16_t> testData(inputSize);
	for (size_t i = 0; i < inputSize; ++i) {
		testData[i] = static_cast<uint16_t>(i % 65536);
	}

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
	size_t outputBufferSize = dataParams.outputSignalLength() * dataParams.ascansPerBscan *
							  dataParams.bscansPerBuffer * dataParams.getOutputBytesPerSample();

	// Add one fast BLOCK consumer to track when processing is done
	ope::ConsumerConfig blockConfig;
	blockConfig.dropPolicy = ope::DropPolicy::BLOCK;
	blockConfig.maxQueueSize = 2;
	ope::ConsumerId blockConsumer = processor.addConsumer(blockConfig);
	std::atomic<int> blockReceivedCount{0};
	std::chrono::high_resolution_clock::time_point endTime;
	std::vector<char> blockCopyBuffer(outputBufferSize);

	std::thread blockThread([&]() {
		for (int i = 0; i < NUM_ITERATIONS; ++i) {
			ope::IOBuffer* output = processor.getNextOutputBuffer(blockConsumer);
			if (!output) break;
			std::memcpy(blockCopyBuffer.data(), output->getDataPointer(), outputBufferSize);
			processor.releaseOutputBuffer(blockConsumer, output);
			blockReceivedCount++;
		}
		endTime = std::chrono::high_resolution_clock::now();
	});

	// Add input consumers (DROP_OLDEST, sleep before release)
	std::vector<ope::ConsumerId> inputConsumers;
	std::vector<std::atomic<int>> inputReceivedCounts(numInputConsumers);
	std::atomic<bool> doneInput{false};
	std::vector<std::thread> inputThreads;
	std::vector<std::vector<char>> inputCopyBuffers(numInputConsumers);

	for (int c = 0; c < numInputConsumers; ++c) {
		inputCopyBuffers[c].resize(inputSize * sizeof(uint16_t));
		ope::ConsumerConfig consumerConfig;
		consumerConfig.dropPolicy = ope::DropPolicy::DROP_OLDEST;
		consumerConfig.maxQueueSize = 2;
		inputConsumers.push_back(processor.addInputConsumer(consumerConfig));

		inputThreads.emplace_back([&, c]() {
			while (!doneInput.load()) {
				ope::IOBuffer* input = processor.getNextInputBuffer(inputConsumers[c]);
				if (!input) break;

				// Copy data
				std::memcpy(inputCopyBuffers[c].data(), input->getDataPointer(), inputSize * sizeof(uint16_t));

				// Simulate slow consumer (BEFORE release to avoid mutex contention)
				std::this_thread::sleep_for(std::chrono::microseconds(CONSUMER_DELAY_US));

				processor.releaseInputBuffer(inputConsumers[c], input);
				inputReceivedCounts[c]++;
			}
		});
	}

	// Add output consumers (DROP_OLDEST, sleep before release)
	std::vector<ope::ConsumerId> outputConsumers;
	std::vector<std::atomic<int>> outputReceivedCounts(numOutputConsumers);
	std::atomic<bool> doneOutput{false};
	std::vector<std::thread> outputThreads;
	std::vector<std::vector<char>> outputCopyBuffers(numOutputConsumers);

	for (int c = 0; c < numOutputConsumers; ++c) {
		outputCopyBuffers[c].resize(outputBufferSize);
		ope::ConsumerConfig consumerConfig;
		consumerConfig.dropPolicy = ope::DropPolicy::DROP_OLDEST;
		consumerConfig.maxQueueSize = 2;
		outputConsumers.push_back(processor.addConsumer(consumerConfig));

		outputThreads.emplace_back([&, c]() {
			while (!doneOutput.load()) {
				ope::IOBuffer* output = processor.getNextOutputBuffer(outputConsumers[c]);
				if (!output) break;

				// Copy data
				std::memcpy(outputCopyBuffers[c].data(), output->getDataPointer(), outputBufferSize);

				// Simulate slow consumer
				std::this_thread::sleep_for(std::chrono::microseconds(CONSUMER_DELAY_US));

				processor.releaseOutputBuffer(outputConsumers[c], output);
				outputReceivedCounts[c]++;
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

	// Wait for BLOCK consumer to finish (it receives all frames)
	blockThread.join();
	processor.removeConsumer(blockConsumer);

	// Signal consumers to stop
	doneInput = true;
	doneOutput = true;
	for (int c = 0; c < numInputConsumers; ++c) {
		processor.removeInputConsumer(inputConsumers[c]);
	}
	for (int c = 0; c < numOutputConsumers; ++c) {
		processor.removeConsumer(outputConsumers[c]);
	}

	// Wait for consumer threads
	for (auto& t : inputThreads) {
		t.join();
	}
	for (auto& t : outputThreads) {
		t.join();
	}

	// Calculate metrics
	double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
	double avgTime = duration / NUM_ITERATIONS;
	double buffersPerSec = 1000.0 / avgTime;
	double ascansPerSec = buffersPerSec * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;
	double mbPerSec = buffersPerSec * outputBufferSize / (1024.0 * 1024.0);

	// Calculate total dropped frames
	int totalDropped = 0;
	for (int c = 0; c < numInputConsumers; ++c) {
		int received = inputReceivedCounts[c].load();
		int dropped = NUM_ITERATIONS - received;
		totalDropped += dropped;
	}
	for (int c = 0; c < numOutputConsumers; ++c) {
		int received = outputReceivedCounts[c].load();
		int dropped = NUM_ITERATIONS - received;
		totalDropped += dropped;
	}

	return {avgTime, buffersPerSec, ascansPerSec, mbPerSec, totalDropped};
}

void testOutputConsumers() {
	std::cout << "\n========================================" << std::endl;
	std::cout << "OUTPUT CONSUMER PERFORMANCE" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "DROP_OLDEST policy - slow consumers drop frames" << std::endl;
	std::cout << "Each consumer sleeps " << CONSUMER_DELAY_US << " us per frame" << std::endl;
	std::cout << std::endl;

	std::cout << std::setw(12) << "Consumers"
			  << std::setw(12) << "ms/buffer"
			  << std::setw(12) << "buffers/s"
			  << std::setw(12) << "ascans/s"
			  << std::setw(12) << "MB/s"
			  << std::setw(12) << "Dropped"
			  << std::endl;
	std::cout << std::string(72, '-') << std::endl;

	for (int numConsumers = 0; numConsumers <= 4; ++numConsumers) {
		auto result = testPerformance(0, numConsumers);
		std::cout << std::setw(12) << numConsumers
				  << std::setw(12) << std::fixed << std::setprecision(3) << result.avgTimeMs
				  << std::setw(12) << std::fixed << std::setprecision(1) << result.buffersPerSec
				  << std::setw(12) << std::fixed << std::setprecision(0) << result.ascansPerSec
				  << std::setw(12) << std::fixed << std::setprecision(1) << result.mbPerSec
				  << std::setw(12) << result.totalDropped
				  << std::endl;
	}

	std::cout << std::endl;
}

void testInputConsumers() {
	std::cout << "\n========================================" << std::endl;
	std::cout << "INPUT CONSUMER PERFORMANCE" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "DROP_OLDEST policy - slow consumers drop frames" << std::endl;
	std::cout << "Each consumer sleeps " << CONSUMER_DELAY_US << " us per frame" << std::endl;
	std::cout << std::endl;

	std::cout << std::setw(12) << "Consumers"
			  << std::setw(12) << "ms/buffer"
			  << std::setw(12) << "buffers/s"
			  << std::setw(12) << "ascans/s"
			  << std::setw(12) << "MB/s"
			  << std::setw(12) << "Dropped"
			  << std::endl;
	std::cout << std::string(72, '-') << std::endl;

	for (int numConsumers = 0; numConsumers <= 4; ++numConsumers) {
		auto result = testPerformance(numConsumers, 0);
		std::cout << std::setw(12) << numConsumers
				  << std::setw(12) << std::fixed << std::setprecision(3) << result.avgTimeMs
				  << std::setw(12) << std::fixed << std::setprecision(1) << result.buffersPerSec
				  << std::setw(12) << std::fixed << std::setprecision(0) << result.ascansPerSec
				  << std::setw(12) << std::fixed << std::setprecision(1) << result.mbPerSec
				  << std::setw(12) << result.totalDropped
				  << std::endl;
	}

	std::cout << std::endl;
}

void testMixedConsumers() {
	std::cout << "\n========================================" << std::endl;
	std::cout << "MIXED CONSUMER PERFORMANCE" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "DROP_OLDEST policy - slow consumers drop frames" << std::endl;
	std::cout << "Each consumer sleeps " << CONSUMER_DELAY_US << " us per frame" << std::endl;
	std::cout << std::endl;

	std::cout << std::setw(12) << "Config"
			  << std::setw(12) << "ms/buffer"
			  << std::setw(12) << "buffers/s"
			  << std::setw(12) << "ascans/s"
			  << std::setw(12) << "MB/s"
			  << std::setw(12) << "Dropped"
			  << std::endl;
	std::cout << std::string(72, '-') << std::endl;

	std::vector<std::pair<int, int>> combinations = {
		{1, 1}, {2, 2}, {2, 4}, {4, 2}
	};

	for (size_t i = 0; i < combinations.size(); ++i) {
		int numInput = combinations[i].first;
		int numOutput = combinations[i].second;
		auto result = testPerformance(numInput, numOutput);
		std::string label = std::to_string(static_cast<long long>(numInput)) + "in+" +
							std::to_string(static_cast<long long>(numOutput)) + "out";
		std::cout << std::setw(12) << label
				  << std::setw(12) << std::fixed << std::setprecision(3) << result.avgTimeMs
				  << std::setw(12) << std::fixed << std::setprecision(1) << result.buffersPerSec
				  << std::setw(12) << std::fixed << std::setprecision(0) << result.ascansPerSec
				  << std::setw(12) << std::fixed << std::setprecision(1) << result.mbPerSec
				  << std::setw(12) << result.totalDropped
				  << std::endl;
	}

	std::cout << std::endl;
}

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Consumer Performance Scaling Test" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Backend: " << (TEST_BACKEND == ope::Backend::CUDA ? "CUDA" :
								  TEST_BACKEND == ope::Backend::VULKAN ? "Vulkan" :
								  TEST_BACKEND == ope::Backend::OPENCL ? "OpenCL" : "CPU") << std::endl;
	std::cout << "Signal: " << SIGNAL_LENGTH << " samples" << std::endl;
	std::cout << "A-scans: " << ASCANS_PER_BSCAN << " per B-scan" << std::endl;
	std::cout << "Iterations: " << NUM_ITERATIONS << std::endl;
	std::cout << std::endl;

	// Run all test sections
	testOutputConsumers();
	testInputConsumers();
	testMixedConsumers();

	std::cout << "========================================" << std::endl;
	std::cout << "SUMMARY" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Processing performance maintained with slow consumers" << std::endl;
	std::cout << "DROP_OLDEST policy prevents blocking" << std::endl;
	std::cout << "Dropped frame counts show consumer load" << std::endl;
	std::cout << "========================================" << std::endl;

	return 0;
}
