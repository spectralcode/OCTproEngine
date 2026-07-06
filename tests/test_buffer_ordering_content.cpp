#include "../include/processor.h"
#include "test_utils.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstring>

const ope::Backend TEST_BACKEND = ope::Backend::VULKAN;

// Simple test: Count A-scans with non-zero data in processed output
void testProgressiveDataContent() {
	std::cout << "Testing progressive data content (in-memory counting)..." << std::endl;

	const int signalLength = 1024;
	const int ascansPerBscan = 512;
	const int bscansPerBuffer = 1;
	const int numBuffers = 20000;
	ope::DataType dataType = ope::DataType::UINT16; // do not change. data generation in loop expects 16 bit

    const int bitDepth = getDataTypeBitDepth(dataType);

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);
	processor.initialize();

	// Print configuration
	std::cout << "\n=== Configuration ===" << std::endl;
	std::cout << "  Backend: ";
	if (TEST_BACKEND == ope::Backend::CUDA) std::cout << "CUDA" << std::endl;
	else if (TEST_BACKEND == ope::Backend::VULKAN) std::cout << "Vulkan" << std::endl;
	else if (TEST_BACKEND == ope::Backend::OPENCL) std::cout << "OpenCL" << std::endl;
	else if (TEST_BACKEND == ope::Backend::CPU) std::cout << "CPU" << std::endl;
	std::cout << "  Samples per signal: " << signalLength << std::endl;
	std::cout << "  A-scans per B-scan: " << ascansPerBscan << std::endl;
	std::cout << "  B-scans per buffer: " << bscansPerBuffer << std::endl;
	std::cout << "  Bit depth: " << bitDepth << std::endl;
	std::cout << "  Number of buffers: " << numBuffers << std::endl;

	// Track counts per buffer IN RECEIVE ORDER (not by buffer ID)
	std::mutex resultsMutex;
	std::vector<int> receivedCounts;  // Counts in receive order
	std::vector<uint64_t> receivedIds;  // Buffer IDs in receive order
	std::atomic<int> callbackCount{0};
	std::chrono::high_resolution_clock::time_point endTime;

	// Add callback that counts A-scans with significant signal
	const float SIGNAL_THRESHOLD = 0.1f;  // Threshold to distinguish signal from background

	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		uint64_t bufferId = buf.getBufferId();

		const float* data = static_cast<const float*>(buf.getDataPointer());
		int outputSignalLength = signalLength / 2;
		int count = 0;

		// Count A-scans that have significant signal (above threshold)
		for (int ascan = 0; ascan < ascansPerBscan; ascan++) {
			bool hasSignal = false;
			for (int sample = 0; sample < outputSignalLength; sample++) {
				size_t idx = ascan * outputSignalLength + sample;
				if (data[idx] > SIGNAL_THRESHOLD) {
					hasSignal = true;
					break;
				}
			}
			if (hasSignal) count++;
		}

		{
			std::lock_guard<std::mutex> lock(resultsMutex);
			receivedCounts.push_back(count);
			receivedIds.push_back(bufferId);
		}
		
		int newCount = ++callbackCount;
		if (newCount == numBuffers) {
			endTime = std::chrono::high_resolution_clock::now();
		}
	});

	// Generate progressive pattern: buffer N has (N+1) A-scans filled
	std::cout << "\nProcessing " << numBuffers << " buffers with progressive pattern..." << std::endl;
	auto startTime = std::chrono::high_resolution_clock::now();
	
	for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		uint16_t* data = static_cast<uint16_t*>(inputBuffer.getDataPointer());

		// Zero-fill entire buffer
		memset(data, 0, signalLength * ascansPerBscan * bscansPerBuffer * sizeof(uint16_t));

		// Fill first (bufferIdx + 1) A-scans with pattern
		int linesToFill = (bufferIdx % ascansPerBscan) + 1;

		for (int ascan = 0; ascan < linesToFill; ascan++) {
			for (int sample = 0; sample < signalLength; sample++) {
				size_t idx = ascan * signalLength + sample;
				data[idx] = static_cast<uint16_t>(1000 + bufferIdx * 50 + ascan * 10 + sample % 100);
			}
		}
        //sleep for 0.1 ms to simulate acquisition delay
       // std::this_thread::sleep_for(std::chrono::microseconds(1000));
       //std::this_thread::sleep_for(std::chrono::milliseconds(1));
		processor.process(inputBuffer);
	}

	// Wait for all callbacks
	std::cout << "Waiting for completion..." << std::endl;
	while (callbackCount < numBuffers) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}

	// === PERFORMANCE ===
	double durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
	double bscansPerSec = (numBuffers * bscansPerBuffer / durationMs) * 1000.0;
	size_t bytesPerBuffer = signalLength * ascansPerBscan * bscansPerBuffer * (bitDepth / 8);
	double mbPerSec = (numBuffers * bytesPerBuffer / durationMs) * 1000.0 / (1024.0 * 1024.0);
	
	std::cout << "\n=== Performance ===" << std::endl;
	std::cout << "  Duration: " << durationMs << " ms" << std::endl;
	std::cout << "  B-scans/s: " << bscansPerSec << std::endl;
	std::cout << "  MB/s: " << mbPerSec << std::endl;

	// Verify exact modulo-based counts (no tolerance)
	std::cout << "\n=== Content Verification ===" << std::endl;
	int errors = 0;
	int mismatchErrors = 0;
	int orderingErrors = 0;
	
	// First check: buffer IDs should be in order [0, 1, 2, ..., numBuffers-1]
	for (int i = 0; i < numBuffers; i++) {
		if (receivedIds[i] != static_cast<uint64_t>(i)) {
			if (orderingErrors < 5) {
				std::cout << "  [ERROR] Position " << i << ": expected buffer ID " << i 
				          << ", got " << receivedIds[i] << std::endl;
			}
			orderingErrors++;
			errors++;
		}
	}
	
	if (orderingErrors > 5) {
		std::cout << "  ... and " << (orderingErrors - 5) << " more ordering errors" << std::endl;
	}
	
	// Second check: counts must match exact modulo-based expectation
	for (int i = 0; i < numBuffers; i++) {
		int expectedCount = (i % ascansPerBscan) + 1;
		int actualCount = receivedCounts[i];
		if (actualCount != expectedCount) {
			if (mismatchErrors < 5) {
				std::cout << "  [ERROR] Position " << i << " (ID " << receivedIds[i]
						  << "): expected count " << expectedCount
						  << ", got " << actualCount << std::endl;
			}
			mismatchErrors++;
			errors++;
		}
	}

	std::cout << "\n=== Summary ===" << std::endl;
	std::cout << "  Total buffers received: " << receivedCounts.size() << std::endl;
	std::cout << "  Ordering errors: " << orderingErrors << std::endl;
	std::cout << "  Mismatch errors: " << mismatchErrors << std::endl;
	std::cout << "  Total errors: " << errors << std::endl;
	std::cout << "  Signal threshold: " << SIGNAL_THRESHOLD << std::endl;

	TEST_ASSERT(errors == 0, "Buffers must be in order and match exact modulo-based signal counts");
	std::cout << "  [OK] All " << numBuffers << " buffers arrived in order with exact modulo signal counts" << std::endl;
}

int main() {
	std::cout << "=== Progressive Data Content Test ===" << std::endl;
	try {
		testProgressiveDataContent();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}
