#include "../include/processor.h"
#include "../include/cudautils.h"
#include <iostream>
#include <fstream>
#include <cstring>

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Testing CUDA API Functionality" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	int totalTests = 0;
	int passedTests = 0;

	// ============================================
	// Test 1: CudaUtils::isAvailable()
	// ============================================
	std::cout << "Test 1: CudaUtils::isAvailable()" << std::endl;
	totalTests++;

	bool cudaAvailable = ope::CudaUtils::isAvailable();
	std::cout << "  CUDA Available: " << (cudaAvailable ? "YES" : "NO") << std::endl;

	if (cudaAvailable) {
		std::cout << "  Result: PASS (CUDA detected)" << std::endl;
		passedTests++;
	} else {
		std::cout << "  Result: PASS (No CUDA - expected on CPU-only builds)" << std::endl;
		passedTests++;
	}
	std::cout << std::endl;

	// ============================================
	// Test 2: CudaUtils::getDeviceCount()
	// ============================================
	std::cout << "Test 2: CudaUtils::getDeviceCount()" << std::endl;
	totalTests++;

	int deviceCount = ope::CudaUtils::getDeviceCount();
	std::cout << "  Device Count: " << deviceCount << std::endl;

	bool test2Pass = (cudaAvailable && deviceCount > 0) || (!cudaAvailable && deviceCount == 0);
	std::cout << "  Result: " << (test2Pass ? "PASS" : "FAIL") << std::endl;
	if (test2Pass) passedTests++;
	std::cout << std::endl;

	// ============================================
	// Test 3: CudaUtils::getAvailableDevices()
	// ============================================
	std::cout << "Test 3: CudaUtils::getAvailableDevices()" << std::endl;
	totalTests++;

	auto devices = ope::CudaUtils::getAvailableDevices();
	std::cout << "  Found " << devices.size() << " device(s)" << std::endl;

	for (size_t i = 0; i < devices.size(); i++) {
		const auto& dev = devices[i];
		std::cout << "  Device " << dev.deviceId << ":" << std::endl;
		std::cout << "    Name: " << dev.name << std::endl;
		std::cout << "    Compute Capability: " << dev.getComputeCapability() << std::endl;
		std::cout << "    Total Memory: " << (dev.totalMemory / (1024*1024)) << " MB" << std::endl;
		std::cout << "    Free Memory: " << (dev.freeMemory / (1024*1024)) << " MB" << std::endl;
		std::cout << "    Max Threads/Block: " << dev.maxThreadsPerBlock << std::endl;
		std::cout << "    Multiprocessor Count: " << dev.multiProcessorCount << std::endl;
	}

	bool test3Pass = (devices.size() == static_cast<size_t>(deviceCount));
	std::cout << "  Result: " << (test3Pass ? "PASS" : "FAIL") << std::endl;
	if (test3Pass) passedTests++;
	std::cout << std::endl;

	// ============================================
	// Test 4: CudaUtils::getDeviceInfo() for each device
	// ============================================
	if (cudaAvailable && deviceCount > 0) {
		std::cout << "Test 4: CudaUtils::getDeviceInfo()" << std::endl;
		totalTests++;

		bool test4Pass = true;
		for (int i = 0; i < deviceCount; i++) {
			try {
				auto info = ope::CudaUtils::getDeviceInfo(i);
				std::cout << "  Device " << i << " info retrieved successfully" << std::endl;

				if (info.deviceId != i || info.name.empty()) {
					test4Pass = false;
					std::cout << "    ERROR: Invalid device info" << std::endl;
				}
			} catch (const std::exception& e) {
				test4Pass = false;
				std::cout << "  ERROR: " << e.what() << std::endl;
			}
		}

		std::cout << "  Result: " << (test4Pass ? "PASS" : "FAIL") << std::endl;
		if (test4Pass) passedTests++;
		std::cout << std::endl;
	}

	// ============================================
	// Test 5: CudaUtils::isDeviceAvailable()
	// ============================================
	if (cudaAvailable && deviceCount > 0) {
		std::cout << "Test 5: CudaUtils::isDeviceAvailable()" << std::endl;
		totalTests++;

		bool device0Available = ope::CudaUtils::isDeviceAvailable(0);
		bool invalidDeviceAvailable = ope::CudaUtils::isDeviceAvailable(999);

		bool test5Pass = device0Available && !invalidDeviceAvailable;
		std::cout << "  Device 0 available: " << (device0Available ? "YES" : "NO") << std::endl;
		std::cout << "  Device 999 available: " << (invalidDeviceAvailable ? "YES" : "NO") << std::endl;
		std::cout << "  Result: " << (test5Pass ? "PASS" : "FAIL") << std::endl;
		if (test5Pass) passedTests++;
		std::cout << std::endl;
	}

	// ============================================
	// Test 6: Processor CUDA settings (before init)
	// ============================================
	if (cudaAvailable && deviceCount > 0) {
		std::cout << "Test 6: Processor CUDA settings (before init)" << std::endl;
		totalTests++;

		try {
			ope::Processor processor(ope::Backend::CUDA);

			// Set CUDA settings before initialization
			processor.setCudaDevice(0);
			processor.setCudaNumStreams(4);
			processor.setCudaBlockSize(256);
			processor.setNumBuffers(3);

			// Verify settings
			int device = processor.getCudaDevice();
			int streams = processor.getCudaNumStreams();
			int blockSize = processor.getCudaBlockSize();
			int buffers = processor.getNumBuffers();

			bool test6Pass = (device == 0) && (streams == 4) && (blockSize == 256) && (buffers == 3);

			std::cout << "  Device ID: " << device << " (expected: 0)" << std::endl;
			std::cout << "  Num Streams: " << streams << " (expected: 4)" << std::endl;
			std::cout << "  Block Size: " << blockSize << " (expected: 256)" << std::endl;
			std::cout << "  Num Buffers: " << buffers << " (expected: 3)" << std::endl;
			std::cout << "  Result: " << (test6Pass ? "PASS" : "FAIL") << std::endl;
			if (test6Pass) passedTests++;
		} catch (const std::exception& e) {
			std::cout << "  ERROR: " << e.what() << std::endl;
			std::cout << "  Result: FAIL" << std::endl;
		}
		std::cout << std::endl;
	}

	// ============================================
	// Test 7: Processor CUDA settings (after init - should fail)
	// ============================================
	if (cudaAvailable && deviceCount > 0) {
		std::cout << "Test 7: Processor CUDA settings (after init - should throw)" << std::endl;
		totalTests++;

		try {
			ope::Processor processor(ope::Backend::CUDA);
			processor.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
			processor.initialize();

			// Try to change settings after init (should throw)
			bool exceptionThrown = false;
			try {
				processor.setCudaDevice(0);
			} catch (const std::runtime_error&) {
				exceptionThrown = true;
			}

			bool test7Pass = exceptionThrown;
			std::cout << "  Exception thrown as expected: " << (exceptionThrown ? "YES" : "NO") << std::endl;
			std::cout << "  Result: " << (test7Pass ? "PASS" : "FAIL") << std::endl;
			if (test7Pass) passedTests++;

			processor.cleanup();
		} catch (const std::exception& e) {
			std::cout << "  ERROR: " << e.what() << std::endl;
			std::cout << "  Result: FAIL" << std::endl;
		}
		std::cout << std::endl;
	}

	// ============================================
	// Test 8: CUDA settings file I/O
	// ============================================
	if (cudaAvailable && deviceCount > 0) {
		std::cout << "Test 8: CUDA settings file I/O" << std::endl;
		totalTests++;

		const std::string testFile = "test_cuda_settings.ini";

		try {
			// Create processor and set CUDA settings
			ope::Processor processor1(ope::Backend::CUDA);
			processor1.setCudaDevice(0);
			processor1.setCudaNumStreams(6);
			processor1.setCudaBlockSize(512);
			processor1.setNumBuffers(4);

			// Save settings
			processor1.saveCudaSettingsToFile(testFile);

			// Create new processor and load settings
			ope::Processor processor2(ope::Backend::CUDA);
			processor2.loadCudaSettingsFromFile(testFile);

			// Verify settings match
			bool test8Pass =
				(processor2.getCudaDevice() == 0) &&
				(processor2.getCudaNumStreams() == 6) &&
				(processor2.getCudaBlockSize() == 512) &&
				(processor2.getNumBuffers() == 4);

			std::cout << "  Device ID: " << processor2.getCudaDevice() << " (expected: 0)" << std::endl;
			std::cout << "  Num Streams: " << processor2.getCudaNumStreams() << " (expected: 6)" << std::endl;
			std::cout << "  Block Size: " << processor2.getCudaBlockSize() << " (expected: 512)" << std::endl;
			std::cout << "  Num Buffers: " << processor2.getNumBuffers() << " (expected: 4)" << std::endl;
			std::cout << "  Result: " << (test8Pass ? "PASS" : "FAIL") << std::endl;
			if (test8Pass) passedTests++;

			// Cleanup test file
			std::remove(testFile.c_str());
		} catch (const std::exception& e) {
			std::cout << "  ERROR: " << e.what() << std::endl;
			std::cout << "  Result: FAIL" << std::endl;
			std::remove(testFile.c_str());
		}
		std::cout << std::endl;
	}

	// ============================================
	// Test 9: CPU backend - CUDA methods should return safe defaults
	// ============================================
	std::cout << "Test 9: CPU backend - CUDA getters return safe defaults" << std::endl;
	totalTests++;

	try {
		ope::Processor cpuProcessor(ope::Backend::CPU);

		int device = cpuProcessor.getCudaDevice();
		int streams = cpuProcessor.getCudaNumStreams();
		int blockSize = cpuProcessor.getCudaBlockSize();
		int gridSize = cpuProcessor.getCudaGridSize();

		bool test9Pass = (device == -1) && (streams == 0) && (blockSize == 0) && (gridSize == 0);

		std::cout << "  Device ID: " << device << " (expected: -1)" << std::endl;
		std::cout << "  Num Streams: " << streams << " (expected: 0)" << std::endl;
		std::cout << "  Block Size: " << blockSize << " (expected: 0)" << std::endl;
		std::cout << "  Grid Size: " << gridSize << " (expected: 0)" << std::endl;
		std::cout << "  Result: " << (test9Pass ? "PASS" : "FAIL") << std::endl;
		if (test9Pass) passedTests++;
	} catch (const std::exception& e) {
		std::cout << "  ERROR: " << e.what() << std::endl;
		std::cout << "  Result: FAIL" << std::endl;
	}
	std::cout << std::endl;

	// ============================================
	// Test 10: CPU backend - CUDA setters should throw
	// ============================================
	std::cout << "Test 10: CPU backend - CUDA setters should throw" << std::endl;
	totalTests++;

	try {
		ope::Processor cpuProcessor(ope::Backend::CPU);

		bool exceptionThrown = false;
		try {
			cpuProcessor.setCudaDevice(0);
		} catch (const std::runtime_error&) {
			exceptionThrown = true;
		}

		bool test10Pass = exceptionThrown;
		std::cout << "  Exception thrown as expected: " << (exceptionThrown ? "YES" : "NO") << std::endl;
		std::cout << "  Result: " << (test10Pass ? "PASS" : "FAIL") << std::endl;
		if (test10Pass) passedTests++;
	} catch (const std::exception& e) {
		std::cout << "  ERROR: " << e.what() << std::endl;
		std::cout << "  Result: FAIL" << std::endl;
	}
	std::cout << std::endl;

	// ============================================
	// Summary
	// ============================================
	std::cout << "========================================" << std::endl;
	std::cout << "Test Summary" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Total Tests: " << totalTests << std::endl;
	std::cout << "Passed: " << passedTests << std::endl;
	std::cout << "Failed: " << (totalTests - passedTests) << std::endl;
	std::cout << "Success Rate: " << (100.0 * passedTests / totalTests) << "%" << std::endl;
	std::cout << std::endl;

	if (passedTests == totalTests) {
		std::cout << "ALL TESTS PASSED!" << std::endl;
		return 0;
	} else {
		std::cout << "SOME TESTS FAILED!" << std::endl;
		return 1;
	}
}
