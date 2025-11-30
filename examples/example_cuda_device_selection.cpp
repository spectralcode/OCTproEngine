/**
 * Example demonstrating CUDA multi-device support
 *
 * Shows how to:
 * 1. Enumerate available CUDA devices
 * 2. Create multiple Processor instances on different GPUs
 * 3. Process data simultaneously on multiple devices
 */

#include "../include/processor.h"
#include "../include/backendconfig.h"
#include <iostream>
#include <vector>

using namespace ope;

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "CUDA Multi-Device Example" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	// Step 1: Check if CUDA is available
	if (!BackendUtils::isCudaAvailable()) {
		std::cout << "CUDA is not available on this system." << std::endl;
		return 0;
	}

	// Step 2: Query available CUDA devices
	std::cout << "Querying CUDA devices..." << std::endl;
	std::cout << std::endl;

	auto devices = BackendUtils::getCudaDevices();

	if (devices.empty()) {
		std::cout << "No CUDA devices found." << std::endl;
		return 0;
	}

	std::cout << "Found " << devices.size() << " CUDA device(s):" << std::endl;
	std::cout << std::endl;

	// Display information about each device
	for (const auto& device : devices) {
		std::cout << "Device " << device.id << ": " << device.name << std::endl;
		std::cout << "  Total Memory: " << (device.totalMemory / (1024*1024)) << " MB" << std::endl;
		std::cout << "  Available Memory: " << (device.availableMemory / (1024*1024)) << " MB" << std::endl;
		std::cout << "  Compute Capability: " << device.computeCapabilityMajor
		          << "." << device.computeCapabilityMinor << std::endl;
		std::cout << std::endl;
	}

	// Step 3: Create processors on different devices
	try {
		if (devices.size() == 1) {
			std::cout << "Single device system - creating one processor..." << std::endl;
			std::cout << std::endl;

			// Single device example
			Processor processor(Backend::CUDA);

			CudaConfig config;
			config.deviceId = 0;
			processor.setBackendConfig(config);

			processor.setInputParameters(
				2048,  // samplesPerRawAscan
				512,   // ascansPerBscan
				1,     // bscansPerBuffer
				DataType::UINT16
			);

			processor.initialize();
			std::cout << "Processor initialized successfully on device 0" << std::endl;

			processor.cleanup();

		} else {
			std::cout << "Multi-device system - creating processors on devices 0 and 1..." << std::endl;
			std::cout << std::endl;

			// Multi-device example: Create two processors on different GPUs
			Processor processor1(Backend::CUDA);
			Processor processor2(Backend::CUDA);

			// Configure first processor to use device 0
			CudaConfig config1;
			config1.deviceId = 0;
			processor1.setBackendConfig(config1);

			// Configure second processor to use device 1
			CudaConfig config2;
			config2.deviceId = 1;
			processor2.setBackendConfig(config2);

			std::cout << "Configured processor1 for device 0" << std::endl;
			std::cout << "Configured processor2 for device 1" << std::endl;
			std::cout << std::endl;

			// Set same parameters for both processors
			processor1.setInputParameters(
				2048,  // samplesPerRawAscan
				512,   // ascansPerBscan
				1,     // bscansPerBuffer
				DataType::UINT16
			);

			processor2.setInputParameters(
				2048,  // samplesPerRawAscan
				512,   // ascansPerBscan
				1,     // bscansPerBuffer
				DataType::UINT16
			);

			// Initialize both processors
			processor1.initialize();
			std::cout << "Processor 1 initialized on GPU " << config1.deviceId << std::endl;

			processor2.initialize();
			std::cout << "Processor 2 initialized on GPU " << config2.deviceId << std::endl;
			std::cout << std::endl;

			std::cout << "Both processors can now process data simultaneously!" << std::endl;
			std::cout << "Each processor will use its configured GPU device." << std::endl;
			std::cout << std::endl;

			// Note: Each processor maintains its device selection throughout its lifetime
			// You can call process() on both processors and they will run on their
			// respective GPUs without interfering with each other

			// Cleanup
			processor1.cleanup();
			processor2.cleanup();
		}

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "========================================" << std::endl;
	std::cout << "Example completed successfully!" << std::endl;
	std::cout << "========================================" << std::endl;

	return 0;
}