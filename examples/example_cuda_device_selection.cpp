/**
 * Minimal example demonstrating CUDA device enumeration and selection
 * using the unified backend configuration API
 */

#include "../include/processor.h"
#include "../include/backendconfig.h"
#include <iostream>

using namespace ope;

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "CUDA Device Selection Example" << std::endl;
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

	// Step 3: Select a specific device
	int selectedDevice = 0;  // Select first device

	if (devices.size() > 1) {
		std::cout << "Multiple devices available. Selecting device " << selectedDevice << "." << std::endl;
	}

	std::cout << "Creating processor with CUDA device " << selectedDevice << "..." << std::endl;

	// Step 4: Create processor with specific CUDA device
	try {
		// Create processor (can start with any backend)
		Processor processor(Backend::CPU);

		// Create CUDA configuration
		CudaConfig cudaConfig;
		cudaConfig.deviceId = selectedDevice;
		cudaConfig.enableZeroCopy = false;  // Optional: configure zero-copy mode, only for Jetson devices

		// Apply configuration (this will switch to CUDA backend)
		processor.setBackendConfig(cudaConfig);

		// Verify the configuration
		auto currentConfig = processor.getBackendConfig();
		if (currentConfig) {
			std::cout << "Processor configured with: " << currentConfig->toString() << std::endl;
		}

		// Step 5: Initialize and use the processor
		processor.setInputParameters(
			2048,  // samplesPerRawAscan
			512,   // ascansPerBscan
			1,     // bscansPerBuffer
			DataType::UINT16
		);

		processor.initialize();
		std::cout << "Processor initialized successfully on CUDA device " << selectedDevice << std::endl;

		// Processor is now ready to use with the selected CUDA device
		// ... your processing code here ...

		// Cleanup
		processor.cleanup();

	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	std::cout << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Example completed successfully!" << std::endl;
	std::cout << "========================================" << std::endl;

	return 0;
}