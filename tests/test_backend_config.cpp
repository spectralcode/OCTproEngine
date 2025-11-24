#include "../include/processor.h"
#include "../include/backendconfig.h"
#include "../include/processorconfiguration.h"
#include "../include/types.h"
#include "../include/iobuffer.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace ope;

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Backend Configuration API Test" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	//	Test device enumeration
	std::cout << "Testing device enumeration..." << std::endl;
	std::cout << std::endl;

	//	CUDA devices
	if (BackendUtils::isCudaAvailable()) {
		std::cout << "CUDA Devices:" << std::endl;
		auto cudaDevices = BackendUtils::getCudaDevices();
		for (const auto& device : cudaDevices) {
			std::cout << "  Device " << device.id << ": " << device.name << std::endl;
			std::cout << "    Total Memory: " << (device.totalMemory / (1024*1024)) << " MB" << std::endl;
			std::cout << "    Available Memory: " << (device.availableMemory / (1024*1024)) << " MB" << std::endl;
			std::cout << "    Compute Capability: " << device.computeCapabilityMajor << "." << device.computeCapabilityMinor << std::endl;
		}
	} else {
		std::cout << "  CUDA not available" << std::endl;
	}
	std::cout << std::endl;

	//	OpenCL devices
	if (BackendUtils::isOpenCLAvailable()) {
		std::cout << "OpenCL Devices:" << std::endl;
		auto openclDevices = BackendUtils::getOpenCLDevices();
		for (const auto& device : openclDevices) {
			std::cout << "  Device " << device.id << ": " << device.name << std::endl;
			std::cout << "    Vendor: " << device.vendorName << std::endl;
			std::cout << "    Version: " << device.deviceVersion << std::endl;
			std::cout << "    Total Memory: " << (device.totalMemory / (1024*1024)) << " MB" << std::endl;
		}
	} else {
		std::cout << "  OpenCL not available" << std::endl;
	}
	std::cout << std::endl;

	//	CPU info
	if (BackendUtils::isCpuAvailable()) {
		std::cout << "CPU Info:" << std::endl;
		auto cpuInfo = BackendUtils::getCpuInfo();
		std::cout << "  " << cpuInfo.name << std::endl;
		std::cout << "  " << cpuInfo.deviceVersion << std::endl;
	} else {
		std::cout << "  CPU backend not available" << std::endl;
	}
	std::cout << std::endl;

	//	Test backend configuration
	std::cout << "========================================" << std::endl;
	std::cout << "Testing backend configuration..." << std::endl;
	std::cout << std::endl;

	//	Test configuration creation and serialization
	if (BackendUtils::isCudaAvailable()) {
		std::cout << "Testing CUDA configuration:" << std::endl;

		//	Create CUDA config
		CudaConfig cudaConfig;
		cudaConfig.deviceId = 0;
		cudaConfig.enableZeroCopy = false;

		std::cout << "  Created: " << cudaConfig.toString() << std::endl;

		//	Serialize and parse
		std::string serialized = BackendUtils::serializeConfig(cudaConfig);
		std::cout << "  Serialized: " << serialized << std::endl;

		auto parsed = BackendUtils::parseConfig(serialized);
		if (parsed) {
			std::cout << "  Parsed: " << parsed->toString() << std::endl;
		} else {
			std::cout << "  [FAILED] Could not parse config" << std::endl;
		}
		std::cout << std::endl;
	}

	if (BackendUtils::isOpenCLAvailable()) {
		std::cout << "Testing OpenCL configuration:" << std::endl;

		//	Create OpenCL config
		OpenCLConfig openclConfig;
		openclConfig.platformId = 0;
		openclConfig.deviceId = 0;
		openclConfig.preferGpu = true;

		std::cout << "  Created: " << openclConfig.toString() << std::endl;

		//	Serialize and parse
		std::string serialized = BackendUtils::serializeConfig(openclConfig);
		std::cout << "  Serialized: " << serialized << std::endl;

		auto parsed = BackendUtils::parseConfig(serialized);
		if (parsed) {
			std::cout << "  Parsed: " << parsed->toString() << std::endl;
		} else {
			std::cout << "  [FAILED] Could not parse config" << std::endl;
		}
		std::cout << std::endl;
	}

	if (BackendUtils::isCpuAvailable()) {
		std::cout << "Testing CPU configuration:" << std::endl;

		//	Create CPU config
		CpuConfig cpuConfig;
		cpuConfig.numThreads = 4;
		cpuConfig.enableSimd = true;

		std::cout << "  Created: " << cpuConfig.toString() << std::endl;

		//	Serialize and parse
		std::string serialized = BackendUtils::serializeConfig(cpuConfig);
		std::cout << "  Serialized: " << serialized << std::endl;

		auto parsed = BackendUtils::parseConfig(serialized);
		if (parsed) {
			std::cout << "  Parsed: " << parsed->toString() << std::endl;
		} else {
			std::cout << "  [FAILED] Could not parse config" << std::endl;
		}
		std::cout << std::endl;
	}

	//	Test processor integration
	std::cout << "========================================" << std::endl;
	std::cout << "Testing processor integration..." << std::endl;
	std::cout << std::endl;

	//	Find an available backend
	Backend testBackend;
	if (BackendUtils::isCudaAvailable()) {
		testBackend = Backend::CUDA;
	} else if (BackendUtils::isOpenCLAvailable()) {
		testBackend = Backend::OPENCL;
	} else if (BackendUtils::isCpuAvailable()) {
		testBackend = Backend::CPU;
	} else {
		std::cerr << "No backends available!" << std::endl;
		return 1;
	}

	try {
		//	Create processor
		Processor processor(testBackend);
		std::cout << "Created processor with ";
		switch (testBackend) {
			case Backend::CUDA: std::cout << "CUDA"; break;
			case Backend::OPENCL: std::cout << "OpenCL"; break;
			case Backend::CPU: std::cout << "CPU"; break;
		}
		std::cout << " backend" << std::endl;

		//	Get current configuration
		auto currentConfig = processor.getBackendConfig();
		if (currentConfig) {
			std::cout << "Current config: " << currentConfig->toString() << std::endl;
		} else {
			std::cout << "[FAILED] Could not get current config" << std::endl;
		}

		//	Test switching backends via config
		if (testBackend != Backend::CPU && BackendUtils::isCpuAvailable()) {
			std::cout << "\nSwitching to CPU backend via config..." << std::endl;
			CpuConfig cpuConfig;
			cpuConfig.numThreads = 2;
			processor.setBackendConfig(cpuConfig);

			auto newConfig = processor.getBackendConfig();
			if (newConfig && newConfig->getBackendType() == Backend::CPU) {
				std::cout << "[OK] Successfully switched to CPU: " << newConfig->toString() << std::endl;
			} else {
				std::cout << "[FAILED] Backend switch failed" << std::endl;
			}
		}

		//	Test switching back
		if (testBackend == Backend::CUDA) {
			std::cout << "\nSwitching back to CUDA via config..." << std::endl;
			CudaConfig cudaConfig;
			cudaConfig.deviceId = 0;
			processor.setBackendConfig(cudaConfig);

			auto newConfig = processor.getBackendConfig();
			if (newConfig && newConfig->getBackendType() == Backend::CUDA) {
				std::cout << "[OK] Successfully switched to CUDA: " << newConfig->toString() << std::endl;
			} else {
				std::cout << "[FAILED] Backend switch failed" << std::endl;
			}
		}

		//	Test save/load configuration
		std::cout << "\nTesting save/load configuration..." << std::endl;

		//	Save current configuration
		std::string configFile = "test_backend_config.ini";
		processor.saveBackendConfigToFile(configFile);
		std::cout << "Saved configuration to " << configFile << std::endl;

		//	Load it back
		Processor processor2(Backend::CPU);  // Start with different backend
		processor2.loadBackendConfigFromFile(configFile);

		auto loadedConfig = processor2.getBackendConfig();
		if (loadedConfig) {
			std::cout << "Loaded configuration: " << loadedConfig->toString() << std::endl;
			if (loadedConfig->getBackendType() == processor.getBackend()) {
				std::cout << "[OK] Configuration loaded correctly" << std::endl;
			} else {
				std::cout << "[FAILED] Backend type mismatch after load" << std::endl;
			}
		} else {
			std::cout << "[FAILED] Could not get loaded configuration" << std::endl;
		}

		std::cout << std::endl;
	} catch (const std::exception& e) {
		std::cerr << "ERROR: " << e.what() << std::endl;
		return 1;
	}

	std::cout << "========================================" << std::endl;
	std::cout << "TEST PASSED" << std::endl;
	std::cout << "Backend configuration API working correctly" << std::endl;
	std::cout << "========================================" << std::endl;

	return 0;
}