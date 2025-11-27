#include "../include/processor.h"
#include "../include/backendconfig.h"
#include <iostream>

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Zero-Copy Runtime Switching Test" << std::endl;
	std::cout << "========================================" << std::endl;

#ifdef __aarch64__
	std::cout << "Platform: Jetson (aarch64)" << std::endl;
#else
	std::cout << "Platform: Desktop (zero-copy has no effect)" << std::endl;
#endif
	std::cout << std::endl;

	try {
		ope::Processor processor(ope::Backend::CUDA);
		processor.setInputParameters(512, 256, 1, ope::DataType::UINT16);

		// Test 1: Portable mode
		std::cout << "[1/3] Testing portable mode..." << std::endl;
		ope::CudaConfig config1;
		config1.enableZeroCopy = false;
		processor.setBackendConfig(config1);
		processor.initialize();

		auto& buffer1 = processor.getNextAvailableInputBuffer();
		if (buffer1.getAllocationHint() != ope::IOBuffer::AllocationHint::PORTABLE) {
			std::cout << "      FAIL: Expected PORTABLE hint" << std::endl;
			return 1;
		}

		processor.cleanup();
		std::cout << "      PASS" << std::endl;

		// Test 2: Zero-copy mode
		std::cout << "[2/3] Testing zero-copy mode..." << std::endl;
		ope::CudaConfig config2;
		config2.enableZeroCopy = true;
		processor.setBackendConfig(config2);
		processor.initialize();

		auto& buffer2 = processor.getNextAvailableInputBuffer();
		if (buffer2.getAllocationHint() != ope::IOBuffer::AllocationHint::DEVICE_MAPPED) {
			std::cout << "      FAIL: Expected DEVICE_MAPPED hint" << std::endl;
			return 1;
		}

		processor.cleanup();
		std::cout << "      PASS" << std::endl;

		// Test 3: Switch back
		std::cout << "[3/3] Testing switch back to portable..." << std::endl;
		ope::CudaConfig config3;
		config3.enableZeroCopy = false;
		processor.setBackendConfig(config3);
		processor.initialize();

		auto& buffer3 = processor.getNextAvailableInputBuffer();
		if (buffer3.getAllocationHint() != ope::IOBuffer::AllocationHint::PORTABLE) {
			std::cout << "      FAIL: Expected PORTABLE hint after switching back" << std::endl;
			return 1;
		}

		processor.cleanup();
		std::cout << "      PASS" << std::endl;

		std::cout << std::endl;
		std::cout << "========================================" << std::endl;
		std::cout << "TEST PASSED" << std::endl;
		std::cout << "========================================" << std::endl;
		return 0;

	} catch (const std::exception& e) {
		std::cout << std::endl;
		std::cout << "TEST FAILED: " << e.what() << std::endl;
		return 1;
	}
}
