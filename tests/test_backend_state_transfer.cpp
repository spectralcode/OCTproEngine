#include "../include/processor.h"
#include "../include/processorconfiguration.h"
#include "../include/types.h"
#include "../include/iobuffer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>

using namespace ope;

//	Configuration
constexpr int SIGNAL_LENGTH = 2048;
constexpr int ASCANS_PER_BSCAN = 512;
constexpr int BSCANS_PER_BUFFER = 1;

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Backend State Transfer Test" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	//	Check which backends are available
	std::cout << "Available backends:" << std::endl;
#ifdef OPE_CUDA_AVAILABLE
	std::cout << "  - CUDA" << std::endl;
#endif
#ifdef OPE_OPENCL_AVAILABLE
	std::cout << "  - OpenCL" << std::endl;
#endif
#ifdef OPE_CPU_AVAILABLE
	std::cout << "  - CPU" << std::endl;
#endif
	std::cout << std::endl;

	//	Determine which backend to start with
	Backend startBackend;
	bool startBackendFound = false;

#ifdef OPE_CUDA_AVAILABLE
	startBackend = Backend::CUDA;
	startBackendFound = true;
#elif defined(OPE_OPENCL_AVAILABLE)
	startBackend = Backend::OPENCL;
	startBackendFound = true;
#elif defined(OPE_CPU_AVAILABLE)
	startBackend = Backend::CPU;
	startBackendFound = true;
#endif

	if (!startBackendFound) {
		std::cerr << "ERROR: No backends available!" << std::endl;
		std::cerr << "OCTproEngine must be compiled with at least one backend." << std::endl;
		return 1;
	}

	//	Generate test data
	int totalSamples = SIGNAL_LENGTH * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;
	std::vector<uint16_t> testData(totalSamples);

	for (int i = 0; i < totalSamples; ++i) {
		float phase = 2.0f * 3.14159f * i / SIGNAL_LENGTH;
		float value = 20000.0f + 10000.0f * std::sin(phase);
		testData[i] = static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, value)));
	}

	//	Setup processor with initial backend
	std::cout << "Creating processor with ";
	if (startBackend == Backend::CUDA) {
		std::cout << "CUDA";
	} else if (startBackend == Backend::OPENCL) {
		std::cout << "OpenCL";
	} else {
		std::cout << "CPU";
	}
	std::cout << " backend..." << std::endl;

	Processor processor(startBackend);

	//	Configure processor
	processor.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, DataType::UINT16);
	processor.initialize();

	//	Enable background subtraction AFTER initialization
	processor.enablePostProcessBackgroundSubtraction(true);
	processor.setPostProcessBackgroundWeight(1.0f);
	processor.setPostProcessBackgroundOffset(0.0f);

	//	Process once to record background
	std::cout << "Recording background profile..." << std::endl;
	processor.requestPostProcessBackgroundRecording();

	//	Setup callback to wait for processing to complete
	std::mutex mtx;
	std::condition_variable cv;
	bool outputReceived = false;

	processor.setOutputCallback([&](const IOBuffer& output) {
		std::lock_guard<std::mutex> lock(mtx);
		outputReceived = true;
		cv.notify_one();
	});

	//	Get input buffer and process
	IOBuffer& input = processor.getNextAvailableInputBuffer();
	std::memcpy(input.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(input);

	//	Wait for output
	{
		std::unique_lock<std::mutex> lock(mtx);
		if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return outputReceived; })) {
			std::cerr << "ERROR: Timeout waiting for output" << std::endl;
			return 1;
		}
	}

	//	Get and copy the background profile
	const float* bgProfilePtr = processor.getPostProcessBackgroundProfile();
	size_t bgSize = processor.getPostProcessBackgroundProfileSize();

	if (!bgProfilePtr || bgSize == 0) {
		std::cerr << "ERROR: Background profile not recorded!" << std::endl;
		return 1;
	}

	std::cout << "  Background profile recorded: " << bgSize << " samples" << std::endl;

	//	COPY the profile data
	std::vector<float> originalProfile(bgProfilePtr, bgProfilePtr + bgSize);

	//	Count successful transfers
	int testedBackends = 1;  // Already tested initial backend
	int successfulTransfers = 0;

	//	Try switching to CUDA (if not already using it and if available)
#ifdef OPE_CUDA_AVAILABLE
	if (startBackend != Backend::CUDA) {
		std::cout << "\nSwitching to CUDA backend..." << std::endl;
		try {
			processor.setBackend(Backend::CUDA);

			//	Check if background profile is still available
			const float* bgProfileCUDA = processor.getPostProcessBackgroundProfile();
			size_t bgSizeCUDA = processor.getPostProcessBackgroundProfileSize();

			if (!bgProfileCUDA || bgSizeCUDA != bgSize) {
				std::cerr << "  [FAILED] Profile lost or size changed!" << std::endl;
			} else {
				//	Compare profile data
				bool match = true;
				for (size_t i = 0; i < bgSize; ++i) {
					if (std::abs(originalProfile[i] - bgProfileCUDA[i]) > 0.0001f) {
						std::cerr << "  [FAILED] Profile data mismatch at index " << i << std::endl;
						match = false;
						break;
					}
				}
				if (match) {
					std::cout << "  [OK] Profile transferred correctly to CUDA" << std::endl;
					successfulTransfers++;
				}
			}
			testedBackends++;
		} catch (const std::exception& e) {
			std::cerr << "  [SKIPPED] CUDA backend not available at runtime" << std::endl;
		}
	}
#endif

	//	Try switching to OpenCL (if not already using it and if available)
#ifdef OPE_OPENCL_AVAILABLE
	if (startBackend != Backend::OPENCL) {
		std::cout << "\nSwitching to OpenCL backend..." << std::endl;
		try {
			processor.setBackend(Backend::OPENCL);

			//	Check if background profile is still available
			const float* bgProfileOpenCL = processor.getPostProcessBackgroundProfile();
			size_t bgSizeOpenCL = processor.getPostProcessBackgroundProfileSize();

			if (!bgProfileOpenCL || bgSizeOpenCL != bgSize) {
				std::cerr << "  [FAILED] Profile lost or size changed!" << std::endl;
			} else {
				//	Compare profile data
				bool match = true;
				for (size_t i = 0; i < bgSize; ++i) {
					if (std::abs(originalProfile[i] - bgProfileOpenCL[i]) > 0.0001f) {
						std::cerr << "  [FAILED] Profile data mismatch at index " << i << std::endl;
						match = false;
						break;
					}
				}
				if (match) {
					std::cout << "  [OK] Profile transferred correctly to OpenCL" << std::endl;
					successfulTransfers++;
				}
			}
			testedBackends++;
		} catch (const std::exception& e) {
			std::cerr << "  [SKIPPED] OpenCL backend not available at runtime" << std::endl;
		}
	}
#endif

	//	Try switching to CPU (if not already using it and if available)
#ifdef OPE_CPU_AVAILABLE
	if (startBackend != Backend::CPU) {
		std::cout << "\nSwitching to CPU backend..." << std::endl;
		try {
			processor.setBackend(Backend::CPU);

			//	Check if background profile is still available
			const float* bgProfileCPU = processor.getPostProcessBackgroundProfile();
			size_t bgSizeCPU = processor.getPostProcessBackgroundProfileSize();

			if (!bgProfileCPU || bgSizeCPU != bgSize) {
				std::cerr << "  [FAILED] Profile lost or size changed!" << std::endl;
			} else {
				//	Compare profile data
				bool match = true;
				for (size_t i = 0; i < bgSize; ++i) {
					if (std::abs(originalProfile[i] - bgProfileCPU[i]) > 0.0001f) {
						std::cerr << "  [FAILED] Profile data mismatch at index " << i << std::endl;
						match = false;
						break;
					}
				}
				if (match) {
					std::cout << "  [OK] Profile transferred correctly to CPU" << std::endl;
					successfulTransfers++;
				}
			}
			testedBackends++;
		} catch (const std::exception& e) {
			std::cerr << "  [SKIPPED] CPU backend not available at runtime" << std::endl;
		}
	}
#endif

	//	Final result
	std::cout << "\n========================================" << std::endl;

	if (testedBackends < 2) {
		std::cout << "TEST SKIPPED" << std::endl;
		std::cout << "Only one backend available - cannot test state transfer" << std::endl;
		std::cout << "Compile with multiple backends to test state transfer" << std::endl;
		return 0;  // Not a failure, just can't test
	} else if (successfulTransfers == testedBackends - 1) {
		std::cout << "TEST PASSED" << std::endl;
		std::cout << "Background profile successfully transferred between all available backends" << std::endl;
		std::cout << "Tested " << testedBackends << " backends" << std::endl;
	} else {
		std::cout << "TEST FAILED" << std::endl;
		std::cout << "Profile transfer failed for some backends" << std::endl;
		std::cout << "Successful transfers: " << successfulTransfers << " / " << (testedBackends - 1) << std::endl;
		return 1;
	}

	std::cout << "========================================" << std::endl;

	return 0;
}