// Basic OCTproEngine example - minimal version

#include "processor.h"
#include "iobuffer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>
#include <atomic>

const int SIGNAL_LENGTH = 2048;
const int ASCANS_PER_BSCAN = 512;

std::atomic<bool> processingDone{false};

// Callback function, called when processing completes
void onProcessingComplete(const ope::IOBuffer& output) {
	std::cout << "Processed: " << output.getSizeInBytes() << " bytes" << std::endl;
	processingDone = true;
}

// Generate simple synthetic data
std::vector<uint16_t> generateTestData() {
	std::vector<uint16_t> data(SIGNAL_LENGTH * ASCANS_PER_BSCAN);
	for (size_t i = 0; i < data.size(); ++i) {
		double val = 2000.0 + 5000.0 * std::sin(i * 0.01) + (rand() % 1000);
		data[i] = static_cast<uint16_t>(std::max(0.0, std::min(65535.0, val)));
	}
	return data;
}

int main() {
	std::cout << "OCTproEngine Basic Example" << std::endl;
	
	// Create and configure processor
	ope::Processor processor(ope::Backend::CUDA);
	processor.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, 1, ope::DataType::UINT16);
	processor.enableResampling(true);
	processor.enableWindowing(true);
	processor.enableLogScaling(true);
	processor.initialize();
	
	// Set callback
	processor.setOutputCallback(onProcessingComplete);
	
	// Generate data, get buffer, process
	auto testData = generateTestData();
	ope::IOBuffer& buffer = processor.getNextAvailableInputBuffer();
	std::memcpy(buffer.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(buffer); 
	
	// Wait for callback (only needed because this example processes one frame and exits)
	// Real applications continuously process in a loop without waiting 
	while (!processingDone) std::this_thread::sleep_for(std::chrono::milliseconds(1));
	
	std::cout << "Done!" << std::endl;
	return 0;
}