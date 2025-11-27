#include "tools/recorder.h"
#include "processor.h"
#include <iostream>
#include <cstring>

// Example demonstrating the Recorder tool

const ope::Backend TEST_BACKEND = ope::Backend::CUDA;

void example_basic_recording() {
	std::cout << "=== Example 1: Basic Recording ===" << std::endl;

	// Create and configure processor
	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(
		1024,  // samplesPerRawAscan
		512,   // ascansPerBscan
		1,     // bscansPerBuffer
		ope::DataType::UINT16
	);
	processor.initialize();

	// Create recorder and attach to processor
	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);

	// Configure recording
	recorder.setMode(ope::tools::Recorder::Mode::BOTH); // BOTH will record raw and processed data
	recorder.setBufferCount(20); // set how many buffers to record
	recorder.setOutputBaseName("example_basic"); // base name for output files. use setOutputDirectory() to change directory if needed

	// Optional: Pre-allocate buffers to avoid allocation overhead during startRecording()
	recorder.setManualAllocation(true);
	recorder.allocateBuffers();

	// Start recording
	recorder.startRecording();
	std::cout << "Recording started..." << std::endl;
	std::cout << "Buffers to record: " << recorder.getBufferCount() << std::endl;

	// Start Processing
	for (int i = 0; i < 30; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();

		// Fill buffer with data (in real application, this comes from your OCT hardware)
		// Here we just fill with dummy data
		memset(inputBuffer.getDataPointer(), i, inputBuffer.getSizeInBytes());

		processor.process(inputBuffer);
		std::cout << "  Processed buffer " << (i + 1) << "/10" << std::endl;
	}

	// Wait for recording to complete (auto-completes at 10 buffers)
	std::cout << "Waiting for recording to complete..." << std::endl;
	bool success = recorder.waitForCompletion(10000); 
	if (success) {
		std::cout << "Recording complete!" << std::endl;
		auto summary = recorder.getLastRecordingSummary();
		std::cout << "  Raw buffers recorded: " << summary.rawRecorded << std::endl;
		std::cout << "  Processed buffers recorded: " << summary.processedRecorded << std::endl;
	} else {
		std::cerr << "Recording failed or timed out!" << std::endl;
		std::cerr << "  Error: " << recorder.getLastError() << std::endl;

		//print buffer ids collected before failure
		auto summary = recorder.getLastRecordingSummary();
		std::cout << "  Raw buffers recorded: " << summary.rawRecorded << std::endl;
		std::cout << "  Processed buffers recorded: " << summary.processedRecorded << std::endl;	
		// You can also print buffer IDs here if needed
		auto rawBufferIds = summary.rawBufferIds;
		auto processedBufferIds = summary.processedBufferIds;
		std::cout << "  Raw Buffer IDs: ";
		for (const auto& id : rawBufferIds) {
			std::cout << id << " ";
		}
		std::cout << std::endl;
		std::cout << "  Processed Buffer IDs: ";
		for (const auto& id : processedBufferIds) {
			std::cout << id << " ";
		}
		std::cout << std::endl;

	}

	std::cout << std::endl;
}

int main() {
	example_basic_recording();

	return 0;
}