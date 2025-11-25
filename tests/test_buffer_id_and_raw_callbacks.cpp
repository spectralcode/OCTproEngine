#include "../include/processor.h"
#include "../include/processortool.h"
#include <iostream>
#include <vector>
#include "test_utils.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <map>

const ope::Backend TEST_BACKEND = ope::Backend::CUDA;

// Simple tool for testing that collects buffer IDs
class TestRecorderTool : public ope::ProcessorTool {
public:
	struct BufferRecord {
		uint64_t bufferId;
		bool isRaw;
		size_t dataSize;
	};

	std::vector<BufferRecord> recordedBuffers;
	std::map<uint64_t, int> idMatchCount;  // Tracks how many times each ID appears

	void recordBuffer(const ope::IOBuffer& buffer, bool isRaw) {
		BufferRecord record;
		record.bufferId = buffer.getBufferId();
		record.isRaw = isRaw;
		record.dataSize = buffer.getSizeInBytes();
		recordedBuffers.push_back(record);

		// Track ID appearances
		idMatchCount[record.bufferId]++;
	}

protected:
	void configureCallbacks() override {
		if (!processor) return;

		// Register both input and output callbacks
		rawCallbackId = processor->addInputCallback(
			[this](const ope::IOBuffer& buf) {
				recordBuffer(buf, true);
			}
		);

		processedCallbackId = processor->addOutputCallback(
			[this](const ope::IOBuffer& buf) {
				recordBuffer(buf, false);
			}
		);
	}
};

void testBufferIdPropagation() {
	std::cout << "Testing buffer ID propagation..." << std::endl;

	// Create processor
	ope::Processor processor(TEST_BACKEND);

	processor.setInputParameters(
		1024,  // samplesPerAscan
		512,   // ascansPerBscan
		2,     // bscansPerBuffer
		ope::DataType::UINT16
	);

	processor.initialize();

	// Create and attach test tool
	TestRecorderTool recorder;
	recorder.attachToProcessor(&processor);

	// Process some buffers
	const int numBuffersToProcess = 8;
	for (int i = 0; i < numBuffersToProcess; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();

		// Fill with test data (not important for this test)
		uint16_t* data = static_cast<uint16_t*>(inputBuffer.getDataPointer());
		int size = processor.getConfig().dataParams.signalLength * processor.getConfig().dataParams.ascansPerBscan * processor.getConfig().dataParams.bscansPerBuffer;
		for (int j = 0; j < size; j++) {
			data[j] = static_cast<uint16_t>(i * 100 + j);
		}
		processor.process(inputBuffer);
	}

	// Wait for processing to complete
	std::this_thread::sleep_for(std::chrono::milliseconds(1750));

	// Verify results
	std::cout << "  Recorded " << recorder.recordedBuffers.size() << " buffers" << std::endl;

	// We should have received both raw and processed for each buffer
	TEST_ASSERT(recorder.recordedBuffers.size() == numBuffersToProcess * 2, "Expected both raw and processed callbacks");

	// Check that each buffer ID appears exactly twice (once raw, once processed)
	int matchedPairs = 0;
	for (const auto& pair : recorder.idMatchCount) {
		if (pair.second == 2) {
			matchedPairs++;
			std::cout << "  Buffer ID " << pair.first << " appeared in both raw and processed callbacks" << std::endl;
		}
	}

	TEST_ASSERT(matchedPairs >= numBuffersToProcess, "Expected sufficient matched buffer pairs");
	std::cout << "  [OK] Successfully matched " << matchedPairs << " buffer pairs" << std::endl;
}

void testInputCallbacks() {
	std::cout << "\nTesting input callbacks..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(512, 256, 1, ope::DataType::UINT16);
	processor.initialize();

	std::atomic<int> inputCallbackCount(0);
	std::atomic<int> processedCallbackCount(0);

	// Add input callback
	auto inputId = processor.addInputCallback(
		[&inputCallbackCount](const ope::IOBuffer& buf) {
			inputCallbackCount++;
			std::cout << "  Input callback received buffer ID: " << buf.getBufferId() << std::endl;
		}
	);

	// Add processed callback
	auto procId = processor.addOutputCallback(
		[&processedCallbackCount](const ope::IOBuffer& buf) {
			processedCallbackCount++;
			std::cout << "  Processed callback received buffer ID: " << buf.getBufferId() << std::endl;
		}
	);

	// Process a buffer
	auto& inputBuffer = processor.getNextAvailableInputBuffer();
	processor.process(inputBuffer);

	// Wait for callbacks
	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	TEST_ASSERT(inputCallbackCount > 0, "Input callback should have been called");
	TEST_ASSERT(processedCallbackCount > 0, "Processed callback should have been called");

	std::cout << "  [OK] Input callbacks: " << inputCallbackCount << std::endl;
	std::cout << "  [OK] Processed callbacks: " << processedCallbackCount << std::endl;

	// Test callback removal
	processor.removeInputCallback(inputId);
	processor.removeOutputCallback(procId);

	// Reset counters
	inputCallbackCount = 0;
	processedCallbackCount = 0;

	// Process another buffer
	auto& inputBuffer2 = processor.getNextAvailableInputBuffer();
	processor.process(inputBuffer2);

	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	// Callbacks should not have been called
	TEST_ASSERT(inputCallbackCount == 0, "Input callback should not have been called after removal");
	TEST_ASSERT(processedCallbackCount == 0, "Processed callback should not have been called after removal");

	std::cout << "  [OK] Callback removal works correctly" << std::endl;
}

void testProcessorToolAttachment() {
	std::cout << "\nTesting ProcessorTool attachment/detachment..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(256, 128, 1, ope::DataType::UINT8);
	processor.initialize();

	TestRecorderTool tool;

	// Test initial state
	TEST_ASSERT(!tool.isAttached(), "Tool should not be attached initially");
	TEST_ASSERT(tool.getProcessor() == nullptr, "Processor should be nullptr initially");

	// Attach to processor
	tool.attachToProcessor(&processor);
	TEST_ASSERT(tool.isAttached(), "Tool should be attached after attachToProcessor");
	TEST_ASSERT(tool.getProcessor() == &processor, "Tool should return correct processor pointer");

	// Process data
	auto& inputBuffer = processor.getNextAvailableInputBuffer();
	processor.process(inputBuffer);

	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	size_t initialRecordCount = tool.recordedBuffers.size();
	TEST_ASSERT(initialRecordCount > 0, "Tool should have recorded some buffers while attached");

	// Detach
	tool.detach();
	TEST_ASSERT(!tool.isAttached(), "Tool should not be attached after detach");
	TEST_ASSERT(tool.getProcessor() == nullptr, "Processor should be nullptr after detach");

	// Process more data. should not be recorded now!
	auto& inputBuffer2 = processor.getNextAvailableInputBuffer();
	processor.process(inputBuffer2);

	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	TEST_ASSERT(tool.recordedBuffers.size() == initialRecordCount, "Tool should not record buffers after detach");

	std::cout << "  [OK] Tool attachment/detachment works correctly" << std::endl;
}

int main() {
	std::cout << "=== Testing Buffer IDs and Input/Output Callbacks ===" << std::endl;

	try {
		testBufferIdPropagation();
		testInputCallbacks();
		testProcessorToolAttachment();

		std::cout << "\n=== All tests passed! ===" << std::endl;
		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Test failed with exception: " << e.what() << std::endl;
		return 1;
	}
}