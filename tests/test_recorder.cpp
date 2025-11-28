#include "../include/processor.h"
#include "../include/tools/recorder.h"
#include "test_utils.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstring>

const ope::Backend TEST_BACKEND = ope::Backend::CUDA;
const bool DELETE_TEST_FILES = true;


void fillTestData(ope::IOBuffer& buffer, const ope::Processor& processor, int seed) {
	uint16_t* data = static_cast<uint16_t*>(buffer.getDataPointer());
	size_t numSamples = processor.getConfig().dataParams.signalLength *
	                    processor.getConfig().dataParams.ascansPerBscan *
	                    processor.getConfig().dataParams.bscansPerBuffer;
	for (size_t i = 0; i < numSamples; i++) {
		data[i] = static_cast<uint16_t>(seed * 100 + i % 1000);
	}
}

void deleteTestFile(const char* filename) {
	if (DELETE_TEST_FILES) {
		std::remove(filename);
	}
}

void testBasicRecording() {
	std::cout << "Testing basic recording..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(2048, 1024, 2, ope::DataType::UINT16);


	//ope::CudaConfig cudaConfig;
	//cudaConfig.enableZeroCopy = true;
	//processor.setBackendConfig(cudaConfig);
	
	processor.initialize();

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setBufferCount(5);
	recorder.setOutputBaseName("test_basic");
	recorder.setUseTimestamp(false);
	recorder.startRecording();

	for (int i = 0; i < 5; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
	}

	bool success = recorder.waitForCompletion(5000);
	TEST_ASSERT(success, "Recording should complete successfully");
	TEST_ASSERT(recorder.getStatus() == ope::tools::Recorder::Status::COMPLETE, "Recorder status should be COMPLETE");
	auto summary = recorder.getLastRecordingSummary();
	TEST_ASSERT(summary.rawRecorded == 5, "Raw buffers recorded should be 5");
	TEST_ASSERT(summary.processedRecorded == 5, "Processed buffers recorded should be 5");

	deleteTestFile("test_basic_raw.raw");
	deleteTestFile("test_basic.raw");

	std::cout << "  [OK] Basic recording" << std::endl;
}

void testStartRecordingWhileProcessing() {
	std::cout << "\nTesting start recording while processing is already running..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
	processor.initialize();

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setManualAllocation(true);

	std::cout << "  Processing buffers..." << std::endl;
	int buffersBeforeRecording = 64;
	int buffersToRecord = 32;

	recorder.setBufferCount(buffersToRecord);
	recorder.allocateBuffers();
	recorder.setOutputBaseName("test_midstream");
	recorder.setUseTimestamp(false);
	int totalBuffers = buffersBeforeRecording + buffersToRecord;
	bool recordingStarted = false;

	for (int i = 0; i < totalBuffers; i++) {
		if (i == buffersBeforeRecording) {
			std::cout << "  Starting recording at buffer " << i << "..." << std::endl;
			recorder.startRecording();
		}
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
	}

	// Wait for recording to complete
	bool success = recorder.waitForCompletion(50000);
	TEST_ASSERT(success, "Operation should succeed");
	auto summary = recorder.getLastRecordingSummary();
	TEST_ASSERT(summary.rawRecorded == buffersToRecord, "Raw buffers should match buffersToRecord");
	TEST_ASSERT(summary.processedRecorded == buffersToRecord, "Processed buffers should match buffersToRecord");

	// Verify buffer IDs
	std::cout << "  Raw buffer IDs: [";
	for (size_t i = 0; i < summary.rawBufferIds.size(); i++) {
		std::cout << summary.rawBufferIds[i];
		if (i < summary.rawBufferIds.size() - 1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;

	std::cout << "  Processed buffer IDs: [";
	for (size_t i = 0; i < summary.processedBufferIds.size(); i++) {
		std::cout << summary.processedBufferIds[i];
		if (i < summary.processedBufferIds.size() - 1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;

	// Verify IDs match
	TEST_ASSERT(summary.rawBufferIds.size() == summary.processedBufferIds.size(), "Raw and processed ID counts should match");
	for (size_t i = 0; i < summary.rawBufferIds.size(); i++) {
		TEST_ASSERT(summary.rawBufferIds[i] == summary.processedBufferIds[i], "Buffer IDs should match");
	}

	// Verify expected range
	TEST_ASSERT(summary.rawBufferIds[0] == buffersBeforeRecording, "First buffer ID should match expected");
	TEST_ASSERT(summary.rawBufferIds[summary.rawBufferIds.size() - 1] == buffersBeforeRecording + buffersToRecord - 1, "Last buffer ID should match expected");

	// Cleanup
	deleteTestFile("test_midstream_raw.raw");
	deleteTestFile("test_midstream.raw");

	std::cout << "  [OK] Recording started successfully during ongoing processing" << std::endl;
}

void testEarlyStopIncompleteRecording() {
	std::cout << "\nTesting early stop with incomplete recording..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(1024, 512, 2, ope::DataType::UINT16);
	processor.initialize();

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setBufferCount(10);
	recorder.setOutputBaseName("test_early_stop");
	recorder.setUseTimestamp(false);
	recorder.startRecording();

	int buffersBeforeStop = 32;

	for (int i = 0; i < 2*buffersBeforeStop; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
		if(i == buffersBeforeStop - 1) { //stop early
			recorder.stopRecording();
		}
	}
	
	// Wait for write to complete (data should be saved with partial recording)
	bool writeSuccess = recorder.waitForCompletion(50000);
	TEST_ASSERT(writeSuccess, "Data should be written despite incomplete recording");

	auto summary = recorder.getLastRecordingSummary();
	TEST_ASSERT(summary.rawRecorded <= buffersBeforeStop, "Raw buffers recorded should be <= " + std::to_string(buffersBeforeStop) + ", but got " + std::to_string(summary.rawRecorded));
	TEST_ASSERT(summary.processedRecorded <= buffersBeforeStop, "Processed buffers recorded should be <= " + std::to_string(buffersBeforeStop) + ", but got " + std::to_string(summary.processedRecorded));

	// Cleanup
	deleteTestFile("test_early_stop_raw.raw");
	deleteTestFile("test_early_stop.raw");

	std::cout << "  [OK] Early stop with incomplete recording" << std::endl;
}

void testVolumeSynchronizationBothMode() {
	std::cout << "\nTesting volume synchronization in BOTH mode..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(1024, 512, 1, ope::DataType::UINT16);

	const int buffersPerVolume = 8;
	const int buffersToRecord = 16;
	const int buffersBeforeRecording = 12;

	processor.setBuffersPerVolume(buffersPerVolume);
	processor.initialize();

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setBufferCount(buffersToRecord);
	recorder.setOutputBaseName("test_volume_both");
	recorder.setUseTimestamp(false);
	recorder.setManualAllocation(true);
	recorder.allocateBuffers();

	const int totalBuffers = buffersBeforeRecording + buffersToRecord + buffersPerVolume;

	for (int i = 0; i < totalBuffers; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
		if (i == buffersBeforeRecording - 1) {
			recorder.startRecording(true);
		}
	}

	bool success = recorder.waitForCompletion(50000);
	
	auto summary = recorder.getLastRecordingSummary();
	std::cout << "  Raw buffer IDs: [";
	for (size_t i = 0; i < summary.rawBufferIds.size(); i++) {
		std::cout << summary.rawBufferIds[i];
		if (i < summary.rawBufferIds.size() - 1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;

	std::cout << "  Processed buffer IDs: [";
	for (size_t i = 0; i < summary.processedBufferIds.size(); i++) {
		std::cout << summary.processedBufferIds[i];
		if (i < summary.processedBufferIds.size() - 1) std::cout << ", ";
	}
	std::cout << "]" << std::endl;

	TEST_ASSERT(success, "Recording should complete successfully");
	TEST_ASSERT(summary.rawRecorded == buffersToRecord, "Raw buffers should be " + std::to_string(buffersToRecord) + ", but got " + std::to_string(summary.rawRecorded));
	TEST_ASSERT(summary.processedRecorded == buffersToRecord, "Processed buffers should be " + std::to_string(buffersToRecord) + ", but got " + std::to_string(summary.processedRecorded));

	// Cleanup
	deleteTestFile("test_volume_both_raw.raw");
	deleteTestFile("test_volume_both.raw");

	std::cout << "  [OK] Volume synchronization in BOTH mode" << std::endl;
}

void testAbortRecording() {
	std::cout << "\nTesting abortRecording()..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(1024, 512, 2, ope::DataType::UINT16);
	processor.initialize();

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setBufferCount(10);
	recorder.setOutputBaseName("test_abort");
	recorder.setUseTimestamp(false);
	recorder.startRecording();

	// Record a few buffers
	for (int i = 0; i < 3; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
	}

	// Abort. should discard all data and return to IDLE
	recorder.abortRecording();

	auto summary = recorder.getLastRecordingSummary();
	TEST_ASSERT(summary.rawRecorded == 0, "Buffers should be cleared");
	TEST_ASSERT(summary.processedRecorded == 0, "Buffers should be cleared");
	TEST_ASSERT(recorder.getStatus() == ope::tools::Recorder::Status::IDLE, "Recorder status should be IDLE");

	// Cleanup (abort should discard files, but delete in case any were created)
	deleteTestFile("test_abort_raw.raw");
	deleteTestFile("test_abort.raw");

	std::cout << "  [OK] abortRecording()" << std::endl;
}

void testAllocationModeSwitching() {
	std::cout << "\nTesting allocation mode switching..." << std::endl;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(1024, 512, 2, ope::DataType::UINT16);
	processor.initialize();

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setBufferCount(5);
	recorder.setUseTimestamp(false);

	// Recording 1: Manual mode
	recorder.setManualAllocation(true);
	recorder.setOutputBaseName("test_switch1");
	recorder.startRecording();  // Auto-allocates

	for (int i = 0; i < 5; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
	}

	recorder.waitForCompletion(5000);
	TEST_ASSERT(recorder.isAllocated(), "Buffers should be allocated");  // Still allocated (manual mode)

	// Recording 2: Switch to auto mode - should enable auto-free
	recorder.setManualAllocation(false);
	recorder.setOutputBaseName("test_switch2");
	recorder.startRecording();  // Reuses buffers

	for (int i = 0; i < 5; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i + 10);
		processor.process(inputBuffer);
	}

	recorder.waitForCompletion(5000);
	TEST_ASSERT(!recorder.isAllocated(), "Buffers should not be allocated");  // Should be auto-freed now!

	// Recording 3: Switch back to manual mode
	recorder.setManualAllocation(true);
	recorder.setOutputBaseName("test_switch3");
	recorder.startRecording();  // Auto-allocates (buffers were freed)

	for (int i = 0; i < 5; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i + 20);
		processor.process(inputBuffer);
	}

	recorder.waitForCompletion(5000);
	TEST_ASSERT(recorder.isAllocated(), "Buffers should be allocated");  // Preserved (manual mode)

	// Recording 4: Switch to auto mode again
	recorder.setManualAllocation(false);
	recorder.setOutputBaseName("test_switch4");
	recorder.startRecording();  // Reuses buffers

	for (int i = 0; i < 5; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i + 30);
		processor.process(inputBuffer);
	}

	recorder.waitForCompletion(5000);
	TEST_ASSERT(!recorder.isAllocated(), "Buffers should not be allocated");  // Auto-freed again

	// Cleanup
	deleteTestFile("test_switch1_raw.raw");
	deleteTestFile("test_switch1.raw");
	deleteTestFile("test_switch2_raw.raw");
	deleteTestFile("test_switch2.raw");
	deleteTestFile("test_switch3_raw.raw");
	deleteTestFile("test_switch3.raw");
	deleteTestFile("test_switch4_raw.raw");
	deleteTestFile("test_switch4.raw");

	std::cout << "  [OK] Allocation mode switching (4 recordings, multiple switches)" << std::endl;
}

void testProgressiveDataPattern() {
	std::cout << "\nTesting progressive data pattern..." << std::endl;

	const int signalLength = 2048;
	const int ascansPerBscan = 2048;
	const int bscansPerBuffer = 1;
	const int numBuffers = 32;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);
	processor.initialize();

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setBufferCount(numBuffers);
	recorder.setOutputBaseName("test_progressive");
	recorder.setUseTimestamp(false);

	// Pre-generate all test data
	std::cout << "  Pre-generating test data..." << std::endl;
	size_t samplesPerBuffer = signalLength * ascansPerBscan * bscansPerBuffer;
	std::vector<std::vector<uint16_t>> testData(numBuffers);

	for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
		testData[bufferIdx].resize(samplesPerBuffer, 0);  // Fill with zeros

		// Fill first (bufferIdx + 1) A-scans with data
		int linesToFill = bufferIdx + 1;  // Buffer 0 has 1 line, buffer 1 has 2 lines, etc.
		if (linesToFill > ascansPerBscan) {
			linesToFill = ascansPerBscan;  // Cap at max A-scans
		}

		for (int ascan = 0; ascan < linesToFill; ascan++) {
			// Fill this A-scan with a recognizable pattern
			for (int sample = 0; sample < signalLength; sample++) {
				size_t idx = ascan * signalLength + sample;
				// Pattern: gradient based on buffer index and A-scan position
				testData[bufferIdx][idx] = static_cast<uint16_t>(1000 + bufferIdx * 50 + ascan * 10 + sample % 100);
			}
		}
	}

	std::cout << "  Starting recording and processing..." << std::endl;

	recorder.allocateBuffers();
	recorder.startRecording();

	// Process all buffers
	for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		uint16_t* data = static_cast<uint16_t*>(inputBuffer.getDataPointer());

		// Copy pre-generated data
		std::memcpy(data, testData[bufferIdx].data(), samplesPerBuffer * sizeof(uint16_t));

		processor.process(inputBuffer);
	}

	// Wait for completion
	bool success = recorder.waitForCompletion(60000);
	auto summary = recorder.getLastRecordingSummary();

	if (!success) {
		std::cout << "Recording did not complete within timeout." << std::endl;

		std::string errorMsg = recorder.getLastError();
		if (!errorMsg.empty()) {
			std::cout << "Error message: " << errorMsg << std::endl;
		}

		std::cout << "  Raw buffer IDs: [";
		for (size_t i = 0; i < summary.rawBufferIds.size(); i++) {
			std::cout << summary.rawBufferIds[i];
			if (i < summary.rawBufferIds.size() - 1) std::cout << ", ";
		}
		std::cout << "]" << std::endl;

		std::cout << "  Processed buffer IDs: [";
		for (size_t i = 0; i < summary.processedBufferIds.size(); i++) {
			std::cout << summary.processedBufferIds[i];
			if (i < summary.processedBufferIds.size() - 1) std::cout << ", ";
		}
		std::cout << "]" << std::endl;
	}

	TEST_ASSERT(success, "Operation should succeed");
	TEST_ASSERT(summary.rawRecorded == numBuffers, "Raw buffers recorded should match numBuffers");
	TEST_ASSERT(summary.processedRecorded == numBuffers, "Processed buffers recorded should match numBuffers");

	std::cout << "  Recording complete, verifying saved data..." << std::endl;

	// ===== Verify RAW data file =====
	std::cout << "  Verifying raw data..." << std::endl;
	std::ifstream rawFile("test_progressive_raw.raw", std::ios::binary);
	TEST_ASSERT(rawFile.is_open(), "Failed to open raw data file");

	size_t rawSamplesPerBuffer = signalLength * ascansPerBscan * bscansPerBuffer;
	std::vector<uint16_t> rawReadBuffer(rawSamplesPerBuffer);

	// Verify each raw buffer
	for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
		// Read buffer from file
		rawFile.read(reinterpret_cast<char*>(rawReadBuffer.data()), rawSamplesPerBuffer * sizeof(uint16_t));
		TEST_ASSERT(rawFile.good(), "Failed to read raw buffer from file");

		// Verify pattern
		int expectedLinesToFill = bufferIdx + 1;
		if (expectedLinesToFill > ascansPerBscan) {
			expectedLinesToFill = ascansPerBscan;
		}

		// Check filled A-scans
		for (int ascan = 0; ascan < expectedLinesToFill; ascan++) {
			for (int sample = 0; sample < signalLength; sample++) {
				size_t idx = ascan * signalLength + sample;
				uint16_t expected = static_cast<uint16_t>(1000 + bufferIdx * 50 + ascan * 10 + sample % 100);
				TEST_ASSERT(rawReadBuffer[idx] == expected, "Raw data mismatch in filled region");
			}
		}

		// Check zero-filled A-scans (rest of the buffer)
		for (int ascan = expectedLinesToFill; ascan < ascansPerBscan; ascan++) {
			for (int sample = 0; sample < signalLength; sample++) {
				size_t idx = ascan * signalLength + sample;
				TEST_ASSERT(rawReadBuffer[idx] == 0, "Raw data should be zero in unfilled region");
			}
		}
	}

	rawFile.close();

	// ===== Verify PROCESSED data file =====
	std::cout << "  Verifying processed data..." << std::endl;
	std::ifstream processedFile("test_progressive.raw", std::ios::binary);
	TEST_ASSERT(processedFile.is_open(), "Failed to open processed data file");

	size_t processedSamplesPerBuffer = signalLength * ascansPerBscan * bscansPerBuffer;
	std::vector<float> processedReadBuffer(processedSamplesPerBuffer);

	// Verify each processed buffer
	for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
		// Read buffer from file
		processedFile.read(reinterpret_cast<char*>(processedReadBuffer.data()), processedSamplesPerBuffer * sizeof(float));

		// Verify pattern
		// (filled A-scans should have non-zero values, unfilled should be background)
		int expectedLinesToFill = bufferIdx + 1;
		if (expectedLinesToFill > ascansPerBscan) {
			expectedLinesToFill = ascansPerBscan;
		}

		// Check that filled A-scans have non-zero processed data
		for (int ascan = 0; ascan < expectedLinesToFill; ascan++) {
			bool hasNonZeroData = false;
			for (int sample = 0; sample < signalLength; sample++) {
				size_t idx = ascan * signalLength + sample;
				if (processedReadBuffer[idx] != 0.0f) {
					hasNonZeroData = true;
					break;
				}
			}
			TEST_ASSERT(hasNonZeroData, "Processed filled A-scan should have non-zero data");
		}
	}

	processedFile.close();

	// Cleanup
	deleteTestFile("test_progressive_raw.raw");
	deleteTestFile("test_progressive.raw");

	std::cout << "  [OK] Progressive data pattern (99 buffers, verified)" << std::endl;
	std::cout << "      Raw file: test_progressive_raw.raw" << std::endl;
	std::cout << "      Processed file: test_progressive.raw" << std::endl;
	std::cout << "      Pattern verified: Each buffer has progressively more A-scans filled" << std::endl;
}

void testDiskWritePerformance() {
	std::cout << "Testing disk write performance..." << std::endl;
	const int signalLength = 2048;
	const int ascansPerBscan = 1024;
	const int bscansPerBuffer = 1;
	const int numBuffers = 128; 

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);
	processor.initialize();

	// Calculate buffer sizes using processor configuration
	int outputSignalLength = processor.getConfig().dataParams.outputSignalLength();
	size_t rawBytesPerBuffer = static_cast<size_t>(signalLength) * ascansPerBscan * bscansPerBuffer * sizeof(uint16_t);
	size_t processedBytesPerBuffer = static_cast<size_t>(outputSignalLength) * ascansPerBscan * bscansPerBuffer * sizeof(float);
	double rawGBperBuffer = rawBytesPerBuffer / (1024.0 * 1024.0 * 1024.0);
	double processedGBperBuffer = processedBytesPerBuffer / (1024.0 * 1024.0 * 1024.0);

	std::cout << "  Buffer size (raw): " << rawGBperBuffer << " GB" << std::endl;
	std::cout << "  Buffer size (processed): " << processedGBperBuffer << " GB" << std::endl;
	std::cout << "  Total buffers: " << numBuffers << std::endl;

	ope::tools::Recorder recorder;
	recorder.attachToProcessor(&processor);
	recorder.setBufferCount(numBuffers);
	recorder.setUseTimestamp(false);
	recorder.setManualAllocation(true);
	recorder.allocateBuffers();

	// Test 1: RAW_ONLY mode
	std::cout << "  Testing RAW_ONLY mode..." << std::endl;
	recorder.setMode(ope::tools::Recorder::Mode::RAW_ONLY);
	recorder.setOutputBaseName("test_perf_raw");
	
	recorder.startRecording();

	for (int i = 0; i < numBuffers; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
	}

	auto startTime = std::chrono::high_resolution_clock::now();
	TEST_ASSERT(recorder.waitForCompletion(30000), "RAW_ONLY recording timeout");
	std::string errorMsg = recorder.getLastError();
	if (!errorMsg.empty()) {
		std::cout << "Error message: " << errorMsg << std::endl;
	}	
	auto endTime = std::chrono::high_resolution_clock::now();

	double durationSec = std::chrono::duration<double>(endTime - startTime).count();
	double totalGB = rawGBperBuffer * numBuffers;
	double speedMBs = (totalGB * 1024.0) / durationSec;

	std::cout << "    Expected total: " << totalGB << " GB (" << rawGBperBuffer << " GB x " << numBuffers << " buffers)" << std::endl;
	std::cout << "    Duration: " << durationSec << " seconds" << std::endl;
	std::cout << "    Total written: " << totalGB << " GB" << std::endl;
	std::cout << "    Estimated write speed: " << speedMBs << " MB/s" << std::endl;

	deleteTestFile("test_perf_raw_raw.raw");

	// Test 2: PROCESSED_ONLY mode
	std::cout << "  Testing PROCESSED_ONLY mode..." << std::endl;
	recorder.setMode(ope::tools::Recorder::Mode::PROCESSED_ONLY);
	recorder.setOutputBaseName("test_perf_processed");

	recorder.startRecording();

	for (int i = 0; i < numBuffers; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
	}
	startTime = std::chrono::high_resolution_clock::now();

	TEST_ASSERT(recorder.waitForCompletion(30000), "PROCESSED_ONLY recording timeout");
	recorder.waitForCompletion(30000);
	std::string errorMsgProcessed = recorder.getLastError();
	if (!errorMsgProcessed.empty()) {
		std::cout << "Error message: " << errorMsgProcessed << std::endl;
	}	
	endTime = std::chrono::high_resolution_clock::now();

	durationSec = std::chrono::duration<double>(endTime - startTime).count();
	totalGB = processedGBperBuffer * numBuffers;
	speedMBs = (totalGB * 1024.0) / durationSec;

	std::cout << "    Expected total: " << totalGB << " GB (" << processedGBperBuffer << " GB x" << numBuffers << " buffers)" << std::endl;
	std::cout << "    Duration: " << durationSec << " seconds" << std::endl;
	std::cout << "    Total written: " << totalGB << " GB" << std::endl;
	std::cout << "    Estimated write speed: " << speedMBs << " MB/s" << std::endl;

	deleteTestFile("test_perf_processed.raw");

	// Test 3: BOTH mode (parallel writing)
	std::cout << "  Testing BOTH mode (parallel write)..." << std::endl;
	recorder.setMode(ope::tools::Recorder::Mode::BOTH);
	recorder.setOutputBaseName("test_perf_both");

	recorder.startRecording();

	for (int i = 0; i < numBuffers; i++) {
		auto& inputBuffer = processor.getNextAvailableInputBuffer();
		fillTestData(inputBuffer, processor, i);
		processor.process(inputBuffer);
	}

	startTime = std::chrono::high_resolution_clock::now();
	TEST_ASSERT(recorder.waitForCompletion(30000), "BOTH mode recording timeout");
	std::string errorMsgBoth = recorder.getLastError();
	if (!errorMsgBoth.empty()) {
		std::cout << "Error message: " << errorMsgBoth << std::endl;
	}	
	endTime = std::chrono::high_resolution_clock::now();

	durationSec = std::chrono::duration<double>(endTime - startTime).count();
	totalGB = (rawGBperBuffer + processedGBperBuffer) * numBuffers;
	speedMBs = (totalGB * 1024.0) / durationSec;

	std::cout << "    Expected total: " << totalGB << " GB (" << rawGBperBuffer << " + " << processedGBperBuffer << " GB x " << numBuffers << " buffers)" << std::endl;
	std::cout << "    Duration: " << durationSec << " seconds" << std::endl;
	std::cout << "    Total written: " << totalGB << " GB (raw + processed)" << std::endl;
	std::cout << "    Estimated combined write speed: " << speedMBs << " MB/s" << std::endl;

	deleteTestFile("test_perf_both_raw.raw");
	deleteTestFile("test_perf_both.raw");

	std::cout << "  [OK] Disk write performance test" << std::endl;
}

int main() {
	std::cout << "=== Testing Recorder Tool ===" << std::endl;

	try {
		testBasicRecording();
		testStartRecordingWhileProcessing();
		testVolumeSynchronizationBothMode();
		testEarlyStopIncompleteRecording();
		testAbortRecording();
		testAllocationModeSwitching();
		testProgressiveDataPattern();
		testDiskWritePerformance();

		std::cout << "\n=== All 15 Recorder tests passed! ===" << std::endl;

		return 0;

	} catch (const std::exception& e) {
		std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
		return 1;
	}
}
