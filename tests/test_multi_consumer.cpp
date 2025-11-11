// Simple test for multi-consumer callback functionality
// Tests that multiple callbacks can be registered and all receive data

#include "../include/processor.h"
#include "../include/processorconfiguration.h"
#include "../include/types.h"
#include "../include/iobuffer.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>
#include <cassert>

// Test configuration
const int SIGNAL_LENGTH = 1024;
const int ASCANS_PER_BSCAN = 256;
const int BSCANS_PER_BUFFER = 1;

// Generate simple test data
std::vector<uint16_t> generateTestData() {
	size_t totalSamples = SIGNAL_LENGTH * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER;
	std::vector<uint16_t> data(totalSamples);
	
	// Simple pattern: ramp from 0 to 65535
	for (size_t i = 0; i < totalSamples; ++i) {
		data[i] = static_cast<uint16_t>((i * 65535) / totalSamples);
	}
	
	return data;
}

// Configure processor with basic settings
void configureProcessor(ope::Processor& processor) {
	auto config = processor.getConfig();
	
	// Data parameters
	config.dataParams.signalLength = SIGNAL_LENGTH;
	config.dataParams.ascansPerBscan = ASCANS_PER_BSCAN;
	config.dataParams.bscansPerBuffer = BSCANS_PER_BUFFER;
	config.dataParams.inputDataType = ope::DataType::UINT16;
	
	// Disable all processing for simple test
	config.resamplingParams.enabled = false;
	config.windowingParams.enabled = false;
	config.dispersionParams.enabled = false;
	config.backgroundRemovalParams.enabled = false;
	config.postProcessingParams.logScaling = false;
	
	processor.setConfig(config);
}

// ============================================
// TEST 1: Basic Multi-Consumer
// ============================================
bool test_basic_multi_consumer() {
	std::cout << "TEST 1: Basic Multi-Consumer" << std::endl;
	std::cout << "  Testing that 3 callbacks all receive data..." << std::endl;
	
	ope::Processor processor(ope::Backend::CPU);
	configureProcessor(processor);
	processor.initialize();
	
	// Counters for each callback
	std::atomic<int> count1{0}, count2{0}, count3{0};
	
	// Add 3 callbacks
	auto id1 = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		count1++;
	});
	
	auto id2 = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		count2++;
	});
	
	auto id3 = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		count3++;
	});
	
	// Verify callback count
	assert(processor.getCallbackCount() == 3);
	std::cout << "  [OK] Registered 3 callbacks" << std::endl;
	
	// Process one frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	// Wait for callbacks to complete
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	// Verify all callbacks were called
	if (count1 != 1 || count2 != 1 || count3 != 1) {
		std::cerr << "  [FAIL] Callbacks not all called!" << std::endl;
		std::cerr << "    count1=" << count1 << ", count2=" << count2 << ", count3=" << count3 << std::endl;
		return false;
	}
	
	std::cout << "  [OK] All 3 callbacks received data" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 2: Remove Callback
// ============================================
bool test_remove_callback() {
	std::cout << "TEST 2: Remove Callback" << std::endl;
	std::cout << "  Testing callback removal..." << std::endl;
	
	ope::Processor processor(ope::Backend::CPU);
	configureProcessor(processor);
	processor.initialize();
	
	std::atomic<int> count1{0}, count2{0}, count3{0};
	
	// Add 3 callbacks
	auto id1 = processor.addOutputCallback([&](const ope::IOBuffer& buf) { count1++; });
	auto id2 = processor.addOutputCallback([&](const ope::IOBuffer& buf) { count2++; });
	auto id3 = processor.addOutputCallback([&](const ope::IOBuffer& buf) { count3++; });
	
	assert(processor.getCallbackCount() == 3);
	std::cout << "  [OK] Added 3 callbacks" << std::endl;
	
	// Remove middle callback
	bool removed = processor.removeOutputCallback(id2);
	if (!removed) {
		std::cerr << "  [FAIL] removeOutputCallback returned false!" << std::endl;
		return false;
	}
	
	assert(processor.getCallbackCount() == 2);
	std::cout << "  [OK] Removed callback 2" << std::endl;
	
	// Process frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	// Verify only callbacks 1 and 3 were called
	if (count1 != 1 || count2 != 0 || count3 != 1) {
		std::cerr << "  [FAIL] Incorrect callback counts!" << std::endl;
		std::cerr << "    count1=" << count1 << " (expected 1)" << std::endl;
		std::cerr << "    count2=" << count2 << " (expected 0)" << std::endl;
		std::cerr << "    count3=" << count3 << " (expected 1)" << std::endl;
		return false;
	}
	
	std::cout << "  [OK] Callbacks 1 and 3 called, callback 2 not called" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 3: Clear All Callbacks
// ============================================
bool test_clear_callbacks() {
	std::cout << "TEST 3: Clear All Callbacks" << std::endl;
	std::cout << "  Testing clearOutputCallbacks()..." << std::endl;
	
	ope::Processor processor(ope::Backend::CPU);
	configureProcessor(processor);
	processor.initialize();
	
	std::atomic<int> count1{0}, count2{0};
	
	// Add callbacks
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count1++; });
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count2++; });
	
	assert(processor.getCallbackCount() == 2);
	std::cout << "  [OK] Added 2 callbacks" << std::endl;
	
	// Clear all
	processor.clearOutputCallbacks();
	
	assert(processor.getCallbackCount() == 0);
	std::cout << "  [OK] Cleared all callbacks" << std::endl;
	
	// Process frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	// Verify no callbacks were called
	if (count1 != 0 || count2 != 0) {
		std::cerr << "  [FAIL] Callbacks were called after clear!" << std::endl;
		std::cerr << "    count1=" << count1 << ", count2=" << count2 << std::endl;
		return false;
	}
	
	std::cout << "  [OK] No callbacks called after clear" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 4: Data Integrity
// ============================================
bool test_data_integrity() {
	std::cout << "TEST 4: Data Integrity" << std::endl;
	std::cout << "  Testing that all callbacks receive same data..." << std::endl;
	
	ope::Processor processor(ope::Backend::CPU);
	configureProcessor(processor);
	processor.initialize();
	
	// Storage for data received by each callback
	std::vector<float> data1, data2, data3;
	std::atomic<bool> done1{false}, done2{false}, done3{false};
	
	// Callbacks that copy data
	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		size_t numFloats = buf.getSizeInBytes() / sizeof(float);
		data1.resize(numFloats);
		std::memcpy(data1.data(), buf.getDataPointer(), buf.getSizeInBytes());
		done1 = true;
	});
	
	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		size_t numFloats = buf.getSizeInBytes() / sizeof(float);
		data2.resize(numFloats);
		std::memcpy(data2.data(), buf.getDataPointer(), buf.getSizeInBytes());
		done2 = true;
	});
	
	processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		size_t numFloats = buf.getSizeInBytes() / sizeof(float);
		data3.resize(numFloats);
		std::memcpy(data3.data(), buf.getDataPointer(), buf.getSizeInBytes());
		done3 = true;
	});
	
	// Process frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	// Wait for all callbacks
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	if (!done1 || !done2 || !done3) {
		std::cerr << "  [FAIL] Not all callbacks completed!" << std::endl;
		return false;
	}
	
	// Verify all received same size data
	if (data1.size() != data2.size() || data2.size() != data3.size()) {
		std::cerr << "  [FAIL] Data sizes don't match!" << std::endl;
		std::cerr << "    data1.size=" << data1.size() << std::endl;
		std::cerr << "    data2.size=" << data2.size() << std::endl;
		std::cerr << "    data3.size=" << data3.size() << std::endl;
		return false;
	}
	
	std::cout << "  [OK] All callbacks received " << data1.size() << " floats" << std::endl;
	
	// Verify data is identical
	for (size_t i = 0; i < data1.size(); ++i) {
		if (data1[i] != data2[i] || data2[i] != data3[i]) {
			std::cerr << "  [FAIL] Data mismatch at index " << i << "!" << std::endl;
			std::cerr << "    data1[" << i << "]=" << data1[i] << std::endl;
			std::cerr << "    data2[" << i << "]=" << data2[i] << std::endl;
			std::cerr << "    data3[" << i << "]=" << data3[i] << std::endl;
			return false;
		}
	}
	
	std::cout << "  [OK] All callbacks received identical data" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 5: Multiple Frames
// ============================================
bool test_multiple_frames() {
	std::cout << "TEST 5: Multiple Frames" << std::endl;
	std::cout << "  Testing multiple frames with multiple callbacks..." << std::endl;
	
	ope::Processor processor(ope::Backend::CPU);
	configureProcessor(processor);
	processor.initialize();
	
	const int NUM_FRAMES = 100;
	std::atomic<int> count1{0}, count2{0};
	
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count1++; });
	processor.addOutputCallback([&](const ope::IOBuffer& buf) { count2++; });
	
	auto testData = generateTestData();
	
	// Process multiple frames
	for (int i = 0; i < NUM_FRAMES; ++i) {
		auto& inputBuf = processor.getNextAvailableInputBuffer();
		std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
		processor.process(inputBuf);
	}
	
	// Wait for all callbacks
	std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	
	if (count1 != NUM_FRAMES || count2 != NUM_FRAMES) {
		std::cerr << "  [FAIL] Incorrect callback counts!" << std::endl;
		std::cerr << "    count1=" << count1 << " (expected " << NUM_FRAMES << ")" << std::endl;
		std::cerr << "    count2=" << count2 << " (expected " << NUM_FRAMES << ")" << std::endl;
		return false;
	}
	
	std::cout << "  [OK] Both callbacks called " << NUM_FRAMES << " times" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// TEST 6: Legacy API Compatibility
// ============================================
bool test_legacy_api() {
	std::cout << "TEST 6: Legacy API Compatibility" << std::endl;
	std::cout << "  Testing setOutputCallback() still works..." << std::endl;
	
	ope::Processor processor(ope::Backend::CPU);
	configureProcessor(processor);
	processor.initialize();
	
	std::atomic<int> count{0};
	
	// Use legacy API
	processor.setOutputCallback([&](const ope::IOBuffer& buf) {
		count++;
	});
	
	assert(processor.getCallbackCount() == 1);
	std::cout << "  [OK] setOutputCallback() registered 1 callback" << std::endl;
	
	// Process frame
	auto testData = generateTestData();
	auto& inputBuf = processor.getNextAvailableInputBuffer();
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	
	if (count != 1) {
		std::cerr << "  [FAIL] Callback count=" << count << " (expected 1)" << std::endl;
		return false;
	}
	
	std::cout << "  [OK] Legacy callback called correctly" << std::endl;
	std::cout << "  PASSED" << std::endl;
	std::cout << std::endl;
	return true;
}

// ============================================
// Main
// ============================================
int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "Multi-Consumer Callback Tests" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;
	
	int passed = 0;
	int total = 0;
	
	#define RUN_TEST(test_func) \
		total++; \
		if (test_func()) { \
			passed++; \
		}
	
	RUN_TEST(test_basic_multi_consumer);
	RUN_TEST(test_remove_callback);
	RUN_TEST(test_clear_callbacks);
	RUN_TEST(test_data_integrity);
	RUN_TEST(test_multiple_frames);
	RUN_TEST(test_legacy_api);
	
	std::cout << "========================================" << std::endl;
	std::cout << "RESULTS: " << passed << "/" << total << " tests passed" << std::endl;
	
	if (passed == total) {
		std::cout << "[OK] ALL TESTS PASSED!" << std::endl;
	} else {
		std::cout << "[FAIL] SOME TESTS FAILED!" << std::endl;
	}
	std::cout << "========================================" << std::endl;
	
	return (passed == total) ? 0 : 1;
}