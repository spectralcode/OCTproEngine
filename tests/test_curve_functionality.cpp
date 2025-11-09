#include "../include/processor.h"
#include "../include/processorconfiguration.h"
#include "../include/types.h"
#include "../include/iobuffer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <mutex>

// Use same dimensions as test_backend_output_comparison
const int SIGNAL_LENGTH = 2048;
const int ASCANS_PER_BSCAN = 1024;
const int BSCANS_PER_BUFFER = 1;

// Test coefficients from existing tests
const float RESAMPLING_COEFFS[4] = {0.5f, 2048.0f, -100.0f, 50.0f};
const float DISPERSION_COEFFS[4] = {0.0f, 0.0f, 1.0f, -3.0f};
const float DISPERSION_FACTOR = 1.0f;

const ope::WindowType WINDOW_TYPE = ope::WindowType::HANN;
const float WINDOW_CENTER = 0.9f;
const float WINDOW_FILL_FACTOR = 0.85f;

// Test control
bool allTestsPassed = true;
int testCounter = 0;

// Helper macros
#define TEST_SECTION(name) \
	std::cout << "\n========================================" << std::endl; \
	std::cout << "TEST SECTION " << (++testCounter) << ": " << name << std::endl; \
	std::cout << "========================================" << std::endl;

#define ASSERT_TRUE(condition, message) \
	if (!(condition)) { \
		std::cerr << "  FAIL: " << message << std::endl; \
		allTestsPassed = false; \
	} else { \
		std::cout << "  PASS: " << message << std::endl; \
	}

#define ASSERT_FALSE(condition, message) \
	if (condition) { \
		std::cerr << "  FAIL: " << message << std::endl; \
		allTestsPassed = false; \
	} else { \
		std::cout << "  PASS: " << message << std::endl; \
	}

#define ASSERT_EQUAL(val1, val2, message) \
	if ((val1) != (val2)) { \
		std::cerr << "  FAIL: " << message << " (expected " << (val2) << ", got " << (val1) << ")" << std::endl; \
		allTestsPassed = false; \
	} else { \
		std::cout << "  PASS: " << message << std::endl; \
	}

// Same test data generation as test_backend_output_comparison
std::vector<uint16_t> generateSyntheticAScan(int signalLength, int ascanIndex) {
	std::vector<uint16_t> ascan(signalLength);
	
	const double peak1Depth = signalLength * 0.2;
	const double peak2Depth = signalLength * 0.5;
	const double peak3Depth = signalLength * 0.7;
	
	const double peak1Width = 50.0;
	const double peak2Width = 30.0;
	const double peak3Width = 40.0;
	
	const double peak1Amp = 8000.0;
	const double peak2Amp = 5000.0;
	const double peak3Amp = 3000.0;
	
	const double lateralPhase = ascanIndex * 0.1;
	
	for (int i = 0; i < signalLength; ++i) {
		double value = 2000.0;
		
		value += peak1Amp * std::exp(-std::pow((i - peak1Depth) / peak1Width, 2));
		value += peak2Amp * std::exp(-std::pow((i - peak2Depth) / peak2Width, 2));
		value += peak3Amp * std::exp(-std::pow((i - peak3Depth) / peak3Width, 2));
		
		value += 1000.0 * std::sin(i * 0.3 + lateralPhase);
		value += 500.0 * std::sin(i * 0.15 + lateralPhase * 0.5);
		
		value += (rand() % 400) - 200;
		
		ascan[i] = static_cast<uint16_t>(std::max(0.0, std::min(65535.0, value)));
	}
	
	return ascan;
}

std::vector<uint16_t> generateTestData(int signalLength, int ascansPerBscan, int bscansPerBuffer) {
	std::vector<uint16_t> data;
	data.reserve(signalLength * ascansPerBscan * bscansPerBuffer);
	
	for (int b = 0; b < bscansPerBuffer; ++b) {
		for (int a = 0; a < ascansPerBscan; ++a) {
			auto ascan = generateSyntheticAScan(signalLength, a);
			data.insert(data.end(), ascan.begin(), ascan.end());
		}
	}
	
	return data;
}

// Result container for async processing
struct ProcessingResult {
	std::vector<float> output;
	bool received = false;
	std::mutex mutex;
	std::condition_variable cv;
	
	void waitForCompletion(int timeoutMs = 5000) {
		std::unique_lock<std::mutex> lock(mutex);
		cv.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this] { return received; });
	}
	
	void reset() {
		std::lock_guard<std::mutex> lock(mutex);
		output.clear();
		received = false;
	}
};

// Process data and get output
bool processData(ope::Processor& processor, const std::vector<uint16_t>& testData, ProcessingResult& result) {
	result.reset();
	
	processor.setOutputCallback([&result](const ope::IOBuffer& output) {
		std::lock_guard<std::mutex> lock(result.mutex);
		
		size_t numFloats = output.getSizeInBytes() / sizeof(float);
		result.output.resize(numFloats);
		std::memcpy(result.output.data(), output.getDataPointer(), output.getSizeInBytes());
		
		result.received = true;
		result.cv.notify_one();
	});
	
	ope::IOBuffer& inputBuf = processor.getInputBuffer(0);
	std::memcpy(inputBuf.getDataPointer(), testData.data(), testData.size() * sizeof(uint16_t));
	processor.process(inputBuf);
	
	result.waitForCompletion();
	
	return result.received && !result.output.empty();
}

// Check if output is valid (no NaN/Inf)
bool isOutputValid(const std::vector<float>& output) {
	for (float val : output) {
		if (std::isnan(val) || std::isinf(val)) {
			return false;
		}
	}
	return true;
}

// Compare two outputs
bool outputsAreDifferent(const std::vector<float>& output1, const std::vector<float>& output2, float threshold = 0.001f) {
	if (output1.size() != output2.size()) return true;
	
	size_t diffCount = 0;
	for (size_t i = 0; i < output1.size(); ++i) {
		if (std::abs(output1[i] - output2[i]) > threshold) {
			diffCount++;
		}
	}
	
	// Consider outputs different if more than 1% of samples differ
	return diffCount > (output1.size() * 0.01);
}

// ============================================
// MAIN TEST
// ============================================

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "COMPREHENSIVE CURVE FUNCTIONALITY TEST" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Signal Length: " << SIGNAL_LENGTH << std::endl;
	std::cout << "A-scans per B-scan: " << ASCANS_PER_BSCAN << std::endl;
	std::cout << "B-scans per buffer: " << BSCANS_PER_BUFFER << std::endl;
	std::cout << std::endl;
	
	// Generate test data once
	std::cout << "Generating test data..." << std::endl;
	auto testData = generateTestData(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER);
	std::cout << "  Generated " << testData.size() << " samples" << std::endl;
	
	// ============================================
	// TEST 1: GETTERS/SETTERS (PRE-INITIALIZATION)
	// ============================================
	
	TEST_SECTION("Getters/Setters (Pre-Initialization)");
	
	ope::Processor processor(ope::Backend::CPU);
	processor.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	processor.enableResampling(true);
	processor.enableWindowing(true);
	processor.enableDispersionCompensation(true);
	processor.enableLogScaling(true);
	processor.setGrayscaleRange(0.0f, 80.0f);
	
	// Create custom curves
	std::vector<float> customResamplingCurve(SIGNAL_LENGTH);
	std::vector<float> customWindowCurve(SIGNAL_LENGTH);
	std::vector<float> customDispersionCurve(SIGNAL_LENGTH);
	
	for (int i = 0; i < SIGNAL_LENGTH; ++i) {
		customResamplingCurve[i] = 7.0f + i * 0.5f; // Linear ramp
		customWindowCurve[i] = (i < SIGNAL_LENGTH/2) ? 1.0f : 0.5f; // Step function
		customDispersionCurve[i] = 0.001f * i * i; // Quadratic phase
	}
	
	// Set custom curves BEFORE initialization
	processor.setCustomResamplingCurve(customResamplingCurve.data(), customResamplingCurve.size());
	processor.setCustomWindowCurve(customWindowCurve.data(), customWindowCurve.size());
	processor.setCustomDispersionCurve(customDispersionCurve.data(), customDispersionCurve.size());
	
	const ope::ProcessorConfiguration& config = processor.getConfig();
	
	// Verify curves are stored
	ASSERT_TRUE(config.hasCustomResamplingCurve(), "Custom resampling curve is stored");
	ASSERT_TRUE(config.hasCustomWindowCurve(), "Custom window curve is stored");
	ASSERT_TRUE(config.hasCustomDispersionCurve(), "Custom dispersion curve is stored");
	
	// Verify sizes
	ASSERT_EQUAL(config.getCustomResamplingCurveSize(), (size_t)SIGNAL_LENGTH, "Resampling curve size correct");
	ASSERT_EQUAL(config.getCustomWindowCurveSize(), (size_t)SIGNAL_LENGTH, "Window curve size correct");
	ASSERT_EQUAL(config.getCustomDispersionCurveSize(), (size_t)SIGNAL_LENGTH, "Dispersion curve size correct");
	
	// Verify data matches
	const float* retrievedResampling = config.getCustomResamplingCurve();
	const float* retrievedWindow = config.getCustomWindowCurve();
	const float* retrievedDispersion = config.getCustomDispersionCurve();
	
	bool resamplingMatch = std::equal(customResamplingCurve.begin(), customResamplingCurve.end(), retrievedResampling);
	bool windowMatch = std::equal(customWindowCurve.begin(), customWindowCurve.end(), retrievedWindow);
	bool dispersionMatch = std::equal(customDispersionCurve.begin(), customDispersionCurve.end(), retrievedDispersion);
	
	ASSERT_TRUE(resamplingMatch, "Resampling curve data matches");
	ASSERT_TRUE(windowMatch, "Window curve data matches");
	ASSERT_TRUE(dispersionMatch, "Dispersion curve data matches");
	
	// Initialize and process
	processor.initialize();
	ProcessingResult result;
	bool processSuccess = processData(processor, testData, result);
	
	ASSERT_TRUE(processSuccess, "Processing succeeded with custom curves");
	ASSERT_TRUE(isOutputValid(result.output), "Output is valid (no NaN/Inf) with pre-init curves");
	
	// ============================================
	// TEST 2: EDGE CASES - SIZE MISMATCHES
	// ============================================
	
	TEST_SECTION("Edge Cases - Size Mismatches");
	
	ope::Processor processor2(ope::Backend::CPU);
	processor2.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	processor2.enableResampling(true);
	processor2.enableWindowing(true);
	processor2.enableDispersionCompensation(true);
	processor2.enableLogScaling(true);
	processor2.setGrayscaleRange(0.0f, 80.0f);
	
	// Test: Curve too small (should zero-pad)
	std::vector<float> tooSmallCurve(SIGNAL_LENGTH / 2);
	for (size_t i = 0; i < tooSmallCurve.size(); ++i) {
		tooSmallCurve[i] = 10.0f + i;
	}
	
	processor2.setCustomResamplingCurve(tooSmallCurve.data(), tooSmallCurve.size());
	
	const ope::ProcessorConfiguration& config2 = processor2.getConfig();
	size_t adjustedSize = config2.getCustomResamplingCurveSize();
	
	ASSERT_EQUAL(adjustedSize, (size_t)SIGNAL_LENGTH, "Too-small curve zero-padded to signalLength");
	
	const float* adjustedCurve = config2.getCustomResamplingCurve();
	bool firstHalfMatches = std::equal(tooSmallCurve.begin(), tooSmallCurve.end(), adjustedCurve);
	bool secondHalfZeros = true;
	for (size_t i = tooSmallCurve.size(); i < SIGNAL_LENGTH; ++i) {
		if (adjustedCurve[i] != 0.0f) {
			secondHalfZeros = false;
			break;
		}
	}
	
	ASSERT_TRUE(firstHalfMatches, "Original data preserved in too-small curve");
	ASSERT_TRUE(secondHalfZeros, "Padding is zeros in too-small curve");
	
	// Test: Curve too large (should truncate)
	std::vector<float> tooLargeCurve(SIGNAL_LENGTH * 2);
	for (size_t i = 0; i < tooLargeCurve.size(); ++i) {
		tooLargeCurve[i] = 20.0f + i;
	}
	
	processor2.setCustomWindowCurve(tooLargeCurve.data(), tooLargeCurve.size());
	
	size_t truncatedSize = config2.getCustomWindowCurveSize();
	ASSERT_EQUAL(truncatedSize, (size_t)SIGNAL_LENGTH, "Too-large curve truncated to signalLength");
	
	const float* truncatedCurve = config2.getCustomWindowCurve();
	bool truncatedMatches = std::equal(truncatedCurve, truncatedCurve + SIGNAL_LENGTH, tooLargeCurve.begin());
	
	ASSERT_TRUE(truncatedMatches, "First signalLength elements preserved in too-large curve");
	
	// Initialize and verify still works
	processor2.initialize();
	ProcessingResult result2;
	bool processSuccess2 = processData(processor2, testData, result2);
	
	ASSERT_TRUE(processSuccess2, "Processing succeeded with mismatched curve sizes");
	ASSERT_TRUE(isOutputValid(result2.output), "Output valid with mismatched curve sizes");
	
	// ============================================
	// TEST 3: HOT-SWAP CUSTOM CURVES
	// ============================================
	
	TEST_SECTION("Hot-Swap Custom Curves (Post-Initialization)");
	
	ope::Processor processor3(ope::Backend::CPU);
	processor3.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	processor3.enableResampling(true);
	processor3.setInterpolationMethod(ope::InterpolationMethod::CUBIC);
	processor3.setResamplingCoefficients(RESAMPLING_COEFFS);
	processor3.enableWindowing(true);
	processor3.setWindowParameters(WINDOW_TYPE, WINDOW_CENTER, WINDOW_FILL_FACTOR);
	processor3.enableDispersionCompensation(true);
	processor3.setDispersionCoefficients(DISPERSION_COEFFS, DISPERSION_FACTOR);
	processor3.enableLogScaling(true);
	processor3.setGrayscaleRange(0.0f, 80.0f);
	
	processor3.initialize();
	
	// Process with generated curves
	ProcessingResult resultGenerated;
	bool genSuccess = processData(processor3, testData, resultGenerated);
	ASSERT_TRUE(genSuccess, "Processing with generated curves succeeded");
	ASSERT_TRUE(isOutputValid(resultGenerated.output), "Generated curves output valid");
	
	// Hot-swap to custom resampling curve
	std::vector<float> hotswapResampling(SIGNAL_LENGTH);
	for (int i = 0; i < SIGNAL_LENGTH; ++i) {
		hotswapResampling[i] = 7.0f + i * 0.6f; // Different from generated
	}
	processor3.setCustomResamplingCurve(hotswapResampling.data(), hotswapResampling.size());
	
	ProcessingResult resultHotswap1;
	bool hotswap1Success = processData(processor3, testData, resultHotswap1);
	ASSERT_TRUE(hotswap1Success, "Processing after resampling hotswap succeeded");
	ASSERT_TRUE(isOutputValid(resultHotswap1.output), "Hotswap resampling output valid");
	ASSERT_TRUE(outputsAreDifferent(resultGenerated.output, resultHotswap1.output), 
	            "Output changed after resampling hotswap");
	
	// Hot-swap to custom window curve
	std::vector<float> hotswapWindow(SIGNAL_LENGTH);
	for (int i = 0; i < SIGNAL_LENGTH; ++i) {
		hotswapWindow[i] = (i < SIGNAL_LENGTH * 0.7f) ? 1.0f : 0.2f;
	}
	processor3.setCustomWindowCurve(hotswapWindow.data(), hotswapWindow.size());
	
	ProcessingResult resultHotswap2;
	bool hotswap2Success = processData(processor3, testData, resultHotswap2);
	ASSERT_TRUE(hotswap2Success, "Processing after window hotswap succeeded");
	ASSERT_TRUE(isOutputValid(resultHotswap2.output), "Hotswap window output valid");
	ASSERT_TRUE(outputsAreDifferent(resultHotswap1.output, resultHotswap2.output),
	            "Output changed after window hotswap");
	
	// Hot-swap to custom dispersion curve
	std::vector<float> hotswapDispersion(SIGNAL_LENGTH);
	for (int i = 0; i < SIGNAL_LENGTH; ++i) {
		hotswapDispersion[i] = 0.0005f * i * i; // Different phase
	}
	processor3.setCustomDispersionCurve(hotswapDispersion.data(), hotswapDispersion.size());
	
	ProcessingResult resultHotswap3;
	bool hotswap3Success = processData(processor3, testData, resultHotswap3);
	ASSERT_TRUE(hotswap3Success, "Processing after dispersion hotswap succeeded");
	ASSERT_TRUE(isOutputValid(resultHotswap3.output), "Hotswap dispersion output valid");
	ASSERT_TRUE(outputsAreDifferent(resultHotswap2.output, resultHotswap3.output),
	            "Output changed after dispersion hotswap");
	
	// ============================================
	// TEST 4: HOT-SWAP COEFFICIENTS (AUTO-REGENERATION)
	// ============================================
	
	TEST_SECTION("Hot-Swap Coefficients (Auto-Regeneration)");
	
	ope::Processor processor4(ope::Backend::CPU);
	processor4.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	processor4.enableResampling(true);
	processor4.setInterpolationMethod(ope::InterpolationMethod::CUBIC);
	processor4.setResamplingCoefficients(RESAMPLING_COEFFS);
	processor4.enableWindowing(true);
	processor4.setWindowParameters(WINDOW_TYPE, WINDOW_CENTER, WINDOW_FILL_FACTOR);
	processor4.enableDispersionCompensation(true);
	processor4.setDispersionCoefficients(DISPERSION_COEFFS, DISPERSION_FACTOR);
	processor4.enableLogScaling(true);
	processor4.setGrayscaleRange(0.0f, 80.0f);
	
	processor4.initialize();
	
	// Process with initial coefficients
	ProcessingResult resultInitialCoeffs;
	bool initialCoeffsSuccess = processData(processor4, testData, resultInitialCoeffs);
	ASSERT_TRUE(initialCoeffsSuccess, "Processing with initial coefficients succeeded");
	ASSERT_TRUE(isOutputValid(resultInitialCoeffs.output), "Initial coefficients output valid");
	
	// Change resampling coefficients
	float newResamplingCoeffs[4] = {1.0f, 2048.0f, -500.0f, 25.0f}; // Different
	processor4.setResamplingCoefficients(newResamplingCoeffs);
	
	ProcessingResult resultNewResamplingCoeffs;
	bool newResamplingSuccess = processData(processor4, testData, resultNewResamplingCoeffs);
	ASSERT_TRUE(newResamplingSuccess, "Processing after changing resampling coeffs succeeded");
	ASSERT_TRUE(isOutputValid(resultNewResamplingCoeffs.output), "New resampling coeffs output valid");
	ASSERT_TRUE(outputsAreDifferent(resultInitialCoeffs.output, resultNewResamplingCoeffs.output),
	            "Output changed after changing resampling coefficients");
	
	// Change window parameters
	processor4.setWindowParameters(ope::WindowType::GAUSS, 0.8f, 0.1f);
	
	ProcessingResult resultNewWindowParams;
	bool newWindowSuccess = processData(processor4, testData, resultNewWindowParams);
	ASSERT_TRUE(newWindowSuccess, "Processing after changing window params succeeded");
	ASSERT_TRUE(isOutputValid(resultNewWindowParams.output), "New window params output valid");
	ASSERT_TRUE(outputsAreDifferent(resultNewResamplingCoeffs.output, resultNewWindowParams.output),
	            "Output changed after changing window parameters");
	
	// Change dispersion coefficients
	float newDispersionCoeffs[4] = {0.0f, 0.0f, 200.0f, -1.5f}; // Different
	processor4.setDispersionCoefficients(newDispersionCoeffs, 0.8f);
	
	ProcessingResult resultNewDispersionCoeffs;
	bool newDispersionSuccess = processData(processor4, testData, resultNewDispersionCoeffs);
	ASSERT_TRUE(newDispersionSuccess, "Processing after changing dispersion coeffs succeeded");
	ASSERT_TRUE(isOutputValid(resultNewDispersionCoeffs.output), "New dispersion coeffs output valid");
	ASSERT_TRUE(outputsAreDifferent(resultNewWindowParams.output, resultNewDispersionCoeffs.output),
	            "Output changed after changing dispersion coefficients");
	
	// ============================================
	// TEST 5: SIGNALLENGTH CHANGE
	// ============================================
	
	TEST_SECTION("SignalLength Change (Curve Adjustment)");
	
	ope::Processor processor5(ope::Backend::CPU);
	
	// Start with SIGNAL_LENGTH
	processor5.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	
	// Set custom curves at original size
	std::vector<float> originalSizeCurve(SIGNAL_LENGTH);
	for (int i = 0; i < SIGNAL_LENGTH; ++i) {
		originalSizeCurve[i] = 100.0f + i;
	}
	processor5.setCustomResamplingCurve(originalSizeCurve.data(), originalSizeCurve.size());
	
	const ope::ProcessorConfiguration& config5 = processor5.getConfig();
	ASSERT_EQUAL(config5.getCustomResamplingCurveSize(), (size_t)SIGNAL_LENGTH, 
	             "Initial curve size matches signalLength");
	
	// Change to smaller signalLength (should truncate)
	int smallerLength = SIGNAL_LENGTH / 2;
	processor5.setInputParameters(smallerLength, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	
	ASSERT_EQUAL(config5.getCustomResamplingCurveSize(), (size_t)smallerLength,
	             "Curve truncated when signalLength decreased");
	
	const float* truncatedData = config5.getCustomResamplingCurve();
	bool truncatedCorrect = std::equal(truncatedData, truncatedData + smallerLength, originalSizeCurve.begin());
	ASSERT_TRUE(truncatedCorrect, "Truncated curve data correct");
	
	// Change to larger signalLength (should zero-pad)
	int largerLength = SIGNAL_LENGTH * 2;
	processor5.setInputParameters(largerLength, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	
	ASSERT_EQUAL(config5.getCustomResamplingCurveSize(), (size_t)largerLength,
	             "Curve zero-padded when signalLength increased");
	
	const float* paddedData = config5.getCustomResamplingCurve();
	bool originalDataPreserved = std::equal(originalSizeCurve.begin(), originalSizeCurve.end(), paddedData);
	bool paddingZeros = true;
	for (int i = SIGNAL_LENGTH; i < largerLength; ++i) {
		if (paddedData[i] != 0.0f) {
			paddingZeros = false;
			break;
		}
	}
	
	ASSERT_TRUE(originalDataPreserved, "Original data preserved when zero-padded");
	ASSERT_TRUE(paddingZeros, "Padding is zeros when signalLength increased");
	
	// ============================================
	// TEST 6: ALL THREE CURVE TYPES (INDIVIDUAL TESTS)
	// ============================================
	
	TEST_SECTION("Individual Curve Type Testing");
	
	// Test resampling curve isolation
	{
		ope::Processor procResampling(ope::Backend::CPU);
		procResampling.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
		procResampling.enableResampling(true);
		procResampling.setInterpolationMethod(ope::InterpolationMethod::CUBIC);
		procResampling.setResamplingCoefficients(RESAMPLING_COEFFS);
		procResampling.enableWindowing(false);
		procResampling.enableDispersionCompensation(false);
		procResampling.enableLogScaling(true);
		procResampling.setGrayscaleRange(0.0f, 80.0f);
		procResampling.initialize();
		
		ProcessingResult resultResamplingGen;
		processData(procResampling, testData, resultResamplingGen);
		
		std::vector<float> customResampling(SIGNAL_LENGTH);
		for (int i = 0; i < SIGNAL_LENGTH; ++i) {
			customResampling[i] = 10.0f + i * 0.8f;
		}
		procResampling.setCustomResamplingCurve(customResampling.data(), customResampling.size());
		
		ProcessingResult resultResamplingCustom;
		processData(procResampling, testData, resultResamplingCustom);
		
		ASSERT_TRUE(isOutputValid(resultResamplingGen.output), "Resampling generated curve output valid");
		ASSERT_TRUE(isOutputValid(resultResamplingCustom.output), "Resampling custom curve output valid");
		ASSERT_TRUE(outputsAreDifferent(resultResamplingGen.output, resultResamplingCustom.output),
		            "Resampling: custom vs generated curves produce different outputs");
	}
	
	// Test window curve isolation
	{
		ope::Processor procWindow(ope::Backend::CPU);
		procWindow.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
		procWindow.enableResampling(false);
		procWindow.enableWindowing(true);
		procWindow.setWindowParameters(WINDOW_TYPE, WINDOW_CENTER, WINDOW_FILL_FACTOR);
		procWindow.enableDispersionCompensation(false);
		procWindow.enableLogScaling(true);
		procWindow.setGrayscaleRange(0.0f, 80.0f);
		procWindow.initialize();
		
		ProcessingResult resultWindowGen;
		processData(procWindow, testData, resultWindowGen);
		
		std::vector<float> customWindow(SIGNAL_LENGTH);
		for (int i = 0; i < SIGNAL_LENGTH; ++i) {
			customWindow[i] = 0.5f + 0.5f * std::sin(i * 3.14159f / SIGNAL_LENGTH);
		}
		procWindow.setCustomWindowCurve(customWindow.data(), customWindow.size());
		
		ProcessingResult resultWindowCustom;
		processData(procWindow, testData, resultWindowCustom);
		
		ASSERT_TRUE(isOutputValid(resultWindowGen.output), "Window generated curve output valid");
		ASSERT_TRUE(isOutputValid(resultWindowCustom.output), "Window custom curve output valid");
		ASSERT_TRUE(outputsAreDifferent(resultWindowGen.output, resultWindowCustom.output),
		            "Window: custom vs generated curves produce different outputs");
	}
	
	// Test dispersion curve isolation
	{
		ope::Processor procDispersion(ope::Backend::CPU);
		procDispersion.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
		procDispersion.enableResampling(false);
		procDispersion.enableWindowing(false);
		procDispersion.enableDispersionCompensation(true);
		procDispersion.setDispersionCoefficients(DISPERSION_COEFFS, DISPERSION_FACTOR);
		procDispersion.enableLogScaling(true);
		procDispersion.setGrayscaleRange(0.0f, 80.0f);
		procDispersion.initialize();
		
		ProcessingResult resultDispersionGen;
		processData(procDispersion, testData, resultDispersionGen);
		
		std::vector<float> customDispersion(SIGNAL_LENGTH);
		for (int i = 0; i < SIGNAL_LENGTH; ++i) {
			customDispersion[i] = 0.002f * i * i; // Different quadratic phase
		}
		procDispersion.setCustomDispersionCurve(customDispersion.data(), customDispersion.size());
		
		ProcessingResult resultDispersionCustom;
		processData(procDispersion, testData, resultDispersionCustom);
		
		ASSERT_TRUE(isOutputValid(resultDispersionGen.output), "Dispersion generated curve output valid");
		ASSERT_TRUE(isOutputValid(resultDispersionCustom.output), "Dispersion custom curve output valid");
		ASSERT_TRUE(outputsAreDifferent(resultDispersionGen.output, resultDispersionCustom.output),
		            "Dispersion: custom vs generated curves produce different outputs");
	}
	
	// ============================================
	// TEST 7: BOTH BACKENDS (CPU AND CUDA)
	// ============================================
	
	TEST_SECTION("Both Backends (CPU and CUDA)");
	
	// Test with CPU backend
	ope::Processor cpuProc(ope::Backend::CPU);
	cpuProc.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	cpuProc.enableResampling(true);
	cpuProc.setInterpolationMethod(ope::InterpolationMethod::CUBIC);
	cpuProc.setResamplingCoefficients(RESAMPLING_COEFFS);
	cpuProc.enableWindowing(true);
	cpuProc.setWindowParameters(WINDOW_TYPE, WINDOW_CENTER, WINDOW_FILL_FACTOR);
	cpuProc.enableDispersionCompensation(true);
	cpuProc.setDispersionCoefficients(DISPERSION_COEFFS, DISPERSION_FACTOR);
	cpuProc.enableLogScaling(true);
	cpuProc.setGrayscaleRange(0.0f, 80.0f);
	cpuProc.initialize();
	
	// Set custom curves on CPU
	std::vector<float> cpuCustomResampling(SIGNAL_LENGTH);
	std::vector<float> cpuCustomWindow(SIGNAL_LENGTH);
	std::vector<float> cpuCustomDispersion(SIGNAL_LENGTH);
	for (int i = 0; i < SIGNAL_LENGTH; ++i) {
		cpuCustomResampling[i] = 7.0f + i * 0.5f;
		cpuCustomWindow[i] = 0.5f + 0.5f * std::cos(i * 6.28f / SIGNAL_LENGTH);
		cpuCustomDispersion[i] = 0.0008f * i * i;
	}
	cpuProc.setCustomResamplingCurve(cpuCustomResampling.data(), cpuCustomResampling.size());
	cpuProc.setCustomWindowCurve(cpuCustomWindow.data(), cpuCustomWindow.size());
	cpuProc.setCustomDispersionCurve(cpuCustomDispersion.data(), cpuCustomDispersion.size());
	
	ProcessingResult cpuResult;
	bool cpuSuccess = processData(cpuProc, testData, cpuResult);
	ASSERT_TRUE(cpuSuccess, "CPU backend with custom curves succeeded");
	ASSERT_TRUE(isOutputValid(cpuResult.output), "CPU backend output valid");
	
	// Test with CUDA backend (same curves)
	ope::Processor cudaProc(ope::Backend::CUDA);
	cudaProc.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);
	cudaProc.enableResampling(true);
	cudaProc.setInterpolationMethod(ope::InterpolationMethod::CUBIC);
	cudaProc.setResamplingCoefficients(RESAMPLING_COEFFS);
	cudaProc.enableWindowing(true);
	cudaProc.setWindowParameters(WINDOW_TYPE, WINDOW_CENTER, WINDOW_FILL_FACTOR);
	cudaProc.enableDispersionCompensation(true);
	cudaProc.setDispersionCoefficients(DISPERSION_COEFFS, DISPERSION_FACTOR);
	cudaProc.enableLogScaling(true);
	cudaProc.setGrayscaleRange(0.0f, 80.0f);
	cudaProc.initialize();
	
	// Set same custom curves on CUDA
	cudaProc.setCustomResamplingCurve(cpuCustomResampling.data(), cpuCustomResampling.size());
	cudaProc.setCustomWindowCurve(cpuCustomWindow.data(), cpuCustomWindow.size());
	cudaProc.setCustomDispersionCurve(cpuCustomDispersion.data(), cpuCustomDispersion.size());
	
	ProcessingResult cudaResult;
	bool cudaSuccess = processData(cudaProc, testData, cudaResult);
	ASSERT_TRUE(cudaSuccess, "CUDA backend with custom curves succeeded");
	ASSERT_TRUE(isOutputValid(cudaResult.output), "CUDA backend output valid");
	
	// Verify CPU and CUDA produce similar results with same curves (should match within tolerance)
	bool backendsSimilar = !outputsAreDifferent(cpuResult.output, cudaResult.output, 0.1f);
	ASSERT_TRUE(backendsSimilar, "CPU and CUDA backends produce similar outputs with same curves");
	
	// ============================================
	// FINAL RESULTS
	// ============================================
	
	std::cout << "\n========================================" << std::endl;
	std::cout << "FINAL RESULTS" << std::endl;
	std::cout << "========================================" << std::endl;
	
	if (allTestsPassed) {
		std::cout << "ALL TESTS PASSED ✓" << std::endl;
		return 0;
	} else {
		std::cout << "SOME TESTS FAILED ✗" << std::endl;
		return 1;
	}
}
