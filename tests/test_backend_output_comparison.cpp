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


const int SIGNAL_LENGTH = 2048;
const int ASCANS_PER_BSCAN = 1024;
const int BSCANS_PER_BUFFER = 1;

const bool ENABLE_RESAMPLING = true;
const bool ENABLE_WINDOWING = true;
const bool ENABLE_DISPERSION = true;
const bool ENABLE_BACKGROUND_REMOVAL = true;
const bool ENABLE_LOG_SCALING = true;
const bool ENABLE_BSCAN_FLIP = false;
const bool ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION = true;

const ope::InterpolationMethod INTERPOLATION_METHOD = ope::InterpolationMethod::CUBIC;
const float RESAMPLING_COEFFS[4] = {0.5f, 2048.0f, -100.0f, 50.0f};

const ope::WindowType WINDOW_TYPE = ope::WindowType::HANN;
const float WINDOW_CENTER = 0.9f;
const float WINDOW_FILL_FACTOR = 0.85f;

const float DISPERSION_COEFFS[4] = {0.0f, 0.0f, 1.0f, -3.0f};
const float DISPERSION_FACTOR = 1.0f;

const float GRAYSCALE_MIN = 0.0f;
const float GRAYSCALE_MAX = 80.0f;

// tolerance for comparison
const float TOLERANCE = 0.05f;

const bool SAVE_OUTPUTS = true;




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
	std::chrono::high_resolution_clock::time_point startTime;
	std::chrono::high_resolution_clock::time_point endTime;
	bool received = false;
	std::mutex mutex;
	std::condition_variable cv;
	
	void waitForCompletion(int timeoutMs = 5000) {
		std::unique_lock<std::mutex> lock(mutex);
		cv.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this] { return received; });
	}
	
	double getDurationMs() const {
		return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0;
	}
};

// Compare two float buffers
struct ComparisonResult {
	bool match;
	double maxAbsDiff;
	double meanAbsDiff;
	double rmsError;
	size_t totalSamples;
	size_t differingSamples;
	
	// Per-A-scan statistics
	std::vector<double> maxDiffPerAscan;
};

ComparisonResult compareBuffers(
	const float* buffer1,
	const float* buffer2,
	size_t numSamples,
	int samplesPerAscan,
	float tolerance)
{
	ComparisonResult result;
	result.match = true;
	result.maxAbsDiff = 0.0;
	result.meanAbsDiff = 0.0;
	result.rmsError = 0.0;
	result.totalSamples = numSamples;
	result.differingSamples = 0;
	
	double sumAbsDiff = 0.0;
	double sumSqDiff = 0.0;
	
	// Calculate per-A-scan max differences
	int numAscans = numSamples / samplesPerAscan;
	result.maxDiffPerAscan.resize(numAscans, 0.0);
	
	for (size_t i = 0; i < numSamples; ++i) {
		float diff = std::abs(buffer1[i] - buffer2[i]);
		
		if (diff > tolerance) {
			result.differingSamples++;
			result.match = false;
		}
		
		result.maxAbsDiff = std::max(result.maxAbsDiff, static_cast<double>(diff));
		sumAbsDiff += diff;
		sumSqDiff += diff * diff;
		
		// Track per-A-scan max
		int ascanIdx = i / samplesPerAscan;
		result.maxDiffPerAscan[ascanIdx] = std::max(result.maxDiffPerAscan[ascanIdx], static_cast<double>(diff));
	}
	
	result.meanAbsDiff = sumAbsDiff / numSamples;
	result.rmsError = std::sqrt(sumSqDiff / numSamples);
	
	return result;
}

bool saveRawData(const std::string& filename, const void* data, size_t sizeInBytes) {
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return false;
	}
	file.write(reinterpret_cast<const char*>(data), sizeInBytes);
	return true;
}

void configureProcessor(ope::Processor& processor) {
	processor.setInputParameters(
		SIGNAL_LENGTH,
		ASCANS_PER_BSCAN,
		BSCANS_PER_BUFFER,
		ope::DataType::UINT16
	);

	processor.enableResampling(ENABLE_RESAMPLING);
	if (ENABLE_RESAMPLING) {
		processor.setInterpolationMethod(INTERPOLATION_METHOD);
		processor.setResamplingCoefficients(RESAMPLING_COEFFS);
	}
	
	processor.enableWindowing(ENABLE_WINDOWING);
	if (ENABLE_WINDOWING) {
		processor.setWindowParameters(
			WINDOW_TYPE,
			WINDOW_CENTER,
			WINDOW_FILL_FACTOR
		);
	}
	
	processor.enableDispersionCompensation(ENABLE_DISPERSION);
	if (ENABLE_DISPERSION) {
		processor.setDispersionCoefficients(
			DISPERSION_COEFFS,
			DISPERSION_FACTOR
		);
	}
	
	processor.enableBackgroundRemoval(ENABLE_BACKGROUND_REMOVAL);
	processor.enableLogScaling(ENABLE_LOG_SCALING);
	processor.setGrayscaleRange(GRAYSCALE_MIN, GRAYSCALE_MAX);
	processor.enableBscanFlip(ENABLE_BSCAN_FLIP);
	processor.enablePostProcessBackgroundSubtraction(ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION);

	if(ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION){
		processor.requestPostProcessBackgroundRecording();
	}
}

// ============================================
// Main Test
// ============================================

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "CPU vs CUDA Comparison Test" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;
	
	// Print configuration
	std::cout << "Configuration:" << std::endl;
	std::cout << "  Dimensions: " << SIGNAL_LENGTH << " x " << ASCANS_PER_BSCAN << " x " << BSCANS_PER_BUFFER << std::endl;
	std::cout << "  Resampling: " << (ENABLE_RESAMPLING ? "ON" : "OFF");
	if (ENABLE_RESAMPLING) {
		std::cout << " (" << (INTERPOLATION_METHOD == ope::InterpolationMethod::LINEAR ? "LINEAR" : 
		                      INTERPOLATION_METHOD == ope::InterpolationMethod::CUBIC ? "CUBIC" : "LANCZOS") << ")";
	}
	std::cout << std::endl;
	std::cout << "  Windowing: " << (ENABLE_WINDOWING ? "ON" : "OFF") << std::endl;
	std::cout << "  Dispersion: " << (ENABLE_DISPERSION ? "ON" : "OFF") << std::endl;
	std::cout << "  Log Scaling: " << (ENABLE_LOG_SCALING ? "ON" : "OFF") << std::endl;
	std::cout << "  Tolerance: " << TOLERANCE << std::endl;
	std::cout << std::endl;
	
	// Generate test data
	std::cout << "Generating test data..." << std::endl;
	auto testData = generateTestData(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER);
	size_t dataSizeBytes = testData.size() * sizeof(uint16_t);
	std::cout << "  Generated " << testData.size() << " samples (" << (dataSizeBytes / 1024.0) << " KB)" << std::endl;
	std::cout << std::endl;
	
	// Save input data if requested
	if (SAVE_OUTPUTS) {
		saveRawData("input.raw", testData.data(), dataSizeBytes);
		std::cout << "  Saved: input.raw" << std::endl;
	}
	
	// ============================================
	// CPU Backend
	// ============================================
	std::cout << "Processing with CPU backend..." << std::endl;
	
	ope::Processor cpuProcessor(ope::Backend::CPU);
	configureProcessor(cpuProcessor);
	cpuProcessor.initialize();
	
	ProcessingResult cpuResult;
	cpuProcessor.setOutputCallback([&cpuResult](const ope::IOBuffer& output) {
		std::lock_guard<std::mutex> lock(cpuResult.mutex);
		cpuResult.endTime = std::chrono::high_resolution_clock::now();
		
		size_t numFloats = output.getSizeInBytes() / sizeof(float);
		cpuResult.output.resize(numFloats);
		std::memcpy(cpuResult.output.data(), output.getDataPointer(), output.getSizeInBytes());
		
		cpuResult.received = true;
		cpuResult.cv.notify_one();
	});
	
	cpuResult.startTime = std::chrono::high_resolution_clock::now();
	ope::IOBuffer& cpuInputBuf = cpuProcessor.getNextAvailableInputBuffer();
	std::memcpy(cpuInputBuf.getDataPointer(), testData.data(), dataSizeBytes);
	cpuProcessor.process(cpuInputBuf);
	
	cpuResult.waitForCompletion();
	
	if (!cpuResult.received || cpuResult.output.empty()) {
		std::cerr << "  ERROR: No output from CPU backend!" << std::endl;
		return 1;
	}
	
	std::cout << "  Output size: " << cpuResult.output.size() << " samples" << std::endl;
	std::cout << "  Processing time: " << std::fixed << std::setprecision(3) 
	          << cpuResult.getDurationMs() << " ms" << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// CUDA Backend
	// ============================================
	std::cout << "Processing with CUDA backend..." << std::endl;
	
	ope::Processor cudaProcessor(ope::Backend::CUDA);
	configureProcessor(cudaProcessor);
	cudaProcessor.initialize();
	
	ProcessingResult cudaResult;
	cudaProcessor.setOutputCallback([&cudaResult](const ope::IOBuffer& output) {
		std::lock_guard<std::mutex> lock(cudaResult.mutex);
		cudaResult.endTime = std::chrono::high_resolution_clock::now();
		
		size_t numFloats = output.getSizeInBytes() / sizeof(float);
		cudaResult.output.resize(numFloats);
		std::memcpy(cudaResult.output.data(), output.getDataPointer(), output.getSizeInBytes());
		
		cudaResult.received = true;
		cudaResult.cv.notify_one();
	});
	
	cudaResult.startTime = std::chrono::high_resolution_clock::now();
	ope::IOBuffer& cudaInputBuf = cudaProcessor.getNextAvailableInputBuffer();
	std::memcpy(cudaInputBuf.getDataPointer(), testData.data(), dataSizeBytes);
	cudaProcessor.process(cudaInputBuf);
	
	cudaResult.waitForCompletion();
	
	if (!cudaResult.received || cudaResult.output.empty()) {
		std::cerr << "  ERROR: No output from CUDA backend!" << std::endl;
		return 1;
	}
	
	std::cout << "  Output size: " << cudaResult.output.size() << " samples" << std::endl;
	std::cout << "  Processing time: " << std::fixed << std::setprecision(3) 
	          << cudaResult.getDurationMs() << " ms" << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Performance Comparison
	// ============================================
	double speedup = cpuResult.getDurationMs() / cudaResult.getDurationMs();
	std::cout << "Performance:" << std::endl;
	std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Compare Results
	// ============================================
	std::cout << "Comparing results..." << std::endl;
	
	if (cpuResult.output.size() != cudaResult.output.size()) {
		std::cerr << "  ERROR: Output size mismatch!" << std::endl;
		std::cerr << "    CPU: " << cpuResult.output.size() << " samples" << std::endl;
		std::cerr << "    CUDA: " << cudaResult.output.size() << " samples" << std::endl;
		return 1;
	}
	
	int samplesPerAscan = SIGNAL_LENGTH / 2;  // Output is half due to truncation
	auto comparison = compareBuffers(
		cpuResult.output.data(),
		cudaResult.output.data(),
		cpuResult.output.size(),
		samplesPerAscan,
		TOLERANCE
	);
	
	std::cout << "  Max absolute difference: " << std::scientific << std::setprecision(6) 
	          << comparison.maxAbsDiff << std::endl;
	std::cout << "  Mean absolute difference: " << comparison.meanAbsDiff << std::endl;
	std::cout << "  RMS error: " << comparison.rmsError << std::endl;
	std::cout << "  Differing samples: " << comparison.differingSamples << " / " 
	          << comparison.totalSamples;
	if (comparison.totalSamples > 0) {
		std::cout << " (" << std::fixed << std::setprecision(2) 
		          << (100.0 * comparison.differingSamples / comparison.totalSamples) << "%)";
	}
	std::cout << std::endl;
	std::cout << std::endl;
	
	// Per-A-scan statistics (important for catching boundary issues!)
	std::cout << "  First A-scan max diff: " << std::scientific << std::setprecision(6) 
	          << comparison.maxDiffPerAscan[0] << std::endl;
	if (comparison.maxDiffPerAscan.size() > 1) {
		std::cout << "  Last A-scan max diff: " << comparison.maxDiffPerAscan.back() << std::endl;
	}
	std::cout << std::endl;
	
	// ============================================
	// Save Outputs
	// ============================================
	if (SAVE_OUTPUTS) {
		std::cout << "Saving outputs..." << std::endl;
		saveRawData("output_cpu.raw", cpuResult.output.data(), cpuResult.output.size() * sizeof(float));
		saveRawData("output_cuda.raw", cudaResult.output.data(), cudaResult.output.size() * sizeof(float));
		
		std::cout << "  Saved: output_cpu.raw" << std::endl;
		std::cout << "  Saved: output_cuda.raw" << std::endl;
		std::cout << std::endl;
	}
	
	// ============================================
	// Result
	// ============================================
	if (comparison.match) {
		std::cout << "TEST PASSED" << std::endl;
		std::cout << "CPU and CUDA outputs match within tolerance." << std::endl;
		return 0;
	} else {
		std::cout << "TEST FAILED" << std::endl;
		std::cout << "CPU and CUDA outputs differ beyond tolerance." << std::endl;
		return 1;
	}
}