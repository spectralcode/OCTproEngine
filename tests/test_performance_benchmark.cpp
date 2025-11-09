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
#include <thread>
#include <sstream>

// ============================================
// CONFIGURE BENCHMARK HERE
// ============================================

// Which backends to benchmark
const bool BENCHMARK_CPU = false;
const bool BENCHMARK_CUDA = true;

// Buffer sizes to test (will test all combinations)
const int SIGNAL_LENGTHS[] = {512, 1024, 2048, 4096};
const int ASCANS_PER_BSCAN[] = {256, 512, 1024, 2048};
const int BSCANS_PER_BUFFER[] = {1, 8};

// Number of iterations per test
const int ITERATIONS = 100;

// Processing configuration
const bool ENABLE_RESAMPLING = true;
const bool ENABLE_WINDOWING = true;
const bool ENABLE_DISPERSION = true;
const bool ENABLE_BACKGROUND_REMOVAL = true;
const bool ENABLE_LOG_SCALING = true;
const bool ENABLE_BSCAN_FLIP = false;

// Resampling configuration
const ope::InterpolationMethod INTERPOLATION_METHOD = ope::InterpolationMethod::CUBIC;
const float RESAMPLING_COEFFS[4] = {0.5f, 2048.0f, -100.0f, 0.0f};

// Windowing configuration
const ope::WindowType WINDOW_TYPE = ope::WindowType::HANN;
const float WINDOW_CENTER = 0.5f;
const float WINDOW_FILL_FACTOR = 0.95f;

// Dispersion configuration
const float DISPERSION_COEFFS[4] = {0.0f, 0.0f, 1.0f, 2.0f};
const float DISPERSION_FACTOR = 1.0f;

// Post-processing
const float GRAYSCALE_MIN = 30.0f;
const float GRAYSCALE_MAX = 100.0f;

// Output options
const bool SAVE_CSV = true;
const char* CSV_FILENAME = "benchmark_results.csv";

// ============================================
// Helper Functions
// ============================================

// Generate synthetic test data
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
	size_t samplesPerBscan = signalLength * ascansPerBscan;
	data.reserve(samplesPerBscan * bscansPerBuffer);
	
	// Generate one B-scan with synthetic data
	std::vector<uint16_t> singleBscan;
	singleBscan.reserve(samplesPerBscan);
	
	for (int a = 0; a < ascansPerBscan; ++a) {
		auto ascan = generateSyntheticAScan(signalLength, a);
		singleBscan.insert(singleBscan.end(), ascan.begin(), ascan.end());
	}
	
	// Copy the same B-scan multiple times (fast!)
	for (int b = 0; b < bscansPerBuffer; ++b) {
		data.insert(data.end(), singleBscan.begin(), singleBscan.end());
	}
	
	return data;
}


void configureProcessor(ope::Processor& processor, int signalLength, int ascansPerBscan, int bscansPerBuffer) {
	processor.setInputParameters(
		signalLength,
		ascansPerBscan,
		bscansPerBuffer,
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
}

struct BenchmarkResult {
	int signalLength;
	int ascansPerBscan;
	int bscansPerBuffer;
	std::string backend;
	
	int iterations;
	double totalTimeMs;
	double avgTimeMs;
	
	double bscansPerSec;
	double ascansPerSec;
	double mbPerSec;
	
	double speedup; 
};

BenchmarkResult runBenchmark(
	ope::Backend backend,
	int signalLength,
	int ascansPerBscan,
	int bscansPerBuffer,
	const std::vector<uint16_t>& testData)
{
	BenchmarkResult result;
	result.signalLength = signalLength;
	result.ascansPerBscan = ascansPerBscan;
	result.bscansPerBuffer = bscansPerBuffer;
	result.backend = (backend == ope::Backend::CPU) ? "CPU" : "CUDA";
	result.iterations = ITERATIONS;
	result.speedup = 1.0;
	
	ope::Processor processor(backend);
	configureProcessor(processor, signalLength, ascansPerBscan, bscansPerBuffer);
	processor.initialize();
	
	size_t dataSizeBytes = testData.size() * sizeof(uint16_t);
	
	std::atomic<int> completedIterations(0);
	processor.setOutputCallback([&completedIterations](const ope::IOBuffer& output) {
		completedIterations++;
	});
	
	auto startTime = std::chrono::high_resolution_clock::now();

	for (int iter = 0; iter < ITERATIONS; ++iter) {
		// Get next available buffer (blocks if all buffers are busy)
		ope::IOBuffer& inputBuf = processor.getNextAvailableInputBuffer();
		
		// Copy data to buffer
		std::memcpy(inputBuf.getDataPointer(), testData.data(), dataSizeBytes);
		
		// Submit for processing (returns immediately, processing happens async)
		processor.process(inputBuf);
	}
	
	// Wait for all iterations to complete
	while (completedIterations < ITERATIONS) {
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
	
	auto endTime = std::chrono::high_resolution_clock::now();
	
	// Calculate statistics
	result.totalTimeMs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0;
	result.avgTimeMs = result.totalTimeMs / ITERATIONS;
	
	// Calculate throughput
	result.bscansPerSec = 1000.0 * bscansPerBuffer / result.avgTimeMs;
	result.ascansPerSec = result.bscansPerSec * ascansPerBscan;
	
	// Calculate bandwidth (input data size)
	double bytesPerIteration = signalLength * ascansPerBscan * bscansPerBuffer * sizeof(uint16_t);
	result.mbPerSec = (bytesPerIteration * 1000.0 / result.avgTimeMs) / (1024.0 * 1024.0);
	
	return result;
}

std::string formatNumber(double value) {
	std::stringstream ss;
	ss.imbue(std::locale(""));
	ss << std::fixed << std::setprecision(0) << value;
	return ss.str();
}

void printResultsTable(const std::vector<BenchmarkResult>& results) {
	std::cout << std::endl;
	std::cout << "+" << std::string(110, '-') << "+" << std::endl;
	std::cout << "| " << std::left << std::setw(6) << "Signal"
	          << " | " << std::setw(6) << "AScans"
	          << " | " << std::setw(6) << "BScans"
	          << " | " << std::setw(7) << "Backend"
	          << " | " << std::right << std::setw(9) << "Time(ms)"
	          << " | " << std::setw(12) << "BScans/s"
	          << " | " << std::setw(14) << "AScans/s"
	          << " | " << std::setw(10) << "MB/s"
	          << " | " << std::setw(8) << "Speedup"
	          << " |" << std::endl;
	std::cout << "+" << std::string(110, '-') << "+" << std::endl;
	
	// Group by buffer configuration
	int lastSignalLength = -1;
	int lastAscans = -1;
	int lastBscans = -1;
	
	for (size_t i = 0; i < results.size(); ++i) {
		const auto& r = results[i];
		
		// Add separator between different configurations
		if (r.signalLength != lastSignalLength || r.ascansPerBscan != lastAscans || r.bscansPerBuffer != lastBscans) {
			if (i > 0) {
				std::cout << "+" << std::string(110, '-') << "+" << std::endl;
			}
			lastSignalLength = r.signalLength;
			lastAscans = r.ascansPerBscan;
			lastBscans = r.bscansPerBuffer;
		}
		
		// Format speedup string
		std::string speedupStr = "-";
		if (r.speedup > 1.0) {
			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << r.speedup << "x";
			speedupStr = ss.str();
		}
		
		std::cout << "| " << std::left << std::setw(6) << r.signalLength
		          << " | " << std::setw(6) << r.ascansPerBscan
		          << " | " << std::setw(6) << r.bscansPerBuffer
		          << " | " << std::setw(7) << r.backend
		          << " | " << std::right << std::setw(9) << std::fixed << std::setprecision(3) << r.avgTimeMs
		          << " | " << std::setw(12) << formatNumber(r.bscansPerSec)
		          << " | " << std::setw(14) << formatNumber(r.ascansPerSec)
		          << " | " << std::setw(10) << std::fixed << std::setprecision(2) << r.mbPerSec
		          << " | " << std::setw(8) << speedupStr
		          << " |" << std::endl;
	}
	
	std::cout << "+" << std::string(110, '-') << "+" << std::endl;
	std::cout << std::endl;
}

void saveResultsCSV(const std::vector<BenchmarkResult>& results, const char* filename) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open CSV file: " << filename << std::endl;
		return;
	}
	
	file << "SignalLength,AScansPerBScan,BScansPerBuffer,Backend,Iterations,"
	     << "TotalTime_ms,AvgTime_ms,BScansPerSec,AScansPerSec,MB_per_sec,Speedup" << std::endl;
	
	for (const auto& r : results) {
		file << r.signalLength << ","
		     << r.ascansPerBscan << ","
		     << r.bscansPerBuffer << ","
		     << r.backend << ","
		     << r.iterations << ","
		     << std::fixed << std::setprecision(6) << r.totalTimeMs << ","
		     << r.avgTimeMs << ","
		     << std::setprecision(2) << r.bscansPerSec << ","
		     << r.ascansPerSec << ","
		     << r.mbPerSec << ","
		     << r.speedup << std::endl;
	}
	
	file.close();
	std::cout << "Results saved to: " << filename << std::endl;
}

void printConfiguration() {
	std::cout << "Configuration:" << std::endl;
	std::cout << "  Processing: ";
	std::vector<std::string> enabled;
	if (ENABLE_RESAMPLING) {
		std::string method = (INTERPOLATION_METHOD == ope::InterpolationMethod::LINEAR ? "LINEAR" : 
		                     INTERPOLATION_METHOD == ope::InterpolationMethod::CUBIC ? "CUBIC" : "LANCZOS");
		enabled.push_back("Resampling(" + method + ")");
	}
	if (ENABLE_WINDOWING) enabled.push_back("Windowing");
	if (ENABLE_DISPERSION) enabled.push_back("Dispersion");
	if (ENABLE_BACKGROUND_REMOVAL) enabled.push_back("BG-Removal");
	if (ENABLE_LOG_SCALING) enabled.push_back("Log-Scale");
	
	for (size_t i = 0; i < enabled.size(); ++i) {
		std::cout << enabled[i];
		if (i < enabled.size() - 1) std::cout << " + ";
	}
	std::cout << std::endl;
	
	std::cout << "  Iterations per test: " << ITERATIONS << std::endl;
	std::cout << "  Backends: ";
	if (BENCHMARK_CPU) std::cout << "CPU ";
	if (BENCHMARK_CUDA) std::cout << "CUDA ";
	std::cout << std::endl;
	std::cout << std::endl;
}

// ============================================
// Main Benchmark
// ============================================

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "OCT Processing Performance Benchmark" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;
	
	printConfiguration();
	
	std::cout << "Running benchmarks..." << std::endl;
	std::cout << std::endl;
	
	std::vector<BenchmarkResult> allResults;
	
	// Calculate total number of tests
	int numSignalLengths = sizeof(SIGNAL_LENGTHS) / sizeof(SIGNAL_LENGTHS[0]);
	int numAscans = sizeof(ASCANS_PER_BSCAN) / sizeof(ASCANS_PER_BSCAN[0]);
	int numBscans = sizeof(BSCANS_PER_BUFFER) / sizeof(BSCANS_PER_BUFFER[0]);
	int numBackends = (BENCHMARK_CPU ? 1 : 0) + (BENCHMARK_CUDA ? 1 : 0);
	int totalTests = numSignalLengths * numAscans * numBscans * numBackends;
	int currentTest = 0;
	
	// Iterate through all combinations
	for (int sl = 0; sl < numSignalLengths; ++sl) {
		int signalLength = SIGNAL_LENGTHS[sl];
		
		for (int ap = 0; ap < numAscans; ++ap) {
			int ascansPerBscan = ASCANS_PER_BSCAN[ap];
			
			for (int bp = 0; bp < numBscans; ++bp) {
				int bscansPerBuffer = BSCANS_PER_BUFFER[bp];
				
				// Generate test data once for this size
				auto testData = generateTestData(signalLength, ascansPerBscan, bscansPerBuffer);
				
				BenchmarkResult cpuResult, cudaResult;
				
				// Benchmark CPU
				if (BENCHMARK_CPU) {
					currentTest++;
					std::cout << "[" << currentTest << "/" << totalTests << "] "
					          << "Testing CPU: " << signalLength << "x" << ascansPerBscan << "x" << bscansPerBuffer
					          << " ... " << std::flush;
					
					cpuResult = runBenchmark(ope::Backend::CPU, signalLength, ascansPerBscan, bscansPerBuffer, testData);
					allResults.push_back(cpuResult);
					
					std::cout << std::fixed << std::setprecision(3) << cpuResult.avgTimeMs << " ms" << std::endl;
				}
				
				// Benchmark CUDA
				if (BENCHMARK_CUDA) {
					currentTest++;
					std::cout << "[" << currentTest << "/" << totalTests << "] "
					          << "Testing CUDA: " << signalLength << "x" << ascansPerBscan << "x" << bscansPerBuffer
					          << " ... " << std::flush;
					
					cudaResult = runBenchmark(ope::Backend::CUDA, signalLength, ascansPerBscan, bscansPerBuffer, testData);
					
					// Calculate speedup if we have CPU result
					if (BENCHMARK_CPU) {
						cudaResult.speedup = cpuResult.avgTimeMs / cudaResult.avgTimeMs;
					}
					
					allResults.push_back(cudaResult);
					
					std::cout << std::fixed << std::setprecision(3) << cudaResult.avgTimeMs << " ms";
					if (BENCHMARK_CPU) {
						std::cout << " (speedup: " << std::setprecision(2) << cudaResult.speedup << "x)";
					}
					std::cout << std::endl;
				}
			}
		}
	}
	
	// Print results table
	std::cout << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Results" << std::endl;
	std::cout << "========================================" << std::endl;
	
	printResultsTable(allResults);
	
	// Save CSV if requested
	if (SAVE_CSV) {
		saveResultsCSV(allResults, CSV_FILENAME);
	}
	
	std::cout << "Benchmark complete!" << std::endl;
	
	return 0;
}