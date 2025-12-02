#include "../include/processor.h"
#include "../include/processorconfiguration.h"
#include "../include/types.h"
#include "../include/iobuffer.h"
#include "../include/version.h"
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
#include <ctime>
#include <sys/utsname.h>
#include <unistd.h>

// ============================================
// CONFIGURE BENCHMARK HERE
// ============================================

// Platform-specific configuration
#ifndef __aarch64__
// ============================================
// Desktop: Full benchmark with all backends
// ============================================
const bool BENCHMARK_CPU = true;
const bool BENCHMARK_CUDA = true;
const bool BENCHMARK_OPENCL = true;
const bool BENCHMARK_VULKAN = true;

const int SIGNAL_LENGTHS[] = {512, 1024, 2048};
const int ASCANS_PER_BSCAN[] = {256, 512, 1024};
const int BSCANS_PER_BUFFER[] = {1};
const int ITERATIONS = 100;

#else
// ============================================
// Jetson Nano: CUDA-only with reduced sizes
// ============================================
const bool BENCHMARK_CPU = false;
const bool BENCHMARK_CUDA = true;
const bool BENCHMARK_OPENCL = false;
const bool BENCHMARK_VULKAN = false;

const int SIGNAL_LENGTHS[] = {512, 1024, 2048};
const int ASCANS_PER_BSCAN[] = {32, 64, 128, 256, 512, 1024, 2048};
const int BSCANS_PER_BUFFER[] = {1};
const int ITERATIONS = 200;
#endif

// Processing configuration
const bool ENABLE_RESAMPLING = true;
const bool ENABLE_WINDOWING = true;
const bool ENABLE_DISPERSION = true;
const bool ENABLE_DC_REMOVAL = true;
const bool ENABLE_LOG_SCALING = true;
const bool ENABLE_FIXED_PATTERN_NOISE_REMOVAL = false;
const bool ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION = true;


const bool ENABLE_BSCAN_FLIP = false;

// DC Removal configuration
const int DC_REMOVAL_WINDOW_SIZE = 64;

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

// Get current timestamp in format YYYYMMDD_HHMMSSmmm (matching Recorder format)
std::string getCurrentTimestamp() {
	auto now = std::chrono::system_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
	auto timer = std::chrono::system_clock::to_time_t(now);
	
	std::tm bt;
#ifdef _WIN32
	localtime_s(&bt, &timer);
#else
	localtime_r(&timer, &bt);
#endif
	
	std::ostringstream oss;
	oss << std::put_time(&bt, "%Y%m%d_%H%M%S");
	oss << std::setfill('0') << std::setw(3) << ms.count();
	
	return oss.str();
}

// Get platform/architecture information
std::string getPlatform() {
	struct utsname info;
	if (uname(&info) == 0) {
		return std::string(info.machine);
	}
	return "unknown";
}

// Get OS information
std::string getOSInfo() {
	struct utsname info;
	if (uname(&info) == 0) {
		return std::string(info.sysname) + " " + std::string(info.release);
	}
	return "unknown";
}

// Get CPU core count
int getCPUCores() {
#ifdef _WIN32
	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	return sysinfo.dwNumberOfProcessors;
#else
	return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

// Get total RAM in GB
double getTotalRAM() {
#ifdef _WIN32
	MEMORYSTATUSEX memInfo;
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	GlobalMemoryStatusEx(&memInfo);
	return memInfo.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
#else
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	return (pages * page_size) / (1024.0 * 1024.0 * 1024.0);
#endif
}

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
	
	// Copy the same B-scan multiple times
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
	
	processor.enableBackgroundRemoval(ENABLE_DC_REMOVAL);
	if (ENABLE_DC_REMOVAL) {
		processor.setBackgroundRemovalWindowSize(DC_REMOVAL_WINDOW_SIZE);
	}

	processor.enableFixedPatternNoiseRemoval(ENABLE_FIXED_PATTERN_NOISE_REMOVAL);
	if (ENABLE_FIXED_PATTERN_NOISE_REMOVAL) {
		processor.requestFixedPatternNoiseDetermination();
	}

	processor.enablePostProcessBackgroundSubtraction(ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION);
	if (ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION) {
		processor.requestPostProcessBackgroundRecording();
	}
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
	result.backend = (backend == ope::Backend::CPU) ? "CPU" :
	                 (backend == ope::Backend::CUDA) ? "CUDA" :
	                 (backend == ope::Backend::VULKAN) ? "Vulkan" : "OpenCL";
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
	// Check if CPU was benchmarked
	bool hasCPU = false;
	for (const auto& r : results) {
		if (r.backend == "CPU") {
			hasCPU = true;
			break;
		}
	}
	
	std::cout << std::endl;
	int tableWidth = hasCPU ? 124 : 114;
	std::cout << "+" << std::string(tableWidth, '-') << "+" << std::endl;
	std::cout << "| " << std::left << std::setw(13) << "Signal length"
	          << " | " << std::setw(14) << "A-Scans/B-scan"
	          << " | " << std::setw(14) << "B-Scans/Buffer"
	          << " | " << std::setw(7) << "Backend"
	          << " | " << std::right << std::setw(10) << "Time in ms"
	          << " | " << std::setw(14) << "A-Scans/s"
	          << " | " << std::setw(12) << "B-Scans/s"
	          << " | " << std::setw(10) << "MB/s";
	if (hasCPU) {
		std::cout << " | " << std::setw(8) << "Speedup";
	}
	std::cout << " |" << std::endl;
	std::cout << "+" << std::string(tableWidth, '-') << "+" << std::endl;
	
	// Group by buffer configuration
	int lastSignalLength = -1;
	int lastAscans = -1;
	int lastBscans = -1;
	
	for (size_t i = 0; i < results.size(); ++i) {
		const auto& r = results[i];
		
		// Add separator between different configurations
		if (r.signalLength != lastSignalLength || r.ascansPerBscan != lastAscans || r.bscansPerBuffer != lastBscans) {
			if (i > 0) {
				std::cout << "+" << std::string(tableWidth, '-') << "+" << std::endl;
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
		
		std::cout << "| " << std::left << std::setw(13) << r.signalLength
		          << " | " << std::setw(14) << r.ascansPerBscan
		          << " | " << std::setw(14) << r.bscansPerBuffer
		          << " | " << std::setw(7) << r.backend
		          << " | " << std::right << std::setw(10) << std::fixed << std::setprecision(3) << r.avgTimeMs
		          << " | " << std::setw(14) << formatNumber(r.ascansPerSec)
		          << " | " << std::setw(12) << formatNumber(r.bscansPerSec)
		          << " | " << std::setw(10) << std::fixed << std::setprecision(2) << r.mbPerSec;
		if (hasCPU) {
			std::cout << " | " << std::setw(8) << speedupStr;
		}
		std::cout << " |" << std::endl;
	}
	
	std::cout << "+" << std::string(tableWidth, '-') << "+" << std::endl;
	std::cout << std::endl;
}

void saveResultsCSV(const std::vector<BenchmarkResult>& results, const std::string& timestamp) {
	// Generate filename with timestamp prefix
	std::string filename = timestamp + "_benchmark_results.csv";
	
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open CSV file: " << filename << std::endl;
		return;
	}
	
	// Check if CPU was benchmarked
	bool hasCPU = false;
	for (const auto& r : results) {
		if (r.backend == "CPU") {
			hasCPU = true;
			break;
		}
	}
	
	// Get actual input/output bit depths from processor configuration
	// Create a temporary processor to get the configuration
	ope::Processor tempProcessor(BENCHMARK_CUDA ? ope::Backend::CUDA : 
	                              BENCHMARK_CPU ? ope::Backend::CPU :
	                              BENCHMARK_OPENCL ? ope::Backend::OPENCL :
	                              ope::Backend::VULKAN);
	configureProcessor(tempProcessor, SIGNAL_LENGTHS[0], ASCANS_PER_BSCAN[0], BSCANS_PER_BUFFER[0]);
	const auto& config = tempProcessor.getConfig();
	int inputBitDepth = ope::getDataTypeBitDepth(config.dataParams.inputDataType);
	int outputBitDepth = ope::getDataTypeBitDepth(config.dataParams.outputDataType);
	
	// Determine number of columns (base 8 + optional speedup)
	int numCols = hasCPU ? 9 : 8;
	std::string emptyColumns = ",";
	for (int i = 1; i < numCols - 1; ++i) {
		emptyColumns += " ,";
	}
	
	// Write configuration header with matching column count
	file << "# Benchmark Configuration" << emptyColumns << std::endl;
	file << "Timestamp," << timestamp << emptyColumns << std::endl;
	file << "OCTproEngine_Version," << OPE_VERSION_STRING << emptyColumns << std::endl;
	file << "Platform," << getPlatform() << emptyColumns << std::endl;
	file << "OS," << getOSInfo() << emptyColumns << std::endl;
	file << "CPU_Cores," << getCPUCores() << emptyColumns << std::endl;
	file << "Total_RAM_GB," << std::fixed << std::setprecision(1) << getTotalRAM() << emptyColumns << std::endl;
	file << "InputBitDepth," << inputBitDepth << emptyColumns << std::endl;
	file << "OutputBitDepth," << outputBitDepth << emptyColumns << std::endl;
	file << "Iterations," << ITERATIONS << emptyColumns << std::endl;
	file << "Resampling," << (ENABLE_RESAMPLING ? "true" : "false") << emptyColumns << std::endl;
	
	std::string interpMethod = "NONE";
	if (ENABLE_RESAMPLING) {
		if (INTERPOLATION_METHOD == ope::InterpolationMethod::LINEAR) interpMethod = "LINEAR";
		else if (INTERPOLATION_METHOD == ope::InterpolationMethod::CUBIC) interpMethod = "CUBIC";
		else if (INTERPOLATION_METHOD == ope::InterpolationMethod::LANCZOS) interpMethod = "LANCZOS";
	}
	file << "ResamplingMethod," << interpMethod << emptyColumns << std::endl;
	
	file << "Windowing," << (ENABLE_WINDOWING ? "true" : "false") << emptyColumns << std::endl;
	file << "Dispersion," << (ENABLE_DISPERSION ? "true" : "false") << emptyColumns << std::endl;
	file << "DC-Removal," << (ENABLE_DC_REMOVAL ? "true" : "false") << emptyColumns << std::endl;
	file << "DC-WindowSize," << DC_REMOVAL_WINDOW_SIZE << emptyColumns << std::endl;
	file << "FPN-Removal," << (ENABLE_FIXED_PATTERN_NOISE_REMOVAL ? "true" : "false") << emptyColumns << std::endl;
	file << "PostProcessBackgroundSubtraction," << (ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION ? "true" : "false") << emptyColumns << std::endl;
	file << "LogScaling," << (ENABLE_LOG_SCALING ? "true" : "false") << emptyColumns << std::endl;
	file << emptyColumns << std::endl;
	
	// Write results header
	file << "# Benchmark Results" << emptyColumns << std::endl;
	file << "Signal length,A-Scans/B-scan,B-Scans/Buffer,Backend,Time in ms,A-Scans/s,B-Scans/s,MB/s";
	if (hasCPU) {
		file << ",Speedup";
	}
	file << std::endl;
	
	for (const auto& r : results) {
		file << r.signalLength << ","
		     << r.ascansPerBscan << ","
		     << r.bscansPerBuffer << ","
		     << r.backend << ","
		     << std::fixed << std::setprecision(3) << r.avgTimeMs << ","
		     << std::setprecision(2) << r.ascansPerSec << ","
		     << r.bscansPerSec << ","
		     << r.mbPerSec;
		if (hasCPU) {
			file << "," << r.speedup;
		}
		file << std::endl;
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
	if (ENABLE_DC_REMOVAL) enabled.push_back("DC-Removal");
	if (ENABLE_LOG_SCALING) enabled.push_back("Log-Scale");
	if (ENABLE_FIXED_PATTERN_NOISE_REMOVAL) enabled.push_back("FPN-Removal");
	if (ENABLE_POST_PROCESS_BACKGROUND_SUBTRACTION) enabled.push_back("PostProcess-BG-Sub");
	
	for (size_t i = 0; i < enabled.size(); ++i) {
		std::cout << enabled[i];
		if (i < enabled.size() - 1) std::cout << " + ";
	}
	std::cout << std::endl;
	
	std::cout << "  Iterations per test: " << ITERATIONS << std::endl;
	std::cout << "  Backends: ";
	if (BENCHMARK_CPU) std::cout << "CPU ";
	if (BENCHMARK_CUDA) std::cout << "CUDA ";
	if (BENCHMARK_OPENCL) std::cout << "OpenCL ";
	if (BENCHMARK_VULKAN) std::cout << "Vulkan ";
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
	
	// Calculate total number of tests
	int numSignalLengths = sizeof(SIGNAL_LENGTHS) / sizeof(SIGNAL_LENGTHS[0]);
	int numAscans = sizeof(ASCANS_PER_BSCAN) / sizeof(ASCANS_PER_BSCAN[0]);
	int numBscans = sizeof(BSCANS_PER_BUFFER) / sizeof(BSCANS_PER_BUFFER[0]);
	int numConfigs = numSignalLengths * numAscans * numBscans;
	int numBackends = (BENCHMARK_CPU ? 1 : 0) + (BENCHMARK_CUDA ? 1 : 0) + (BENCHMARK_OPENCL ? 1 : 0) + (BENCHMARK_VULKAN ? 1 : 0);
	int totalTests = numConfigs;// * numBackends;
	int currentTest = 0;

	// Store results per backend
	std::vector<BenchmarkResult> cpuResults;
	std::vector<BenchmarkResult> cudaResults;
	std::vector<BenchmarkResult> openclResults;
	std::vector<BenchmarkResult> vulkanResults;
	
	// Helper lambda to run all configurations for a backend
	auto runAllConfigs = [&](ope::Backend backend, const std::string& backendName, std::vector<BenchmarkResult>& results) {
		std::cout << "--- " << backendName << " Backend ---" << std::endl;
		
		for (int sl = 0; sl < numSignalLengths; ++sl) {
			int signalLength = SIGNAL_LENGTHS[sl];
			
			for (int ap = 0; ap < numAscans; ++ap) {
				int ascansPerBscan = ASCANS_PER_BSCAN[ap];
				
				for (int bp = 0; bp < numBscans; ++bp) {
					int bscansPerBuffer = BSCANS_PER_BUFFER[bp];
					
					currentTest++;
					std::cout << "[" << currentTest << "/" << totalTests << "] "
					          << "Testing " << backendName << ": " << signalLength << "x" << ascansPerBscan << "x" << bscansPerBuffer
					          << " ... " << std::flush;
					
					auto testData = generateTestData(signalLength, ascansPerBscan, bscansPerBuffer);
					auto result = runBenchmark(backend, signalLength, ascansPerBscan, bscansPerBuffer, testData);
					results.push_back(result);
					
					std::cout << std::fixed << std::setprecision(3) << result.avgTimeMs << " ms" << std::endl;
				}
			}
		}
		std::cout << std::endl;
	};
	
	// Run benchmarks backend by backend
	if (BENCHMARK_CPU) {
		runAllConfigs(ope::Backend::CPU, "CPU", cpuResults);
	}
	
	if (BENCHMARK_CUDA) {
		runAllConfigs(ope::Backend::CUDA, "CUDA", cudaResults);
	}
	
	if (BENCHMARK_OPENCL) {
		runAllConfigs(ope::Backend::OPENCL, "OpenCL", openclResults);
	}

	if (BENCHMARK_VULKAN) {
		runAllConfigs(ope::Backend::VULKAN, "Vulkan", vulkanResults);
	}

	// Calculate speedups relative to CPU
	if (BENCHMARK_CPU) {
		for (size_t i = 0; i < cpuResults.size(); ++i) {
			if (BENCHMARK_CUDA && i < cudaResults.size()) {
				cudaResults[i].speedup = cpuResults[i].avgTimeMs / cudaResults[i].avgTimeMs;
			}
			if (BENCHMARK_OPENCL && i < openclResults.size()) {
				openclResults[i].speedup = cpuResults[i].avgTimeMs / openclResults[i].avgTimeMs;
			}
			if (BENCHMARK_VULKAN && i < vulkanResults.size()) {
				vulkanResults[i].speedup = cpuResults[i].avgTimeMs / vulkanResults[i].avgTimeMs;
			}
		}
	}
	
	// Merge results grouped by configuration (CPU, CUDA, OpenCL, Vulkan for each config)
	std::vector<BenchmarkResult> allResults;
	for (int i = 0; i < numConfigs; ++i) {
		if (BENCHMARK_CPU && i < static_cast<int>(cpuResults.size())) {
			allResults.push_back(cpuResults[i]);
		}
		if (BENCHMARK_CUDA && i < static_cast<int>(cudaResults.size())) {
			allResults.push_back(cudaResults[i]);
		}
		if (BENCHMARK_OPENCL && i < static_cast<int>(openclResults.size())) {
			allResults.push_back(openclResults[i]);
		}
		if (BENCHMARK_VULKAN && i < static_cast<int>(vulkanResults.size())) {
			allResults.push_back(vulkanResults[i]);
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
		std::string timestamp = getCurrentTimestamp();
		saveResultsCSV(allResults, timestamp);
	}
	
	std::cout << "Benchmark complete!" << std::endl;
	
	return 0;
}