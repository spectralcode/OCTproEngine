#include "octproviewer_benchmark.h"

#include "imgui.h"
#include <GLFW/glfw3.h>

#include "processor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

namespace {

#ifndef __aarch64__
const int BENCHMARK_SIGNAL_LENGTHS[] = {512, 1024, 2048};
const int BENCHMARK_ASCANS_PER_BSCAN[] = {256, 512, 1024};
const int BENCHMARK_BSCANS_PER_BUFFER[] = {1};
const int BENCHMARK_ITERATIONS = 1000;
#else
const int BENCHMARK_SIGNAL_LENGTHS[] = {512, 1024, 2048};
const int BENCHMARK_ASCANS_PER_BSCAN[] = {32, 64, 128, 256, 512, 1024, 2048};
const int BENCHMARK_BSCANS_PER_BUFFER[] = {1};
const int BENCHMARK_ITERATIONS = 20000;
#endif

const ope::InterpolationMethod BENCHMARK_INTERPOLATION = ope::InterpolationMethod::CUBIC;
const float BENCHMARK_RESAMPLING_COEFFS[4] = {0.5f, 2048.0f, -100.0f, 0.0f};
const ope::WindowType BENCHMARK_WINDOW_TYPE = ope::WindowType::HANN;
const float BENCHMARK_WINDOW_CENTER = 0.5f;
const float BENCHMARK_WINDOW_FILL = 0.95f;
const float BENCHMARK_DISPERSION_COEFFS[4] = {0.0f, 0.0f, 1.0f, 2.0f};
const float BENCHMARK_DISPERSION_FACTOR = 1.0f;
const float BENCHMARK_GRAYSCALE_MIN = 30.0f;
const float BENCHMARK_GRAYSCALE_MAX = 100.0f;

std::string backendName(ope::Backend backend) {
	return backend == ope::Backend::CPU ? "CPU" :
	       backend == ope::Backend::CUDA ? "CUDA" :
	       backend == ope::Backend::OPENCL ? "OpenCL" : "Vulkan";
}

bool sameBenchmarkConfig(const BenchmarkResult& a, const BenchmarkResult& b) {
	return a.signalLength == b.signalLength &&
	       a.ascansPerBscan == b.ascansPerBscan &&
	       a.bscansPerBuffer == b.bscansPerBuffer;
}

void calculateBenchmarkSpeedups(std::vector<BenchmarkResult>& results) {
	for (auto& result : results) {
		result.speedup = result.backend == "CPU" ? 1.0 : 0.0;
	}

	for (auto& result : results) {
		if (result.backend == "CPU") continue;

		for (const auto& candidate : results) {
			if (candidate.backend == "CPU" && sameBenchmarkConfig(result, candidate) && result.avgTimeMs > 0.0) {
				result.speedup = candidate.avgTimeMs / result.avgTimeMs;
				break;
			}
		}
	}
}

std::string buildPresetSummary() {
	std::ostringstream ss;
	ss << "Signal lengths: ";
	for (size_t i = 0; i < sizeof(BENCHMARK_SIGNAL_LENGTHS) / sizeof(BENCHMARK_SIGNAL_LENGTHS[0]); ++i) {
		if (i > 0) ss << ", ";
		ss << BENCHMARK_SIGNAL_LENGTHS[i];
	}

	ss << " | A-scans/B-scan: ";
	for (size_t i = 0; i < sizeof(BENCHMARK_ASCANS_PER_BSCAN) / sizeof(BENCHMARK_ASCANS_PER_BSCAN[0]); ++i) {
		if (i > 0) ss << ", ";
		ss << BENCHMARK_ASCANS_PER_BSCAN[i];
	}

	ss << " | B-scans/Buffer: ";
	for (size_t i = 0; i < sizeof(BENCHMARK_BSCANS_PER_BUFFER) / sizeof(BENCHMARK_BSCANS_PER_BUFFER[0]); ++i) {
		if (i > 0) ss << ", ";
		ss << BENCHMARK_BSCANS_PER_BUFFER[i];
	}

	ss << " | Iterations: " << BENCHMARK_ITERATIONS;
	return ss.str();
}

void setBenchmarkStatus(BenchmarkState& benchmark, const std::string& status) {
	std::lock_guard<std::mutex> lock(benchmark.mutex);
	benchmark.status = status;
}

void appendBenchmarkError(BenchmarkState& benchmark, const std::string& error) {
	std::lock_guard<std::mutex> lock(benchmark.mutex);
	if (!benchmark.error.empty()) {
		benchmark.error += "\n";
	}
	benchmark.error += error;
}

void updateSharedBenchmarkResults(BenchmarkState& benchmark, const std::vector<BenchmarkResult>& results) {
	std::lock_guard<std::mutex> lock(benchmark.mutex);
	benchmark.results = results;
}

std::string getBenchmarkTimestamp() {
	auto now = std::chrono::system_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
	auto timer = std::chrono::system_clock::to_time_t(now);

	std::tm tmValue;
#ifdef _WIN32
	localtime_s(&tmValue, &timer);
#else
	localtime_r(&timer, &tmValue);
#endif

	std::ostringstream ss;
	ss << std::put_time(&tmValue, "%Y%m%d_%H%M%S")
	   << std::setfill('0') << std::setw(3) << ms.count();
	return ss.str();
}

bool saveBenchmarkResultsCsv(const std::vector<BenchmarkResult>& results, std::string* outputPath, std::string* errorMessage) {
	const std::string filename = getBenchmarkTimestamp() + "_viewer_benchmark_results.csv";
	std::ofstream file(filename);
	if (!file.is_open()) {
		if (errorMessage) *errorMessage = "Failed to open CSV file for writing: " + filename;
		return false;
	}

	file << "SignalLength,AscansPerBscan,BscansPerBuffer,Backend,Iterations,AvgTimeMs,BscansPerSec,MBPerSec,Speedup\n";
	for (const auto& result : results) {
		file << result.signalLength << ","
		     << result.ascansPerBscan << ","
		     << result.bscansPerBuffer << ","
		     << result.backend << ","
		     << result.iterations << ","
		     << std::fixed << std::setprecision(6) << result.avgTimeMs << ","
		     << result.bscansPerSec << ","
		     << result.mbPerSec << ","
		     << result.speedup << "\n";
	}

	if (outputPath) *outputPath = filename;
	return true;
}

std::vector<uint16_t> generateBenchmarkAScan(int signalLength, int ascanIndex) {
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
		value += (std::rand() % 400) - 200;
		ascan[i] = static_cast<uint16_t>(std::max(0.0, std::min(65535.0, value)));
	}

	return ascan;
}

std::vector<uint16_t> generateBenchmarkData(int signalLength, int ascansPerBscan, int bscansPerBuffer) {
	std::vector<uint16_t> data;
	const size_t samplesPerBscan = static_cast<size_t>(signalLength) * ascansPerBscan;
	data.reserve(samplesPerBscan * bscansPerBuffer);

	std::vector<uint16_t> singleBscan;
	singleBscan.reserve(samplesPerBscan);

	for (int ascan = 0; ascan < ascansPerBscan; ++ascan) {
		auto generated = generateBenchmarkAScan(signalLength, ascan);
		singleBscan.insert(singleBscan.end(), generated.begin(), generated.end());
	}

	for (int bscan = 0; bscan < bscansPerBuffer; ++bscan) {
		data.insert(data.end(), singleBscan.begin(), singleBscan.end());
	}

	return data;
}

void configureBenchmarkProcessor(ope::Processor& processor, int signalLength, int ascansPerBscan, int bscansPerBuffer) {
	processor.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);

	processor.enableResampling(true);
	processor.setInterpolationMethod(BENCHMARK_INTERPOLATION);
	processor.setResamplingCoefficients(BENCHMARK_RESAMPLING_COEFFS);

	processor.enableWindowing(true);
	processor.setWindowParameters(BENCHMARK_WINDOW_TYPE, BENCHMARK_WINDOW_CENTER, BENCHMARK_WINDOW_FILL);

	processor.enableDispersionCompensation(true);
	processor.setDispersionCoefficients(BENCHMARK_DISPERSION_COEFFS, BENCHMARK_DISPERSION_FACTOR);

	processor.enableBackgroundRemoval(false);
	processor.enableFixedPatternNoiseRemoval(false);
	processor.enablePostProcessBackgroundSubtraction(false);
	processor.enableLogScaling(true);
	processor.setGrayscaleRange(BENCHMARK_GRAYSCALE_MIN, BENCHMARK_GRAYSCALE_MAX);
	processor.enableBscanFlip(false);
}

bool runBenchmarkPreset(
	ope::Backend backend,
	int signalLength,
	int ascansPerBscan,
	int bscansPerBuffer,
	const std::vector<uint16_t>& testData,
	std::atomic<bool>& cancelRequested,
	BenchmarkResult* result)
{
	BenchmarkResult localResult;
	localResult.signalLength = signalLength;
	localResult.ascansPerBscan = ascansPerBscan;
	localResult.bscansPerBuffer = bscansPerBuffer;
	localResult.backend = backendName(backend);

	std::atomic<int> completedIterations(0);
	const size_t outputBufferSize = static_cast<size_t>(signalLength / 2) * ascansPerBscan * bscansPerBuffer * sizeof(float);
	std::vector<uint8_t> tempBuffer(outputBufferSize);

	ope::Processor processor(backend);
	configureBenchmarkProcessor(processor, signalLength, ascansPerBscan, bscansPerBuffer);
	processor.initialize();

	processor.addOutputCallback([&completedIterations, &tempBuffer](const ope::IOBuffer& output) {
		const size_t copySize = std::min(output.getSizeInBytes(), tempBuffer.size());
		std::memcpy(tempBuffer.data(), output.getDataPointer(), copySize);
		completedIterations++;
	});

	const size_t dataSizeBytes = testData.size() * sizeof(uint16_t);
	int submittedIterations = 0;
	auto startTime = std::chrono::high_resolution_clock::now();

	for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
		if (cancelRequested.load()) break;

		ope::IOBuffer& inputBuffer = processor.getNextAvailableInputBuffer();
		std::memcpy(inputBuffer.getDataPointer(), testData.data(), dataSizeBytes);
		processor.process(inputBuffer);
		submittedIterations++;
	}

	while (completedIterations.load() < submittedIterations) {
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	if (cancelRequested.load() || submittedIterations == 0) {
		return false;
	}

	localResult.iterations = submittedIterations;
	const double totalTimeMs = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0;
	localResult.avgTimeMs = totalTimeMs / submittedIterations;
	localResult.bscansPerSec = 1000.0 * bscansPerBuffer / localResult.avgTimeMs;

	const double bytesPerIteration = static_cast<double>(signalLength) * ascansPerBscan * bscansPerBuffer * sizeof(uint16_t);
	localResult.mbPerSec = (bytesPerIteration * 1000.0 / localResult.avgTimeMs) / (1024.0 * 1024.0);

	*result = localResult;
	return true;
}

} // namespace

void refreshBenchmarkAvailability(BenchmarkState& benchmark) {
	benchmark.cpuAvailable = ope::BackendUtils::isCpuAvailable();
	benchmark.cudaAvailable = ope::BackendUtils::isCudaAvailable();
	benchmark.openclAvailable = ope::BackendUtils::isOpenCLAvailable();
	benchmark.vulkanAvailable = ope::BackendUtils::isVulkanAvailable();

	if (!benchmark.optionsInitialized) {
		benchmark.includeCpu = benchmark.cpuAvailable;
		benchmark.includeCuda = benchmark.cudaAvailable;
		benchmark.includeOpencl = benchmark.openclAvailable;
		benchmark.includeVulkan = benchmark.vulkanAvailable;
		benchmark.optionsInitialized = true;
	} else {
		if (!benchmark.cpuAvailable) benchmark.includeCpu = false;
		if (!benchmark.cudaAvailable) benchmark.includeCuda = false;
		if (!benchmark.openclAvailable) benchmark.includeOpencl = false;
		if (!benchmark.vulkanAvailable) benchmark.includeVulkan = false;
	}
}

void finalizeBenchmarkThread(BenchmarkState& benchmark) {
	if (!benchmark.running.load() && benchmark.worker.joinable()) {
		benchmark.worker.join();
	}
}

void stopBenchmarkThread(BenchmarkState& benchmark) {
	if (benchmark.worker.joinable()) {
		benchmark.cancelRequested = true;
		benchmark.worker.join();
		benchmark.running = false;
	}
}

void startBenchmarkThread(BenchmarkState& benchmark) {
	refreshBenchmarkAvailability(benchmark);

	const bool runCpu = benchmark.includeCpu && benchmark.cpuAvailable;
	const bool runCuda = benchmark.includeCuda && benchmark.cudaAvailable;
	const bool runOpencl = benchmark.includeOpencl && benchmark.openclAvailable;
	const bool runVulkan = benchmark.includeVulkan && benchmark.vulkanAvailable;

	if (!(runCpu || runCuda || runOpencl || runVulkan) || benchmark.running.load()) {
		return;
	}

	finalizeBenchmarkThread(benchmark);

	const int numConfigs =
		static_cast<int>(sizeof(BENCHMARK_SIGNAL_LENGTHS) / sizeof(BENCHMARK_SIGNAL_LENGTHS[0])) *
		static_cast<int>(sizeof(BENCHMARK_ASCANS_PER_BSCAN) / sizeof(BENCHMARK_ASCANS_PER_BSCAN[0])) *
		static_cast<int>(sizeof(BENCHMARK_BSCANS_PER_BUFFER) / sizeof(BENCHMARK_BSCANS_PER_BUFFER[0]));
	const int selectedBackends =
		(runCpu ? 1 : 0) + (runCuda ? 1 : 0) + (runOpencl ? 1 : 0) + (runVulkan ? 1 : 0);

	benchmark.cancelRequested = false;
	benchmark.running = true;
	benchmark.completedTests = 0;
	benchmark.totalTests = numConfigs * selectedBackends;
	benchmark.showResultsWindow = true;

	{
		std::lock_guard<std::mutex> lock(benchmark.mutex);
		benchmark.results.clear();
		benchmark.status = "Preparing benchmark...";
		benchmark.csvPath.clear();
		benchmark.error.clear();
	}

	benchmark.worker = std::thread([&benchmark, runCpu, runCuda, runOpencl, runVulkan]() {
		try {
			std::vector<ope::Backend> selected;
			if (runCpu) selected.push_back(ope::Backend::CPU);
			if (runCuda) selected.push_back(ope::Backend::CUDA);
			if (runOpencl) selected.push_back(ope::Backend::OPENCL);
			if (runVulkan) selected.push_back(ope::Backend::VULKAN);

			std::vector<BenchmarkResult> collectedResults;

			for (int signalLength : BENCHMARK_SIGNAL_LENGTHS) {
				for (int ascansPerBscan : BENCHMARK_ASCANS_PER_BSCAN) {
					for (int bscansPerBuffer : BENCHMARK_BSCANS_PER_BUFFER) {
						if (benchmark.cancelRequested.load()) break;

						auto testData = generateBenchmarkData(signalLength, ascansPerBscan, bscansPerBuffer);

						for (ope::Backend backend : selected) {
							if (benchmark.cancelRequested.load()) break;

							std::ostringstream status;
							status << "Running " << backendName(backend)
							       << " benchmark: " << signalLength
							       << " x " << ascansPerBscan
							       << " x " << bscansPerBuffer;
							setBenchmarkStatus(benchmark, status.str());

							try {
								BenchmarkResult result;
								if (!runBenchmarkPreset(
										backend,
										signalLength,
										ascansPerBscan,
										bscansPerBuffer,
										testData,
										benchmark.cancelRequested,
										&result)) {
									break;
								}

								collectedResults.push_back(result);
								calculateBenchmarkSpeedups(collectedResults);
								updateSharedBenchmarkResults(benchmark, collectedResults);
								benchmark.completedTests++;
							} catch (const std::exception& e) {
								appendBenchmarkError(
									benchmark,
									backendName(backend) + " benchmark failed: " + e.what()
								);
								benchmark.completedTests++;
							}
						}
					}
					if (benchmark.cancelRequested.load()) break;
				}
				if (benchmark.cancelRequested.load()) break;
			}

			if (benchmark.cancelRequested.load()) {
				setBenchmarkStatus(benchmark, "Benchmark canceled.");
			} else {
				calculateBenchmarkSpeedups(collectedResults);
				updateSharedBenchmarkResults(benchmark, collectedResults);

				setBenchmarkStatus(benchmark, "Benchmark completed.");
			}
		} catch (const std::exception& e) {
			appendBenchmarkError(benchmark, std::string("Benchmark failed: ") + e.what());
			setBenchmarkStatus(benchmark, "Benchmark failed.");
		}

		benchmark.running = false;
	});
}

void renderBenchmarkMenuBar(BenchmarkState& benchmark, GLFWwindow* window) {
	if (!ImGui::BeginMainMenuBar()) return;

	if (ImGui::BeginMenu("App")) {
		if (ImGui::MenuItem("Run Performance Benchmark...")) {
			refreshBenchmarkAvailability(benchmark);
			benchmark.openDialogRequested = true;
			benchmark.showResultsWindow = true;
		}

		ImGui::MenuItem("Show Benchmark Results", nullptr, &benchmark.showResultsWindow);
		ImGui::Separator();

		if (ImGui::MenuItem("Exit")) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}

		ImGui::EndMenu();
	}

	ImGui::EndMainMenuBar();
}

void renderBenchmarkDialog(BenchmarkState& benchmark) {
	if (benchmark.openDialogRequested) {
		ImGui::OpenPopup("Run Performance Benchmark");
		benchmark.openDialogRequested = false;
	}

	if (!ImGui::BeginPopupModal("Run Performance Benchmark", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
		return;
	}

	ImGui::TextWrapped("Run the preset sweep used for throughput benchmarking.");
	ImGui::Spacing();
	ImGui::TextWrapped("%s", buildPresetSummary().c_str());
	ImGui::Spacing();
	ImGui::SeparatorText("Backends");

	ImGui::BeginDisabled(!benchmark.cpuAvailable);
	ImGui::Checkbox("CPU", &benchmark.includeCpu);
	ImGui::EndDisabled();
	if (!benchmark.cpuAvailable) ImGui::TextDisabled("CPU backend not available");

	ImGui::BeginDisabled(!benchmark.cudaAvailable);
	ImGui::Checkbox("CUDA", &benchmark.includeCuda);
	ImGui::EndDisabled();
	if (!benchmark.cudaAvailable) ImGui::TextDisabled("CUDA backend not available");

	ImGui::BeginDisabled(!benchmark.openclAvailable);
	ImGui::Checkbox("OpenCL", &benchmark.includeOpencl);
	ImGui::EndDisabled();
	if (!benchmark.openclAvailable) ImGui::TextDisabled("OpenCL backend not available");

	ImGui::BeginDisabled(!benchmark.vulkanAvailable);
	ImGui::Checkbox("Vulkan", &benchmark.includeVulkan);
	ImGui::EndDisabled();
	if (!benchmark.vulkanAvailable) ImGui::TextDisabled("Vulkan backend not available");

	ImGui::Spacing();

	const bool anyBackendSelected =
		benchmark.includeCpu || benchmark.includeCuda || benchmark.includeOpencl || benchmark.includeVulkan;

	ImGui::BeginDisabled(!anyBackendSelected || benchmark.running.load());
	if (ImGui::Button("Start", ImVec2(140, 0))) {
		startBenchmarkThread(benchmark);
		ImGui::CloseCurrentPopup();
	}
	ImGui::EndDisabled();

	ImGui::SameLine();
	if (ImGui::Button("Cancel", ImVec2(140, 0))) {
		ImGui::CloseCurrentPopup();
	}

	if (!anyBackendSelected) {
		ImGui::Spacing();
		ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.3f, 1.0f), "Select at least one backend to start.");
	}

	ImGui::EndPopup();
}

void renderBenchmarkResultsWindow(BenchmarkState& benchmark) {
	if (!benchmark.showResultsWindow) return;

	std::vector<BenchmarkResult> results;
	std::string status;
	std::string csvPath;
	std::string error;
	{
		std::lock_guard<std::mutex> lock(benchmark.mutex);
		results = benchmark.results;
		status = benchmark.status;
		csvPath = benchmark.csvPath;
		error = benchmark.error;
	}

	if (!ImGui::Begin("Benchmark Results", &benchmark.showResultsWindow)) {
		ImGui::End();
		return;
	}

	ImGui::TextWrapped("%s", buildPresetSummary().c_str());
	ImGui::Separator();
	ImGui::Text("Status: %s", status.c_str());

	const int totalTests = benchmark.totalTests.load();
	const int completedTests = benchmark.completedTests.load();
	if (benchmark.running.load() && totalTests > 0) {
		ImGui::ProgressBar(static_cast<float>(completedTests) / static_cast<float>(totalTests), ImVec2(-1, 0));
		ImGui::Text("%d / %d completed", completedTests, totalTests);
		if (ImGui::Button("Cancel Benchmark")) {
			benchmark.cancelRequested = true;
		}
	} else if (totalTests > 0) {
		ImGui::Text("%d / %d completed", completedTests, totalTests);
	}

	ImGui::Spacing();
	ImGui::BeginDisabled(results.empty());
	if (ImGui::Button("Save as CSV")) {
		std::string savedPath;
		std::string saveError;
		if (saveBenchmarkResultsCsv(results, &savedPath, &saveError)) {
			std::lock_guard<std::mutex> lock(benchmark.mutex);
			benchmark.csvPath = savedPath;
		} else {
			appendBenchmarkError(benchmark, saveError);
			error = saveError;
		}
	}
	ImGui::EndDisabled();

	if (!csvPath.empty()) {
		ImGui::Spacing();
		ImGui::TextWrapped("CSV saved to: %s", csvPath.c_str());
	}

	if (!error.empty()) {
		ImGui::Spacing();
		ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.45f, 1.0f), "%s", error.c_str());
	}

	ImGui::Spacing();
	if (results.empty()) {
		ImGui::TextDisabled("No benchmark results yet.");
		ImGui::End();
		return;
	}

	ImGuiTableFlags tableFlags =
		ImGuiTableFlags_Borders |
		ImGuiTableFlags_RowBg |
		ImGuiTableFlags_Resizable |
		ImGuiTableFlags_SizingStretchProp |
		ImGuiTableFlags_ScrollY;

	if (ImGui::BeginTable("benchmark_results_table", 8, tableFlags, ImVec2(0, 320))) {
		ImGui::TableSetupColumn("Signal");
		ImGui::TableSetupColumn("A-scans");
		ImGui::TableSetupColumn("B-scans");
		ImGui::TableSetupColumn("Backend");
		ImGui::TableSetupColumn("Time (ms)");
		ImGui::TableSetupColumn("B-scans/s");
		ImGui::TableSetupColumn("MB/s");
		ImGui::TableSetupColumn("Speedup");
		ImGui::TableHeadersRow();

		for (const auto& result : results) {
			ImGui::TableNextRow();

			ImGui::TableSetColumnIndex(0);
			ImGui::Text("%d", result.signalLength);

			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%d", result.ascansPerBscan);

			ImGui::TableSetColumnIndex(2);
			ImGui::Text("%d", result.bscansPerBuffer);

			ImGui::TableSetColumnIndex(3);
			ImGui::TextUnformatted(result.backend.c_str());

			ImGui::TableSetColumnIndex(4);
			ImGui::Text("%.3f", result.avgTimeMs);

			ImGui::TableSetColumnIndex(5);
			ImGui::Text("%.1f", result.bscansPerSec);

			ImGui::TableSetColumnIndex(6);
			ImGui::Text("%.1f", result.mbPerSec);

			ImGui::TableSetColumnIndex(7);
			if (result.speedup > 0.0) {
				ImGui::Text("%.2fx", result.speedup);
			} else {
				ImGui::TextUnformatted("-");
			}
		}

		ImGui::EndTable();
	}

	ImGui::End();
}
