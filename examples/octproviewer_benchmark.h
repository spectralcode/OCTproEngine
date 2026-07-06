#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct GLFWwindow;

struct BenchmarkResult {
	int signalLength = 0;
	int ascansPerBscan = 0;
	int bscansPerBuffer = 0;
	std::string backend;
	int iterations = 0;
	double avgTimeMs = 0.0;
	double bscansPerSec = 0.0;
	double mbPerSec = 0.0;
	double speedup = 0.0;
};

struct BenchmarkState {
	bool cpuAvailable = false;
	bool cudaAvailable = false;
	bool openclAvailable = false;
	bool vulkanAvailable = false;

	bool includeCpu = false;
	bool includeCuda = false;
	bool includeOpencl = false;
	bool includeVulkan = false;

	bool optionsInitialized = false;
	bool openDialogRequested = false;
	bool showResultsWindow = false;

	std::atomic<bool> running{false};
	std::atomic<bool> cancelRequested{false};
	std::atomic<int> completedTests{0};
	std::atomic<int> totalTests{0};

	std::thread worker;
	std::mutex mutex;
	std::vector<BenchmarkResult> results;
	std::string status = "Idle";
	std::string csvPath;
	std::string error;
};

void refreshBenchmarkAvailability(BenchmarkState& benchmark);
void finalizeBenchmarkThread(BenchmarkState& benchmark);
void stopBenchmarkThread(BenchmarkState& benchmark);
void startBenchmarkThread(BenchmarkState& benchmark);

void renderBenchmarkMenuBar(BenchmarkState& benchmark, GLFWwindow* window);
void renderBenchmarkDialog(BenchmarkState& benchmark);
void renderBenchmarkResultsWindow(BenchmarkState& benchmark);
