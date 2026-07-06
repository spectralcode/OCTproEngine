// Measures single-processor versus dual-simultaneous-processor throughput on
// CPU, CUDA and Vulkan. Two instances model the dual-camera use case: each
// processor owns a full pipeline and is driven by its own producer thread,
// so the comparison shows whether two pipelines overlap or serialize against
// each other. Reported per instance: ms/frame, frames/s, MB/s; per scenario:
// aggregate throughput and dual/solo scaling. The test fails on functional
// problems only (exceptions, missing or duplicated outputs, hangs), never on
// slow numbers.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "processor.h"

namespace {

const int SIGNAL_LENGTH = 2048;
const int ASCANS_PER_BSCAN = 512;
const int BSCANS_PER_BUFFER = 1;
const int ITERATIONS_GPU = 500;
const int ITERATIONS_CPU = 50;
const int WARMUP_ITERATIONS = 10;
const int COMPLETION_TIMEOUT_MS = 120000;

// Processing preset identical to test_performance_benchmark and the
// OCTproViewer benchmark so the numbers stay comparable
const ope::InterpolationMethod INTERPOLATION_METHOD = ope::InterpolationMethod::CUBIC;
const float RESAMPLING_COEFFS[4] = {0.5f, 2048.0f, -100.0f, 0.0f};
const ope::WindowType WINDOW_TYPE = ope::WindowType::HANN;
const float WINDOW_CENTER = 0.5f;
const float WINDOW_FILL_FACTOR = 0.95f;
const float DISPERSION_COEFFS[4] = {0.0f, 0.0f, 1.0f, 2.0f};
const float DISPERSION_FACTOR = 1.0f;
const float GRAYSCALE_MIN = 30.0f;
const float GRAYSCALE_MAX = 100.0f;

struct InstanceResult {
	double msPerFrame = 0.0;
	double framesPerSec = 0.0;
	double mbPerSec = 0.0;
};

struct ScenarioResult {
	bool ok = false;
	std::string error;
	std::vector<InstanceResult> instances;
	double aggregateFramesPerSec = 0.0;
};

struct ProcessorHarness {
	std::unique_ptr<ope::Processor> processor;
	std::vector<uint8_t> outputCopy;
	std::atomic<int> completed{0};
	double elapsedMs = 0.0;
};

std::vector<uint16_t> generateTestData() {
	std::vector<uint16_t> data(static_cast<size_t>(SIGNAL_LENGTH) * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER);
	for (int ascan = 0; ascan < ASCANS_PER_BSCAN * BSCANS_PER_BUFFER; ++ascan) {
		for (int i = 0; i < SIGNAL_LENGTH; ++i) {
			double envelope = std::exp(-std::pow((i - SIGNAL_LENGTH * 0.5) / (SIGNAL_LENGTH * 0.25), 2.0));
			double value = 2000.0 + 6000.0 * envelope * std::sin(i * 0.25 + ascan * 0.1);
			data[static_cast<size_t>(ascan) * SIGNAL_LENGTH + i] =
				static_cast<uint16_t>(std::max(0.0, std::min(65535.0, value)));
		}
	}
	return data;
}

void configureProcessor(ope::Processor& processor) {
	processor.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, ope::DataType::UINT16);

	processor.enableResampling(true);
	processor.setInterpolationMethod(INTERPOLATION_METHOD);
	processor.setResamplingCoefficients(RESAMPLING_COEFFS);

	processor.enableWindowing(true);
	processor.setWindowParameters(WINDOW_TYPE, WINDOW_CENTER, WINDOW_FILL_FACTOR);

	processor.enableDispersionCompensation(true);
	processor.setDispersionCoefficients(DISPERSION_COEFFS, DISPERSION_FACTOR);

	processor.enableBackgroundRemoval(false);
	processor.enableFixedPatternNoiseRemoval(false);
	processor.enablePostProcessBackgroundSubtraction(false);
	processor.enableLogScaling(true);
	processor.setGrayscaleRange(GRAYSCALE_MIN, GRAYSCALE_MAX);
	processor.enableBscanFlip(false);
}

bool waitForCount(const std::atomic<int>& counter, int target, int timeoutMs) {
	auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(timeoutMs);
	while (counter.load() < target) {
		if (std::chrono::high_resolution_clock::now() > deadline) return false;
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
	return true;
}

ScenarioResult runScenario(ope::Backend backend, int processorCount, int iterations, const std::vector<uint16_t>& testData) {
	ScenarioResult scenario;
	const size_t inputBytes = testData.size() * sizeof(uint16_t);
	const size_t outputBytes = static_cast<size_t>(SIGNAL_LENGTH / 2) * ASCANS_PER_BSCAN * BSCANS_PER_BUFFER * sizeof(float);

	try {
		std::vector<std::unique_ptr<ProcessorHarness>> harnesses;
		for (int p = 0; p < processorCount; ++p) {
			auto harness = std::make_unique<ProcessorHarness>();
			harness->processor.reset(new ope::Processor(backend));
			configureProcessor(*harness->processor);
			harness->processor->initialize();
			harness->outputCopy.resize(outputBytes);
			ProcessorHarness* raw = harness.get();
			harness->processor->addOutputCallback([raw](const ope::IOBuffer& output) {
				const size_t copySize = std::min(output.getSizeInBytes(), raw->outputCopy.size());
				std::memcpy(raw->outputCopy.data(), output.getDataPointer(), copySize);
				raw->completed.fetch_add(1);
			});
			harnesses.push_back(std::move(harness));
		}

		// Warmup settles GPU clocks and lazy allocations before timing starts
		for (auto& harness : harnesses) {
			for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
				ope::IOBuffer& buffer = harness->processor->getNextAvailableInputBuffer();
				std::memcpy(buffer.getDataPointer(), testData.data(), inputBytes);
				harness->processor->process(buffer);
			}
			if (!waitForCount(harness->completed, WARMUP_ITERATIONS, COMPLETION_TIMEOUT_MS)) {
				scenario.error = "warmup timed out";
				return scenario;
			}
			harness->completed.store(0);
		}

		std::atomic<int> readyCount{0};
		std::atomic<bool> go{false};
		std::atomic<bool> failed{false};
		std::mutex errorMutex;
		std::string threadError;

		auto worker = [&](ProcessorHarness* harness) {
			try {
				readyCount.fetch_add(1);
				while (!go.load()) {
					std::this_thread::yield();
				}
				auto start = std::chrono::high_resolution_clock::now();
				for (int i = 0; i < iterations; ++i) {
					ope::IOBuffer& buffer = harness->processor->getNextAvailableInputBuffer();
					std::memcpy(buffer.getDataPointer(), testData.data(), inputBytes);
					harness->processor->process(buffer);
				}
				auto deadline = start + std::chrono::milliseconds(COMPLETION_TIMEOUT_MS);
				while (harness->completed.load() < iterations) {
					if (std::chrono::high_resolution_clock::now() > deadline) {
						std::lock_guard<std::mutex> lock(errorMutex);
						threadError = "timed out waiting for outputs";
						failed.store(true);
						return;
					}
					std::this_thread::sleep_for(std::chrono::microseconds(100));
				}
				auto end = std::chrono::high_resolution_clock::now();
				harness->elapsedMs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
			} catch (const std::exception& e) {
				std::lock_guard<std::mutex> lock(errorMutex);
				threadError = e.what();
				failed.store(true);
			}
		};

		std::vector<std::thread> threads;
		for (auto& harness : harnesses) {
			threads.emplace_back(worker, harness.get());
		}
		while (readyCount.load() < processorCount) {
			std::this_thread::yield();
		}
		go.store(true);
		for (auto& thread : threads) {
			thread.join();
		}

		if (failed.load()) {
			scenario.error = threadError.empty() ? "worker thread failed" : threadError;
			return scenario;
		}

		double maxElapsedMs = 0.0;
		for (auto& harness : harnesses) {
			if (harness->completed.load() != iterations) {
				scenario.error = "output count mismatch (duplicated or lost outputs)";
				return scenario;
			}
			InstanceResult result;
			result.msPerFrame = harness->elapsedMs / iterations;
			result.framesPerSec = 1000.0 * iterations / harness->elapsedMs;
			result.mbPerSec = result.framesPerSec * inputBytes / (1024.0 * 1024.0);
			scenario.instances.push_back(result);
			maxElapsedMs = std::max(maxElapsedMs, harness->elapsedMs);
		}
		scenario.aggregateFramesPerSec = 1000.0 * processorCount * iterations / maxElapsedMs;
		scenario.ok = true;
	} catch (const std::exception& e) {
		scenario.error = e.what();
	}
	return scenario;
}

void printScenario(const char* label, const ScenarioResult& scenario) {
	if (!scenario.ok) {
		printf("  %s: FAILED (%s)\n", label, scenario.error.c_str());
		return;
	}
	for (size_t i = 0; i < scenario.instances.size(); ++i) {
		const InstanceResult& result = scenario.instances[i];
		printf("  %s processor %zu: %8.3f ms/frame, %9.1f frames/s, %9.1f MB/s\n",
		       label, i, result.msPerFrame, result.framesPerSec, result.mbPerSec);
	}
	if (scenario.instances.size() > 1) {
		printf("  %s aggregate: %9.1f frames/s\n", label, scenario.aggregateFramesPerSec);
	}
}

double averageMsPerFrame(const ScenarioResult& scenario) {
	double sum = 0.0;
	for (const auto& result : scenario.instances) {
		sum += result.msPerFrame;
	}
	return scenario.instances.empty() ? 0.0 : sum / scenario.instances.size();
}

} // namespace

int main() {
	printf("Dual processor performance test\n");
	printf("Geometry: %d x %d x %d UINT16, output %d x %d FLOAT32\n",
	       SIGNAL_LENGTH, ASCANS_PER_BSCAN, BSCANS_PER_BUFFER, SIGNAL_LENGTH / 2, ASCANS_PER_BSCAN);

	auto testData = generateTestData();

	struct BackendPlan {
		ope::Backend backend;
		const char* name;
		bool available;
		int iterations;
	};
	const BackendPlan plans[] = {
		{ope::Backend::CPU, "CPU", ope::BackendUtils::isCpuAvailable(), ITERATIONS_CPU},
		{ope::Backend::CUDA, "CUDA", ope::BackendUtils::isCudaAvailable(), ITERATIONS_GPU},
		{ope::Backend::VULKAN, "Vulkan", ope::BackendUtils::isVulkanAvailable(), ITERATIONS_GPU},
	};

	struct Row {
		const char* name;
		ScenarioResult solo;
		ScenarioResult dual;
	};
	std::vector<Row> rows;
	bool anyFailure = false;

	for (const auto& plan : plans) {
		if (!plan.available) {
			printf("\n=== %s: backend not available, skipped ===\n", plan.name);
			continue;
		}
		printf("\n=== %s (%d frames per processor) ===\n", plan.name, plan.iterations);

		ScenarioResult solo = runScenario(plan.backend, 1, plan.iterations, testData);
		printScenario("solo", solo);

		ScenarioResult dual = runScenario(plan.backend, 2, plan.iterations, testData);
		printScenario("dual", dual);

		if (!solo.ok || !dual.ok) {
			anyFailure = true;
		}
		rows.push_back({plan.name, std::move(solo), std::move(dual)});
	}

	printf("\n%-8s %15s %15s %15s %18s %9s\n",
	       "Backend", "solo ms/frame", "dual ms/frame", "solo frames/s", "dual frames/s sum", "scaling");
	for (const auto& row : rows) {
		if (!row.solo.ok || !row.dual.ok) {
			printf("%-8s FAILED (%s)\n", row.name, (!row.solo.ok ? row.solo.error : row.dual.error).c_str());
			continue;
		}
		double soloFps = row.solo.instances[0].framesPerSec;
		double scaling = soloFps > 0.0 ? row.dual.aggregateFramesPerSec / soloFps : 0.0;
		printf("%-8s %15.3f %15.3f %15.1f %18.1f %8.2fx\n",
		       row.name,
		       row.solo.instances[0].msPerFrame,
		       averageMsPerFrame(row.dual),
		       soloFps,
		       row.dual.aggregateFramesPerSec,
		       scaling);
	}
	printf("\nScaling 2.00x = both pipelines fully overlap, 1.00x = they serialize\n");
	printf("(or a single pipeline already saturates the device)\n");

	if (rows.empty()) {
		printf("FAIL: no backend available\n");
		return 1;
	}
	if (anyFailure) {
		printf("FAIL\n");
		return 1;
	}
	printf("PASS\n");
	return 0;
}
