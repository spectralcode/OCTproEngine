#include "backends/cpu/cpu_backend.h"
#include "test_utils.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using CurveSetter = void (ope::CpuBackend::*)(const float*, size_t);
const CurveSetter SETTERS[] = {
	&ope::CpuBackend::updateResamplingCurve,
	&ope::CpuBackend::updateWindowCurve,
	&ope::CpuBackend::updateDispersionCurve
};

std::vector<float> makeCurve(int kind, int length, bool alternate) {
	std::vector<float> curve(kind == 2 ? length * 2 : length);
	for (int i = 0; i < length; ++i) {
		if (kind == 0) {
			curve[i] = alternate ? i * 0.7f : static_cast<float>(i);
		} else if (kind == 1) {
			curve[i] = alternate ? 0.3f + 0.5f * i / length : 1.0f;
		} else {
			float phase = alternate ? 0.04f * i : 0.0f;
			curve[2 * i] = std::cos(phase);
			curve[2 * i + 1] = std::sin(phase);
		}
	}
	return curve;
}

void upload(ope::CpuBackend& backend, int kind, const std::vector<float>& curve) {
	(backend.*SETTERS[kind])(curve.data(), curve.size());
}

ope::ProcessorConfiguration makeConfig(int length, ope::InterpolationMethod method) {
	ope::ProcessorConfiguration config;
	config.dataParams.signalLength = length;
	config.dataParams.ascansPerBscan = 17;
	config.dataParams.bscansPerBuffer = 2;
	config.processingParams.resampling.enabled = true;
	config.processingParams.resampling.method = method;
	config.processingParams.windowing.enabled = true;
	config.processingParams.dispersion.enabled = true;
	config.processingParams.intensity.logScale = false;
	config.processingParams.intensity.rangeMin = 0.0f;
	config.processingParams.intensity.rangeMax = 1.0f;
	return config;
}

bool matches(const std::vector<float>& actual, const std::vector<float>& expected) {
	if (actual.size() != expected.size()) {
		return false;
	}
	for (size_t i = 0; i < actual.size(); ++i) {
		if (!std::isfinite(actual[i]) ||
			std::abs(actual[i] - expected[i]) > 1e-5f + 1e-5f * std::abs(expected[i])) {
			return false;
		}
	}
	return true;
}

struct Capture {
	std::mutex mutex;
	std::condition_variable cv;
	bool received = false;
	std::vector<float> output;
	ope::CpuBackend backend;

	Capture() {
		this->backend.setOutputCallback([this](const ope::IOBuffer& buffer) {
			std::lock_guard<std::mutex> lock(this->mutex);
			const float* data = static_cast<const float*>(buffer.getDataPointer());
			this->output.assign(data, data + buffer.getSizeInBytes() / sizeof(float));
			this->backend.releaseOutputBuffer(const_cast<ope::IOBuffer*>(&buffer));
			this->received = true;
			this->cv.notify_one();
		});
	}

	~Capture() {
		this->backend.cleanup();
	}

	std::vector<float> process(const std::vector<uint16_t>& input) {
		std::unique_lock<std::mutex> lock(this->mutex);
		this->received = false;
		auto& buffer = this->backend.getNextAvailableInputBuffer();
		std::memcpy(buffer.getDataPointer(), input.data(), input.size() * sizeof(uint16_t));
		this->backend.process(buffer);
		TEST_ASSERT(this->cv.wait_for(lock, std::chrono::seconds(10), [this] { return this->received; }),
			"CPU curve test timed out");
		return this->output;
	}
};

std::vector<uint16_t> initialize(Capture& capture, const ope::ProcessorConfiguration& config) {
	capture.backend.initialize(config);
	for (int kind = 0; kind < 3; ++kind) {
		upload(capture.backend, kind, makeCurve(kind, config.dataParams.signalLength, false));
	}
	std::vector<uint16_t> input(config.dataParams.samplesPerBuffer());
	for (size_t i = 0; i < input.size(); ++i) {
		input[i] = static_cast<uint16_t>(1 + (i * 13 + i / config.dataParams.signalLength * 7) % 61);
	}
	return input;
}

void testLengths() {
	Capture capture;
	for (auto setter : SETTERS) {
		(capture.backend.*setter)(nullptr, 0);
	}
	for (auto method : {ope::InterpolationMethod::LINEAR, ope::InterpolationMethod::CUBIC,
		ope::InterpolationMethod::LANCZOS}) {
		for (int length : {64, 96}) {
			auto config = makeConfig(length, method);
			auto input = initialize(capture, config);
			for (int kind = 0; kind < 3; ++kind) {
				auto original = makeCurve(kind, length, false);
				auto replacement = makeCurve(kind, length, true);
				auto before = capture.process(input);
				upload(capture.backend, kind, replacement);
				auto expected = capture.process(input);
				TEST_ASSERT(!matches(expected, before), "Valid replacement must change the output");

				// Includes odd interleaved dispersion counts, and both forms of empty upload.
				for (size_t count : {size_t(0), replacement.size() - 1, replacement.size() + 1}) {
					std::vector<float> invalid(replacement.size() + 1, 0.0f);
					(capture.backend.*SETTERS[kind])(invalid.data(), count);
					TEST_ASSERT(matches(capture.process(input), expected), "Wrong length changed the curve");
				}
				(capture.backend.*SETTERS[kind])(nullptr, 0);
				bool threw = false;
				try {
					(capture.backend.*SETTERS[kind])(nullptr, replacement.size());
				} catch (const std::runtime_error&) {
					threw = true;
				}
				TEST_ASSERT(threw, "Nonempty null upload must throw");
				for (int repeat = 0; repeat < 3; ++repeat) {
					TEST_ASSERT(matches(capture.process(input), expected), "Rejected upload left stale FFT rows");
				}

				// Validate against the allocation even if a direct caller changes the config.
				auto changed = config;
				changed.dataParams.signalLength = length + 1;
				capture.backend.updateConfig(changed);
				upload(capture.backend, kind, makeCurve(kind, length + 1, false));
				capture.backend.updateConfig(config);
				TEST_ASSERT(matches(capture.process(input), expected), "Guard followed mutable config dimensions");
				upload(capture.backend, kind, original);
				TEST_ASSERT(matches(capture.process(input), before), "Valid curve could not be restored");
			}
			capture.backend.cleanup();
			for (auto setter : SETTERS) {
				(capture.backend.*setter)(nullptr, 0);
			}
		}
	}
}

void testConcurrentUploads() {
	Capture capture;
	auto input = initialize(capture, makeConfig(1024, ope::InterpolationMethod::LINEAR));
	for (int kind = 0; kind < 3; ++kind) {
		auto original = makeCurve(kind, 1024, false);
		auto replacement = makeCurve(kind, 1024, true);
		auto first = capture.process(input);
		upload(capture.backend, kind, replacement);
		auto second = capture.process(input);
		TEST_ASSERT(!matches(first, second), "Concurrent test needs distinguishable reference outputs");
		std::atomic<bool> stop{false};
		std::atomic<int> updates{0};
		auto updater = std::async(std::launch::async, [&] {
			while (!stop) {
				upload(capture.backend, kind, updates % 2 ? original : replacement);
				++updates;
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		});
		bool intact = true;
		try {
			for (int repeat = 0; repeat < 48; ++repeat) {
				auto output = capture.process(input);
				intact = intact && (matches(output, first) || matches(output, second));
			}
		} catch (...) {
			stop = true;
			updater.wait();
			throw;
		}
		stop = true;
		updater.get();
		TEST_ASSERT(updates > 0, "No concurrent curve uploads ran");
		TEST_ASSERT(intact, "Output contains a partially updated curve");
		upload(capture.backend, kind, original);
	}
}

} // namespace

int main() {
	try {
		testLengths();
		testConcurrentUploads();
		std::cout << "PASS: CPU curve lengths, reinitialization and concurrent uploads" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
}
