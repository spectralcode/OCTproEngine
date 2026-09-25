#include "../include/processor.h"
#include "test_backend.h"
#include "test_utils.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstring>
#include <cmath>

// Tests for post-FFT frame correction (line-field OCT):
// corrected = IFFT_output * sqrt(2^bitDepth) / sqrt(raw_spectral_average),
// applied only where sqrt(average) > 1.
// Note the correction divides by sqrt of the gain: a spectrum scaled by 4 leaves a
// corrected magnitude ratio of 2, not 1.

namespace {

const int SIGNAL_LENGTH = 64;
const int ASCANS_PER_BSCAN = 8;
const int SAMPLES_PER_BSCAN = SIGNAL_LENGTH * ASCANS_PER_BSCAN;

void configurePassthrough(ope::Processor& processor) {
	processor.setInputParameters(SIGNAL_LENGTH, ASCANS_PER_BSCAN, 1, ope::DataType::UINT16);
	processor.enableLogScaling(false);
	processor.setGrayscaleRange(0.0f, 1.0f);
	processor.setSignalMultiplicatorAndAddend(1.0f, 0.0f);
}

std::vector<float> processOneBuffer(ope::Processor& processor, const std::vector<uint16_t>& input) {
	size_t outputSamples = SAMPLES_PER_BSCAN / 2;
	std::vector<float> output(outputSamples);
	std::atomic<bool> done{false};

	int callbackId = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		const float* data = static_cast<const float*>(buf.getDataPointer());
		std::copy(data, data + outputSamples, output.begin());
		done = true;
	});

	auto& buffer = processor.getNextAvailableInputBuffer();
	memcpy(buffer.getDataPointer(), input.data(), input.size() * sizeof(uint16_t));
	processor.process(buffer);

	while (!done) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	processor.removeOutputCallback(callbackId);
	return output;
}

bool nearlyEqual(float a, float b, float tolerance) {
	return std::abs(a - b) <= tolerance;
}

} // namespace

// A-scan 1 is 4x A-scan 0: raw DC ratio 4, corrected DC ratio must be 2
void testCorrectionFormula(ope::Backend backend) {
	std::cout << "  Correction formula (gain 4 -> corrected ratio 2)..." << std::endl;

	std::vector<uint16_t> input(SAMPLES_PER_BSCAN, 0);
	std::fill(input.begin(), input.begin() + SIGNAL_LENGTH, static_cast<uint16_t>(100));
	std::fill(input.begin() + SIGNAL_LENGTH, input.begin() + 2 * SIGNAL_LENGTH, static_cast<uint16_t>(400));

	ope::Processor processor(backend);
	configurePassthrough(processor);
	processor.enablePostFftFrameCorrection(true);
	processor.initialize();

	std::vector<float> output = processOneBuffer(processor, input);

	// Constant A-scan of value v: raw DC output = 2v, spectral average = v,
	// corrected DC = 2v * scale / sqrt(v)
	float scale = std::sqrt(std::pow(2.0f, 16.0f));
	float expectedDc0 = 2.0f * 100.0f * scale / std::sqrt(100.0f);
	float expectedDc1 = 2.0f * 400.0f * scale / std::sqrt(400.0f);
	int outputAscanLength = SIGNAL_LENGTH / 2;
	float dc0 = output[0];
	float dc1 = output[outputAscanLength];

	TEST_ASSERT(nearlyEqual(dc0, expectedDc0, expectedDc0 * 0.001f), "Corrected DC of A-scan 0 must match the formula");
	TEST_ASSERT(nearlyEqual(dc1, expectedDc1, expectedDc1 * 0.001f), "Corrected DC of A-scan 1 must match the formula");
	TEST_ASSERT(nearlyEqual(dc1 / dc0, 2.0f, 0.01f), "Gain 4 must leave a corrected ratio of 2 (divides by sqrt of gain)");
}

// Where sqrt(average) <= 1 the A-scan must pass through uncorrected
void testPassthroughBranch(ope::Backend backend) {
	std::cout << "  sqrt(average) <= 1 passthrough..." << std::endl;

	std::vector<uint16_t> input(SAMPLES_PER_BSCAN, 0);
	// Average of A-scan 0 = 1 -> sqrt(average) = 1, not > 1 -> no correction
	std::fill(input.begin(), input.begin() + SIGNAL_LENGTH, static_cast<uint16_t>(1));

	ope::Processor reference(backend);
	configurePassthrough(reference);
	reference.initialize();
	std::vector<float> referenceOutput = processOneBuffer(reference, input);

	ope::Processor processor(backend);
	configurePassthrough(processor);
	processor.enablePostFftFrameCorrection(true);
	processor.initialize();
	std::vector<float> output = processOneBuffer(processor, input);

	for (size_t i = 0; i < output.size(); ++i) {
		TEST_ASSERT(nearlyEqual(output[i], referenceOutput[i], 0.0001f),
			"A-scans with sqrt(average) <= 1 must pass through uncorrected");
	}
}

// Disabled correction must leave the output identical to the baseline
void testDisabledIsBaseline(ope::Backend backend) {
	std::cout << "  Disabled correction equals baseline..." << std::endl;

	std::vector<uint16_t> input(SAMPLES_PER_BSCAN);
	for (int i = 0; i < SAMPLES_PER_BSCAN; ++i) {
		input[i] = static_cast<uint16_t>(500 + 300 * std::sin(i * 0.13));
	}

	ope::Processor reference(backend);
	configurePassthrough(reference);
	reference.initialize();
	std::vector<float> referenceOutput = processOneBuffer(reference, input);

	ope::Processor processor(backend);
	configurePassthrough(processor);
	processor.enablePostFftFrameCorrection(true);
	processor.initialize();
	processor.enablePostFftFrameCorrection(false);
	std::vector<float> output = processOneBuffer(processor, input);

	for (size_t i = 0; i < output.size(); ++i) {
		TEST_ASSERT(output[i] == referenceOutput[i], "Disabled correction must not alter the output");
	}
}

void runBackendSuite(ope::Backend backend, const char* name) {
	std::cout << "\n=== Backend: " << name << " ===" << std::endl;
	testCorrectionFormula(backend);
	testPassthroughBranch(backend);
	testDisabledIsBaseline(backend);
}

int main(int argc, char** argv) {
	ope::Backend backend;
	const int status = selectTestBackend(argc, argv, backend);
	if (status != 0) return status;
	std::cout << "=== Post-FFT Frame Correction Tests (Line-Field OCT) ===" << std::endl;
	try {
		// Availability is decided by BackendUtils, not by catching exceptions:
		// once a backend is available, every failure inside the suite fails the test
		runBackendSuite(backend, argv[1]);

		std::cout << "\nAll frame correction tests passed" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}
