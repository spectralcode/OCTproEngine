#include "../include/processor.h"
#include "test_utils.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>
#include <cmath>

const ope::Backend TEST_BACKEND = ope::Backend::CPU;

// Regression test: setConfig() with unchanged dimensions must forward processing
// parameter changes to an already initialized backend (not just the curves).

namespace {

std::vector<float> processOneBuffer(ope::Processor& processor, const std::vector<uint16_t>& inputData, int outputSamples) {
	std::vector<float> output(outputSamples);
	std::atomic<bool> done{false};

	int callbackId = processor.addOutputCallback([&](const ope::IOBuffer& buf) {
		const float* data = static_cast<const float*>(buf.getDataPointer());
		std::copy(data, data + outputSamples, output.begin());
		done = true;
	});

	auto& inputBuffer = processor.getNextAvailableInputBuffer();
	memcpy(inputBuffer.getDataPointer(), inputData.data(), inputData.size() * sizeof(uint16_t));
	processor.process(inputBuffer);

	while (!done) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	processor.removeOutputCallback(callbackId);
	return output;
}

} // namespace

void testSetConfigPropagatesProcessingFlags() {
	std::cout << "Testing setConfig() propagation of processing flags to initialized backend..." << std::endl;

	const int signalLength = 1024;
	const int ascansPerBscan = 16;
	const int bscansPerBuffer = 1;
	const int samplesPerBuffer = signalLength * ascansPerBscan * bscansPerBuffer;
	const int outputSamples = samplesPerBuffer / 2;

	ope::Processor processor(TEST_BACKEND);
	processor.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);
	processor.enableLogScaling(true);
	processor.initialize();

	// Deterministic input signal
	std::vector<uint16_t> inputData(samplesPerBuffer);
	for (int i = 0; i < samplesPerBuffer; i++) {
		inputData[i] = static_cast<uint16_t>(1000 + 500 * std::sin(i * 0.05));
	}

	std::vector<float> outputLog = processOneBuffer(processor, inputData, outputSamples);

	// Toggle log scaling off via bulk setConfig() with unchanged dimensions
	ope::ProcessorConfiguration config = processor.getConfig();
	config.processingParams.intensity.logScale = false;
	processor.setConfig(config);

	std::vector<float> outputLin = processOneBuffer(processor, inputData, outputSamples);

	// If the flag reached the backend, the outputs must differ
	bool identical = true;
	for (int i = 0; i < outputSamples; i++) {
		if (outputLog[i] != outputLin[i]) {
			identical = false;
			break;
		}
	}

	TEST_ASSERT(!identical, "setConfig() flag change must reach the initialized backend and alter the output");
	std::cout << "  [OK] logScale change via setConfig() reached the backend" << std::endl;
}

// Changing only the input data type must reallocate the backend buffers: the caller
// must receive a buffer of the new size from getNextAvailableInputBuffer(), and the
// output must match a processor that was built with the new type from the start
void testDataTypeChangeReinitializes() {
	std::cout << "Testing data type change reinitializes backend buffers..." << std::endl;

	const int signalLength = 1024;
	const int ascansPerBscan = 16;
	const int bscansPerBuffer = 1;
	const int samplesPerBuffer = signalLength * ascansPerBscan * bscansPerBuffer;
	const int outputSamples = samplesPerBuffer / 2;

	std::vector<uint16_t> inputData(samplesPerBuffer);
	for (int i = 0; i < samplesPerBuffer; i++) {
		inputData[i] = static_cast<uint16_t>(2000 + 900 * std::sin(i * 0.07));
	}

	// Reference: built as UINT16 from the start
	ope::Processor reference(ope::Backend::CPU);
	reference.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);
	reference.initialize();
	std::vector<float> referenceOutput = processOneBuffer(reference, inputData, outputSamples);

	// Path 1: type change via bulk setConfig() on an initialized processor
	ope::Processor viaSetConfig(ope::Backend::CPU);
	viaSetConfig.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT8);
	viaSetConfig.initialize();
	ope::ProcessorConfiguration config = viaSetConfig.getConfig();
	config.dataParams.inputDataType = ope::DataType::UINT16;
	viaSetConfig.setConfig(config);
	TEST_ASSERT(viaSetConfig.getNextAvailableInputBuffer().getSizeInBytes() ==
		static_cast<size_t>(samplesPerBuffer) * sizeof(uint16_t),
		"setConfig() type change must reallocate input buffers before hand-out");
	std::vector<float> output1 = processOneBuffer(viaSetConfig, inputData, outputSamples);

	// Path 2: type change via setInputParameters() followed directly by buffer acquisition
	ope::Processor viaSetParams(ope::Backend::CPU);
	viaSetParams.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT8);
	viaSetParams.initialize();
	viaSetParams.setInputParameters(signalLength, ascansPerBscan, bscansPerBuffer, ope::DataType::UINT16);
	TEST_ASSERT(viaSetParams.getNextAvailableInputBuffer().getSizeInBytes() ==
		static_cast<size_t>(samplesPerBuffer) * sizeof(uint16_t),
		"setInputParameters() type change must reallocate input buffers before hand-out");
	std::vector<float> output2 = processOneBuffer(viaSetParams, inputData, outputSamples);

	for (int i = 0; i < outputSamples; i++) {
		TEST_ASSERT(output1[i] == referenceOutput[i], "setConfig() type change output must match fresh processor");
		TEST_ASSERT(output2[i] == referenceOutput[i], "setInputParameters() type change output must match fresh processor");
	}
	std::cout << "  [OK] data type changes reallocate buffers on both paths" << std::endl;
}

int main() {
	if (!ope::BackendUtils::isCpuAvailable()) {
		std::cout << "SKIP: CPU backend required" << std::endl;
		return 77;
	}
	std::cout << "=== setConfig() Propagation Test ===" << std::endl;
	try {
		testSetConfigPropagatesProcessingFlags();
		testDataTypeChangeReinitializes();
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Test failed: " << e.what() << std::endl;
		return 1;
	}
}
