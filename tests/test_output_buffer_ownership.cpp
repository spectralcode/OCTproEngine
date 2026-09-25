#include "processor.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <vector>

namespace {

void check(bool condition, const char* message) {
	if (!condition) {
		std::cerr << "FAIL: " << message << std::endl;
		// A broken ownership path can leave backend workers blocked during teardown.
		std::_Exit(1);
	}
}

struct Snapshot {
	ope::IOBuffer* buffer;
	uint64_t id;
	std::vector<unsigned char> contents;

	explicit Snapshot(ope::IOBuffer* output) : buffer(output), id(output->getBufferId()) {
		const auto* data = static_cast<const unsigned char*>(output->getDataPointer());
		this->contents.assign(data, data + output->getSizeInBytes());
	}

	void verify() const {
		check(this->buffer->getBufferId() == this->id, "Held output ID changed");
		check(std::memcmp(this->contents.data(), this->buffer->getDataPointer(), this->contents.size()) == 0,
			"Held output contents changed");
	}
};

void testOwnership(ope::Processor& processor, int outputCount, ope::DropPolicy policy) {
	ope::ConsumerConfig config;
	config.maxQueueSize = 16;
	config.dropPolicy = policy;
	const auto first = processor.addConsumer(config);
	const auto second = processor.addConsumer(config);
	uint64_t nextId = 0;
	auto submit = [&]() {
		auto& input = processor.getNextAvailableInputBuffer();
		auto* data = static_cast<uint16_t*>(input.getDataPointer());
		for (size_t i = 0; i < input.getSizeInBytes() / sizeof(uint16_t); ++i) {
			data[i] = static_cast<uint16_t>(1000 + nextId * 13 + i % 251);
		}
		processor.process(input);
		++nextId;
	};
	auto receive = [&]() {
		auto* output = processor.getNextOutputBuffer(first);
		check(output && output->getBufferId() == nextId - 1, "Incorrect output ID");
		check(processor.getNextOutputBuffer(second) == output, "Consumers received different outputs");
		return output;
	};
	auto release = [&](ope::IOBuffer* output) {
		processor.releaseOutputBuffer(first, output);
		processor.releaseOutputBuffer(second, output);
	};

	submit();
	Snapshot held(receive());
	processor.releaseOutputBuffer(first, held.buffer);
	for (int i = 0; i < 100; ++i) {
		submit();
		auto* output = receive();
		check(output != held.buffer, "Reused an output still held by a consumer");
		held.verify();
		release(output);
	}
	processor.releaseOutputBuffer(second, held.buffer);

	std::vector<Snapshot> allHeld;
	for (int i = 0; i < outputCount; ++i) {
		submit();
		auto* output = receive();
		for (const auto& previous : allHeld) check(previous.buffer != output, "Reused a held output");
		allHeld.emplace_back(output);
	}

	std::promise<void> started;
	auto ready = started.get_future();
	auto pending = std::async(std::launch::async, [&]() {
		started.set_value();
		submit();
		return receive();
	});
	ready.wait();
	check(pending.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout,
		"Produced an output while all outputs were held");
	for (const auto& snapshot : allHeld) snapshot.verify();

	// Release out of submission order, one consumer at a time.
	auto* released = allHeld.back().buffer;
	processor.releaseOutputBuffer(first, released);
	check(pending.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout,
		"Reused an output before its last consumer released it");
	for (const auto& snapshot : allHeld) snapshot.verify();
	processor.releaseOutputBuffer(second, released);
	check(pending.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
		"Processing did not resume after output release");
	check(pending.get() == released, "Did not reuse the released output");
	release(released);
	allHeld.pop_back();
	for (auto it = allHeld.rbegin(); it != allHeld.rend(); ++it) {
		it->verify();
		release(it->buffer);
	}
	processor.removeConsumer(first);
	processor.removeConsumer(second);
}

} // namespace

int main(int argc, char** argv) {
	check(argc == 2, "Expected backend name");
	const std::string name = argv[1];
	ope::Backend backend;
	bool available = false;
	if (name == "CPU") { backend = ope::Backend::CPU; available = ope::BackendUtils::isCpuAvailable(); }
	else if (name == "CUDA") { backend = ope::Backend::CUDA; available = ope::BackendUtils::isCudaAvailable(); }
	else if (name == "OPENCL") { backend = ope::Backend::OPENCL; available = ope::BackendUtils::isOpenCLAvailable(); }
	else if (name == "VULKAN") { backend = ope::Backend::VULKAN; available = ope::BackendUtils::isVulkanAvailable(); }
	else { check(false, "Unknown backend"); return 1; }
	if (!available) { std::cout << "SKIP: " << name << " unavailable\n"; return 77; }

	try {
		for (auto policy : {ope::DropPolicy::BLOCK, ope::DropPolicy::DROP_OLDEST}) {
			ope::Processor processor(backend);
			const int outputCount = backend == ope::Backend::CPU ? 2 : 3;
			if (backend == ope::Backend::CUDA) {
				ope::CudaConfig config;
				config.numOutputBuffers = outputCount;
				processor.setBackendConfig(config);
			} else if (backend == ope::Backend::VULKAN) {
				ope::VulkanConfig config;
				config.numOutputBuffers = outputCount;
				processor.setBackendConfig(config);
			} else if (backend == ope::Backend::OPENCL) {
				ope::OpenCLConfig config;
				config.numOutputBuffers = outputCount;
				processor.setBackendConfig(config);
			}
			processor.setInputParameters(512, 32, 1, ope::DataType::UINT16);
			processor.enableLogScaling(false);
			processor.initialize();
			testOwnership(processor, outputCount, policy);
			processor.cleanup();
			if (backend == ope::Backend::CPU || backend == ope::Backend::OPENCL) {
				processor.initialize();
				testOwnership(processor, outputCount, policy);
				processor.cleanup();
			}
		}
		std::cout << "PASS: " << name << " output ownership (both consumer policies)\n";
		return 0;
	} catch (const std::exception& error) {
		std::cerr << "FAIL: " << error.what() << std::endl;
		return 1;
	}
}
