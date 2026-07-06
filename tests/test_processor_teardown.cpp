// Destroys a Processor while frames are still in flight and callbacks are
// registered. Teardown must neither crash nor hang: the destructor races the
// backend's output delivery, the input publication and the callback worker
// threads. Other tests reach the destructor only after their workers went
// idle (they sleep before returning), so this covers the hot teardown path.
#include <iostream>
#include <atomic>
#include <cstring>
#include "processor.h"

const ope::Backend TEST_BACKEND = ope::Backend::CUDA;

int main() {
	std::cout << "Processor teardown under load" << std::endl;

	const int ITERATIONS = 15;
	const int FRAMES_PER_ITERATION = 8;

	// Counters outlive every processor so callbacks racing the destruction
	// still write valid memory
	std::atomic<int> outputs{0};
	std::atomic<int> inputs{0};

	for (int iter = 0; iter < ITERATIONS; iter++) {
		{
			ope::Processor processor(TEST_BACKEND);
			processor.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
			processor.initialize();

			processor.addOutputCallback([&outputs](const ope::IOBuffer&) { outputs.fetch_add(1); });
			processor.addInputCallback([&inputs](const ope::IOBuffer&) { inputs.fetch_add(1); });

			for (int i = 0; i < FRAMES_PER_ITERATION; i++) {
				auto& buf = processor.getNextAvailableInputBuffer();
				std::memset(buf.getDataPointer(), i, buf.getSizeInBytes());
				processor.process(buf);
			}
			// Deliberately no sleep: destruction starts while the GPU still
			// processes and the callback workers still deliver
		}
	}

	// No count assertions: frames undelivered at destruction are legally
	// dropped. The test asserts by finishing: no crash, no deadlock.
	std::cout << "outputs=" << outputs.load() << " inputs=" << inputs.load()
	          << " (of " << (ITERATIONS * FRAMES_PER_ITERATION) << " submitted)" << std::endl;
	std::cout << "PASS" << std::endl;
	return 0;
}
