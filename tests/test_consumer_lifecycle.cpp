// Validates the callback-thread pattern: worker blocks in getNext(),
// removeConsumer() from another thread must wake it with nullptr.
// Also validates stale-ID rejection after slot reuse (ABA protection).
#undef NDEBUG
#include "core/buffer_manager.h"
#include "iobuffer.h"
#include <thread>
#include <cstdio>
#include <cassert>

using namespace ope;

int main() {
	BufferManager manager;
	manager.setBufferCount(2);
	std::vector<IOBuffer> buffers(2);
	for (auto& b : buffers) b.allocateMemory(64);
	std::atomic<int> released{0};
	manager.setReleaseCallback([&](IOBuffer*) { released.fetch_add(1); });

	for (int iter = 0; iter < 500; iter++) {
		ConsumerId id = manager.addConsumer({});
		std::atomic<int> got{0};
		std::thread worker([&]() {
			while (true) {
				IOBuffer* buf = manager.getNext(id);
				if (!buf) break;
				got.fetch_add(1);
				manager.release(id, buf);
			}
		});
		manager.publish(&buffers[0]);
		manager.waitUntilReleased(&buffers[0]);
		manager.publish(&buffers[1]);
		manager.waitUntilReleased(&buffers[1]);
		manager.removeConsumer(id);
		worker.join();
		assert(got.load() == 2);
	}
	printf("released=%d (expected 1000)\n", released.load());
	assert(released.load() == 1000);

	// Stale ID after slot reuse must be rejected (ABA protection)
	ConsumerId a = manager.addConsumer({});
	manager.removeConsumer(a);
	ConsumerId b = manager.addConsumer({});
	assert(a != b);
	IOBuffer* out = nullptr;
	manager.publish(&buffers[0]);
	assert(!manager.tryGet(a, &out));  // stale ID: must fail
	assert(manager.tryGet(b, &out));   // current ID: must succeed
	manager.release(b, out);
	manager.removeConsumer(b);
	printf("PASS\n");
	return 0;
}
