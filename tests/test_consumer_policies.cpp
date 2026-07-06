// Asserting coverage of the consumer policies on BufferManager directly:
// - DROP_OLDEST drops the oldest frames, keeps the newest, counts every drop
// - BLOCK producer backpressure (publish blocks when full, resumes on consume)
// - maxQueueSize is honored beyond 8 (regression: the old SPSC queue silently
//   capped every queue at 8 entries and dropped BLOCK frames beyond it)
// - waitUntilReleased() gates producer reuse on held buffers, but reclaims
//   frames still queued at DROP_OLDEST consumers instead of blocking
// - consumer slot exhaustion throws
#undef NDEBUG
#include "core/buffer_manager.h"
#include "iobuffer.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <vector>

using namespace ope;

namespace {

struct Fixture {
	BufferManager manager;
	std::vector<IOBuffer> buffers;
	std::atomic<int> releases{0};

	explicit Fixture(int bufferCount) : buffers(bufferCount) {
		this->manager.setBufferCount(bufferCount);
		for (size_t i = 0; i < this->buffers.size(); i++) {
			this->buffers[i].allocateMemory(sizeof(uint64_t));
			this->buffers[i].setBufferId(i);
		}
		this->manager.setReleaseCallback([this](IOBuffer*) { this->releases.fetch_add(1); });
	}
};

// Wait for a flag with timeout; returns true if the flag became true
bool waitFor(const std::atomic<bool>& flag, int timeoutMs) {
	auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
	while (!flag.load()) {
		if (std::chrono::steady_clock::now() > deadline) return false;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return true;
}

void testDropOldestDirection() {
	printf("TEST 1: DROP_OLDEST drops oldest, keeps newest, counts drops\n");
	Fixture f(8);
	ConsumerConfig cfg;
	cfg.dropPolicy = DropPolicy::DROP_OLDEST;
	cfg.maxQueueSize = 2;
	ConsumerId id = f.manager.addConsumer(cfg);

	// Publish 5 distinct buffers without any consumer polling
	for (int i = 0; i < 5; i++) {
		f.manager.publish(&f.buffers[i]);
	}

	// Capacity 2: buffers 0,1,2 must have been dropped (oldest first), 3,4 kept
	assert(f.manager.getQueueSize(id) == 2);
	assert(f.manager.getDroppedCount(id) == 3);
	assert(f.releases.load() == 3); // each dropped buffer released exactly once

	IOBuffer* out = nullptr;
	assert(f.manager.tryGet(id, &out) && out->getBufferId() == 3);
	f.manager.release(id, out);
	assert(f.manager.tryGet(id, &out) && out->getBufferId() == 4);
	f.manager.release(id, out);
	assert(!f.manager.tryGet(id, &out));
	assert(f.releases.load() == 5); // exactly one release per publish overall

	f.manager.removeConsumer(id);
	printf("  PASS\n");
}

void testBlockBackpressure() {
	printf("TEST 2: BLOCK backpressures the producer, never drops\n");
	Fixture f(4);
	ConsumerConfig cfg;
	cfg.dropPolicy = DropPolicy::BLOCK;
	cfg.maxQueueSize = 1;
	ConsumerId id = f.manager.addConsumer(cfg);

	f.manager.publish(&f.buffers[0]); // fills the queue

	std::atomic<bool> secondPublishDone{false};
	std::thread producer([&]() {
		f.manager.publish(&f.buffers[1]); // must block until space frees up
		secondPublishDone.store(true);
	});

	std::this_thread::sleep_for(std::chrono::milliseconds(150));
	assert(!secondPublishDone.load()); // still blocked on the full queue

	IOBuffer* out = nullptr;
	assert(f.manager.tryGet(id, &out) && out->getBufferId() == 0);
	f.manager.release(id, out);

	assert(waitFor(secondPublishDone, 5000)); // space freed -> publish completed
	producer.join();

	assert(f.manager.tryGet(id, &out) && out->getBufferId() == 1);
	f.manager.release(id, out);
	assert(f.manager.getDroppedCount(id) == 0); // BLOCK never drops
	f.manager.removeConsumer(id);
	printf("  PASS\n");
}

void testQueueCapacityBeyondEight() {
	printf("TEST 3: maxQueueSize honored beyond 8 (old silent cap regression)\n");
	Fixture f(16);
	ConsumerConfig cfg;
	cfg.dropPolicy = DropPolicy::BLOCK;
	cfg.maxQueueSize = 12;
	ConsumerId id = f.manager.addConsumer(cfg);

	// All 12 publishes must fit without blocking and without silent drops
	for (int i = 0; i < 12; i++) {
		f.manager.publish(&f.buffers[i]);
	}
	assert(f.manager.getQueueSize(id) == 12);
	assert(f.manager.getDroppedCount(id) == 0);

	// The 13th publish must block (queue genuinely full now)
	std::atomic<bool> extraPublishDone{false};
	std::thread producer([&]() {
		f.manager.publish(&f.buffers[12]);
		extraPublishDone.store(true);
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(150));
	assert(!extraPublishDone.load());

	// Drain everything in FIFO order
	IOBuffer* out = nullptr;
	for (int i = 0; i < 13; i++) {
		while (!f.manager.tryGet(id, &out)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		assert(out->getBufferId() == static_cast<uint64_t>(i));
		f.manager.release(id, out);
	}
	assert(waitFor(extraPublishDone, 5000));
	producer.join();
	assert(f.releases.load() == 13);
	f.manager.removeConsumer(id);
	printf("  PASS\n");
}

void testProducerReuseGating() {
	printf("TEST 4: waitUntilReleased gates on held buffers, reclaims DROP_OLDEST queues\n");
	Fixture f(4);

	// (a) a buffer held in hand by a BLOCK consumer gates the producer
	ConsumerConfig blockCfg;
	blockCfg.dropPolicy = DropPolicy::BLOCK;
	blockCfg.maxQueueSize = 2;
	ConsumerId blockId = f.manager.addConsumer(blockCfg);

	f.manager.publish(&f.buffers[0]);
	IOBuffer* held = nullptr;
	assert(f.manager.tryGet(blockId, &held) && held == &f.buffers[0]);

	std::atomic<bool> waitDone{false};
	std::thread producer([&]() {
		f.manager.waitUntilReleased(&f.buffers[0]);
		waitDone.store(true);
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(150));
	assert(!waitDone.load()); // consumer still reads the buffer
	f.manager.release(blockId, held);
	assert(waitFor(waitDone, 5000));
	producer.join();
	f.manager.removeConsumer(blockId);

	// (b) a buffer only queued at a DROP_OLDEST consumer is reclaimed, not waited for
	ConsumerConfig dropCfg;
	dropCfg.dropPolicy = DropPolicy::DROP_OLDEST;
	dropCfg.maxQueueSize = 2;
	ConsumerId dropId = f.manager.addConsumer(dropCfg);

	f.manager.publish(&f.buffers[1]); // sits in the queue, consumer never polls
	uint64_t droppedBefore = f.manager.getDroppedCount(dropId);
	f.manager.waitUntilReleased(&f.buffers[1]); // must return without blocking
	assert(f.manager.getDroppedCount(dropId) == droppedBefore + 1);
	assert(f.manager.getQueueSize(dropId) == 0);
	f.manager.removeConsumer(dropId);
	printf("  PASS\n");
}

void testConsumerSlotExhaustion() {
	printf("TEST 5: consumer slot exhaustion throws\n");
	Fixture f(2);
	std::vector<ConsumerId> ids;
	for (int i = 0; i < MAX_CONSUMERS; i++) {
		ids.push_back(f.manager.addConsumer({}));
	}
	bool threw = false;
	try {
		f.manager.addConsumer({});
	} catch (const std::runtime_error&) {
		threw = true;
	}
	assert(threw);
	for (ConsumerId id : ids) {
		f.manager.removeConsumer(id);
	}
	printf("  PASS\n");
}

} // namespace

int main() {
	testDropOldestDirection();
	testBlockBackpressure();
	testQueueCapacityBeyondEight();
	testProducerReuseGating();
	testConsumerSlotExhaustion();
	printf("ALL CONSUMER POLICY TESTS PASSED\n");
	return 0;
}
