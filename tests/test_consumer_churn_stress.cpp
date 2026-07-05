// Standalone stress test: consumers added/removed while publishing.
// Catches the refcount underflow race and validates the publisher-reference fix.
// Also covers the hold-buffer-while-slow pattern via a long-hold BLOCK consumer.
#undef NDEBUG
#include "core/buffer_manager.h"
#include "iobuffer.h"
#include <thread>
#include <random>
#include <cstdio>
#include <cassert>
#include <queue>
#include <chrono>

using namespace ope;

int main() {
	constexpr int NUM_BUFFERS = 4;
	constexpr int NUM_PUBLISHES = 20000;

	BufferManager manager;
	manager.setBufferCount(NUM_BUFFERS);

	std::vector<IOBuffer> buffers(NUM_BUFFERS);
	for (auto& b : buffers) b.allocateMemory(64);

	// Simulated backend pool
	std::mutex poolMutex;
	std::condition_variable poolCV;
	std::queue<IOBuffer*> pool;
	for (auto& b : buffers) pool.push(&b);
	std::atomic<uint64_t> releases{0};

	manager.setReleaseCallback([&](IOBuffer* buf) {
		releases.fetch_add(1);
		{
			std::lock_guard<std::mutex> lock(poolMutex);
			pool.push(buf);
		}
		poolCV.notify_one();
	});

	std::atomic<bool> done{false};

	// Churn threads: add/remove consumers with both policies while publishing
	std::vector<std::thread> churn;
	for (int t = 0; t < 4; t++) {
		churn.emplace_back([&, t]() {
			std::mt19937 rng(t);
			while (!done.load()) {
				ConsumerConfig cfg;
				cfg.dropPolicy = (rng() % 2) ? DropPolicy::DROP_OLDEST : DropPolicy::BLOCK;
				cfg.maxQueueSize = 1 + rng() % 3;
				ConsumerId id = manager.addConsumer(cfg);
				int toConsume = rng() % 20;
				for (int i = 0; i < toConsume && !done.load(); i++) {
					IOBuffer* buf = nullptr;
					if (manager.tryGet(id, &buf)) {
						assert(buf != nullptr);
						volatile char sink = *static_cast<char*>(buf->getDataPointer());
						(void)sink;
						manager.release(id, buf);
					} else {
						std::this_thread::yield();
					}
				}
				manager.removeConsumer(id);
			}
		});
	}

	// Long-hold consumer: holds every 50th buffer for a few ms before releasing
	// while the producer keeps cycling the pool. Covers the pattern of a slow
	// consumer that keeps its reference during processing (previously exercised
	// by test_drop_policy_performance's copy -> sleep -> release loop).
	ConsumerConfig holdCfg;
	holdCfg.dropPolicy = DropPolicy::BLOCK;
	holdCfg.maxQueueSize = 1;
	ConsumerId holdId = manager.addConsumer(holdCfg);
	std::atomic<uint64_t> heldCount{0};
	std::thread longHold([&]() {
		int n = 0;
		while (true) {
			IOBuffer* buf = manager.getNext(holdId);
			if (!buf) break;
			if (++n % 50 == 0) {
				heldCount.fetch_add(1);
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
			}
			manager.release(holdId, buf);
		}
	});

	// Producer: acquire from pool, publish, wait for full release before reuse
	for (int i = 0; i < NUM_PUBLISHES; i++) {
		IOBuffer* buf;
		{
			std::unique_lock<std::mutex> lock(poolMutex);
			poolCV.wait(lock, [&]() { return !pool.empty(); });
			buf = pool.front();
			pool.pop();
		}
		manager.waitUntilReleased(buf);
		*static_cast<char*>(buf->getDataPointer()) = static_cast<char>(i); // simulate new frame
		manager.publish(buf);
	}

	done.store(true);
	for (auto& t : churn) t.join();
	manager.removeConsumer(holdId);
	longHold.join();
	manager.shutdown();

	// Every publish must come back to the pool exactly once
	printf("publishes=%d releases=%llu held=%llu\n", NUM_PUBLISHES,
	       (unsigned long long)releases.load(), (unsigned long long)heldCount.load());
	assert(releases.load() == NUM_PUBLISHES);
	printf("PASS\n");
	return 0;
}
