#include "output_buffer_manager.h"
#include <iostream>

namespace ope {

OutputBufferManager::ConsumerSlot::ConsumerSlot()
	: queue(8)
{
}

OutputBufferManager::OutputBufferManager() {
}

OutputBufferManager::~OutputBufferManager() {
	this->shutdown();
}

void OutputBufferManager::setBufferCount(size_t count) {
	this->bufferCount = count;
	this->refCounts = std::vector<std::atomic<int>>(count);
	for (auto& ref : this->refCounts) {
		ref.store(0, std::memory_order_relaxed);
	}

	// Update existing consumers that have maxQueueSize=0 (using default)
	// This handles the case where consumers were added before initialize()
	for (int i = 0; i < MAX_CONSUMERS; i++) {
		auto& slot = this->slots[i];
		if (slot.active.load(std::memory_order_acquire)) {
			std::lock_guard<std::mutex> lock(slot.blockMutex);
			if (slot.config.maxQueueSize == 0) {
				slot.config.maxQueueSize = count;
			}
		}
	}
}

void OutputBufferManager::setReleaseCallback(ReleaseCallback callback) {
	this->releaseCallback = std::move(callback);
}

ConsumerId OutputBufferManager::addConsumer(ConsumerConfig config) {
	for (int i = 0; i < MAX_CONSUMERS; i++) {
		if (!this->slots[i].active.load(std::memory_order_acquire)) {
			auto& slot = this->slots[i];
			std::lock_guard<std::mutex> lock(slot.blockMutex);
			if (!slot.active.load(std::memory_order_acquire)) {
				slot.config = config;
				if (slot.config.maxQueueSize == 0) {
					slot.config.maxQueueSize = this->bufferCount;
				}
				slot.droppedCount.store(0, std::memory_order_relaxed);
				slot.queueDepth.store(0, std::memory_order_relaxed);
				slot.active.store(true, std::memory_order_release);
				this->activeCount.fetch_add(1, std::memory_order_relaxed);
				return i;
			}
		}
	}
	throw std::runtime_error("Maximum number of consumers reached");
}

void OutputBufferManager::removeConsumer(ConsumerId id) {
	if (id < 0 || id >= MAX_CONSUMERS) return;

	auto& slot = this->slots[id];
	{
		std::lock_guard<std::mutex> lock(slot.blockMutex);
		if (!slot.active.load(std::memory_order_acquire)) return;

		slot.active.store(false, std::memory_order_release);
		this->activeCount.fetch_sub(1, std::memory_order_relaxed);

		// Release refs for any queued buffers
		IOBuffer* buf;
		while (slot.queue.try_dequeue(buf)) {
			slot.queueDepth.fetch_sub(1, std::memory_order_relaxed);
			this->decrementRef(buf);
		}
	}
	slot.blockCV.notify_all();
}

size_t OutputBufferManager::getConsumerCount() const {
	return static_cast<size_t>(this->activeCount.load(std::memory_order_acquire));
}

void OutputBufferManager::publish(IOBuffer* buffer) {
	int consumers = this->activeCount.load(std::memory_order_acquire);
	if (consumers == 0) {
		if (this->releaseCallback) {
			this->releaseCallback(buffer);
		}
		return;
	}

	int idx = buffer->getBackendIndex();

	// Set initial refCount BEFORE pushing (so any immediate consumer release has valid refCount)
	// Use seq_cst to ensure store is visible to all threads before any queue operation
	if (idx >= 0 && idx < static_cast<int>(this->bufferCount)) {
		this->refCounts[idx].store(consumers, std::memory_order_seq_cst);
	}

	// Push to all active consumer queues, count how many we actually pushed to
	int actualPushes = 0;
	for (auto& slot : this->slots) {
		if (slot.active.load(std::memory_order_acquire)) {
			this->pushToSlot(slot, buffer);
			actualPushes++;
		}
	}

	// Adjust for any consumers that became inactive between reading activeCount and iterating
	// This handles the race where a consumer is removed during publish
	int missed = consumers - actualPushes;
	for (int i = 0; i < missed; i++) {
		this->decrementRef(buffer);
	}
}

void OutputBufferManager::pushToSlot(ConsumerSlot& slot, IOBuffer* buffer) {
	// For DROP_OLDEST policy, drop old buffers if at capacity
	// NOTE: Must use mutex because ReaderWriterQueue is SPSC, but both publisher
	// (dropping old buffers) and consumer (getting buffers) dequeue from it
	if (slot.config.dropPolicy == DropPolicy::DROP_OLDEST) {
		std::vector<IOBuffer*> toDrop;
		{
			std::lock_guard<std::mutex> lock(slot.blockMutex);
			// Collect buffers to drop while holding the lock
			while (slot.queueDepth.load(std::memory_order_acquire) >= slot.config.maxQueueSize) {
				IOBuffer* dropped;
				if (slot.queue.try_dequeue(dropped)) {
					slot.queueDepth.fetch_sub(1, std::memory_order_relaxed);
					toDrop.push_back(dropped);
				} else {
					break;
				}
			}
			// Enqueue new buffer while holding lock
			if (slot.queue.try_enqueue(buffer)) {
				slot.queueDepth.fetch_add(1, std::memory_order_relaxed);
			} else {
				toDrop.push_back(buffer);
			}
		}
		// Release dropped buffers outside the lock
		for (IOBuffer* dropped : toDrop) {
			this->decrementRef(dropped);
			slot.droppedCount.fetch_add(1, std::memory_order_relaxed);
		}
		// Notify consumer
		slot.blockCV.notify_one();
		return;
	}

	// BLOCK policy: wait for space in queue, then enqueue
	{
		std::unique_lock<std::mutex> lock(slot.blockMutex);
		// Wait until there's space in the queue or consumer is deactivated
		slot.blockCV.wait(lock, [&]() {
			return !slot.active.load(std::memory_order_acquire) ||
			       slot.queueDepth.load(std::memory_order_acquire) < slot.config.maxQueueSize;
		});

		if (!slot.active.load(std::memory_order_acquire)) {
			// Consumer was removed while waiting
			this->decrementRef(buffer);
			return;
		}

		if (slot.queue.try_enqueue(buffer)) {
			slot.queueDepth.fetch_add(1, std::memory_order_relaxed);
		} else {
			// Enqueue failed (very rare - only if out of memory)
			this->decrementRef(buffer);
			return;
		}
	}
	// Notify consumer that buffer is available
	slot.blockCV.notify_one();
}

bool OutputBufferManager::tryGet(ConsumerId id, IOBuffer** output) {
	if (id < 0 || id >= MAX_CONSUMERS) return false;

	auto& slot = this->slots[id];
	if (!slot.active.load(std::memory_order_acquire)) return false;

	// For DROP_OLDEST, need mutex since publisher also dequeues (SPSC violation otherwise)
	if (slot.config.dropPolicy == DropPolicy::DROP_OLDEST) {
		std::lock_guard<std::mutex> lock(slot.blockMutex);
		if (slot.queue.try_dequeue(*output)) {
			slot.queueDepth.fetch_sub(1, std::memory_order_relaxed);
			return true;
		}
		return false;
	}

	// BLOCK policy: lock-free dequeue is safe (only consumer dequeues)
	if (slot.queue.try_dequeue(*output)) {
		// Use release so producer's acquire load sees the decrement before waking
		slot.queueDepth.fetch_sub(1, std::memory_order_release);
		// Notify any waiting publisher that space is available
		slot.blockCV.notify_one();
		return true;
	}
	return false;
}

IOBuffer* OutputBufferManager::getNext(ConsumerId id) {
	if (id < 0 || id >= MAX_CONSUMERS) return nullptr;

	auto& slot = this->slots[id];
	IOBuffer* buffer;

	// For DROP_OLDEST, must always use mutex since publisher also dequeues
	if (slot.config.dropPolicy == DropPolicy::DROP_OLDEST) {
		std::unique_lock<std::mutex> lock(slot.blockMutex);
		while (!slot.queue.try_dequeue(buffer)) {
			if (!slot.active.load(std::memory_order_acquire) || !this->running.load(std::memory_order_acquire)) {
				return nullptr;
			}
			slot.blockCV.wait_for(lock, std::chrono::microseconds(100));
		}
		slot.queueDepth.fetch_sub(1, std::memory_order_relaxed);
		return buffer;
	}

	// BLOCK policy: fast path without mutex (only consumer dequeues)
	if (slot.queue.try_dequeue(buffer)) {
		// Use release so producer's acquire load sees the decrement before waking
		slot.queueDepth.fetch_sub(1, std::memory_order_release);
		slot.blockCV.notify_one();
		return buffer;
	}

	// Slow path: wait for buffer with timeout to handle lost wakeups
	std::unique_lock<std::mutex> lock(slot.blockMutex);

	while (!slot.queue.try_dequeue(buffer)) {
		if (!slot.active.load(std::memory_order_acquire) || !this->running.load(std::memory_order_acquire)) {
			return nullptr;
		}
		slot.blockCV.wait_for(lock, std::chrono::microseconds(100));
	}

	// Use release so producer's acquire load sees the decrement before waking
	slot.queueDepth.fetch_sub(1, std::memory_order_release);

	// Notify any waiting publisher that space is available
	lock.unlock();
	slot.blockCV.notify_one();

	return buffer;
}

void OutputBufferManager::release(ConsumerId id, IOBuffer* buffer) {
	if (id < 0 || id >= MAX_CONSUMERS) return;

	this->decrementRef(buffer);
}

void OutputBufferManager::decrementRef(IOBuffer* buffer) {
	if (!buffer) return;

	int idx = buffer->getBackendIndex();
	if (idx < 0 || idx >= static_cast<int>(this->bufferCount)) {
		// Invalid index, just call release callback directly
		if (this->releaseCallback) {
			this->releaseCallback(buffer);
		}
		return;
	}

	int prevCount = this->refCounts[idx].fetch_sub(1, std::memory_order_acq_rel);
	if (prevCount == 1) {
		// Last reference - return buffer to pool
		if (this->releaseCallback) {
			this->releaseCallback(buffer);
		}
	}
}

uint64_t OutputBufferManager::getDroppedCount(ConsumerId id) const {
	if (id < 0 || id >= MAX_CONSUMERS) return 0;
	return this->slots[id].droppedCount.load(std::memory_order_acquire);
}

size_t OutputBufferManager::getQueueSize(ConsumerId id) const {
	if (id < 0 || id >= MAX_CONSUMERS) return 0;
	return this->slots[id].queueDepth.load(std::memory_order_acquire);
}

void OutputBufferManager::shutdown() {
	this->running.store(false, std::memory_order_release);

	// Wake all waiting consumers
	for (auto& slot : this->slots) {
		slot.blockCV.notify_all();
	}

	// Release all queued buffers
	for (auto& slot : this->slots) {
		if (slot.active.load(std::memory_order_acquire)) {
			IOBuffer* buf;
			while (slot.queue.try_dequeue(buf)) {
				slot.queueDepth.fetch_sub(1, std::memory_order_relaxed);
				this->decrementRef(buf);
			}
		}
	}
}

} // namespace ope
