#include "buffer_manager.h"
#include <iostream>

namespace ope {

BufferManager::BufferManager() {
}

BufferManager::~BufferManager() {
	this->shutdown();
}

void BufferManager::setBufferCount(size_t count) {
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

void BufferManager::setReleaseCallback(ReleaseCallback callback) {
	this->releaseCallback = std::move(callback);
}

ConsumerId BufferManager::addConsumer(ConsumerConfig config) {
	for (int i = 0; i < MAX_CONSUMERS; i++) {
		if (!this->slots[i].active.load(std::memory_order_acquire)) {
			auto& slot = this->slots[i];
			std::lock_guard<std::mutex> lock(slot.blockMutex);
			if (!this->slots[i].active.load(std::memory_order_acquire)) {
				slot.config = config;
				if (slot.config.maxQueueSize == 0) {
					slot.config.maxQueueSize = this->bufferCount;
				}
				slot.droppedCount.store(0, std::memory_order_relaxed);
				slot.active.store(true, std::memory_order_release);
				this->activeCount.fetch_add(1, std::memory_order_relaxed);
				return i;
			}
		}
	}
	throw std::runtime_error("Maximum number of consumers reached");
}

void BufferManager::removeConsumer(ConsumerId id) {
	if (id < 0 || id >= MAX_CONSUMERS) return;

	auto& slot = this->slots[id];
	std::vector<IOBuffer*> drained;
	{
		std::lock_guard<std::mutex> lock(slot.blockMutex);
		if (!slot.active.load(std::memory_order_acquire)) return;

		slot.active.store(false, std::memory_order_release);
		this->activeCount.fetch_sub(1, std::memory_order_relaxed);

		// Release refs for any queued buffers
		while (!slot.queue.empty()) {
			drained.push_back(slot.queue.front());
			slot.queue.pop_front();
		}
	}
	// Decrement outside the lock: the release callback may call back into the backend
	for (IOBuffer* buf : drained) {
		this->decrementRef(buf);
	}
	slot.blockCV.notify_all();
}

size_t BufferManager::getConsumerCount() const {
	return static_cast<size_t>(this->activeCount.load(std::memory_order_acquire));
}

void BufferManager::publish(IOBuffer* buffer) {
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

void BufferManager::pushToSlot(ConsumerSlot& slot, IOBuffer* buffer) {
	std::vector<IOBuffer*> toDrop;
	{
		// slot.config must only be read under the lock, addConsumer() writes it
		std::unique_lock<std::mutex> lock(slot.blockMutex);
		if (!slot.active.load(std::memory_order_acquire)) {
			// Consumer was removed between publish()'s check and here
			this->decrementRef(buffer);
			return;
		}

		// For DROP_OLDEST policy, drop old buffers if at capacity
		if (slot.config.dropPolicy == DropPolicy::DROP_OLDEST) {
			// Collect buffers to drop while holding the lock
			while (!slot.queue.empty() && slot.queue.size() >= slot.config.maxQueueSize) {
				toDrop.push_back(slot.queue.front());
				slot.queue.pop_front();
			}
			// Enqueue new buffer while holding lock
			slot.queue.push_back(buffer);
		} else {
			// BLOCK policy: wait for space in queue, then enqueue
			// Wait until there's space in the queue or consumer is deactivated
			slot.blockCV.wait(lock, [&]() {
				return !slot.active.load(std::memory_order_acquire) ||
				       !this->running.load(std::memory_order_acquire) ||
				       slot.queue.size() < slot.config.maxQueueSize;
			});

			if (!slot.active.load(std::memory_order_acquire) || !this->running.load(std::memory_order_acquire)) {
				// Consumer was removed while waiting
				this->decrementRef(buffer);
				return;
			}

			slot.queue.push_back(buffer);
		}
	}
	// Release dropped buffers outside the lock
	for (IOBuffer* dropped : toDrop) {
		this->decrementRef(dropped);
		slot.droppedCount.fetch_add(1, std::memory_order_relaxed);
	}
	// Notify consumer that buffer is available
	slot.blockCV.notify_one();
}

bool BufferManager::tryGet(ConsumerId id, IOBuffer** output) {
	if (id < 0 || id >= MAX_CONSUMERS) return false;

	auto& slot = this->slots[id];
	std::lock_guard<std::mutex> lock(slot.blockMutex);
	if (!slot.active.load(std::memory_order_acquire)) return false;
	if (slot.queue.empty()) return false;

	*output = slot.queue.front();
	slot.queue.pop_front();
	// Notify any waiting publisher that space is available
	slot.blockCV.notify_one();
	return true;
}

IOBuffer* BufferManager::getNext(ConsumerId id) {
	if (id < 0 || id >= MAX_CONSUMERS) return nullptr;

	auto& slot = this->slots[id];
	std::unique_lock<std::mutex> lock(slot.blockMutex);

	slot.blockCV.wait(lock, [&]() {
		return !slot.queue.empty() ||
		       !slot.active.load(std::memory_order_acquire) ||
		       !this->running.load(std::memory_order_acquire);
	});

	if (slot.queue.empty()) {
		return nullptr;
	}

	IOBuffer* buffer = slot.queue.front();
	slot.queue.pop_front();

	// Notify any waiting publisher that space is available
	lock.unlock();
	slot.blockCV.notify_one();

	return buffer;
}

void BufferManager::release(ConsumerId id, IOBuffer* buffer) {
	if (id < 0 || id >= MAX_CONSUMERS) return;

	this->decrementRef(buffer);
}

void BufferManager::decrementRef(IOBuffer* buffer) {
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

uint64_t BufferManager::getDroppedCount(ConsumerId id) const {
	if (id < 0 || id >= MAX_CONSUMERS) return 0;
	return this->slots[id].droppedCount.load(std::memory_order_acquire);
}

size_t BufferManager::getQueueSize(ConsumerId id) const {
	if (id < 0 || id >= MAX_CONSUMERS) return 0;
	auto& slot = this->slots[id];
	std::lock_guard<std::mutex> lock(slot.blockMutex);
	return slot.queue.size();
}

void BufferManager::shutdown() {
	this->running.store(false, std::memory_order_release);

	// Wake all waiting consumers
	for (auto& slot : this->slots) {
		{
			// Empty critical section prevents lost wakeups for threads that
			// evaluated their wait predicate just before running was cleared
			std::lock_guard<std::mutex> lock(slot.blockMutex);
		}
		slot.blockCV.notify_all();
	}

	// Release all queued buffers
	for (auto& slot : this->slots) {
		std::vector<IOBuffer*> drained;
		{
			std::lock_guard<std::mutex> lock(slot.blockMutex);
			while (!slot.queue.empty()) {
				drained.push_back(slot.queue.front());
				slot.queue.pop_front();
			}
		}
		for (IOBuffer* buf : drained) {
			this->decrementRef(buf);
		}
	}
}

} // namespace ope
