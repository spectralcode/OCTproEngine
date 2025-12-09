#include "output_buffer_manager.h"

namespace ope {

OutputBufferManager::ConsumerSlot::ConsumerSlot()
	: queue(DEFAULT_QUEUE_SIZE)
{
}

OutputBufferManager::OutputBufferManager() {
	for (auto& refCount : this->refCounts) {
		refCount.store(0, std::memory_order_relaxed);
	}
}

OutputBufferManager::~OutputBufferManager() {
	this->shutdown();
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
					slot.config.maxQueueSize = DEFAULT_QUEUE_SIZE;
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

void OutputBufferManager::removeConsumer(ConsumerId id) {
	if (id < 0 || id >= MAX_CONSUMERS) return;

	auto& slot = this->slots[id];
	{
		std::lock_guard<std::mutex> lock(slot.blockMutex);
		if (!slot.active.load(std::memory_order_acquire)) return;

		bool isDropOldest = (slot.config.dropPolicy == DropPolicy::DROP_OLDEST);

		slot.active.store(false, std::memory_order_release);
		this->activeCount.fetch_sub(1, std::memory_order_relaxed);

		// Release refs for any queued buffers (only for BLOCK consumers)
		IOBuffer* buf;
		while (slot.queue.try_dequeue(buf)) {
			if (!isDropOldest) {
				this->decrementRef(buf);
			}
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

	// Count only BLOCK consumers for reference counting
	// DROP_OLDEST consumers don't hold references - they access data opportunistically
	int blockConsumers = 0;
	for (auto& slot : this->slots) {
		if (slot.active.load(std::memory_order_acquire) &&
		    slot.config.dropPolicy == DropPolicy::BLOCK) {
			blockConsumers++;
		}
	}

	// Set reference count based on BLOCK consumers only
	int idx = buffer->getBackendIndex();
	if (idx >= 0 && idx < static_cast<int>(MAX_OUTPUT_BUFFERS)) {
		// If no BLOCK consumers, set to 1 and release after pushing
		this->refCounts[idx].store(blockConsumers > 0 ? blockConsumers : 1, std::memory_order_release);
	}

	// Push to all active consumer queues
	for (auto& slot : this->slots) {
		if (slot.active.load(std::memory_order_acquire)) {
			this->pushToSlot(slot, buffer);
		}
	}

	// If no BLOCK consumers, release immediately after pushing to DROP_OLDEST
	if (blockConsumers == 0) {
		this->decrementRef(buffer);
	}
}

void OutputBufferManager::pushToSlot(ConsumerSlot& slot, IOBuffer* buffer) {
	// For DROP_OLDEST policy, drop old buffers if at capacity
	// Note: DROP_OLDEST consumers don't hold references, so no decrementRef() calls
	if (slot.config.dropPolicy == DropPolicy::DROP_OLDEST) {
		while (slot.queue.size_approx() >= slot.config.maxQueueSize) {
			IOBuffer* dropped;
			if (slot.queue.try_dequeue(dropped)) {
				// No decrementRef - DROP_OLDEST doesn't hold references
				slot.droppedCount.fetch_add(1, std::memory_order_relaxed);
			} else {
				break;
			}
		}
		// DROP_OLDEST: lock-free enqueue, notify without lock
		if (!slot.queue.try_enqueue(buffer)) {
			// No decrementRef - DROP_OLDEST doesn't hold references
			slot.droppedCount.fetch_add(1, std::memory_order_relaxed);
		} else {
			slot.blockCV.notify_one();
		}
		return;
	}

	// BLOCK policy: Enqueue under lock to synchronize with getNext() - prevents lost wakeups
	{
		std::lock_guard<std::mutex> lock(slot.blockMutex);
		if (!slot.queue.try_enqueue(buffer)) {
			// Enqueue failed (very rare - only if out of memory)
			this->decrementRef(buffer);
			return;
		}
	}
	// Notify after releasing lock
	slot.blockCV.notify_one();
}

bool OutputBufferManager::tryGet(ConsumerId id, IOBuffer** output) {
	if (id < 0 || id >= MAX_CONSUMERS) return false;

	auto& slot = this->slots[id];
	if (!slot.active.load(std::memory_order_acquire)) return false;

	return slot.queue.try_dequeue(*output);
}

IOBuffer* OutputBufferManager::getNext(ConsumerId id) {
	if (id < 0 || id >= MAX_CONSUMERS) return nullptr;

	auto& slot = this->slots[id];
	IOBuffer* buffer;

	// Fast path: try lock-free dequeue
	if (slot.queue.try_dequeue(buffer)) {
		return buffer;
	}

	// Slow path: wait for buffer with timeout to handle lost wakeups
	std::unique_lock<std::mutex> lock(slot.blockMutex);

	while (!slot.queue.try_dequeue(buffer)) {
		if (!slot.active.load(std::memory_order_acquire) || !this->running.load(std::memory_order_acquire)) {
			return nullptr;
		}
		// Use wait_for with short timeout to handle potential lost wakeups
		slot.blockCV.wait_for(lock, std::chrono::microseconds(100));
	}

	return buffer;
}

void OutputBufferManager::release(ConsumerId id, IOBuffer* buffer) {
	if (id < 0 || id >= MAX_CONSUMERS) return;

	// DROP_OLDEST consumers don't hold references, so don't decrement
	if (this->slots[id].config.dropPolicy == DropPolicy::DROP_OLDEST) {
		return;
	}

	this->decrementRef(buffer);
}

void OutputBufferManager::decrementRef(IOBuffer* buffer) {
	if (!buffer) return;

	int idx = buffer->getBackendIndex();
	if (idx < 0 || idx >= static_cast<int>(MAX_OUTPUT_BUFFERS)) {
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
	return this->slots[id].queue.size_approx();
}

void OutputBufferManager::shutdown() {
	this->running.store(false, std::memory_order_release);

	// Wake all waiting consumers
	for (auto& slot : this->slots) {
		slot.blockCV.notify_all();
	}

	// Release all queued buffers (only decrement refs for BLOCK consumers)
	for (auto& slot : this->slots) {
		if (slot.active.load(std::memory_order_acquire)) {
			bool isDropOldest = (slot.config.dropPolicy == DropPolicy::DROP_OLDEST);
			IOBuffer* buf;
			while (slot.queue.try_dequeue(buf)) {
				if (!isDropOldest) {
					this->decrementRef(buf);
				}
			}
		}
	}
}

} // namespace ope
