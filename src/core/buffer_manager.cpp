#include "buffer_manager.h"
#include <cassert>
#include <climits>
#include <iostream>

namespace ope {

BufferManager::BufferManager() {
}

BufferManager::~BufferManager() {
	this->shutdown();
}

int BufferManager::slotIndexFromId(ConsumerId id) {
	if (id < 0) {
		return -1;
	}
	// Low part encodes the slot index, upper part the slot generation (see addConsumer)
	return id % MAX_CONSUMERS;
}

void BufferManager::setBufferCount(size_t count) {
	this->bufferCount = count;
	// Recreating the table also clears stale pointer bindings from a previous
	// backend initialization (buffers get new addresses after reinitialization)
	this->refSlots = std::vector<RefSlot>(count);

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

BufferManager::RefSlot* BufferManager::findRefSlot(IOBuffer* buffer) {
	for (size_t i = 0; i < this->refSlots.size(); i++) {
		if (this->refSlots[i].buffer.load(std::memory_order_acquire) == buffer) {
			return &this->refSlots[i];
		}
	}
	return nullptr;
}

BufferManager::RefSlot* BufferManager::acquireRefSlot(IOBuffer* buffer) {
	// Fast path: buffer was published before and already has a slot
	if (RefSlot* existing = this->findRefSlot(buffer)) {
		return existing;
	}
	// Bind the buffer to a free slot. Only the publisher claims slots, but CAS
	// keeps this safe even if two threads ever race on the same table.
	for (size_t i = 0; i < this->refSlots.size(); i++) {
		IOBuffer* expected = nullptr;
		if (this->refSlots[i].buffer.compare_exchange_strong(expected, buffer, std::memory_order_acq_rel)) {
			return &this->refSlots[i];
		}
		if (expected == buffer) {
			return &this->refSlots[i];
		}
	}
	return nullptr;
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
				slot.queue.clear();
				slot.droppedCount.store(0, std::memory_order_relaxed);
				// Encode a per-slot generation into the ID so a stale ID of a removed
				// consumer can never address a new consumer in the same slot
				slot.generation = (slot.generation + 1) % (INT_MAX / MAX_CONSUMERS);
				slot.currentId = static_cast<ConsumerId>(slot.generation) * MAX_CONSUMERS + i;
				slot.active.store(true, std::memory_order_release);
				this->activeCount.fetch_add(1, std::memory_order_relaxed);
				return slot.currentId;
			}
		}
	}
	throw std::runtime_error("Maximum number of consumers reached");
}

void BufferManager::removeConsumer(ConsumerId id) {
	int index = slotIndexFromId(id);
	if (index < 0) return;

	auto& slot = this->slots[index];
	std::vector<IOBuffer*> drained;
	{
		std::lock_guard<std::mutex> lock(slot.blockMutex);
		if (!slot.active.load(std::memory_order_acquire) || slot.currentId != id) return;

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
	if (!buffer) return;

	RefSlot* ref = this->acquireRefSlot(buffer);
	if (!ref) {
		// More distinct buffers than announced via setBufferCount().
		// The buffer cannot be tracked, so it must not be handed to consumers.
		assert(false && "publish(): no free reference slot, check setBufferCount()");
		if (this->releaseCallback) {
			this->releaseCallback(buffer);
		}
		return;
	}

	// The publisher holds its own reference while distributing. This way the
	// refcount can never drop to zero before all pushes are accounted for,
	// no matter how consumers are added or removed concurrently.
	ref->count.store(1, std::memory_order_seq_cst);

	// Push to all active consumer queues
	for (auto& slot : this->slots) {
		if (!slot.active.load(std::memory_order_acquire)) {
			continue;
		}
		ref->count.fetch_add(1, std::memory_order_acq_rel);
		if (!this->pushToSlot(slot, buffer)) {
			// Consumer was removed while we tried to push; take its reference back.
			// Safe as a plain decrement because the publisher reference is still held.
			ref->count.fetch_sub(1, std::memory_order_acq_rel);
		}
	}

	// Drop the publisher reference. If no consumer accepted the buffer,
	// this releases it back to the pool immediately.
	this->decrementRef(buffer);
}

bool BufferManager::pushToSlot(ConsumerSlot& slot, IOBuffer* buffer) {
	std::vector<IOBuffer*> toDrop;
	{
		// slot.config must only be read under the lock, addConsumer() writes it
		std::unique_lock<std::mutex> lock(slot.blockMutex);
		if (!slot.active.load(std::memory_order_acquire)) {
			// Consumer was removed between publish()'s check and here
			return false;
		}

		// For DROP_OLDEST policy, drop old buffers if at capacity
		// (maxQueueSize == 0 means the consumer was added before setBufferCount(); treat as capacity 1)
		if (slot.config.dropPolicy == DropPolicy::DROP_OLDEST) {
			// Collect buffers to drop while holding the lock
			while (!slot.queue.empty() && slot.queue.size() >= (slot.config.maxQueueSize > 0 ? slot.config.maxQueueSize : 1)) {
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
				       slot.queue.size() < (slot.config.maxQueueSize > 0 ? slot.config.maxQueueSize : 1);
			});

			if (!slot.active.load(std::memory_order_acquire) || !this->running.load(std::memory_order_acquire)) {
				// Consumer was removed while waiting
				return false;
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
	return true;
}

bool BufferManager::tryGet(ConsumerId id, IOBuffer** output) {
	int index = slotIndexFromId(id);
	if (index < 0) return false;

	auto& slot = this->slots[index];
	std::lock_guard<std::mutex> lock(slot.blockMutex);
	if (!slot.active.load(std::memory_order_acquire) || slot.currentId != id) return false;
	if (slot.queue.empty()) return false;

	*output = slot.queue.front();
	slot.queue.pop_front();
	// Notify any waiting publisher that space is available
	slot.blockCV.notify_one();
	return true;
}

IOBuffer* BufferManager::getNext(ConsumerId id) {
	int index = slotIndexFromId(id);
	if (index < 0) return nullptr;

	auto& slot = this->slots[index];
	std::unique_lock<std::mutex> lock(slot.blockMutex);

	slot.blockCV.wait(lock, [&]() {
		return !slot.queue.empty() ||
		       !slot.active.load(std::memory_order_acquire) ||
		       slot.currentId != id ||
		       !this->running.load(std::memory_order_acquire);
	});

	if (slot.currentId != id || slot.queue.empty()) {
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
	int index = slotIndexFromId(id);
	if (index < 0) return;

	// Deliberately no currentId validation: a consumer must be able to release
	// an in-flight buffer after removeConsumer() (removal only drains queued buffers)
	this->decrementRef(buffer);
}

void BufferManager::decrementRef(IOBuffer* buffer) {
	if (!buffer) return;

	RefSlot* ref = this->findRefSlot(buffer);
	if (!ref) {
		// Buffer was never published through this manager, nothing to release
		return;
	}

	int prevCount = ref->count.fetch_sub(1, std::memory_order_acq_rel);
	if (prevCount == 1) {
		// Last reference - return buffer to pool
		if (this->releaseCallback) {
			this->releaseCallback(buffer);
		}
		// Empty critical section pairs with the predicate check in waitUntilReleased()
		// so the notification cannot be lost
		{
			std::lock_guard<std::mutex> lock(this->refReleaseMutex);
		}
		this->refReleaseCV.notify_all();
	}
}

void BufferManager::waitUntilReleased(IOBuffer* buffer) {
	if (!buffer) return;

	RefSlot* ref = this->findRefSlot(buffer);
	if (!ref) {
		// Never published, no consumer can hold a reference
		return;
	}

	std::unique_lock<std::mutex> lock(this->refReleaseMutex);
	this->refReleaseCV.wait(lock, [&]() {
		return ref->count.load(std::memory_order_acquire) <= 0 ||
		       !this->running.load(std::memory_order_acquire);
	});
}

uint64_t BufferManager::getDroppedCount(ConsumerId id) const {
	int index = slotIndexFromId(id);
	if (index < 0) return 0;
	auto& slot = this->slots[index];
	std::lock_guard<std::mutex> lock(slot.blockMutex);
	if (slot.currentId != id) return 0;
	return slot.droppedCount.load(std::memory_order_acquire);
}

size_t BufferManager::getQueueSize(ConsumerId id) const {
	int index = slotIndexFromId(id);
	if (index < 0) return 0;
	auto& slot = this->slots[index];
	std::lock_guard<std::mutex> lock(slot.blockMutex);
	if (slot.currentId != id) return 0;
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

	// Wake any producer blocked in waitUntilReleased()
	{
		std::lock_guard<std::mutex> lock(this->refReleaseMutex);
	}
	this->refReleaseCV.notify_all();

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
