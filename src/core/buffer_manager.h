#ifndef OPE_BUFFER_MANAGER_H
#define OPE_BUFFER_MANAGER_H

#include <array>
#include <deque>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include "../../include/processor.h"

namespace ope {

constexpr int MAX_CONSUMERS = 32;

using ReleaseCallback = std::function<void(IOBuffer*)>;

/**
 * @brief Manages buffer distribution to multiple consumers
 *
 * Used for both input buffers (raw data distribution) and output buffers (processed data distribution).
 * The only difference is the release handling:
 * - Input: no release callback; waitUntilReleased() gates producer-side buffer reuse
 * - Output: releaseCallback returns buffer to backend pool
 *
 * Uses one FIFO queue per consumer, guarded by a per-consumer mutex, with
 * reference counting keyed by buffer pointer.
 *
 * Drop Policies:
 * - BLOCK: Holds buffer references. Safe but slow consumers block producer.
 *   Publishing is sequential across consumers, so one full BLOCK queue also
 *   delays delivery to the consumers after it (head-of-line blocking).
 * - DROP_OLDEST: Holds buffer references but drops oldest when queue full.
 *   Never blocks processing, dropped buffers are released and can be reused by backend.
 */
class BufferManager {
public:
	BufferManager();
	~BufferManager();

	BufferManager(const BufferManager&) = delete;
	BufferManager& operator=(const BufferManager&) = delete;

	/**
	 * @brief Set the number of buffers from backend
	 * @note Internal - called by Processor during initialization
	 * Must be called before any buffers are published
	 */
	void setBufferCount(size_t count);

	/**
	 * @brief Set callback for buffer release
	 * Called when refCount reaches 0
	 * - For output buffers: Returns buffer to backend pool
	 * - For input buffers: no callback is installed; producer-side reuse is
	 *   gated via waitUntilReleased() instead
	 */
	void setReleaseCallback(ReleaseCallback callback);

	/**
	 * @brief Register a new consumer
	 * @param config Consumer configuration (queue size, drop policy)
	 * @return Unique consumer ID
	 */
	ConsumerId addConsumer(ConsumerConfig config = {});

	/**
	 * @brief Remove a consumer
	 * Releases references for any queued buffers
	 */
	void removeConsumer(ConsumerId id);

	/**
	 * @brief Get number of active consumers
	 */
	size_t getConsumerCount() const;

	/**
	 * @brief Publish buffer to all consumers
	 * Non-blocking for DROP_OLDEST policy, may block for BLOCK policy
	 */
	void publish(IOBuffer* buffer);

	/**
	 * @brief Try to get next buffer (non-blocking)
	 * @return true if buffer available
	 */
	bool tryGet(ConsumerId id, IOBuffer** output);

	/**
	 * @brief Get next buffer (blocking)
	 * @return Buffer pointer, or nullptr if consumer removed/shutdown
	 */
	IOBuffer* getNext(ConsumerId id);

	/**
	 * @brief Release buffer after processing
	 * Decrements refCount, returns to pool when all consumers done
	 */
	void release(ConsumerId id, IOBuffer* buffer);

	/**
	 * @brief Get number of dropped frames for consumer
	 */
	uint64_t getDroppedCount(ConsumerId id) const;

	/**
	 * @brief Get current queue size for consumer
	 */
	size_t getQueueSize(ConsumerId id) const;

	/**
	 * @brief Block until no consumer holds a reference to the given buffer
	 *
	 * Returns immediately if the buffer was never published through this manager
	 * or if shutdown() was called. Used to gate producer-side buffer reuse
	 * (e.g. before the acquisition loop writes new data into a recycled input buffer).
	 */
	void waitUntilReleased(IOBuffer* buffer);

	/**
	 * @brief Shutdown manager, wake all waiting consumers
	 */
	void shutdown();

private:
	struct ConsumerSlot {
		std::deque<IOBuffer*> queue;
		mutable std::mutex blockMutex;
		std::condition_variable blockCV;
		ConsumerConfig config;
		std::atomic<bool> active{false};
		std::atomic<uint64_t> droppedCount{0};
		// Generation counter makes ConsumerIds unique across slot reuse,
		// so a stale ID of a removed consumer can never address a new consumer
		// that happens to occupy the same slot (ABA problem)
		int generation = 0;
		ConsumerId currentId = -1;
	};

	// Reference counting is keyed by buffer pointer instead of IOBuffer::getBackendIndex(),
	// because backendIndex is an internal backend concept (staging/command buffer mapping)
	// that is not guaranteed to be set or unique per buffer (e.g. OpenCL maps several
	// input buffers to the same command queue index, CUDA and CPU do not set it at all
	// for some buffer types).
	struct RefSlot {
		std::atomic<IOBuffer*> buffer{nullptr};
		std::atomic<int> count{0};
	};

	std::array<ConsumerSlot, MAX_CONSUMERS> slots;
	std::atomic<int> activeCount{0};

	std::vector<RefSlot> refSlots; //tracks how many consumers still hold a reference to a specific buffer.
	size_t bufferCount = 0;

	// Signaled whenever a refcount drops to zero; used by waitUntilReleased()
	std::mutex refReleaseMutex;
	std::condition_variable refReleaseCV;

	ReleaseCallback releaseCallback;
	std::atomic<bool> running{true};

	static int slotIndexFromId(ConsumerId id);
	RefSlot* findRefSlot(IOBuffer* buffer);
	RefSlot* acquireRefSlot(IOBuffer* buffer);
	bool pushToSlot(ConsumerSlot& slot, IOBuffer* buffer);
	void decrementRef(IOBuffer* buffer);
};

} // namespace ope

#endif // OPE_BUFFER_MANAGER_H
