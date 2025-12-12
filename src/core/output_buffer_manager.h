#ifndef OPE_OUTPUT_BUFFER_MANAGER_H
#define OPE_OUTPUT_BUFFER_MANAGER_H

#include <array>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include "readerwriterqueue.h"
#include "../../include/processor.h"

namespace ope {

constexpr int MAX_CONSUMERS = 32;

using ReleaseCallback = std::function<void(IOBuffer*)>;

/**
 * @brief Manages output buffer distribution to multiple consumers
 *
 * Uses lock-free SPSC queues per consumer with reference counting.
 *
 * Drop Policies:
 * - BLOCK: Holds buffer references. Safe but slow consumers block producer.
 * - DROP_OLDEST: Holds buffer references but drops oldest when queue full.
 *   Never blocks processing, dropped buffers are released and can be reused by backend.
 */
class OutputBufferManager {
public:
	OutputBufferManager();
	~OutputBufferManager();

	OutputBufferManager(const OutputBufferManager&) = delete;
	OutputBufferManager& operator=(const OutputBufferManager&) = delete;

	/**
	 * @brief Set the number of output buffers from backend
	 * @note Internal - called by Processor during initialization
	 * Must be called before any buffers are published
	 */
	void setBufferCount(size_t count);

	/**
	 * @brief Set callback to return buffers to backend pool
	 * Called when refCount reaches 0
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
	 * @brief Shutdown manager, wake all waiting consumers
	 */
	void shutdown();

private:
	struct ConsumerSlot {
		moodycamel::ReaderWriterQueue<IOBuffer*> queue;
		std::mutex blockMutex;
		std::condition_variable blockCV;
		ConsumerConfig config;
		std::atomic<bool> active{false};
		std::atomic<uint64_t> droppedCount{0};
		std::atomic<size_t> queueDepth{0}; //used to enforce maxQueueSize per consumer

		ConsumerSlot();
	};

	std::array<ConsumerSlot, MAX_CONSUMERS> slots;
	std::atomic<int> activeCount{0};

	std::vector<std::atomic<int>> refCounts; //tracks how many consumers still hold a reference to a specific buffer.
	size_t bufferCount = 0;

	ReleaseCallback releaseCallback;
	std::atomic<bool> running{true};

	void pushToSlot(ConsumerSlot& slot, IOBuffer* buffer);
	void decrementRef(IOBuffer* buffer);
};

} // namespace ope

#endif // OPE_OUTPUT_BUFFER_MANAGER_H
