#ifndef OPE_OUTPUT_BUFFER_MANAGER_H
#define OPE_OUTPUT_BUFFER_MANAGER_H

#include <array>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include "readerwriterqueue.h"
#include "../../include/processor.h"

namespace ope {

constexpr int MAX_CONSUMERS = 32;
constexpr size_t DEFAULT_QUEUE_SIZE = 64;
// Must be >= backend output buffer count (CUDA: numStreams * 4, Vulkan/OpenCL: numCommandBuffers)
// Used for refCounts array. Buffers with backendIndex >= this value won't be refcounted.
// todo: somehow couple this with backend buffer count dynamically
constexpr size_t MAX_OUTPUT_BUFFERS = 16;

using ReleaseCallback = std::function<void(IOBuffer*)>;

/**
 * @brief Manages output buffer distribution to multiple consumers
 *
 * Uses lock-free SPSC queues per consumer with reference counting.
 *
 * Drop Policies:
 * - BLOCK: Holds buffer references. Safe but slow consumers block producer.
 * - DROP_OLDEST: No references held. Never blocks, but buffer may be
 *   overwritten while consumer reads (data race). 
 *   Use for non-critical things like live visualization.
 */
class OutputBufferManager {
public:
	OutputBufferManager();
	~OutputBufferManager();

	OutputBufferManager(const OutputBufferManager&) = delete;
	OutputBufferManager& operator=(const OutputBufferManager&) = delete;

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

		ConsumerSlot();
	};

	std::array<ConsumerSlot, MAX_CONSUMERS> slots;
	std::atomic<int> activeCount{0};

	std::array<std::atomic<int>, MAX_OUTPUT_BUFFERS> refCounts;

	ReleaseCallback releaseCallback;
	std::atomic<bool> running{true};

	void pushToSlot(ConsumerSlot& slot, IOBuffer* buffer);
	void decrementRef(IOBuffer* buffer);
};

} // namespace ope

#endif // OPE_OUTPUT_BUFFER_MANAGER_H
