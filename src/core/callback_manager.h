#ifndef OPE_CALLBACK_MANAGER_H
#define OPE_CALLBACK_MANAGER_H

#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory>
#include <atomic>
#include "../../include/iobuffer.h"

namespace ope {

/**
 * @brief Manages multiple output callbacks with dedicated consumer threads
 * 
 * Each registered callback gets its own worker thread and queue.
 * When invokeAll() is called, buffer pointers are posted to all consumer queues,
 * and each callback executes in parallel on its dedicated thread.
 * 
 * Thread-safe for concurrent add/remove/invoke operations.
 */
class CallbackManager {
public:
	using CallbackId = int;
	using OutputCallback = std::function<void(const IOBuffer&)>;
	
	CallbackManager();
	~CallbackManager();
	
	// No copy/move
	CallbackManager(const CallbackManager&) = delete;
	CallbackManager& operator=(const CallbackManager&) = delete;
	
	/**
	 * @brief Add a new output callback
	 * 
	 * Creates a dedicated worker thread for this callback.
	 * The callback will be invoked on the worker thread when invokeAll() is called.
	 * 
	 * @param callback Function to call when output is ready
	 * @return Unique ID for this callback (use with removeCallback)
	 */
	CallbackId addCallback(OutputCallback callback);
	
	/**
	 * @brief Remove a callback by ID
	 * 
	 * Stops and destroys the associated worker thread.
	 * Blocks until the thread has finished its current callback (if any).
	 * 
	 * @param id Callback ID returned from addCallback()
	 * @return true if callback was found and removed, false otherwise
	 */
	bool removeCallback(CallbackId id);
	
	/**
	 * @brief Remove all callbacks
	 * 
	 * Stops and destroys all worker threads.
	 */
	void clear();
	
	/**
	 * @brief Invoke all registered callbacks with the given buffer
	 * 
	 * Posts a pointer to the buffer to all consumer queues.
	 * Returns immediately (~0.1ms for 10 consumers).
	 * 
	 * Callbacks execute in parallel on their dedicated threads.
	 * 
	 * WARNING: Buffer lifetime management is caller's responsibility!
	 * Callbacks must copy data if they need to keep it beyond the callback.
	 * 
	 * @param buffer Output buffer to pass to all callbacks
	 */
	void invokeAll(const IOBuffer& buffer);
	
	/**
	 * @brief Get number of registered callbacks
	 */
	size_t getCallbackCount() const;
	
private:
	struct ConsumerThread {
		std::thread thread;
		std::queue<const IOBuffer*> queue;
		std::mutex mutex;
		std::condition_variable cv;
		OutputCallback callback;
		std::atomic<bool> running;
		CallbackId id;
		
		ConsumerThread(CallbackId callbackId, OutputCallback cb);
		~ConsumerThread();
		
		void run();
		void post(const IOBuffer* buffer);
		void stop();
	};
	
	std::vector<std::unique_ptr<ConsumerThread>> consumers;
	mutable std::mutex consumersMutex;
	std::atomic<CallbackId> nextCallbackId;
};

} // namespace ope

#endif // OPE_CALLBACK_MANAGER_H
