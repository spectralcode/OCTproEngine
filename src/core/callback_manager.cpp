#include "callback_manager.h"
#include <algorithm>

namespace ope {

CallbackManager::ConsumerThread::ConsumerThread(CallbackId callbackId, OutputCallback cb)
	: callback(std::move(cb))
	, running(true)
	, id(callbackId)
{
	this->thread = std::thread([this]() {
		this->run();
	});
}

CallbackManager::ConsumerThread::~ConsumerThread() {
	this->stop();
}

void CallbackManager::ConsumerThread::run() {
	while (this->running) {
		const IOBuffer* buffer = nullptr;
		
		// Wait for data in queue
		{
			std::unique_lock<std::mutex> lock(this->mutex);
			this->cv.wait(lock, [this]() {
				return !this->queue.empty() || !this->running;
			});
			
			if (!this->running) {
				break;
			}
			
			buffer = this->queue.front();
			this->queue.pop();
		}
		
		// Call user's callback on this thread
		if (buffer) {
			try {
				this->callback(*buffer);
			} catch (const std::exception& e) {
				// Log error but don't crash
				// TODO: Add proper error handling/logging
			} catch (...) {
				// Catch all to prevent thread termination
			}
		}
	}
}

void CallbackManager::ConsumerThread::post(const IOBuffer* buffer) {
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		this->queue.push(buffer);
	}
	this->cv.notify_one();
}

void CallbackManager::ConsumerThread::stop() {
	if (this->running.exchange(false)) {
		// Signal thread to stop
		{
			std::lock_guard<std::mutex> lock(this->mutex);
		}
		this->cv.notify_one();
		
		// Wait for thread to finish
		if (this->thread.joinable()) {
			this->thread.join();
		}
	}
}



CallbackManager::CallbackManager()
	: nextCallbackId(0)
{
}

CallbackManager::~CallbackManager() {
	this->clear();
}

CallbackManager::CallbackId CallbackManager::addCallback(OutputCallback callback) {
	std::lock_guard<std::mutex> lock(this->consumersMutex);
	
	CallbackId id = this->nextCallbackId++;
	auto consumer = std::make_unique<ConsumerThread>(id, std::move(callback));
	this->consumers.push_back(std::move(consumer));
	
	return id;
}

bool CallbackManager::removeCallback(CallbackId id) {
	std::lock_guard<std::mutex> lock(this->consumersMutex);
	
	auto it = std::find_if(
		this->consumers.begin(),
		this->consumers.end(),
		[id](const std::unique_ptr<ConsumerThread>& consumer) {
			return consumer->id == id;
		}
	);
	
	if (it != this->consumers.end()) {
		(*it)->stop(); // Stop thread (blocking until current callback finishes)
		this->consumers.erase(it);
		return true;
	}
	
	return false;
}

void CallbackManager::clear() {
	std::lock_guard<std::mutex> lock(this->consumersMutex);
	for (auto& consumer : this->consumers) {
		consumer->stop();
	}
	this->consumers.clear();
}

void CallbackManager::invokeAll(const IOBuffer& buffer) {
	std::lock_guard<std::mutex> lock(this->consumersMutex);
	for (auto& consumer : this->consumers) {
		consumer->post(&buffer);
	}
}

size_t CallbackManager::getCallbackCount() const {
	std::lock_guard<std::mutex> lock(this->consumersMutex);
	return this->consumers.size();
}

} // namespace ope