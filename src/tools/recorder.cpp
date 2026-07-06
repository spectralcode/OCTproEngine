#include "../../include/tools/recorder.h"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <iostream>

namespace ope {
namespace tools {

// ============================================
// Constructor / Destructor
// ============================================

Recorder::Recorder() {
	this->threadsRunning = true;

	this->rawCompletionThread = std::thread([this]() { this->rawCompletionThreadFunc(); });
	this->processedCompletionThread = std::thread([this]() { this->processedCompletionThreadFunc(); });
	this->rawWriterThread = std::thread([this]() { this->rawWriterFunc(); });
	this->processedWriterThread = std::thread([this]() { this->processedWriterFunc(); });
	this->status = Status::IDLE;
}

Recorder::~Recorder() {
	// Stop all threads
	this->threadsRunning = false;

	// Wake up all threads
	this->rawCompletionCV.notify_one();
	this->processedCompletionCV.notify_one();
	this->rawQueueCV.notify_one();
	this->processedQueueCV.notify_one();

	// Wait for threads to finish
	if (this->rawCompletionThread.joinable()) {
		this->rawCompletionThread.join();
	}
	if (this->processedCompletionThread.joinable()) {
		this->processedCompletionThread.join();
	}
	if (this->rawWriterThread.joinable()) {
		this->rawWriterThread.join();
	}
	if (this->processedWriterThread.joinable()) {
		this->processedWriterThread.join();
	}
}

// ============================================
// Configuration
// ============================================

void Recorder::setMode(Mode newMode) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot change mode while recording");
	}
	if (this->mode != newMode) {
		this->mode = newMode;
		this->buffersAllocated = false;
		if (this->processor) {
			this->registerCallbacksForCurrentMode();
		}
	}
}

void Recorder::setOutputDirectory(const std::string& directory) {
	this->outputDirectory = directory;
}

void Recorder::setOutputBaseName(const std::string& baseName) {
	this->baseName = baseName;
}

void Recorder::setRawFormat(Format format) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot change format while recording");
	}
	this->rawFormat = format;
}

void Recorder::setProcessedFormat(Format format) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot change format while recording");
	}
	this->processedFormat = format;
}

void Recorder::setBufferCount(size_t count) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot change buffer count while recording");
	}
	if (count == 0) {
		throw RecordingException("Number of buffers must be greater than 0");
	}
	this->numBuffersToRecord = count;
	if (this->buffersAllocated && this->allocatedNumBuffers != count) {
		this->buffersAllocated = false;
	}
}

size_t Recorder::getBufferCount() const {
	return this->numBuffersToRecord;
}

void Recorder::setWriteMetadata(bool enabled) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot change metadata setting while recording");
	}
	this->writeMetadata = enabled;
}

void Recorder::setUseTimestamp(bool enabled) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot change timestamp setting while recording");
	}
	this->useTimestamp = enabled;
}

void Recorder::setManualAllocation(bool manual) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot change allocation mode while recording");
	}
	this->manualAllocation = manual;
}

void Recorder::allocateBuffers() {
	//if (this->isActivelyRecording()) {
	//	throw RecordingException("Cannot allocate buffers while recording");
	//}
	if (!this->processor) {
		throw RecordingException("Not attached to processor");
	}
	if (this->numBuffersToRecord == 0) {
		throw RecordingException("Number of buffers to record not set");
	}

	auto& config = this->processor->getConfig();
	this->buffersPerVolume = config.dataParams.buffersPerVolume;

	size_t newRawBufferSize = config.dataParams.signalLength *
	                          config.dataParams.ascansPerBscan *
	                          config.dataParams.bscansPerBuffer *
	                          config.dataParams.getBytesPerSample();


	size_t newProcessedBufferSize = config.dataParams.outputSignalLength() *
	                                config.dataParams.ascansPerBscan *
	                                config.dataParams.bscansPerBuffer *
	                                config.dataParams.getOutputBytesPerSample();

	bool needsReallocation = !this->buffersAllocated ||
	                         this->allocatedNumBuffers != this->numBuffersToRecord ||
	                         this->rawBufferSize != newRawBufferSize ||
	                         this->processedBufferSize != newProcessedBufferSize;

	if (!needsReallocation) {
		return;
	}

	this->rawBufferSize = newRawBufferSize;
	this->processedBufferSize = newProcessedBufferSize;
	this->allocatedNumBuffers = this->numBuffersToRecord;

	if (this->mode == Mode::RAW_ONLY || this->mode == Mode::BOTH) {
		this->rawData.resize(this->numBuffersToRecord * this->rawBufferSize);
		this->rawBufferIds.resize(this->numBuffersToRecord);
	} else {
		this->rawData.clear();
		this->rawData.shrink_to_fit();
		this->rawBufferIds.clear();
		this->rawBufferIds.shrink_to_fit();
	}

	if (this->mode == Mode::PROCESSED_ONLY || this->mode == Mode::BOTH) {
		this->processedData.resize(this->numBuffersToRecord * this->processedBufferSize);
		this->processedBufferIds.resize(this->numBuffersToRecord);
	} else {
		this->processedData.clear();
		this->processedData.shrink_to_fit();
		this->processedBufferIds.clear();
		this->processedBufferIds.shrink_to_fit();
	}

	this->buffersAllocated = true;
}

void Recorder::freeBuffers() {
	if (this->isActivelyRecording()) {
		throw RecordingException("Cannot free buffers while recording");
	}
	this->freeBuffersInternal();
}

void Recorder::freeBuffersInternal() {
	this->rawData.clear();
	this->rawData.shrink_to_fit();
	this->processedData.clear();
	this->processedData.shrink_to_fit();
	this->rawBufferIds.clear();
	this->rawBufferIds.shrink_to_fit();
	this->processedBufferIds.clear();
	this->processedBufferIds.shrink_to_fit();

	this->buffersAllocated = false;
	this->allocatedNumBuffers = 0;
	this->rawBufferSize = 0;
	this->processedBufferSize = 0;
}

bool Recorder::isAllocated() const {
	return this->buffersAllocated.load();
}

// ============================================
// Recording Control
// ============================================

void Recorder::startRecording(bool waitForVolumeStart) {
	if (this->isActivelyRecording()) {
		throw RecordingException("Already recording");
	}
	if (!this->processor) {
		throw RecordingException("Not attached to processor");
	}
	if (this->numBuffersToRecord == 0) {
		throw RecordingException("Number of buffers to record not set");
	}
	if (this->baseName.empty()) {
		throw RecordingException("Output path not set");
	}

	if (!this->buffersAllocated) {
		this->allocateBuffers();
	} else {
		// Already allocated, verify sizes match. This is for programmers who enable manualAllocation but forget to reallocate buffers after change of data/recording size. You are welcome!
		auto& config = this->processor->getConfig();
		this->buffersPerVolume = config.dataParams.buffersPerVolume;

		size_t expectedRawSize = config.dataParams.signalLength *
		                         config.dataParams.ascansPerBscan *
		                         config.dataParams.bscansPerBuffer *
		                         config.dataParams.getBytesPerSample();

		int processedSignalLength = config.dataParams.signalLength;
		size_t expectedProcessedSize = processedSignalLength *
		                               config.dataParams.ascansPerBscan *
		                               config.dataParams.bscansPerBuffer *
		                               config.dataParams.getOutputBytesPerSample();

		if (this->allocatedNumBuffers != this->numBuffersToRecord ||
		    this->rawBufferSize != expectedRawSize ||
		    this->processedBufferSize != expectedProcessedSize) {
			this->allocateBuffers();
		}
	}

	this->rawBuffersRecorded = 0;
	this->processedBuffersRecorded = 0;
	this->rawPartComplete = false;
	this->processedPartComplete = false;
	this->rawWriteComplete = false;
	this->processedWriteComplete = false;
	this->firstRawBufferId = UINT64_MAX;

	{
		std::lock_guard<std::mutex> lock(this->errorMutex);
		this->lastError.clear();
	}

	if (this->useTimestamp) {
		this->recordingTimestamp = this->generateTimestamp();
	}

	this->waitingForVolumeStartRaw = waitForVolumeStart;
	this->waitingForVolumeStartProcessed = waitForVolumeStart;

	// ID fence: callbacks stay registered between recordings (cheap restarts),
	// so frames published before this call may still sit in the callback queues.
	// Recording only accepts buffers processed from this point on.
	this->recordStartId.store(this->processor->getNextBufferId());

	this->status = Status::RECORDING;
}

void Recorder::stopRecording() {
	if (this->status.load() != Status::RECORDING) {
		return;
	}

	{
		std::lock_guard<std::mutex> lock(this->statusMutex);
		this->status = Status::WRITING;  // this stops recording. todo: rethink if this is the best appraoch, because here we do not start writing to disk but nevertheless we set status to WRITING.
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for any in-flight callbacks

	this->updateRecordingSummary();

	size_t actualRawCount = this->rawBuffersRecorded.load();
	size_t actualProcessedCount = this->processedBuffersRecorded.load();

	if (this->mode == Mode::RAW_ONLY || this->mode == Mode::BOTH) {
		if (actualRawCount > 0) {
			{
				std::lock_guard<std::mutex> lock(this->rawCompletionMutex);
				this->rawCompletionQueue.push(true);
			}
			this->rawCompletionCV.notify_one();
		}
	}

	if (this->mode == Mode::PROCESSED_ONLY || this->mode == Mode::BOTH) {
		if (actualProcessedCount > 0) {
			{
				std::lock_guard<std::mutex> lock(this->processedCompletionMutex);
				this->processedCompletionQueue.push(true);
			}
			this->processedCompletionCV.notify_one();
		}
	}
}

void Recorder::abortRecording() {
	if (this->status.load() != Status::RECORDING) { //todo: think about if abortRecording should also abort writing process. Currently it only aborts buffer collection.
		return;
	}

	{
		std::lock_guard<std::mutex> lock(this->statusMutex);
		this->status = Status::IDLE;  // this stops recording
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for any in-flight callbacks

	// Discarding follows the same storage policy as normal completion:
	// automatic allocation frees the buffers, manual allocation keeps them
	// sized for reuse. Deliberately no clear() on the vectors: a cleared
	// vector reports size 0 while startRecording() would skip reallocation,
	// so the next recording's summary would lose its buffer IDs.
	if (!this->manualAllocation) {
		this->freeBuffersInternal();
	}

	this->rawBuffersRecorded = 0;
	this->processedBuffersRecorded = 0;
	this->rawPartComplete = false;
	this->processedPartComplete = false;
}

// ============================================
// Status
// ============================================

bool Recorder::isRecording() const {
	return this->isActivelyRecording();
}

Recorder::Status Recorder::getStatus() const {
	return this->status.load();
}

std::string Recorder::getLastError() const {
	std::lock_guard<std::mutex> lock(this->errorMutex);
	return this->lastError;
}

Recorder::RecordingSummary Recorder::getLastRecordingSummary() const {
	std::lock_guard<std::mutex> lock(this->summaryMutex);
	return this->lastRecordingSummary;
}

std::vector<uint64_t> Recorder::getRawBufferIds() const {
	size_t count = this->rawBuffersRecorded.load();
	if (count == 0 || this->rawBufferIds.empty()) {
		return std::vector<uint64_t>();
	}
	return std::vector<uint64_t>(this->rawBufferIds.begin(),
	                              this->rawBufferIds.begin() + count);
}

std::vector<uint64_t> Recorder::getProcessedBufferIds() const {
	size_t count = this->processedBuffersRecorded.load();
	if (count == 0 || this->processedBufferIds.empty()) {
		return std::vector<uint64_t>();
	}
	return std::vector<uint64_t>(this->processedBufferIds.begin(),
	                              this->processedBufferIds.begin() + count);
}

bool Recorder::waitForCompletion(int timeoutMs) {
	std::unique_lock<std::mutex> lock(this->statusMutex);

	if (timeoutMs > 0) {
		this->statusCV.wait_for(lock, std::chrono::milliseconds(timeoutMs), [this]() {
			Status currentStatus = this->status.load();
			return (currentStatus == Status::COMPLETE ||
			        currentStatus == Status::ERROR ||
			        currentStatus == Status::IDLE);
		});
		Status finalStatus = this->status.load();
		return finalStatus == Status::COMPLETE;
	} else {
		// Wait indefinitely
		this->statusCV.wait(lock, [this]() {
			Status currentStatus = this->status.load();
			return (currentStatus == Status::COMPLETE ||
			        currentStatus == Status::ERROR ||
			        currentStatus == Status::IDLE);
		});
		Status finalStatus = this->status.load();
		return finalStatus == Status::COMPLETE;
	}
}

// ============================================
// Callbacks
// ============================================

void Recorder::configureCallbacks() {
	if (!this->processor) return;
	this->registerCallbacksForCurrentMode();
}

void Recorder::registerCallbacksForCurrentMode() {
	this->cleanupCallbacks();

	if (this->mode == Mode::RAW_ONLY || this->mode == Mode::BOTH) {
		this->rawCallbackId = this->processor->addInputCallback(
			[this](const IOBuffer& buf) { this->collectRawBuffer(buf); }
		);
	}

	if (this->mode == Mode::PROCESSED_ONLY || this->mode == Mode::BOTH) {
		this->processedCallbackId = this->processor->addOutputCallback(
			[this](const IOBuffer& buf) { this->collectProcessedBuffer(buf); }
		);
	}
}

void Recorder::collectRawBuffer(const IOBuffer& buffer) {
	if (this->status.load() != Status::RECORDING) return;

	uint64_t bufferId = buffer.getBufferId();
	const void* dataPtr = buffer.getDataPointer();
	size_t dataSize = buffer.getSizeInBytes();

	if (bufferId < this->recordStartId.load()) {
		return; // stale frame from before startRecording(), see recordStartId
	}

	if (this->waitingForVolumeStartRaw) {
		if (bufferId % this->buffersPerVolume != 0) {
			return;
		}
		this->waitingForVolumeStartRaw = false;
	}

	// Claim next index atomically
	// this loop handles the case where collectRawBuffer is called extremy fast in succession.
	// it avoids overshooting the numBuffersToRecord
	size_t index = this->rawBuffersRecorded.load();
	while (index < this->numBuffersToRecord) {
			// compare_exchange_weak attempts to change rawBuffersRecorded from 'index' to 'index + 1'.
			// On success it returns true. On failure (another thread updated the value or a so called 'spurious failure').
			// it returns false and updates 'index' to the current value of rawBuffersRecorded.
			// see: - Wikipedia, atomic adder example: https://en.wikipedia.org/wiki/Compare-and-swap
			//      - cppreference: https://en.cppreference.com/w/cpp/atomic/atomic/compare_exchange.html
			//      - Raymond Chen blog post: https://devblogs.microsoft.com/oldnewthing/20180330-00/?p=98395)
		if (this->rawBuffersRecorded.compare_exchange_weak(index, index + 1)) { //compare_exchane_weak increases rawBuffersRecorded if not already increased by another thread
			break; // Successfully claimed this index
		}
	}
	if (index >= this->numBuffersToRecord) {
	    return; // Too late, recording full
	}

	if (index == 0) {
		this->firstRawBufferId.store(bufferId); //used for syncing processed buffers in BOTH mode
	}

	size_t offset = index * this->rawBufferSize;
	std::memcpy(this->rawData.data() + offset,
	            dataPtr,
	            dataSize);
	this->rawBufferIds[index] = bufferId;

	// Check if this was the last buffer and signal raw completion thread
	if (index + 1 == this->numBuffersToRecord) {
		{
			std::lock_guard<std::mutex> lock(this->rawCompletionMutex);
			this->rawCompletionQueue.push(true);
		}
		this->rawCompletionCV.notify_one();
	}
}

void Recorder::collectProcessedBuffer(const IOBuffer& buffer) {
	if (this->status.load() != Status::RECORDING) return;

	uint64_t bufferId = buffer.getBufferId();
	const void* dataPtr = buffer.getDataPointer();

	if (bufferId < this->recordStartId.load()) {
		return; // stale frame from before startRecording(), see recordStartId
	}

	if (this->waitingForVolumeStartProcessed) {
		if (bufferId % this->buffersPerVolume != 0) {
			return;
		}
		this->waitingForVolumeStartProcessed = false;
	}

	if (this->mode == Mode::BOTH) {
		uint64_t targetId = this->firstRawBufferId.load();
		// The raw side may not have claimed its first buffer yet: input callbacks
		// run asynchronously, so a processed buffer can overtake the raw callback.
		// Wait for the raw side instead of wrongly skipping this frame (safe here:
		// this runs on a callback worker thread, not inside process())
		while (targetId == UINT64_MAX && this->status.load() == Status::RECORDING) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			targetId = this->firstRawBufferId.load();
		}
		if (targetId == UINT64_MAX) {
			return; // recording stopped before any raw buffer arrived
		}
		if (bufferId < targetId) {
			// here we assume that raw buffers always arrive before processed buffers
			// (in other words: processed buffers lag behind raw buffers)
			// if this is true and recording is started mid-stream,
			// it could happen that that raw and processed buffers are out of sync,
			// so we skip processed buffers until we reach the first raw buffer ID
			return;
		}
	}

	// Claim next index atomically. see comments in collectRawBuffer
	// Unlike to collectRawBuffer, it is unlikely for multiple processed buffers to arrive at the same time
	// but we use the same pattern for consistency. no performance penalty was observed in testing.
	size_t index = this->processedBuffersRecorded.load();
	while (index < this->numBuffersToRecord) {
		if (this->processedBuffersRecorded.compare_exchange_weak(index, index + 1)) { //compare_exchane_weak increases rawBuffersRecorded if not already increased by another thread
			break; // Successfully claimed this index
		}
	}
	if (index >= this->numBuffersToRecord) {
		return; // Too late, recording full
	}

	size_t offset = index * this->processedBufferSize;
 	std::memcpy(this->processedData.data() + offset,
	            dataPtr,
	            this->processedBufferSize);

	this->processedBufferIds[index] = bufferId; 

	// Check if this was the last buffer and signal processed completion thread
	if (index + 1 == this->numBuffersToRecord) {
		{
			std::lock_guard<std::mutex> lock(this->processedCompletionMutex);
			this->processedCompletionQueue.push(true);
		}
		this->processedCompletionCV.notify_one();
	}
}

// ============================================
// Thread Functions
// ============================================

void Recorder::rawCompletionThreadFunc() {
	while (this->threadsRunning) {
		{
			std::unique_lock<std::mutex> lock(this->rawCompletionMutex);
			this->rawCompletionCV.wait(lock, [this]() {
				return !this->rawCompletionQueue.empty() || !this->threadsRunning;
			});
			if (!this->threadsRunning && this->rawCompletionQueue.empty()) {
				break;
			}
			if (!this->rawCompletionQueue.empty()) {
				this->rawCompletionQueue.pop();
			} else {
				continue;
			}
		}
		this->handleRawCompletion();
	}
}

void Recorder::processedCompletionThreadFunc() {
	while (this->threadsRunning) {
		{
			std::unique_lock<std::mutex> lock(this->processedCompletionMutex);
			this->processedCompletionCV.wait(lock, [this]() {
				return !this->processedCompletionQueue.empty() || !this->threadsRunning;
			});
			if (!this->threadsRunning && this->processedCompletionQueue.empty()) {
				break;
			}
			if (!this->processedCompletionQueue.empty()) {
				this->processedCompletionQueue.pop();
			} else {
				continue;
			}
		}
		this->handleProcessedCompletion();
	}
}

// todo: think about if it makes sense to have two writer threads 
// one for writing raw and one for writing processed data to disk
// maybe this harms performance on old HDDs and systems using SD cards
// todo: profile sequencial vs parallel writing speeds on different systems
void Recorder::rawWriterFunc() {
	while (this->threadsRunning) {
		WriteTask task;
		{
			std::unique_lock<std::mutex> lock(this->rawQueueMutex);
			this->rawQueueCV.wait(lock, [this]() {
				return !this->rawWriteQueue.empty() || !this->threadsRunning;
			});
			if (!this->threadsRunning && this->rawWriteQueue.empty()) {
				break;
			}
			if (!this->rawWriteQueue.empty()) {
				task = std::move(this->rawWriteQueue.front());
				this->rawWriteQueue.pop();
			} else {
				continue;
			}
		}

		bool success = this->writeToFile(task.filename, task.dataPtr, task.dataSize);

		if (!success) {
			if (this->status.load() != Status::ERROR) {
				std::lock_guard<std::mutex> lock(this->statusMutex);
				this->status = Status::ERROR;
				this->statusCV.notify_all();
			}
			this->appendError("Failed to write raw data to: " + task.filename); //we can not throw here, as we are in a separate thread. user can use waitForCompletion and getLastError to check for errors
		} else if (!task.validation.success) {
			if (this->status.load() != Status::ERROR) {
				std::lock_guard<std::mutex> lock(this->statusMutex);
				this->status = Status::ERROR;
				this->statusCV.notify_all();
			}
			this->appendError("Raw data validation failed: " + task.validation.errorMessage);
		} else {
			this->rawWriteComplete = true;

			// Check if all writes are done
			if (this->mode == Mode::RAW_ONLY && this->rawWriteComplete) {
				// Do all cleanup BEFORE setting status to COMPLETE to prevent race with setMode()
				if (!this->manualAllocation) {
					this->freeBuffersInternal();
				}
				this->recordingTimestamp.clear();
				// Now safe to notify - all cleanup complete
				{
					std::lock_guard<std::mutex> lock(this->statusMutex);
					this->status = Status::COMPLETE;
					this->statusCV.notify_all();
				}
			} else if (this->mode == Mode::BOTH && this->rawWriteComplete && this->processedWriteComplete) {
				// BOTH mode: Both writers finished
				// Use atomic exchange to ensure only one thread frees buffers
				Status expected = Status::WRITING;
				if (this->status.compare_exchange_strong(expected, Status::COMPLETE)) {
					// This thread won the race. Do all cleanup BEFORE notify to prevent race with setMode()
					if (!this->manualAllocation) {
						this->freeBuffersInternal();
					}
					this->recordingTimestamp.clear();
					// Now safe to notify - all cleanup complete
					{
						std::lock_guard<std::mutex> lock(this->statusMutex);
						this->statusCV.notify_all();
					}
				}
			}
		}
	}
}

void Recorder::processedWriterFunc() {
	while (this->threadsRunning) {
		WriteTask task;
		{
			std::unique_lock<std::mutex> lock(this->processedQueueMutex);
			this->processedQueueCV.wait(lock, [this]() {
				return !this->processedWriteQueue.empty() || !this->threadsRunning;
			});
			if (!this->threadsRunning && this->processedWriteQueue.empty()) {
				break;
			}
			if (!this->processedWriteQueue.empty()) {
				task = std::move(this->processedWriteQueue.front());
				this->processedWriteQueue.pop();
			} else {
				continue;
			}
		}

		bool success = this->writeToFile(task.filename, task.dataPtr, task.dataSize);

		if (!success) {
			if (this->status.load() != Status::ERROR) {
				std::lock_guard<std::mutex> lock(this->statusMutex);
				this->status = Status::ERROR;
				this->statusCV.notify_all();
			}
			this->appendError("Failed to write processed data to: " + task.filename); //we can not throw here, as we are in a separate thread. user can use waitForCompletion and getLastError to check for errors
		} else if (!task.validation.success) {
			if (this->status.load() != Status::ERROR) {
				std::lock_guard<std::mutex> lock(this->statusMutex);
				this->status = Status::ERROR;
				this->statusCV.notify_all();
			}
			this->appendError("Processed data validation failed: " + task.validation.errorMessage);
		} else {

			this->processedWriteComplete = true;

			// Check if all writes are done
			if (this->mode == Mode::PROCESSED_ONLY && this->processedWriteComplete) {
				// Do all cleanup BEFORE setting status to COMPLETE to prevent race with setMode()
				if (!this->manualAllocation) {
					this->freeBuffersInternal();
				}
				this->recordingTimestamp.clear();
				// Now safe to notify - all cleanup complete
				{
					std::lock_guard<std::mutex> lock(this->statusMutex);
					this->status = Status::COMPLETE;
					this->statusCV.notify_all();
				}
			} else if (this->mode == Mode::BOTH && this->rawWriteComplete && this->processedWriteComplete) {
				// BOTH mode: Both writers finished
				// Use atomic exchange to ensure only one thread frees buffers
				Status expected = Status::WRITING;
				if (this->status.compare_exchange_strong(expected, Status::COMPLETE)) {
					// This thread won the race. Do all cleanup BEFORE notify to prevent race with setMode()
					if (!this->manualAllocation) {
						this->freeBuffersInternal();
					}
					this->recordingTimestamp.clear();
					// Now safe to notify - all cleanup complete
					{
						std::lock_guard<std::mutex> lock(this->statusMutex);
						this->statusCV.notify_all();
					}
				}
			}
		}
	}
}

// ============================================
// Completion Handlers
// ============================================

void Recorder::handleRawCompletion() {
	ValidationResult rawValidation = this->validateRawBufferIds();

	WriteTask task;
	std::string filename;
	if (this->useTimestamp) {
		filename = this->recordingTimestamp + "_" + this->baseName + "_raw." + this->getExtension(this->rawFormat);
	} else {
		filename = this->baseName + "_raw." + this->getExtension(this->rawFormat);
	}
	task.filename = this->buildFilePath(filename);
	size_t actualBytes = this->rawBuffersRecorded.load() * this->rawBufferSize;
	task.dataPtr = this->rawData.data();
	task.dataSize = actualBytes;
	task.validation = rawValidation;

	{
		std::lock_guard<std::mutex> lock(this->rawQueueMutex);
		this->rawWriteQueue.push(std::move(task));
	}
	this->rawQueueCV.notify_one();

	this->rawPartComplete = true;

	if (this->mode == Mode::RAW_ONLY) {
		this->finalizeRecording();
	} else if (this->mode == Mode::BOTH) {
		if (this->processedPartComplete.load()) {
			this->finalizeRecording();
		}
	}
}

void Recorder::handleProcessedCompletion() {
	ValidationResult processedValidation = this->validateProcessedBufferIds();

	WriteTask task;
	std::string filename;
	if (this->useTimestamp) {
		filename = this->recordingTimestamp + "_" + this->baseName + "." + this->getExtension(this->processedFormat);
	} else {
		filename = this->baseName + "." + this->getExtension(this->processedFormat);
	}
	task.filename = this->buildFilePath(filename);
	size_t actualBytes = this->processedBuffersRecorded.load() * this->processedBufferSize;
	task.dataPtr = this->processedData.data();
	task.dataSize = actualBytes;
	task.validation = processedValidation;

	{
		std::lock_guard<std::mutex> lock(this->processedQueueMutex);
		this->processedWriteQueue.push(std::move(task));
	}
	this->processedQueueCV.notify_one();

	this->processedPartComplete = true;

	if (this->mode == Mode::PROCESSED_ONLY) {
		this->finalizeRecording();
	} else if (this->mode == Mode::BOTH) {
		if (this->rawPartComplete.load()) {
			this->finalizeRecording();
		}
	}
}

void Recorder::finalizeRecording() {
	this->updateRecordingSummary();

	{
		std::lock_guard<std::mutex> lock(this->statusMutex);
		this->status = Status::WRITING;
	}

	// Write metadata (configuration) if enabled
	if (this->writeMetadata) {
		std::string metadataFilename;
		if (this->useTimestamp) {
			metadataFilename = this->recordingTimestamp + "_" + this->baseName;
		} else {
			metadataFilename = this->baseName;
		}
		this->writeMetadataFile(metadataFilename);
		// Metadata write failure is non-fatal (not implemented yet)
		// Once ProcessorConfiguration::saveToFile() is implemented, this will work
		// Note: saveToFile() will add the .ini extension automatically
	}

	// Reset for next recording
	this->rawPartComplete = false;
	this->processedPartComplete = false;

	// Note: Status will transition to COMPLETE when writer threads finish
	// (see rawWriterFunc and processedWriterFunc)
}

// ============================================
// Validation
// ============================================

Recorder::ValidationResult Recorder::validateRawBufferIds() {
	ValidationResult result;

	size_t actualCount = this->rawBuffersRecorded.load();

	if (actualCount == 0) {
		result.success = false;
		result.errorMessage = "No raw buffers recorded";
		return result;
	}

	// Check for sequential IDs
	for (size_t i = 1; i < actualCount; i++) {
		if (this->rawBufferIds[i] != this->rawBufferIds[i - 1] + 1) {
			result.gapIndices.push_back(i);
			result.success = false;
		}
	}

	if (!result.success && !result.gapIndices.empty()) {
		result.errorMessage = "Buffer ID gaps detected in raw data at indices: ";
		for (size_t idx : result.gapIndices) {
			result.errorMessage += std::to_string(idx) + " ";
		}
	}

	return result;
}

Recorder::ValidationResult Recorder::validateProcessedBufferIds() {
	ValidationResult result;

	size_t actualCount = this->processedBuffersRecorded.load();

	if (actualCount == 0) {
		result.success = false;
		result.errorMessage = "No processed buffers recorded";
		return result;
	}

	// Check for sequential IDs
	for (size_t i = 1; i < actualCount; i++) {
		if (this->processedBufferIds[i] != this->processedBufferIds[i - 1] + 1) {
			result.gapIndices.push_back(i);
			result.success = false;
		}
	}

	if (!result.success && !result.gapIndices.empty()) {
		result.errorMessage = "Buffer ID gaps detected in processed data at indices: ";
		for (size_t idx : result.gapIndices) {
			result.errorMessage += std::to_string(idx) + " ";
		}
	}

	// For BOTH mode, also check if raw and processed IDs match
	if (this->mode == Mode::BOTH) {
		size_t rawCount = this->rawBuffersRecorded.load();
		size_t minCount = std::min(actualCount, rawCount);

		for (size_t i = 0; i < minCount; i++) {
			if (this->processedBufferIds[i] != this->rawBufferIds[i]) {
				result.success = false;
				result.errorMessage += "Raw/Processed ID mismatch at index " + std::to_string(i) +
				                       " (raw=" + std::to_string(this->rawBufferIds[i]) +
				                       ", processed=" + std::to_string(this->processedBufferIds[i]) + ") ";
			}
		}
	}

	return result;
}

// ============================================
// Recording Summary
// ============================================

void Recorder::updateRecordingSummary() {
	size_t actualRawCount = this->rawBuffersRecorded.load();
	size_t actualProcessedCount = this->processedBuffersRecorded.load();

	std::lock_guard<std::mutex> lock(this->summaryMutex);
	this->lastRecordingSummary.expectedBuffers = this->numBuffersToRecord;
	this->lastRecordingSummary.rawRecorded = actualRawCount;
	this->lastRecordingSummary.processedRecorded = actualProcessedCount;

	bool rawComplete = (this->mode == Mode::PROCESSED_ONLY) || 
	                   (actualRawCount >= this->numBuffersToRecord);
	bool processedComplete = (this->mode == Mode::RAW_ONLY) || 
	                         (actualProcessedCount >= this->numBuffersToRecord);
	this->lastRecordingSummary.complete = rawComplete && processedComplete;

	this->lastRecordingSummary.rawBufferIds = this->getRawBufferIds(); //todo: remove getRawBufferIds() method
	this->lastRecordingSummary.processedBufferIds = this->getProcessedBufferIds();
}

// ============================================
// File Writing
// ============================================

bool Recorder::writeToFile(const std::string& filename, const uint8_t* data, size_t size) {
	try {
		std::ofstream file(filename, std::ios::binary);
		if (!file) {
			return false;
		}

		file.write(reinterpret_cast<const char*>(data), size);

		return file.good();
	} catch (...) {
		return false;
	}
}

bool Recorder::writeMetadataFile(const std::string& filename) {
	if (!this->processor) {
		return false;
	}
	auto& config = this->processor->getConfig();
	return config.saveToFile(filename);
}

std::string Recorder::getExtension(Format format) const {
	switch (format) {
		case Format::RAW_BINARY:
			return "raw";
		default:
			return "raw";
	}
}

std::string Recorder::generateTimestamp() const {
	auto now = std::chrono::system_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
	auto timer = std::chrono::system_clock::to_time_t(now);

	std::tm bt;
#ifdef _WIN32
	localtime_s(&bt, &timer);
#else
	localtime_r(&timer, &bt);
#endif

	std::ostringstream oss;
	oss << std::put_time(&bt, "%Y%m%d_%H%M%S");
	oss << std::setfill('0') << std::setw(3) << ms.count();

	return oss.str();
}

std::string Recorder::buildFilePath(const std::string& filename) const {
	if (this->outputDirectory.empty()) {
		return filename;
	}

	// Add path separator if needed
	char sep = '/';
#ifdef _WIN32
	sep = '\\';
#endif

	if (this->outputDirectory.back() == '/' || this->outputDirectory.back() == '\\') {
		return this->outputDirectory + filename;
	}
	return this->outputDirectory + sep + filename;
}

// ============================================
// Internal Helpers
// ============================================

bool Recorder::isActivelyRecording() const {
	Status s = this->status.load();
	return s == Status::RECORDING || s == Status::WRITING;
}

void Recorder::appendError(const std::string& msg) {
	std::lock_guard<std::mutex> lock(this->errorMutex);
	if (this->lastError.empty()) {
		this->lastError = msg;
	} else {
		// Avoid duplicate identical message append
		if (this->lastError.find(msg) == std::string::npos) {
			this->lastError += "\n" + msg;
		}
	}
}

} // namespace tools
} // namespace ope
