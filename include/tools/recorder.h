#ifndef OPE_TOOLS_RECORDER_H
#define OPE_TOOLS_RECORDER_H

#include "../../include/processortool.h"
#include "../../include/processor.h"
#include "../../include/iobuffer.h"
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>

namespace ope {
namespace tools {

/**
 * @brief Recorder that collects raw and processed OCT data and saves the data to disk.
 *
 * The recoder needs to be attached to a processor and can then intercept 
 * raw data going into the processor and processed data coming out of it.
 * The intercepted data is collected until the desired buffer count set by 
 * setBufferCount() is reached or the recording is stopped manually via stopRecording() 
 * or aborted via abortRecording(). Stopping will save already collected data to disk. 
 * Aborting will discard all collected data.
 * 
 * With setMode() it is possible to select which data to record: raw only, processed only, or both.
 * 
 * For large recordings, it may make sense to pre-allocate recording buffers via 
 * setManualAllocation(true) and allocateBuffers() before starting the recording 
 * to avoid allocation overhead directly after starting the recording via startRecording().
 * In manual allocation mode, the user is also responsible for freeing the buffers 
 * via freeBuffers() when they are no longer needed. * 
 *
 * Usage:
 * @code
 * Recorder recorder;
 * recorder.attachToProcessor(&processor);
 * recorder.setMode(Recorder::Mode::BOTH);
 * recorder.setBufferCount(1000);
 * recorder.setOutputPath("experiment1");
 * recorder.startRecording();
 *
 * // ... processing happens, auto-completes at 1000 buffers ...
 *
 * if (recorder.getStatus() == Recorder::Status::COMPLETE) {
 *     std::cout << "Recording saved!" << std::endl;
 * }
 *
 * // Reuse for next recording
 * recorder.setMode(Recorder::Mode::PROCESSED_ONLY);
 * recorder.setBufferCount(500);
 * recorder.setOutputPath("experiment2");
 * recorder.startRecording();
 * @endcode
 */
class OPE_API Recorder : public ProcessorTool {
public:
	/**
	 * @brief Recording mode
	 */
	enum class Mode {
		RAW_ONLY,       ///< Record only raw (input) data
		PROCESSED_ONLY, ///< Record only processed (output) data
		BOTH            ///< Record both raw and processed data
	};

	/**
	 * @brief File format for recording
	 */
	enum class Format {
		RAW_BINARY  ///< Raw binary format (default)
		// todo: maybe add HDF5, TIFF, etc.
	};

	/**
	 * @brief Recording status
	 */
	enum class Status {
		IDLE,      ///< Newly created or last recording aborted. Ready for new recording
		RECORDING, ///< Currently collecting buffers
		WRITING,   ///< Recording complete, writing to disk
		COMPLETE,  ///< Last recording succeeded. Ready for new recording
		ERROR      ///< Last recording failed
	};

	/**
	 * @brief Buffer ID validation result
	 */
	struct ValidationResult {
		bool success = true;
		std::string errorMessage;
		std::vector<size_t> gapIndices;  // Indices where gaps in IDs occur
	};

	/**
	 * @brief Summary of a completed or stopped recording
	 */
	struct RecordingSummary {
		size_t expectedBuffers = 0;    ///< Number of buffers requested
		size_t rawRecorded = 0;         ///< Number of raw buffers actually recorded
		size_t processedRecorded = 0;   ///< Number of processed buffers actually recorded
		bool complete = false;          ///< True if all requested buffers were recorded for active mode(s)
		std::vector<uint64_t> rawBufferIds;      ///< Buffer IDs of recorded raw buffers
		std::vector<uint64_t> processedBufferIds; ///< Buffer IDs of recorded processed buffers
};

	Recorder();
	~Recorder();

	// Configuration (must be called before startRecording)

	/**
	 * @brief Set recording mode
	 * @param mode Recording mode (RAW_ONLY, PROCESSED_ONLY, or BOTH)
	 * @throws RecordingException if called while recording
	 */
	void setMode(Mode mode);

	/**
	 * @brief Set output directory for recorded files
	 * @param directory Directory path where files will be saved (default: current directory)
	 */
	void setOutputDirectory(const std::string& directory);

	/**
	 * @brief Set base name for output files (without extension or directory)
	 * @param baseName Base name for output files
	 *
	 * Files will be named: <baseName>_raw.ext and <baseName>.ext
	 * Or with timestamp: YYYYMMDD_HHMMSSmmm_<baseName>_raw.ext
	 */
	void setOutputBaseName(const std::string& baseName);

	/**
	 * @brief Set format for raw data files
	 * @param format File format
	 */
	void setRawFormat(Format format);

	/**
	 * @brief Set format for processed data files
	 * @param format File format
	 */
	void setProcessedFormat(Format format);

	/**
	 * @brief Set number of buffers to record
	 * @param count Number of buffers (must be > 0)
	 *
	 * Recording will automatically stop after this many buffers are collected.
	 */
	void setBufferCount(size_t count);

	/**
	 * @brief Get number of buffers to record
	 * @return Number of buffers that will be recorded
	 */
	size_t getBufferCount() const;

	/**
	 * @brief Enable or disable metadata recording
	 * @param enabled If true, saves ProcessorConfiguration alongside data files
	 *
	 */
	void setWriteMetadata(bool enabled);

	/**
	 * @brief Enable or disable timestamp in output filenames
	 * @param enabled If true, filenames include YYYYMMDD_HHMMSSmmm prefix
	 *
	 * When enabled: YYYYMMDD_HHMMSSmmm_<basename>_raw.raw (raw) and YYYYMMDD_HHMMSSmmm_<basename>.raw (processed)
	 * When disabled: <basename>_raw.raw (raw) and <basename>.raw (processed)
	 * Default: enabled (true)
	 */
	void setUseTimestamp(bool enabled);

	/**
	 * @brief Enable or disable manual buffer allocation mode
	 * @param manual If true, disables auto-free (user controls buffer lifetime)
	 *
	 * When manual==false (default, automatic mode):
	 * - startRecording() auto-allocates buffers if needed
	 * - Buffers ALWAYS auto-freed when write completes (status == COMPLETE)
	 * - Auto-reallocates if buffer size/count changes
	 *
	 * When manual==true (manual mode):
	 * - User can call allocateBuffers() early to pre-allocate
	 * - User MUST call freeBuffers() to release memory
* -  * - Buffers NEVER auto-freed (user controls memory lifetime)
	 * - startRecording() only auto-allocates if needed (convenience feature)

	 *
	 * Use manual mode when reusing buffers across multiple recordings
	 * to avoid repeated allocation/deallocation overhead.
	 *
	 * Manual storage also survives abortRecording(). Automatic mode frees the
	 * buffers after every completed or aborted recording and allocates inside
	 * startRecording(), which delays the recording start for large buffers.
	 *
	 * Default: false (automatic mode)
	 */
	void setManualAllocation(bool manual);

	/**
	 * @brief Pre-allocate recording buffers
	 *
	 * Allocates RAM for recording based on current configuration.
	 * Call this before startRecording() to avoid allocation overhead during recording.
	 * For high-speed systems, call this during setup phase to ensure deterministic performance.
	 * Only valid in manual allocation mode.
	 *
	 * @throws RecordingException if not attached to processor or configuration invalid
	 */
	void allocateBuffers();

	/**
	 * @brief Free allocated recording buffers
	 *
	 * Releases RAM allocated by allocateBuffers().
	 * Buffers are automatically freed in destructor.
	 * Only call this if you want to free memory between recordings.
	 * Only valid in manual allocation mode.
	 */
	void freeBuffers();

	/**
	 * @brief Check if buffers are allocated
	 * @return true if buffers are pre-allocated and ready
	 */
	bool isAllocated() const;

	// Recording control

	/**
	 * @brief Start recording
	 *
	 * Pre-allocates RAM and starts recording buffers.
	 * Recording automatically stops after numBuffersToRecord buffers.
	 *
	 * @param waitForVolumeStart If true, waits until bufferID % buffersPerVolume == 0 before starting
	 * @throws RecordingException if already recording or not attached to processor
	 */
	void startRecording(bool waitForVolumeStart = false);

	/**
	 * @brief Stop recording early and save already collected data to disk
	 *
	 * Use to stop recording before reaching numBuffersToRecord.
	 */
	void stopRecording();

	/**
	 * @brief Abort recording and discard all data
	 *
	 * Frees memory without writing to disk.
	 */
	void abortRecording();

	// Status

	/**
	 * @brief Check if currently collecting data or writing to disk
	 */
	bool isRecording() const;

	/**
	 * @brief Get current status
	 */
	Status getStatus() const;

	/**
	 * @brief Get last error message. This can help diagnose errors when status==ERROR
	 * @return Error string, or empty if no error
	 */
	std::string getLastError() const;

	/**
	 * @brief Get summary of the last recording
	 * @return Summary with expected vs actual buffer counts and completion status
	 *
	 * Updated when stopRecording() is called or when recording auto-completes.
	 * Use this to check if you want to verify a recording was complete and check the completeness and order of buffer IDs.
	 */
	RecordingSummary getLastRecordingSummary() const;

	/**
	 * @brief Wait for current recording to finish writing
	 * @param timeoutMs Timeout in milliseconds (<=0 wait forever)
	 * @return true if write completed successfully
	 */
	bool waitForCompletion(int timeoutMs = 0);

protected:
	void configureCallbacks() override;

private:
	Mode mode = Mode::BOTH;
	Format rawFormat = Format::RAW_BINARY;
	Format processedFormat = Format::RAW_BINARY;
	std::string outputDirectory;
	std::string baseName;
	size_t numBuffersToRecord = 0;
	bool writeMetadata = false;
	bool useTimestamp = true;
	bool manualAllocation = false; 
	std::string recordingTimestamp;

	std::atomic<bool> waitingForVolumeStartRaw{false};
	std::atomic<bool> waitingForVolumeStartProcessed{false};
	int buffersPerVolume = 1;
	std::atomic<Status> status{Status::IDLE};
	std::atomic<size_t> rawBuffersRecorded{0};
	std::atomic<size_t> processedBuffersRecorded{0};
	std::atomic<uint64_t> firstRawBufferId{UINT64_MAX};
	// ID fence captured at startRecording(): only buffers with an ID >= this
	// are recorded, so stale frames still queued at the always-registered
	// callbacks can never leak into a new recording
	std::atomic<uint64_t> recordStartId{0};

	size_t rawBufferSize = 0;
	size_t processedBufferSize = 0;
	size_t allocatedNumBuffers = 0;
	std::atomic<bool> buffersAllocated{false};

	std::vector<uint8_t> rawData;
	std::vector<uint8_t> processedData;
	std::vector<uint64_t> rawBufferIds;
	std::vector<uint64_t> processedBufferIds;

	std::atomic<bool> rawPartComplete{false};
	std::atomic<bool> processedPartComplete{false};

	std::atomic<bool> rawWriteComplete{false};
	std::atomic<bool> processedWriteComplete{false};

	mutable std::mutex errorMutex;
	std::string lastError;

	mutable std::mutex summaryMutex;
	RecordingSummary lastRecordingSummary;

	std::atomic<bool> threadsRunning{false};
	std::thread rawCompletionThread;
	std::thread processedCompletionThread;
	std::thread rawWriterThread;
	std::thread processedWriterThread;

	std::queue<bool> rawCompletionQueue;
	std::mutex rawCompletionMutex;
	std::condition_variable rawCompletionCV;

	std::queue<bool> processedCompletionQueue;
	std::mutex processedCompletionMutex;
	std::condition_variable processedCompletionCV;

	struct WriteTask {
		std::string filename;
		const uint8_t* dataPtr;
		size_t dataSize;
		ValidationResult validation;
	};

	std::queue<WriteTask> rawWriteQueue;
	std::mutex rawQueueMutex;
	std::condition_variable rawQueueCV;

	std::queue<WriteTask> processedWriteQueue;
	std::mutex processedQueueMutex;
	std::condition_variable processedQueueCV;

	std::mutex statusMutex;
	std::condition_variable statusCV;

	void collectRawBuffer(const IOBuffer& buffer);
	void collectProcessedBuffer(const IOBuffer& buffer);

	std::vector<uint64_t> getRawBufferIds() const;
	std::vector<uint64_t> getProcessedBufferIds() const;

	void rawCompletionThreadFunc();
	void processedCompletionThreadFunc();
	void rawWriterFunc();
	void processedWriterFunc();

	void handleRawCompletion();
	void handleProcessedCompletion();
	void finalizeRecording();

	ValidationResult validateRawBufferIds();
	ValidationResult validateProcessedBufferIds();

	void updateRecordingSummary();

	bool writeToFile(const std::string& filename, const uint8_t* data, size_t size);
	bool writeMetadataFile(const std::string& filename);
	std::string getExtension(Format format) const;
	std::string generateTimestamp() const;
	std::string buildFilePath(const std::string& filename) const;

	void freeBuffersInternal();  //internal free without status checks (for auto-free)
	void appendError(const std::string& msg);
	bool isActivelyRecording() const;

	void registerCallbacksForCurrentMode();
};

/**
 * @brief Exception thrown by Recorder
 */
class RecordingException : public std::runtime_error {
public:
	RecordingException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace tools
} // namespace ope

#endif // OPE_TOOLS_RECORDER_H
