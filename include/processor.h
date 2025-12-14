#ifndef OPE_PROCESSOR_H
#define OPE_PROCESSOR_H

#include <memory>
#include <functional>
#include <vector>
#include <string>
#include "processorconfiguration.h"
#include "iobuffer.h"
#include "backendconfig.h"
#include "export.h"

namespace ope {

// Drop policy for output buffer consumers
enum class DropPolicy {
	BLOCK,       // Block producer if consumer queue is full (default, backward compatible)
	DROP_OLDEST  // Drop oldest buffer in queue if full (non-blocking)
};

// Configuration for output buffer consumers
struct ConsumerConfig {
	size_t maxQueueSize = 0;                  // 0 = default queue size
	DropPolicy dropPolicy = DropPolicy::BLOCK;
};

using ConsumerId = int;

class OPE_API Processor {
public:
// todo: Consider redesigning public API for cross-compiler/OS binary compatibility (pure C ABI).
// maybe keep cpp style API but don't use STL (std::function, std::vector, etc.) in public API?
// maybe c api and optional cpp header-only wrapper?
	using OutputCallback = std::function<void(const IOBuffer&)>;
	using InputCallback = std::function<void(const IOBuffer&)>;
	using CallbackId = int;

	// Construction
	Processor(Backend backend);
	~Processor();

	Processor(const Processor&) = delete;
	Processor& operator=(const Processor&) = delete;

	// Initialize processor and allocate buffers.
	// OPTIONAL - automatically called on first use if not called explicitly.
	// Call this to control when memory allocation occurs.
	void initialize();

	// Free all allocated resources.
	// OPTIONAL - automatically called when needed (e.g., setBackend, destructor).
	void cleanup();

	// Query initialization state
	bool isInitialized() const;

	// todo: implement load/save ini files (load should be backwards compatible with OCTproZ settings.ini)
	void loadConfigurationFromFile(const std::string& filepath);
	void saveConfigurationToFile(const std::string& filepath) const;
	
	// Get read-only reference to current configuration
	// Useful for inspection, debugging, and GUI state synchronization
	const ProcessorConfiguration& getConfig() const;

	// Set entire configuration at once
	void setConfig(const ProcessorConfiguration& config);

	// Set input buffer parameters (requires reinitialization)
	void setInputParameters(
		int samplesPerRawAscan,
		int ascansPerBscan,
		int bscansPerBuffer,
		DataType type
	);

	// Set buffers per volume (for volume-synchronized recording)
	void setBuffersPerVolume(int buffersPerVolume);

	// Get buffers per volume
	int getBuffersPerVolume() const;

	// Switch backend (CUDA <-> CPU)
	// Preserves all configuration, automatically cleanup old backend
	void setBackend(Backend backend);

	// Get current backend
	Backend getBackend() const;

	// Process input buffer that was previously acquired via getNextAvailableInputBuffer()
	// processing is asynchronous; output is delivered via registered callbacks
	void process(IOBuffer& input);

	// Add an output callback for processed data
	// Each callback runs on its own dedicated thread. Callbacks execute
	// in parallel when processing completes.
	CallbackId addOutputCallback(OutputCallback callback);

	// Remove previously added output callback by its ID
	// Stops and destroys the associated worker thread.
	// Blocks until the thread finishes its current callback (if any)
	bool removeOutputCallback(CallbackId id);

	// Remove all output callbacks and stop and destroy their threads
	void clearOutputCallbacks();

	// Get number of registered output callbacks
	size_t getOutputCallbackCount() const;

	// Add output callback with custom configuration
	CallbackId addOutputCallback(OutputCallback callback, ConsumerConfig config);

	// ============================================
	// POLLING API (alternative to callbacks)
	// ============================================

	// Register a consumer for polling-based output retrieval
	ConsumerId addConsumer(ConsumerConfig config = {});

	// Remove a consumer (releases any queued buffers)
	void removeConsumer(ConsumerId id);

	// Non-blocking: returns true if buffer available
	bool tryGetOutputBuffer(ConsumerId id, IOBuffer** output);

	// Blocking: waits for next processed buffer
	// Returns nullptr if consumer was removed or shutdown
	IOBuffer* getNextOutputBuffer(ConsumerId id);

	// Release buffer back to pool (required after tryGetOutputBuffer/getNextOutputBuffer)
	void releaseOutputBuffer(ConsumerId id, IOBuffer* buffer);

	// Get number of dropped frames for consumer (only for DROP_OLDEST policy)
	uint64_t getDroppedFrameCount(ConsumerId id) const;

	// Input callbacks - receive input buffer before processing
	// WARNING: Buffer is still in use by backend, copy data if needed beyond callback
	CallbackId addInputCallback(InputCallback callback);
	bool removeInputCallback(CallbackId id);
	void clearInputCallbacks();
	size_t getInputCallbackCount() const;

	//dont use this, only for testing. will be removed
	IOBuffer& getInputBuffer(int index);
	int getNumInputBuffers() const;

	// Get next available input buffer for processing
	// Blocks if no buffer is available
	IOBuffer& getNextAvailableInputBuffer();

	
	// ============================================
	// HOT-SWAP METHODS (real-time parameter updates)
	// These update parameters immediately without reinitialization
	// ============================================
	
	// Resampling
	void setResamplingCoefficients(const float coefficients[4]);
	void setCustomResamplingCurve(const float* curve, size_t length); //todo: think about renaming it to resamplingLut
	void useCustomResamplingCurve(bool useCustom);
	void enableResampling(bool enable);
	void setInterpolationMethod(InterpolationMethod method);
	
	// Dispersion
	void setDispersionCoefficients(const float coefficients[4], float factor = 1.0f);
	void setCustomDispersionCurve(const float* curve, size_t length); //todo: think about renaming it to dispersionPhase
	void useCustomDispersionCurve(bool useCustom);
	void enableDispersionCompensation(bool enable);
	
	// Windowing
	void setWindowParameters(WindowType type, float centerPosition, float fillFactor);
	void setCustomWindowCurve(const float* curve, size_t length); //todo: think about renaming it to windowFunction
	void useCustomWindowCurve(bool useCustom);
	void enableWindowing(bool enable);
	
	// Post-processing
	void setGrayscaleRange(float min, float max);
	void setSignalMultiplicatorAndAddend(float multiplicator, float addend);
	void enableLogScaling(bool enable);
	
	// Background removal // todo: rename to DC removal to distinguish from post-process background subraction
	void enableBackgroundRemoval(bool enable);
	void setBackgroundRemovalWindowSize(int windowSize);

	// Post-process background profile subtraction
	void requestPostProcessBackgroundRecording();
	void setPostProcessBackgroundWeight(float weight);
	void setPostProcessBackgroundOffset(float offset);
	const float* getPostProcessBackgroundProfile() const;
	size_t getPostProcessBackgroundProfileSize() const;
	bool hasPostProcessBackgroundProfile() const;
	void setPostProcessBackgroundProfile(const float* data, size_t size);
	void savePostProcessBackgroundProfileToFile(const std::string& filepath) const;
	void loadPostProcessBackgroundProfileFromFile(const std::string& filepath);
	void enablePostProcessBackgroundSubtraction(bool enable); 
	
	// Other toggles
	void enableBscanFlip(bool enable);
	void enableSinusoidalScanCorrection(bool enable);

	// ============================================
	// BACKEND CONFIGURATION
	// Unified API for backend-specific settings
	// ============================================

	// Set backend configuration
	// Automatically switches backend if type differs from current
	// Preserves all processing configuration
	// @param config Backend-specific configuration (CudaConfig, OpenCLConfig, or CpuConfig)
	void setBackendConfig(const BackendConfig& config);

	// Get current backend configuration
	// @returns Current backend configuration or nullptr if not initialized
	std::unique_ptr<BackendConfig> getBackendConfig() const;

	// Save backend configuration to file
	// @param filepath Path to save configuration
	void saveBackendConfigToFile(const std::string& filepath) const;

	// Load backend configuration from file
	// Automatically switches backend if type differs from current
	// @param filepath Path to load configuration from
	void loadBackendConfigFromFile(const std::string& filepath);

	// Buffer settings (all backends)
	// Set number of input buffers for pipelining
	// Must be called before initialize() or requires cleanup() + reinitialize()
	// @param numBuffers Number of buffers (default: 2)
	void setNumBuffers(int numBuffers);
	int getNumBuffers() const;

	// Fixed-pattern noise
	void enableFixedPatternNoiseRemoval(bool enable);
	void requestFixedPatternNoiseDetermination();

	/// Set the number of B-scans to accumulate for fixed-pattern noise determination
	// Note: the CPU backend can accumulate A-scans across multiple process() calls;
	// the CUDA backend currently operates only on the A-scans contained in the
	// current GPU input buffer.
	void setFixedPatternNoiseBscanCount(int numberOfBscans);

	void enableContinuousFixedPatternNoiseDetermination(bool enable);
	void setFixedPatternNoiseProfile(const float* data, size_t complexPairs);
	const float* getFixedPatternNoiseProfile() const;
	size_t getFixedPatternNoiseProfileSize() const;
	bool hasFixedPatternNoiseProfile() const;
	void saveFixedPatternNoiseProfileToFile(const std::string& filepath) const;
	void loadFixedPatternNoiseProfileFromFile(const std::string& filepath); 
	

private:
	class Impl;
	std::unique_ptr<Impl> impl;
};

} // namespace ope

#endif // OPE_PROCESSOR_H