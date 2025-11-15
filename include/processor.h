#ifndef OPE_PROCESSOR_H
#define OPE_PROCESSOR_H

#include <memory>
#include <functional>
#include <vector>
#include <string>
#include "processorconfiguration.h"
#include "iobuffer.h"
#include "export.h"

namespace ope {

class OPE_API Processor {
public:
	using OutputCallback = std::function<void(const IOBuffer&)>;
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

	// Switch backend (CUDA <-> CPU)
	// Preserves all configuration, automatically cleanup old backend
	void setBackend(Backend backend);

	// Get current backend
	Backend getBackend() const;

	// Process input buffer that was previously acquired via getNextAvailableInputBuffer()
	// processing is asynchronous; output is delivered via registered callbacks
	void process(IOBuffer& input);

	// todo: remove this method and update python bindings, tests etc
	// Set single output callback (legacy method)
	void setOutputCallback(OutputCallback callback);

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

	// Get number of registered callback
	size_t getCallbackCount() const;
	
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
	
	// ============================================
	// LOW-LEVEL API - Individual Operations (for testing)
	// ============================================
	
	std::vector<float> convertInput(
		const void* input,
		IOBuffer::DataType inputType,
		int bitDepth,
		int samples,
		bool applyBitshift = false
	);
	
	std::vector<float> rollingAverageBackgroundRemoval(
		const float* input,
		int windowSize,
		int lineWidth,
		int numLines
	);
	
	std::vector<float> kLinearization(
		const float* input,
		const float* resampleCurve,
		InterpolationMethod method,
		int lineWidth,
		int samples
	);
	
	std::vector<float> windowing(
		const float* input,
		const float* windowCurve,
		int lineWidth,
		int samples
	);
	
	std::vector<float> dispersionCompensation(
		const float* input,
		const float* phaseComplex,
		int lineWidth,
		int samples
	);
	
	std::vector<float> kLinearizationAndWindowing(
		const float* input,
		const float* resampleCurve,
		const float* windowCurve,
		InterpolationMethod method,
		int lineWidth,
		int samples
	);
	
	std::vector<float> kLinearizationAndWindowingAndDispersion(
		const float* input,
		const float* resampleCurve,
		const float* windowCurve,
		const float* phaseComplex,
		InterpolationMethod method,
		int lineWidth,
		int samples
	);
	
	std::vector<float> dispersionCompensationAndWindowing(
		const float* input,
		const float* phaseComplex,
		const float* windowCurve,
		int lineWidth,
		int samples
	);
	
	std::vector<float> fft(const float* input, int lineWidth, int samples);
	std::vector<float> ifft(const float* input, int lineWidth, int samples);
	
	std::vector<float> getMinimumVarianceMean(
		const float* input,
		int width,
		int height,
		int segments
	);
	
	std::vector<float> fixedPatternNoiseRemoval(
		const float* input,
		const float* meanALine,
		int lineWidth,
		int numLines
	);
	
	std::vector<float> postProcessTruncate(
		const float* input,
		bool logScaling,
		float grayscaleMax,
		float grayscaleMin,
		float addend,
		float multiplicator,
		int lineWidth,
		int samples
	);
	
	std::vector<float> bscanFlip(
		const float* input,
		int lineWidth,
		int linesPerBscan,
		int numBscans
	);
	
	std::vector<float> sinusoidalScanCorrection(
		const float* input,
		const float* resampleCurve,
		int lineWidth,
		int linesPerBscan,
		int numBscans
	);
	
	std::vector<float> postProcessBackgroundSubtraction(
		const float* input,
		const float* backgroundLine,
		float weight,
		float offset,
		int lineWidth,
		int samples
	);

private:
	class Impl;
	std::unique_ptr<Impl> impl;
};

} // namespace ope

#endif // OPE_PROCESSOR_H