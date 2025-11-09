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
	
	// Construction
	Processor(Backend backend);
	~Processor();
	
	Processor(const Processor&) = delete;
	Processor& operator=(const Processor&) = delete;
	
	// ============================================
	// LIFECYCLE
	// ============================================
	
	// Initialize processor and allocate buffers.
	// OPTIONAL - automatically called on first use if not called explicitly.
	// Call this to control when memory allocation occurs.
	void initialize();
	
	// Free all allocated resources.
	// OPTIONAL - automatically called when needed (e.g., setBackend, destructor).
	void cleanup();
	
	// Query initialization state
	bool isInitialized() const;
	
	// ============================================
	// CONFIGURATION - FILE-BASED
	// ============================================
	
	void loadConfigurationFromFile(const std::string& filepath);
	void saveConfigurationToFile(const std::string& filepath) const;
	
	// ============================================
	// CONFIGURATION - READ ACCESS
	// ============================================
	
	// Get read-only reference to current configuration
	// Useful for inspection, debugging, and GUI state synchronization
	const ProcessorConfiguration& getConfig() const;

	// Set entire configuration at once
	void setConfig(const ProcessorConfiguration& config);

	// ============================================
	// CONFIGURATION - CONTROLLED WRITE ACCESS
	// ============================================
	
	// Set input buffer parameters (requires reinitialization)
	void setInputParameters(
		int samplesPerRawAscan,
		int ascansPerBscan,
		int bscansPerBuffer,
		DataType type
	);
	
	// ============================================
	// BACKEND MANAGEMENT
	// ============================================
	
	// Switch backend (CUDA <-> CPU)
	// Preserves all configuration, automatically cleanup old backend
	void setBackend(Backend backend);
	
	// Get current backend
	Backend getBackend() const;
	
	// ============================================
	// PROCESSING
	// ============================================
	
	void process(IOBuffer& input);
	void setOutputCallback(OutputCallback callback);
	
	// ============================================
	// BUFFER MANAGEMENT
	// ============================================
	
	IOBuffer& getInputBuffer(int index); //dont use this, only for testing. will be removed
	IOBuffer& getNextAvailableInputBuffer();
	int getNumInputBuffers() const;
	
	// ============================================
	// HOT-SWAP METHODS (real-time parameter updates)
	// These update parameters immediately without reinitialization
	// ============================================
	
	// Resampling
	void setResamplingCoefficients(const float coefficients[4]);
	void setCustomResamplingCurve(const float* curve, size_t length);
	void useCustomResamplingCurve(bool useCustom);
	void enableResampling(bool enable);
	void setInterpolationMethod(InterpolationMethod method);
	
	// Dispersion
	void setDispersionCoefficients(const float coefficients[4], float factor = 1.0f);
	void setCustomDispersionCurve(const float* curve, size_t length);
	void useCustomDispersionCurve(bool useCustom);
	void enableDispersionCompensation(bool enable);
	
	// Windowing
	void setWindowParameters(WindowType type, float centerPosition, float fillFactor);
	void setCustomWindowCurve(const float* curve, size_t length);
	void useCustomWindowCurve(bool useCustom);
	void enableWindowing(bool enable);
	
	// Post-processing
	void setGrayscaleRange(float min, float max);
	void setSignalMultiplicatorAndAddend(float multiplicator, float addend);
	void enableLogScaling(bool enable);
	
	// Background removal
	void enableBackgroundRemoval(bool enable);
	void setBackgroundRemovalWindowSize(int windowSize);
	
	// Other toggles
	void enableBscanFlip(bool enable);
	void enableSinusoidalScanCorrection(bool enable);
	void enableFixedPatternNoiseRemoval(bool enable);
	void enablePostProcessBackgroundRemoval(bool enable);
	
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
	
	std::vector<float> postProcessBackgroundRemoval(
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