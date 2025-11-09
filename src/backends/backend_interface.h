#ifndef OPE_BACKEND_INTERFACE_H
#define OPE_BACKEND_INTERFACE_H

#include <functional>
#include <vector>
#include "../../include/processorconfiguration.h"
#include "../../include/iobuffer.h"

namespace ope {

// Abstract interface that all backends must implement
class ProcessingBackend {
public:
	virtual ~ProcessingBackend() = default;
	
	// Lifecycle
	virtual void initialize(const ProcessorConfiguration& config) = 0;
	virtual void cleanup() = 0;
	
	virtual void setOutputCallback(std::function<void(const IOBuffer&)> callback) = 0;
	
	// Main processing pipeline
	// Backend manages output buffers internally and calls callback when done
	virtual void process(IOBuffer& input) = 0;
	
	// Configuration updates
	virtual void updateConfig(const ProcessorConfiguration& config) = 0;
	
	// Hot-swap curve updates
	virtual void updateResamplingCurve(const float* curve, size_t length) = 0;
	virtual void updateDispersionCurve(const float* curve, size_t length) = 0;
	virtual void updateWindowCurve(const float* curve, size_t length) = 0;
	
	// Buffer management - for thread-safe, high-throughput acquisition
	virtual IOBuffer& getInputBuffer(int index) = 0;
	virtual IOBuffer& getNextAvailableInputBuffer() = 0;
	virtual int getNumInputBuffers() const = 0;
	
	// Individual operations for testing
	virtual std::vector<float> convertInput(
		const void* input,
		IOBuffer::DataType inputType,
		int bitDepth,
		int samples,
		bool applyBitshift
	) = 0;
	
	virtual std::vector<float> rollingAverageBackgroundRemoval(
		const float* input,
		int windowSize,
		int lineWidth,
		int numLines
	) = 0;
	
	virtual std::vector<float> kLinearization(
		const float* input,
		const float* resampleCurve,
		InterpolationMethod method,
		int lineWidth,
		int samples
	) = 0;
	
	virtual std::vector<float> windowing(
		const float* input,
		const float* windowCurve,
		int lineWidth,
		int samples
	) = 0;
	
	virtual std::vector<float> dispersionCompensation(
		const float* input,
		const float* phaseComplex,
		int lineWidth,
		int samples
	) = 0;
	
	virtual std::vector<float> kLinearizationAndWindowing(
		const float* input,
		const float* resampleCurve,
		const float* windowCurve,
		InterpolationMethod method,
		int lineWidth,
		int samples
	) = 0;
	
	virtual std::vector<float> kLinearizationAndWindowingAndDispersion(
		const float* input,
		const float* resampleCurve,
		const float* windowCurve,
		const float* phaseComplex,
		InterpolationMethod method,
		int lineWidth,
		int samples
	) = 0;
	
	virtual std::vector<float> dispersionCompensationAndWindowing(
		const float* input,
		const float* phaseComplex,
		const float* windowCurve,
		int lineWidth,
		int samples
	) = 0;
	
	virtual std::vector<float> fft(const float* input, int lineWidth, int samples) = 0;
	virtual std::vector<float> ifft(const float* input, int lineWidth, int samples) = 0;
	
	virtual std::vector<float> getMinimumVarianceMean(
		const float* input,
		int width,
		int height,
		int segments
	) = 0;
	
	virtual std::vector<float> fixedPatternNoiseRemoval(
		const float* input,
		const float* meanALine,
		int lineWidth,
		int numLines
	) = 0;
	
	virtual std::vector<float> postProcessTruncate(
		const float* input,
		bool logScaling,
		float grayscaleMax,
		float grayscaleMin,
		float addend,
		float multiplicator,
		int lineWidth,
		int samples
	) = 0;
	
	virtual std::vector<float> bscanFlip(
		const float* input,
		int lineWidth,
		int linesPerBscan,
		int numBscans
	) = 0;
	
	virtual std::vector<float> sinusoidalScanCorrection(
		const float* input,
		const float* resampleCurve,
		int lineWidth,
		int linesPerBscan,
		int numBscans
	) = 0;
	
	virtual std::vector<float> postProcessBackgroundRemoval(
		const float* input,
		const float* backgroundLine,
		float weight,
		float offset,
		int lineWidth,
		int samples
	) = 0;
};

} // namespace ope

#endif // OPE_BACKEND_INTERFACE_H