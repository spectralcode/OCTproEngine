#ifndef OPE_CPU_BACKEND_H
#define OPE_CPU_BACKEND_H

#include "../backend_interface.h"
#include "fftw3.h"
#include <memory>
#include <vector>
#include <complex>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace ope {

class CpuBackend : public ProcessingBackend {
public:
	CpuBackend();
	~CpuBackend() override;
	
	void initialize(const ProcessorConfiguration& config) override;
	void cleanup() override;
	
	void setOutputCallback(std::function<void(const IOBuffer&)> callback) override;
	void process(IOBuffer& input) override;
	
	void updateConfig(const ProcessorConfiguration& config) override;
	
	void updateResamplingCurve(const float* curve, size_t length) override;
	void updateDispersionCurve(const float* curve, size_t length) override;
	void updateWindowCurve(const float* curve, size_t length) override;

	// Post-process background management
	void requestPostProcessBackgroundRecording() override;
	void setPostProcessBackgroundProfile(const float* background, size_t length) override;
	const std::vector<float>& getPostProcessBackgroundProfile() const override;

	// Buffer management
	IOBuffer& getInputBuffer(int index) override;
	IOBuffer& getNextAvailableInputBuffer() override;
	int getNumInputBuffers() const override;
	
	// Individual operations
	std::vector<float> convertInput(
		const void* input,
		IOBuffer::DataType inputType,
		int bitDepth,
		int samples,
		bool applyBitshift
	) override;
	
	std::vector<float> kLinearization(
		const float* input,
		const float* resampleCurve,
		InterpolationMethod method,
		int lineWidth,
		int samples
	) override;
	
	std::vector<float> windowing(
		const float* input,
		const float* windowCurve,
		int lineWidth,
		int samples
	) override;
	
	std::vector<float> dispersionCompensation(
		const float* input,
		const float* phaseComplex,
		int lineWidth,
		int samples
	) override;
	
	std::vector<float> fft(const float* input, int lineWidth, int samples) override;
	std::vector<float> ifft(const float* input, int lineWidth, int samples) override;
	
	// Stubs for other operations (implement as needed)
	std::vector<float> rollingAverageBackgroundRemoval(const float* input, int windowSize, int lineWidth, int numLines) override;
	std::vector<float> kLinearizationAndWindowing(const float* input, const float* resampleCurve, const float* windowCurve, InterpolationMethod method, int lineWidth, int samples) override;
	std::vector<float> kLinearizationAndWindowingAndDispersion(const float* input, const float* resampleCurve, const float* windowCurve, const float* phaseComplex, InterpolationMethod method, int lineWidth, int samples) override;
	std::vector<float> dispersionCompensationAndWindowing(const float* input, const float* phaseComplex, const float* windowCurve, int lineWidth, int samples) override;
	std::vector<float> getMinimumVarianceMean(const float* input, int width, int height, int segments) override;
	std::vector<float> fixedPatternNoiseRemoval(const float* input, const float* meanALine, int lineWidth, int numLines) override;
	std::vector<float> postProcessTruncate(const float* input, bool logScaling, float grayscaleMax, float grayscaleMin, float addend, float multiplicator, int lineWidth, int samples) override;
	std::vector<float> bscanFlip(const float* input, int lineWidth, int linesPerBscan, int numBscans) override;
	std::vector<float> sinusoidalScanCorrection(const float* input, const float* resampleCurve, int lineWidth, int linesPerBscan, int numBscans) override;
	std::vector<float> postProcessBackgroundSubtraction(const float* input, const float* backgroundLine, float weight, float offset, int lineWidth, int samples) override;

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
	
	// Helper functions
	float cubicHermiteInterpolation(float y0, float y1, float y2, float y3, float t);
	float clamp(float value, float low, float high);
};

} // namespace ope

#endif // OPE_CPU_BACKEND_H