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

	// Fixed-pattern noise management
	void requestFixedPatternNoiseDetermination() override;
	void setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) override;
	const std::vector<float>& getFixedPatternNoiseProfile() const override;

	// Buffer management
	IOBuffer& getInputBuffer(int index) override;
	IOBuffer& getNextAvailableInputBuffer() override;
	int getNumInputBuffers() const override;
	int getOutputBufferCount() const override;
	void releaseOutputBuffer(IOBuffer* buffer) override;

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
	
	// Helper functions
	float cubicHermiteInterpolation(float y0, float y1, float y2, float y3, float t);
	float clamp(float value, float low, float high);
};

} // namespace ope

#endif // OPE_CPU_BACKEND_H