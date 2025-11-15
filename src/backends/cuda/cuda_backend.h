#ifndef OPE_CUDA_BACKEND_H
#define OPE_CUDA_BACKEND_H

#include "../backend_interface.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace ope {

// ============================================
// GPU Device Information Structure
// ============================================

struct GpuDeviceInfo {
	int deviceId;
	std::string name;
	size_t totalMemory;
	size_t freeMemory;
	int computeCapabilityMajor;
	int computeCapabilityMinor;
	int maxThreadsPerBlock;
	int multiProcessorCount;
	bool isAvailable;
	
	std::string getComputeCapability() const {
		return std::to_string(computeCapabilityMajor) + "." + std::to_string(computeCapabilityMinor);
	}
};

// ============================================
// CUDA Backend Implementation
// ============================================

class CudaBackend : public ProcessingBackend {
public:
	CudaBackend();
	~CudaBackend() override;
	
	// Lifecycle
	void initialize(const ProcessorConfiguration& config) override;
	void cleanup() override;
	
	void setOutputCallback(std::function<void(const IOBuffer&)> callback) override;
	void process(IOBuffer& input) override;
	
	// Configuration updates
	void updateConfig(const ProcessorConfiguration& config) override;
	
	// Hot-swap curve updates
	void updateResamplingCurve(const float* curve, size_t length) override;
	void updateDispersionCurve(const float* curve, size_t length) override;
	void updateWindowCurve(const float* curve, size_t length) override;
	
	// Buffer management
	IOBuffer& getInputBuffer(int index) override;
	IOBuffer& getNextAvailableInputBuffer() override;
	int getNumInputBuffers() const override;
	
	// Post-process background methods
	void requestPostProcessBackgroundRecording() override;
	void setPostProcessBackgroundProfile(const float* background, size_t length) override;
	const std::vector<float>& getPostProcessBackgroundProfile() const override;

	// Fixed-pattern noise management
	void requestFixedPatternNoiseDetermination() override;
	void setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) override;
	const std::vector<float>& getFixedPatternNoiseProfile() const override;

	
	// Individual operations (for testing/debugging)
	std::vector<float> convertInput(
		const void* input,
		IOBuffer::DataType inputType,
		int bitDepth,
		int samples,
		bool applyBitshift
	) override;
	
	std::vector<float> rollingAverageBackgroundRemoval(
		const float* input,
		int windowSize,
		int lineWidth,
		int numLines
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
	
	std::vector<float> kLinearizationAndWindowing(
		const float* input,
		const float* resampleCurve,
		const float* windowCurve,
		InterpolationMethod method,
		int lineWidth,
		int samples
	) override;
	
	std::vector<float> kLinearizationAndWindowingAndDispersion(
		const float* input,
		const float* resampleCurve,
		const float* windowCurve,
		const float* phaseComplex,
		InterpolationMethod method,
		int lineWidth,
		int samples
	) override;
	
	std::vector<float> dispersionCompensationAndWindowing(
		const float* input,
		const float* phaseComplex,
		const float* windowCurve,
		int lineWidth,
		int samples
	) override;
	
	std::vector<float> fft(const float* input, int lineWidth, int samples) override;
	std::vector<float> ifft(const float* input, int lineWidth, int samples) override;
	
	std::vector<float> getMinimumVarianceMean(
		const float* input,
		int width,
		int height,
		int segments
	) override;
	
	std::vector<float> fixedPatternNoiseRemoval(
		const float* input,
		const float* meanALine,
		int lineWidth,
		int numLines
	) override;
	
	std::vector<float> postProcessTruncate(
		const float* input,
		bool logScaling,
		float grayscaleMax,
		float grayscaleMin,
		float addend,
		float multiplicator,
		int lineWidth,
		int samples
	) override;
	
	std::vector<float> bscanFlip(
		const float* input,
		int lineWidth,
		int linesPerBscan,
		int numBscans
	) override;
	
	std::vector<float> sinusoidalScanCorrection(
		const float* input,
		const float* resampleCurve,
		int lineWidth,
		int linesPerBscan,
		int numBscans
	) override;
	
	std::vector<float> postProcessBackgroundSubtraction(
		const float* input,
		const float* backgroundLine,
		float weight,
		float offset,
		int lineWidth,
		int samples
	) override;
	
	// ============================================
	// CUDA-Specific Configuration Methods
	// ============================================
	
	void setNumInputBuffers(int count);  // Must be called before initialize()
	void setNumStreams(int numStreams);
	void setBlockSize(int blockSize);
	void setDeviceId(int deviceId);
	
	int getNumStreams() const;
	int getBlockSize() const;
	int getCurrentDeviceId() const;
	
	// ============================================
	// Static GPU Management Methods
	// ============================================
	
	static std::vector<GpuDeviceInfo> getAvailableDevices();
	static bool setDevice(int deviceId);
	static int getCurrentDevice();
	static bool isDeviceAvailable(int deviceId);
	static GpuDeviceInfo getDeviceInfo(int deviceId);
	
private:
	struct Impl;
	std::unique_ptr<Impl> impl;
	
	// Helper methods
	void checkCudaError(cudaError_t error, const char* context);
	void checkCufftError(cufftResult error, const char* context);
	void allocateDeviceBuffers();
	void releaseDeviceBuffers();
	void createStreamsAndEvents();
	void destroyStreamsAndEvents();
	void registerHostMemory();
	void unregisterHostMemory();
	
	// Callback wrappers for CUDA stream
	static void CUDART_CB returnBufferCallback(void* userData);
	static void CUDART_CB outputCallback(void* userData);
};

} // namespace ope

#endif // OPE_CUDA_BACKEND_H