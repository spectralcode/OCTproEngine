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
	int getOutputBufferCount() const override;
	void releaseOutputBuffer(IOBuffer* buffer) override;
	
	// Post-process background methods
	void requestPostProcessBackgroundRecording() override;
	void setPostProcessBackgroundProfile(const float* background, size_t length) override;
	const std::vector<float>& getPostProcessBackgroundProfile() const override;

	// Fixed-pattern noise management
	void requestFixedPatternNoiseDetermination() override;
	void setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) override;
	const std::vector<float>& getFixedPatternNoiseProfile() const override;

	
	// ============================================
	// CUDA-Specific Configuration Methods
	// ============================================
	
	void setNumInputBuffers(int count);  // Must be called before initialize()
	void setNumOutputBuffers(int count);  // Must be called before initialize() (0 = auto)
	void setNumStreams(int numStreams);
	void setBlockSize(int blockSize);
	void setDeviceId(int deviceId);
	void setEnableZeroCopy(bool enable);  // Must be called before initialize()

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