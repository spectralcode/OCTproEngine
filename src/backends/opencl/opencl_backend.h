#ifndef OPE_OPENCL_BACKEND_H
#define OPE_OPENCL_BACKEND_H

#include "../backend_interface.h"
#include <CL/cl.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace ope {

// ============================================
// OpenCL Device Information Structure
// ============================================

struct OpenClDeviceInfo {
	int platformId;
	int deviceId;
	std::string platformName;
	std::string deviceName;
	cl_device_type deviceType;
	size_t globalMemorySize;
	size_t localMemorySize;
	size_t maxWorkGroupSize;
	cl_uint maxComputeUnits;
	cl_uint maxClockFrequency;
	bool isAvailable;

	std::string getDeviceTypeString() const {
		if (deviceType & CL_DEVICE_TYPE_GPU) return "GPU";
		if (deviceType & CL_DEVICE_TYPE_CPU) return "CPU";
		if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) return "Accelerator";
		return "Unknown";
	}
};

// ============================================
// OpenCL Backend Implementation
// ============================================

class OpenClBackend : public ProcessingBackend {
public:
	OpenClBackend();
	~OpenClBackend() override;

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
	// OpenCL-Specific Configuration Methods
	// ============================================

	void setNumInputBuffers(int count);  // Must be called before initialize()  //todo: remove and only use setNumCommandQueues
	void setNumCommandQueues(int numQueues);
	void setWorkGroupSize(int workGroupSize);
	void setPlatformId(int platformId);
	void setDeviceId(int deviceId);  // -1 = auto-select based on preferGpu
	void setPreferGpu(bool prefer);  // Used when deviceId is -1

	int getNumCommandQueues() const;
	int getWorkGroupSize() const;
	int getCurrentPlatformId() const;
	int getCurrentDeviceId() const;

	// ============================================
	// Static OpenCL Management Methods
	// ============================================

	static std::vector<OpenClDeviceInfo> getAvailableDevices();
	static bool selectDevice(int platformId, int deviceId);
	static OpenClDeviceInfo getDeviceInfo(int platformId, int deviceId);

private:
	struct Impl;
	std::unique_ptr<Impl> impl;

	// Helper methods
	void checkOpenClError(cl_int error, const char* context);
	void allocateDeviceBuffers();
	void releaseDeviceBuffers();
	void createCommandQueues();
	void destroyCommandQueues();
	void loadAndBuildKernels();
	void releaseKernels();
	void registerHostMemory();
	void unregisterHostMemory();

	// Callback wrappers for OpenCL events
	static void CL_CALLBACK returnBufferCallback(cl_event event, cl_int status, void* userData);
	static void CL_CALLBACK outputCallback(cl_event event, cl_int status, void* userData);
};

} // namespace ope

#endif // OPE_OPENCL_BACKEND_H
