#ifndef OPE_VULKAN_BACKEND_H
#define OPE_VULKAN_BACKEND_H

#include "../backend_interface.h"
#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace ope {

// ============================================
// Vulkan Device Information Structure
// ============================================

struct VulkanDeviceInfo {
	int deviceId;
	std::string name;
	size_t totalMemory;
	size_t freeMemory;
	uint32_t apiVersionMajor;
	uint32_t apiVersionMinor;
	uint32_t apiVersionPatch;
	uint32_t driverVersion;
	int maxWorkGroupSize;
	int maxComputeSharedMemorySize;
	bool isAvailable;

	std::string getApiVersion() const {
		return std::to_string(apiVersionMajor) + "." +
		       std::to_string(apiVersionMinor) + "." +
		       std::to_string(apiVersionPatch);
	}
};

// ============================================
// Vulkan Backend Implementation
// ============================================

class VulkanBackend : public ProcessingBackend {
public:
	VulkanBackend();
	~VulkanBackend() override;

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
	// Vulkan-Specific Configuration Methods
	// ============================================

	void setNumInputBuffers(int count);  // Must be called before initialize()
	void setNumOutputBuffers(int count);  // Must be called before initialize() (0 = auto)
	void setNumStagingInputBuffers(int count);  // Must be called before initialize() (0 = auto, default: numCommandBuffers * 2)
	void setNumCommandBuffers(int count);  // Equivalent to CUDA streams
	void setDeviceId(int deviceId);  // Select physical device

	int getNumCommandBuffers() const;
	int getCurrentDeviceId() const;

	// ============================================
	// Static Vulkan Device Management Methods
	// ============================================

	static std::vector<VulkanDeviceInfo> getAvailableDevices();

private:
	struct Impl;
	std::unique_ptr<Impl> impl;

	// Helper methods
	void checkVulkanError(VkResult result, const char* context);
	void allocateDeviceBuffers();
	void releaseDeviceBuffers();
	void createCommandBuffersAndFences();
	void destroyCommandBuffersAndFences();
	void recordCommandBuffers();  // Pre-record command buffers for reuse
	void recordSingleCommandBuffer(VkCommandBuffer cmd, int idx);  // Helper to record one command buffer
	void createComputePipelines();
	void destroyComputePipelines();
};

} // namespace ope

#endif // OPE_VULKAN_BACKEND_H
