#include "vulkan_backend.h"

//	VkFFT backend selection: 0 = Vulkan
#define VKFFT_BACKEND 0
#include <vkFFT/vkFFT.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <cstring>
#include <fstream>
#include <vector>
#include <shaderc/shaderc.hpp>
#include <glslang_c_interface.h>

// Helper macro for checking Vulkan errors
#define checkVulkanErrors(call) \
	do { \
		VkResult err = call; \
		if (err != VK_SUCCESS) { \
			std::stringstream ss; \
			ss << "Vulkan error at " << __FILE__ << ":" << __LINE__ << " - code: " << err; \
			throw std::runtime_error(ss.str()); \
		} \
	} while(0)

// Helper macro for checking VkFFT errors
#define checkVkFFTErrors(call) \
	do { \
		VkFFTResult err = call; \
		if (err != VKFFT_SUCCESS) { \
			std::stringstream ss; \
			ss << "VkFFT error at " << __FILE__ << ":" << __LINE__ << " - code: " << err; \
			throw std::runtime_error(ss.str()); \
		} \
	} while(0)

namespace ope {

// ============================================
// Static Variables
// ============================================

// glslang is process-wide, not per-instance
static bool s_glslangInitialized = false;
static std::mutex s_glslangMutex;

// ============================================
// Forward Declarations
// ============================================

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size,
                  VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                  VkBuffer& buffer, VkDeviceMemory& bufferMemory);

// ============================================
// Implementation Structure
// ============================================

struct VulkanBackend::Impl {
	// ============================================
	// Pipeline Index Enumeration
	// ============================================

	// Enum for indexing into computePipelines vector
	// Reflects the exact order pipelines are created in createComputePipelines()
	// IMPORTANT: Keep this in sync with the order of push_back calls in createComputePipelines()
	enum class PipelineIndex : size_t {
		InputConversion = 0,       // Convert input data to float
		Truncate = 1,              // Post-FFT magnitude/truncate
		Windowing = 2,             // Apply window function
		Postprocess = 3,           // Log scaling, grayscale normalization
		DcRemoval = 4,             // Rolling average background removal
		KLinearization = 5,        // K-space linearization (resampling)
		Dispersion = 6,            // Dispersion compensation
		MergedKLinearWindowDispCubic = 7,    // Combined k-linear + window + dispersion (cubic interpolation)
		MergedKLinearWindowDispLinear = 8,   // Combined k-linear + window + dispersion (linear interpolation)
		MergedKLinearWindowDispLanczos = 9,  // Combined k-linear + window + dispersion (lanczos interpolation)

		Count = 10  // Total number of pipelines
	};

	// Helper function to get pipeline with type-safe indexing
	inline VkPipeline getPipeline(PipelineIndex idx) const {
		return this->computePipelines[static_cast<size_t>(idx)];
	}

	// Configuration
	ProcessorConfiguration config;

	// Vulkan parameters
	int numCommandBuffers = 2;  // Equivalent to CUDA streams
	int deviceId = 0;
	bool vulkanInitialized = false;

	// Data dimensions
	int signalLength = 0;
	int ascansPerBscan = 0;
	int bscansPerBuffer = 0;
	int samplesPerBuffer = 0;
	int bytesPerSample = 0;

	// Vulkan core objects
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue computeQueue = VK_NULL_HANDLE;
	uint32_t queueFamilyIndex = 0;

	// Command buffers and synchronization
	VkCommandPool commandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkFence> fences;
	int currentCommandBuffer = 0;

	// Input buffer management (queue-based, thread-safe)
	int numInputBuffers = 2;  // Default 2
	std::vector<IOBuffer> hostInputBuffers;
	std::queue<IOBuffer*> freeBuffersQueue;
	std::mutex freeQueueMutex;
	std::condition_variable freeQueueCV;

	// Staging buffers (host-visible, one per command buffer)
	std::vector<VkBuffer> stagingInputBuffers;
	std::vector<VkDeviceMemory> stagingInputMemory;
	std::vector<void*> stagingInputMapped;

	std::vector<VkBuffer> stagingOutputBuffers;
	std::vector<VkDeviceMemory> stagingOutputMemory;
	std::vector<void*> stagingOutputMapped;

	// Device buffers (device-local)
	std::vector<VkBuffer> deviceInputBuffers;
	std::vector<VkDeviceMemory> deviceInputMemory;

	VkBuffer deviceFftBuffer = VK_NULL_HANDLE;
	VkDeviceMemory deviceFftMemory = VK_NULL_HANDLE;

	VkBuffer deviceIntermediateBuffer = VK_NULL_HANDLE;  // For preprocessing ping-pong
	VkDeviceMemory deviceIntermediateMemory = VK_NULL_HANDLE;

	VkBuffer deviceProcessedBuffer = VK_NULL_HANDLE;
	VkDeviceMemory deviceProcessedMemory = VK_NULL_HANDLE;

	// Curve buffers
	VkBuffer resampleCurveBuffer = VK_NULL_HANDLE;
	VkDeviceMemory resampleCurveMemory = VK_NULL_HANDLE;

	VkBuffer windowCurveBuffer = VK_NULL_HANDLE;
	VkDeviceMemory windowCurveMemory = VK_NULL_HANDLE;

	VkBuffer dispersionCurveBuffer = VK_NULL_HANDLE;
	VkDeviceMemory dispersionCurveMemory = VK_NULL_HANDLE;

	// Fixed pattern noise removal
	VkBuffer meanALineBuffer = VK_NULL_HANDLE;
	VkDeviceMemory meanALineMemory = VK_NULL_HANDLE;
	bool fixedPatternNoiseDetermined = false;
	std::vector<float> recordedFixedPatternNoise;

	// Post-processing background
	VkBuffer postProcBackgroundBuffer = VK_NULL_HANDLE;
	VkDeviceMemory postProcBackgroundMemory = VK_NULL_HANDLE;
	bool postProcessBackgroundRecordingRequested = false;
	std::vector<float> recordedPostProcessBackground;

	// Compute pipelines (will be created later)
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> descriptorSets;

	// Truncate pipeline resources
	VkPipelineLayout truncatePipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout truncateDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> truncateDescriptorSets;

	// Windowing pipeline resources
	VkPipelineLayout windowingPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout windowingDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> windowingDescriptorSets;

	// Post-processing pipeline resources
	VkPipelineLayout postprocessPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout postprocessDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> postprocessDescriptorSets;

	// DC removal pipeline resources
	VkPipelineLayout dcRemovalPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout dcRemovalDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> dcRemovalDescriptorSets;

	// K-linearization pipeline resources
	VkPipelineLayout klinearizationPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout klinearizationDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> klinearizationDescriptorSets;

	// Dispersion compensation pipeline resources
	VkPipelineLayout dispersionPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout dispersionDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> dispersionDescriptorSets;

	// Merged K-linearization+Windowing+Dispersion pipeline resources (cubic variant)
	VkPipelineLayout klinearCubicWindowingDispersionPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout klinearCubicWindowingDispersionDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> klinearCubicWindowingDispersionDescriptorSets;

	// Individual pipeline objects for each interpolation method
	VkPipeline mergedKlinearCubicWindowingDispersionPipeline = VK_NULL_HANDLE;
	VkPipeline mergedKlinearLinearWindowingDispersionPipeline = VK_NULL_HANDLE;
	VkPipeline mergedKlinearLanczosWindowingDispersionPipeline = VK_NULL_HANDLE;

	// Merged K-linearization+Windowing pipeline resources (cubic variant, no dispersion)
	VkPipelineLayout klinearCubicWindowingPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout klinearCubicWindowingDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> klinearCubicWindowingDescriptorSets;

	// Merged Dispersion+Windowing pipeline resources
	VkPipelineLayout dispersionWindowingPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout dispersionWindowingDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> dispersionWindowingDescriptorSets;

	// Individual merged pipelines (not stored in computePipelines vector)
	VkPipeline klinearCubicWindowingDispersionPipeline = VK_NULL_HANDLE;
	VkPipeline klinearLinearWindowingDispersionPipeline = VK_NULL_HANDLE;
	VkPipeline klinearLanczosWindowingDispersionPipeline = VK_NULL_HANDLE;

	// Shader modules (will be created later)
	std::vector<VkShaderModule> shaderModules;
	std::vector<VkPipeline> computePipelines;

	// VkFFT
	VkFFTConfiguration fftConfig;
	VkFFTApplication fftApp;
	uint64_t fftBufferSize = 0;  // Must persist for VkFFT lifetime
	VkFence fftFence = VK_NULL_HANDLE;  // Dedicated fence for VkFFT operations

	// Output buffers for callback (one per command buffer)
	std::vector<IOBuffer> outputBuffers;

	// Async completion tracking (similar to CUDA stream callbacks)
	struct PendingWork {
		VkFence fence;
		int commandBufferIdx;
		IOBuffer* outputBuffer;
		size_t outputSize;
		int outputSignalLength;
	};
	std::queue<PendingWork> pendingWorkQueue;
	std::mutex pendingWorkMutex;
	std::condition_variable pendingWorkCV;
	std::thread completionThread;
	std::atomic<bool> completionThreadRunning{false};

	// Callback
	std::function<void(const IOBuffer&)> callback;

	Impl() = default;

	~Impl() {
		// Cleanup is handled in cleanup() method
	}

	// Completion thread function (runs asynchronously, similar to CUDA stream callbacks)
	void completionThreadFunc() {
		while (this->completionThreadRunning) {
			PendingWork work;
			{
				std::unique_lock<std::mutex> lock(this->pendingWorkMutex);
				// Wait for work or shutdown signal
				this->pendingWorkCV.wait(lock, [this] {
					return !this->pendingWorkQueue.empty() || !this->completionThreadRunning;
				});

				if (!this->completionThreadRunning && this->pendingWorkQueue.empty()) {
					break;  // Shutdown
				}

				if (this->pendingWorkQueue.empty()) {
					continue;  // Spurious wakeup
				}

				work = this->pendingWorkQueue.front();
				this->pendingWorkQueue.pop();
			}

			// Wait for fence (outside lock to allow concurrent submissions)
			VkResult result = vkWaitForFences(this->device, 1, &work.fence, VK_TRUE, UINT64_MAX);
			if (result != VK_SUCCESS) {
				std::cerr << "Vulkan fence wait failed in completion thread: " << result << std::endl;
				continue;
			}

			// Copy output from staging to output buffer
			std::memcpy(work.outputBuffer->getDataPointer(),
			            this->stagingOutputMapped[work.commandBufferIdx],
			            work.outputSize);

			// Invoke callback if registered
			if (this->callback) {
				this->callback(*work.outputBuffer);
			}
		}
	}
};

// ============================================
// Constructor / Destructor
// ============================================

VulkanBackend::VulkanBackend() : impl(std::make_unique<Impl>()) {
	// Vulkan initialization will happen in initialize()
}

VulkanBackend::~VulkanBackend() {
	this->cleanup();
}

// ============================================
// Configuration Methods (before initialize)
// ============================================

void VulkanBackend::setNumInputBuffers(int count) {
	if (this->impl->vulkanInitialized) {
		throw std::runtime_error("Cannot change number of input buffers after initialization");
	}
	if (count < 1) {
		throw std::invalid_argument("Number of input buffers must be at least 1");
	}
	this->impl->numInputBuffers = count;
}

void VulkanBackend::setNumCommandBuffers(int count) {
	if (this->impl->vulkanInitialized) {
		throw std::runtime_error("Cannot change number of command buffers after initialization");
	}
	if (count < 1) {
		throw std::invalid_argument("Number of command buffers must be at least 1");
	}
	this->impl->numCommandBuffers = count;
}

void VulkanBackend::setDeviceId(int deviceId) {
	if (this->impl->vulkanInitialized) {
		throw std::runtime_error("Cannot change device ID after initialization");
	}
	if (deviceId < 0) {
		throw std::invalid_argument("Device ID must be non-negative");
	}
	this->impl->deviceId = deviceId;
}

int VulkanBackend::getNumCommandBuffers() const {
	return this->impl->numCommandBuffers;
}

int VulkanBackend::getCurrentDeviceId() const {
	return this->impl->deviceId;
}

// ============================================
// Lifecycle
// ============================================

void VulkanBackend::initialize(const ProcessorConfiguration& config) {
	// Store configuration
	this->impl->config = config;

	// Extract dimensions
	this->impl->signalLength = config.dataParams.signalLength;
	this->impl->ascansPerBscan = config.dataParams.ascansPerBscan;
	this->impl->bscansPerBuffer = config.dataParams.bscansPerBuffer;
	this->impl->samplesPerBuffer = this->impl->signalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer;

	// Determine bytes per sample based on input data type
	switch (config.dataParams.inputDataType) {
		case IOBuffer::DataType::UINT8:
			this->impl->bytesPerSample = 1;
			break;
		case IOBuffer::DataType::UINT16:
			this->impl->bytesPerSample = 2;
			break;
		case IOBuffer::DataType::FLOAT32:
			this->impl->bytesPerSample = 4;
			break;
		default:
			throw std::runtime_error("Unsupported input data type");
	}

	// Create Vulkan instance
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "OCTproEngine";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "OCTproEngine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_1;  // Use Vulkan 1.1 for better VkFFT compatibility

	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pApplicationInfo = &appInfo;

	checkVulkanErrors(vkCreateInstance(&instanceCreateInfo, nullptr, &this->impl->instance));

	// Select physical device
	uint32_t deviceCount = 0;
	checkVulkanErrors(vkEnumeratePhysicalDevices(this->impl->instance, &deviceCount, nullptr));

	if (deviceCount == 0) {
		throw std::runtime_error("No Vulkan devices found");
	}

	std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
	checkVulkanErrors(vkEnumeratePhysicalDevices(this->impl->instance, &deviceCount, physicalDevices.data()));

	if (this->impl->deviceId >= static_cast<int>(deviceCount)) {
		throw std::runtime_error("Invalid device ID");
	}

	this->impl->physicalDevice = physicalDevices[this->impl->deviceId];

	// Find compute queue family
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(this->impl->physicalDevice, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(this->impl->physicalDevice, &queueFamilyCount, queueFamilies.data());

	bool foundQueue = false;
	for (uint32_t i = 0; i < queueFamilyCount; i++) {
		if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			this->impl->queueFamilyIndex = i;
			foundQueue = true;
			break;
		}
	}

	if (!foundQueue) {
		throw std::runtime_error("No compute queue family found");
	}

	// Create logical device
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = this->impl->queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	// Query device features and enable what we need
	VkPhysicalDeviceFeatures deviceFeatures = {};
	vkGetPhysicalDeviceFeatures(this->impl->physicalDevice, &deviceFeatures);

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

	checkVulkanErrors(vkCreateDevice(this->impl->physicalDevice, &deviceCreateInfo, nullptr, &this->impl->device));

	// Get compute queue
	vkGetDeviceQueue(this->impl->device, this->impl->queueFamilyIndex, 0, &this->impl->computeQueue);

	// Create command pool
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = this->impl->queueFamilyIndex;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	checkVulkanErrors(vkCreateCommandPool(this->impl->device, &poolInfo, nullptr, &this->impl->commandPool));

	// Allocate buffers
	this->allocateDeviceBuffers();

	// Create command buffers and fences (BEFORE VkFFT initialization)
	this->createCommandBuffersAndFences();

	// Initialize glslang for VkFFT shader compilation (process-wide, only once)
	{
		std::lock_guard<std::mutex> lock(s_glslangMutex);
		if (!s_glslangInitialized) {
			glslang_initialize_process();
			s_glslangInitialized = true;
		}
	}

	// Initialize VkFFT library (BEFORE our own pipelines/descriptor pool)
	memset(&this->impl->fftConfig, 0, sizeof(VkFFTConfiguration));
	memset(&this->impl->fftApp, 0, sizeof(VkFFTApplication));

	// Configure VkFFT for 1D IFFT
	this->impl->fftConfig.FFTdim = 1;  // 1D FFT
	this->impl->fftConfig.size[0] = this->impl->signalLength;
	this->impl->fftConfig.size[1] = 1;
	this->impl->fftConfig.size[2] = 1;

	// Vulkan backend specific configuration
	this->impl->fftConfig.physicalDevice = &this->impl->physicalDevice;
	this->impl->fftConfig.device = &this->impl->device;
	this->impl->fftConfig.queue = &this->impl->computeQueue;
	this->impl->fftConfig.commandPool = &this->impl->commandPool;
	this->impl->fftConfig.fence = &this->impl->fftFence;  // Use dedicated fence for FFT operations
	this->impl->fftConfig.isCompilerInitialized = 1;  // glslang initialized above

	// Provide FFT buffer for VkFFT to use
	this->impl->fftConfig.buffer = &this->impl->deviceFftBuffer;
	this->impl->fftBufferSize = this->impl->samplesPerBuffer * sizeof(float) * 2;  // Complex float (2 floats per sample)
	this->impl->fftConfig.bufferSize = &this->impl->fftBufferSize;

	// Precision (single precision float)
	this->impl->fftConfig.doublePrecision = 0;

	// Set number of batches
	this->impl->fftConfig.numberBatches = this->impl->ascansPerBscan * this->impl->bscansPerBuffer;

	// Disable normalization to match cuFFT behavior (we handle it manually in truncate shader)
	this->impl->fftConfig.normalize = 0;

	// Performance optimizations - set coalescedMemory based on vendor
	// Query device properties to detect vendor
	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(this->impl->physicalDevice, &deviceProperties);

	// For Nvidia and AMD: 32 bytes, for Intel: 64 bytes (per VkFFT documentation)
	if (deviceProperties.vendorID == 0x8086) {  // Intel
		this->impl->fftConfig.coalescedMemory = 64;
	} else {
		// Nvidia (0x10DE), AMD (0x1002), and others
		this->impl->fftConfig.coalescedMemory = 32;
	}

	// Use automatic LUT selection (let VkFFT decide based on FFT size)
	// -1 = off, 0 = auto, 1 = on
	this->impl->fftConfig.useLUT = 0;

	// Target threads per block to match our shader work group size
	this->impl->fftConfig.aimThreads = 128;

	// Number of shared memory banks (NVIDIA has 32)
	this->impl->fftConfig.numSharedBanks = 32;

	// Try bandwidth boost optimization for better memory coalescing
	// This reduces coalesced number to get bigger sequences in one upload
	this->impl->fftConfig.performBandwidthBoost = 2;

	// Initialize VkFFT
	checkVkFFTErrors(initializeVkFFT(&this->impl->fftApp, this->impl->fftConfig));

	// Create compute pipelines and shaders (AFTER VkFFT so it gets its descriptor pool first)
	this->createComputePipelines();

	// Allocate and initialize host input buffers
	size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
	this->impl->hostInputBuffers.resize(this->impl->numInputBuffers);

	for (int i = 0; i < this->impl->numInputBuffers; ++i) {
		this->impl->hostInputBuffers[i].setDataType(config.dataParams.inputDataType);
		this->impl->hostInputBuffers[i].allocateMemory(inputSize);
		this->impl->freeBuffersQueue.push(&this->impl->hostInputBuffers[i]);
	}

	// Allocate output buffers (one per command buffer)
	// Output is truncated to half signal length after FFT
	int outputSignalLength = this->impl->signalLength / 2;
	size_t outputSamplesPerBuffer = outputSignalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer;
	size_t outputSize = outputSamplesPerBuffer * sizeof(float);  // Output is always float
	this->impl->outputBuffers.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		this->impl->outputBuffers[i].setDataType(IOBuffer::DataType::FLOAT32);
		this->impl->outputBuffers[i].allocateMemory(outputSize);
	}

	// Pre-record all command buffers for maximum performance
	this->recordCommandBuffers();

	// Start async completion thread (handles fence polling and callbacks)
	this->impl->completionThreadRunning = true;
	this->impl->completionThread = std::thread(&Impl::completionThreadFunc, this->impl.get());

	this->impl->vulkanInitialized = true;
}

void VulkanBackend::cleanup() {
	if (!this->impl->vulkanInitialized) {
		return;
	}

	// Stop completion thread
	if (this->impl->completionThread.joinable()) {
		this->impl->completionThreadRunning = false;
		this->impl->pendingWorkCV.notify_all();  // Wake up thread
		this->impl->completionThread.join();
	}

	// Wait for all operations to complete
	vkDeviceWaitIdle(this->impl->device);

	// Destroy VkFFT
	deleteVkFFT(&this->impl->fftApp);

	// Note: Do NOT call glslang_finalize_process() here
	// glslang is process-wide and should persist for the application lifetime
	// It will be finalized when the process exits

	// Destroy pipelines and shaders
	this->destroyComputePipelines();

	// Free command buffers and destroy fences
	this->destroyCommandBuffersAndFences();

	// Release buffers
	this->releaseDeviceBuffers();

	// Release host buffers
	for (auto& buffer : this->impl->hostInputBuffers) {
		buffer.releaseMemory();
	}
	this->impl->hostInputBuffers.clear();

	for (auto& buffer : this->impl->outputBuffers) {
		buffer.releaseMemory();
	}
	this->impl->outputBuffers.clear();

	// Clear queue
	while (!this->impl->freeBuffersQueue.empty()) {
		this->impl->freeBuffersQueue.pop();
	}

	// Destroy command pool
	if (this->impl->commandPool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(this->impl->device, this->impl->commandPool, nullptr);
		this->impl->commandPool = VK_NULL_HANDLE;
	}

	// Destroy device
	if (this->impl->device != VK_NULL_HANDLE) {
		vkDestroyDevice(this->impl->device, nullptr);
		this->impl->device = VK_NULL_HANDLE;
	}

	// Destroy instance
	if (this->impl->instance != VK_NULL_HANDLE) {
		vkDestroyInstance(this->impl->instance, nullptr);
		this->impl->instance = VK_NULL_HANDLE;
	}

	this->impl->vulkanInitialized = false;
}

// ============================================
// Main Processing
// ============================================

void VulkanBackend::setOutputCallback(std::function<void(const IOBuffer&)> callback) {
	this->impl->callback = callback;
}

void VulkanBackend::process(IOBuffer& input) {
	if (!this->impl->vulkanInitialized) {
		throw std::runtime_error("Backend not initialized");
	}

	// Select command buffer (round-robin)
	int idx = this->impl->currentCommandBuffer;
	this->impl->currentCommandBuffer = (this->impl->currentCommandBuffer + 1) % this->impl->numCommandBuffers;

	VkCommandBuffer cmd = this->impl->commandBuffers[idx];
	VkFence fence = this->impl->fences[idx];

	// Copy input data to staging buffer (CPU → GPU transfer preparation)
	// Note: This is a CPU-side memcpy to pinned memory, happens asynchronously with GPU work
	size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
	std::memcpy(this->impl->stagingInputMapped[idx], input.getDataPointer(), inputSize);

	// Return input buffer immediately - we've copied the data, don't need it anymore
	// This allows the producer to start filling the next buffer while GPU processes current frame
	{
		std::lock_guard<std::mutex> lock(this->impl->freeQueueMutex);
		this->impl->freeBuffersQueue.push(&input);
	}
	this->impl->freeQueueCV.notify_one();

	// Wait ONLY if this specific command buffer is still in use (allows overlap between command buffers!)
	// This ensures we don't overwrite staging buffers while GPU is reading them
	checkVulkanErrors(vkWaitForFences(this->impl->device, 1, &fence, VK_TRUE, UINT64_MAX));
	checkVulkanErrors(vkResetFences(this->impl->device, 1, &fence));

	// Get output buffer for this command buffer
	IOBuffer* outputBuf = &this->impl->outputBuffers[idx];
	outputBuf->setBufferId(input.getBufferId());  // Correlation ID

	#if 1  // Re-enable command recording
	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;  // Reusable command buffer

	checkVulkanErrors(vkBeginCommandBuffer(cmd, &beginInfo));

	// Copy from staging to device buffer
	VkBufferCopy copyRegion = {};
	copyRegion.size = inputSize;
	vkCmdCopyBuffer(cmd, this->impl->stagingInputBuffers[idx], this->impl->deviceInputBuffers[idx], 1, &copyRegion);

	// Add memory barrier (ensure copy completes before any compute operations)
	VkBufferMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = this->impl->deviceInputBuffers[idx];
	barrier.offset = 0;
	barrier.size = inputSize;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0,
	                     0, nullptr,
	                     1, &barrier,
	                     0, nullptr);

	// ============================================
	// Dispatch Input Conversion Shader
	// ============================================

	// Bind the input conversion pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::InputConversion));

	// Bind descriptor set for this command buffer
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->pipelineLayout,
	                        0, 1, &this->impl->descriptorSets[idx], 0, nullptr);

	// Push constants: samplesPerBuffer, inputBitDepth, bytesPerSample
	uint32_t pushConstants[3] = {
		static_cast<uint32_t>(this->impl->samplesPerBuffer),
		static_cast<uint32_t>(this->impl->config.dataParams.getBitDepth()),
		static_cast<uint32_t>(this->impl->bytesPerSample)
	};
	vkCmdPushConstants(cmd, this->impl->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(pushConstants), pushConstants);

	// Dispatch compute shader (256 threads per workgroup, as defined in shader)
	uint32_t numWorkgroups = (this->impl->samplesPerBuffer + 127) / 128;
	vkCmdDispatch(cmd, numWorkgroups, 1, 1);

	// Barrier after input conversion
	VkBufferMemoryBarrier preprocessBarrier = {};
	preprocessBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	preprocessBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	preprocessBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	preprocessBarrier.buffer = this->impl->deviceFftBuffer;
	preprocessBarrier.offset = 0;
	preprocessBarrier.size = VK_WHOLE_SIZE;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0,
	                     0, nullptr,
	                     1, &preprocessBarrier,
	                     0, nullptr);

	// ============================================
	// Preprocessing Pipeline (before FFT)
	// ============================================
	// Note: Descriptor sets enforce fixed buffer routing:
	//   K-linearization: deviceFftBuffer → deviceIntermediateBuffer
	//   Windowing: deviceIntermediateBuffer → deviceFftBuffer
	//   Dispersion: deviceFftBuffer → deviceIntermediateBuffer
	// Data ends in deviceIntermediateBuffer if dispersion is enabled, deviceFftBuffer otherwise

	const ProcessorConfiguration& config = this->impl->config;
	bool dataInFftBuffer = true;  // Track final data location

	// Step 1: DC Removal (if enabled)
	//   Input: deviceFftBuffer, Output: deviceIntermediateBuffer
	if (config.processingParams.dcRemoval.enabled) {
		// Bind DC removal pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::DcRemoval));

		// Bind DC removal descriptor set
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->dcRemovalPipelineLayout,
		                        0, 1, &this->impl->dcRemovalDescriptorSets[idx], 0, nullptr);

		// Push constants: windowSize, signalLength, ascansPerBscan, samplesPerBuffer
		uint32_t dcRemovalPushConstants[4] = {
			static_cast<uint32_t>(config.processingParams.dcRemoval.windowSize),
			static_cast<uint32_t>(this->impl->signalLength),
			static_cast<uint32_t>(this->impl->ascansPerBscan),
			static_cast<uint32_t>(this->impl->samplesPerBuffer)
		};
		vkCmdPushConstants(cmd, this->impl->dcRemovalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(dcRemovalPushConstants), dcRemovalPushConstants);

		// Dispatch DC removal shader
		vkCmdDispatch(cmd, numWorkgroups, 1, 1);

		// Barrier after DC removal
		preprocessBarrier.buffer = this->impl->deviceIntermediateBuffer;
		preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0,
		                     0, nullptr,
		                     1, &preprocessBarrier,
		                     0, nullptr);

		// Data is now in deviceIntermediateBuffer
		dataInFftBuffer = false;
	}

	// Step 2-4: K-Linearization, Windowing, and Dispersion Compensation (use merged pipelines when possible)
	// Determine which operations are enabled
	bool resampling = config.processingParams.resampling.enabled && (this->impl->resampleCurveBuffer != VK_NULL_HANDLE);
	bool windowing = config.processingParams.windowing.enabled && (this->impl->windowCurveBuffer != VK_NULL_HANDLE);
	bool dispersion = config.processingParams.dispersion.enabled && (this->impl->dispersionCurveBuffer != VK_NULL_HANDLE);
	InterpolationMethod interpMethod = config.processingParams.resampling.method;

	bool usedMergedPipeline = false;

	// Use merged pipelines when possible to eliminate intermediate memory transfers
	if (resampling && windowing && dispersion) {
		// 3-operation merge: klinearization + windowing + dispersion
		// Input: deviceFftBuffer, Output: deviceIntermediateBuffer

		// If DC removal ran, data is in deviceIntermediateBuffer - copy it back to deviceFftBuffer
		// since the merged pipeline descriptor sets are hardcoded to read from deviceFftBuffer
		if (!dataInFftBuffer) {
			VkBufferCopy copyRegion = {};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = 0;
			copyRegion.size = this->impl->samplesPerBuffer * sizeof(float) * 2;  // Complex float

			vkCmdCopyBuffer(cmd, this->impl->deviceIntermediateBuffer, this->impl->deviceFftBuffer, 1, &copyRegion);

			// Barrier after copy
			VkBufferMemoryBarrier copyBarrier = {};
			copyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
			copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			copyBarrier.buffer = this->impl->deviceFftBuffer;
			copyBarrier.offset = 0;
			copyBarrier.size = VK_WHOLE_SIZE;

			vkCmdPipelineBarrier(cmd,
			                     VK_PIPELINE_STAGE_TRANSFER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     0,
			                     0, nullptr,
			                     1, &copyBarrier,
			                     0, nullptr);

			dataInFftBuffer = true;  // Data is now back in FFT buffer
		}

		// Select the appropriate merged pipeline based on interpolation method
		VkPipeline mergedPipeline;
		if (interpMethod == InterpolationMethod::CUBIC) {
			mergedPipeline = this->impl->klinearCubicWindowingDispersionPipeline;
		} else if (interpMethod == InterpolationMethod::LINEAR) {
			mergedPipeline = this->impl->klinearLinearWindowingDispersionPipeline;
		} else {  // LANCZOS
			mergedPipeline = this->impl->klinearLanczosWindowingDispersionPipeline;
		}

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mergedPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->klinearCubicWindowingDispersionPipelineLayout,
		                        0, 1, &this->impl->klinearCubicWindowingDispersionDescriptorSets[idx], 0, nullptr);

		uint32_t pushConstants[2] = {
			static_cast<uint32_t>(this->impl->signalLength),
			static_cast<uint32_t>(this->impl->samplesPerBuffer)
		};
		vkCmdPushConstants(cmd, this->impl->klinearCubicWindowingDispersionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(pushConstants), pushConstants);
		vkCmdDispatch(cmd, numWorkgroups, 1, 1);

		// Barrier after merged operation
		preprocessBarrier.buffer = this->impl->deviceIntermediateBuffer;
		preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0, 0, nullptr, 1, &preprocessBarrier, 0, nullptr);

		dataInFftBuffer = false;
		usedMergedPipeline = true;
	}

	// Fall back to separate operations for combinations without merged pipelines

	// Fall back to separate operations for combinations without merged pipelines
	if (!usedMergedPipeline) {
		// Step 2: K-Linearization (if enabled)
		//   Input: deviceFftBuffer, Output: deviceIntermediateBuffer
		if (config.processingParams.resampling.enabled) {
			// Bind k-linearization pipeline
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::KLinearization));

			// Bind k-linearization descriptor set
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->klinearizationPipelineLayout,
			                        0, 1, &this->impl->klinearizationDescriptorSets[idx], 0, nullptr);

			// Push constants: signalLength, samplesPerBuffer
			uint32_t klinearPushConstants[2] = {
				static_cast<uint32_t>(this->impl->signalLength),
				static_cast<uint32_t>(this->impl->samplesPerBuffer)
			};
			vkCmdPushConstants(cmd, this->impl->klinearizationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(klinearPushConstants), klinearPushConstants);

			// Dispatch k-linearization shader
			vkCmdDispatch(cmd, numWorkgroups, 1, 1);

			// Barrier after k-linearization
			preprocessBarrier.buffer = this->impl->deviceIntermediateBuffer;
			preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmd,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     0,
			                     0, nullptr,
			                     1, &preprocessBarrier,
			                     0, nullptr);

			// Data is now in deviceIntermediateBuffer
			dataInFftBuffer = false;
		}

		// Step 3: Windowing (if enabled)
		//   Input: deviceIntermediateBuffer, Output: deviceFftBuffer
		if (config.processingParams.windowing.enabled) {
			// If data is still in FFT buffer, copy it to intermediate buffer first
			if (dataInFftBuffer) {
				VkBufferCopy copyRegion = {};
				copyRegion.srcOffset = 0;
				copyRegion.dstOffset = 0;
				copyRegion.size = this->impl->samplesPerBuffer * sizeof(float) * 2;  // Complex float

				vkCmdCopyBuffer(cmd, this->impl->deviceFftBuffer, this->impl->deviceIntermediateBuffer, 1, &copyRegion);

				// Barrier after copy
				VkBufferMemoryBarrier copyBarrier = {};
				copyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copyBarrier.buffer = this->impl->deviceIntermediateBuffer;
				copyBarrier.offset = 0;
				copyBarrier.size = VK_WHOLE_SIZE;

				vkCmdPipelineBarrier(cmd,
				                     VK_PIPELINE_STAGE_TRANSFER_BIT,
				                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0,
				                     0, nullptr,
				                     1, &copyBarrier,
				                     0, nullptr);

				dataInFftBuffer = false;
			}

			// Bind windowing pipeline
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::Windowing));

			// Bind windowing descriptor set
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->windowingPipelineLayout,
			                        0, 1, &this->impl->windowingDescriptorSets[idx], 0, nullptr);

			// Push constants: signalLength, samplesPerBuffer
			uint32_t windowingPushConstants[2] = {
				static_cast<uint32_t>(this->impl->signalLength),
				static_cast<uint32_t>(this->impl->samplesPerBuffer)
			};
			vkCmdPushConstants(cmd, this->impl->windowingPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(windowingPushConstants), windowingPushConstants);

			// Dispatch windowing shader
			vkCmdDispatch(cmd, numWorkgroups, 1, 1);

			// Barrier after windowing
			preprocessBarrier.buffer = this->impl->deviceFftBuffer;
			preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmd,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     0,
			                     0, nullptr,
			                     1, &preprocessBarrier,
			                     0, nullptr);

			// Data is now in deviceFftBuffer
			dataInFftBuffer = true;
		}

		// Step 4: Dispersion Compensation (if enabled)
		//   Input: deviceFftBuffer, Output: deviceIntermediateBuffer
		if (config.processingParams.dispersion.enabled) {
			// If data is in intermediate buffer, copy it to FFT buffer first
			if (!dataInFftBuffer) {
				VkBufferCopy copyRegion = {};
				copyRegion.srcOffset = 0;
				copyRegion.dstOffset = 0;
				copyRegion.size = this->impl->samplesPerBuffer * sizeof(float) * 2;  // Complex float

				vkCmdCopyBuffer(cmd, this->impl->deviceIntermediateBuffer, this->impl->deviceFftBuffer, 1, &copyRegion);

				// Barrier after copy
				VkBufferMemoryBarrier copyBarrier = {};
				copyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				copyBarrier.buffer = this->impl->deviceFftBuffer;
				copyBarrier.offset = 0;
				copyBarrier.size = VK_WHOLE_SIZE;

				vkCmdPipelineBarrier(cmd,
				                     VK_PIPELINE_STAGE_TRANSFER_BIT,
				                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				                     0,
				                     0, nullptr,
				                     1, &copyBarrier,
				                     0, nullptr);

				dataInFftBuffer = true;
			}

			// Bind dispersion pipeline (index 6)
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::Dispersion));

			// Bind dispersion descriptor set
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->dispersionPipelineLayout,
			                        0, 1, &this->impl->dispersionDescriptorSets[idx], 0, nullptr);

			// Push constants: signalLength, samplesPerBuffer
			uint32_t dispersionPushConstants[2] = {
				static_cast<uint32_t>(this->impl->signalLength),
				static_cast<uint32_t>(this->impl->samplesPerBuffer)
			};
			vkCmdPushConstants(cmd, this->impl->dispersionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			                   0, sizeof(dispersionPushConstants), dispersionPushConstants);

			// Dispatch dispersion shader
			vkCmdDispatch(cmd, numWorkgroups, 1, 1);

			// Barrier after dispersion
			preprocessBarrier.buffer = this->impl->deviceIntermediateBuffer;
			preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(cmd,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			                     0,
			                     0, nullptr,
			                     1, &preprocessBarrier,
			                     0, nullptr);

			// Data is now in deviceIntermediateBuffer
			dataInFftBuffer = false;
		}
	}

	// Ensure data is in deviceFftBuffer for FFT
	if (!dataInFftBuffer) {
		// Copy from intermediate to FFT buffer
		VkBufferCopy copyRegion = {};
		copyRegion.size = this->impl->samplesPerBuffer * sizeof(float) * 2;  // Complex data
		vkCmdCopyBuffer(cmd, this->impl->deviceIntermediateBuffer, this->impl->deviceFftBuffer, 1, &copyRegion);

		// Barrier after copy
		preprocessBarrier.buffer = this->impl->deviceFftBuffer;
		preprocessBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0,
		                     0, nullptr,
		                     1, &preprocessBarrier,
		                     0, nullptr);
	}

	// Final barrier before FFT
	VkBufferMemoryBarrier fftInputBarrier = {};
	fftInputBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	fftInputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	fftInputBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	fftInputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	fftInputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	fftInputBarrier.buffer = this->impl->deviceFftBuffer;
	fftInputBarrier.offset = 0;
	fftInputBarrier.size = VK_WHOLE_SIZE;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0,
	                     0, nullptr,
	                     1, &fftInputBarrier,
	                     0, nullptr);

	// ============================================
	// Execute VkFFT (Inverse FFT)
	// ============================================

	VkFFTLaunchParams fftLaunchParams = {};
	fftLaunchParams.commandBuffer = &cmd;
	fftLaunchParams.buffer = &this->impl->deviceFftBuffer;

	checkVkFFTErrors(VkFFTAppend(&this->impl->fftApp, 1, &fftLaunchParams));  // +1 = inverse FFT

	// Barrier after FFT (wait for FFT to complete before truncate shader)
	VkBufferMemoryBarrier fftOutputBarrier = {};
	fftOutputBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	fftOutputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	fftOutputBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	fftOutputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	fftOutputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	fftOutputBarrier.buffer = this->impl->deviceFftBuffer;
	fftOutputBarrier.offset = 0;
	fftOutputBarrier.size = VK_WHOLE_SIZE;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0,
	                     0, nullptr,
	                     1, &fftOutputBarrier,
	                     0, nullptr);

	// ============================================
	// Dispatch Truncate Shader
	// ============================================

	// Bind truncate pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::Truncate));

	// Bind truncate descriptor set
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->truncatePipelineLayout,
	                        0, 1, &this->impl->truncateDescriptorSets[idx], 0, nullptr);

	// Push constants for truncate: fullSignalLength, outputSignalLength, samplesPerBuffer
	int outputSignalLength = this->impl->signalLength / 2;
	uint32_t truncatePushConstants[3] = {
		static_cast<uint32_t>(this->impl->signalLength),
		static_cast<uint32_t>(outputSignalLength),
		static_cast<uint32_t>(this->impl->samplesPerBuffer)
	};
	vkCmdPushConstants(cmd, this->impl->truncatePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(truncatePushConstants), truncatePushConstants);

	// Dispatch truncate shader
	uint32_t truncateWorkgroups = (this->impl->samplesPerBuffer + 127) / 128;
	vkCmdDispatch(cmd, truncateWorkgroups, 1, 1);


	// Barrier after truncate (wait for writes to deviceProcessedBuffer to complete before postprocess)
	VkBufferMemoryBarrier truncateToPostprocessBarrier = {};
	truncateToPostprocessBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	truncateToPostprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	truncateToPostprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	truncateToPostprocessBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	truncateToPostprocessBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	truncateToPostprocessBarrier.buffer = this->impl->deviceProcessedBuffer;
	truncateToPostprocessBarrier.offset = 0;
	truncateToPostprocessBarrier.size = VK_WHOLE_SIZE;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0,
	                     0, nullptr,
	                     1, &truncateToPostprocessBarrier,
	                     0, nullptr);

	// ============================================
	// Dispatch Post-Process Shader (Log Scaling, Grayscale Normalization)
	// ============================================

	// Bind postprocess pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::Postprocess));

	// Bind postprocess descriptor set
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->postprocessPipelineLayout,
	                        0, 1, &this->impl->postprocessDescriptorSets[idx], 0, nullptr);

	// Push constants for postprocess: samplesPerBuffer, logScaling, outputSignalLength, grayscaleMax, grayscaleMin, addend, multiplicator
	size_t truncatedSamples = outputSignalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer;
	struct PostprocessPushConstants {
		uint32_t samplesPerBuffer;
		uint32_t logScaling;  // 1 for log, 0 for linear
		uint32_t outputSignalLength;  // FFT size for normalization (1024)
		float grayscaleMax;
		float grayscaleMin;
		float addend;
		float multiplicator;
	} postprocessPush;

	postprocessPush.samplesPerBuffer = static_cast<uint32_t>(truncatedSamples);
	postprocessPush.logScaling = this->impl->config.processingParams.intensity.logScale ? 1 : 0;
	postprocessPush.outputSignalLength = static_cast<uint32_t>(outputSignalLength);
	postprocessPush.grayscaleMax = this->impl->config.processingParams.intensity.rangeMax;
	postprocessPush.grayscaleMin = this->impl->config.processingParams.intensity.rangeMin;
	postprocessPush.addend = this->impl->config.processingParams.intensity.postOffset;
	postprocessPush.multiplicator = this->impl->config.processingParams.intensity.preScale;

	vkCmdPushConstants(cmd, this->impl->postprocessPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(postprocessPush), &postprocessPush);

	// Dispatch postprocess shader
	uint32_t postprocessWorkgroups = (truncatedSamples + 127) / 128;
	vkCmdDispatch(cmd, postprocessWorkgroups, 1, 1);

	// Barrier after postprocess (wait for writes to complete before copy)
	VkBufferMemoryBarrier postprocessOutputBarrier = {};
	postprocessOutputBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	postprocessOutputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	postprocessOutputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	postprocessOutputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	postprocessOutputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	postprocessOutputBarrier.buffer = this->impl->deviceProcessedBuffer;
	postprocessOutputBarrier.offset = 0;
	postprocessOutputBarrier.size = VK_WHOLE_SIZE;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0,
	                     0, nullptr,
	                     1, &postprocessOutputBarrier,
	                     0, nullptr);


	// ============================================
	// Copy Truncated Output to Staging
	// ============================================

	// Copy processed buffer (truncated) to staging output (GPU → CPU transfer)
	size_t truncatedOutputSize = outputSignalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer * sizeof(float);
	VkBufferCopy finalCopy = {};
	finalCopy.size = truncatedOutputSize;
	vkCmdCopyBuffer(cmd, this->impl->deviceProcessedBuffer, this->impl->stagingOutputBuffers[idx], 1, &finalCopy);

	checkVulkanErrors(vkEndCommandBuffer(cmd));
	#endif  // End of OLD CODE (re-recording every frame)

	// Submit pre-recorded command buffer
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;

	checkVulkanErrors(vkQueueSubmit(this->impl->computeQueue, 1, &submitInfo, fence));

	// Queue work for async completion (completion thread will wait for fence and invoke callback)
	// This enables true async overlap: process() returns immediately, allowing next frame to start
	size_t outputSize = outputSignalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer * sizeof(float);
	{
		std::lock_guard<std::mutex> lock(this->impl->pendingWorkMutex);
		this->impl->pendingWorkQueue.push({
			fence,
			idx,
			outputBuf,
			outputSize,
			outputSignalLength
		});
	}
	this->impl->pendingWorkCV.notify_one();  // Wake completion thread
}

// ============================================
// Configuration Updates
// ============================================

void VulkanBackend::updateConfig(const ProcessorConfiguration& config) {
	this->impl->config = config;

	// Re-record command buffers with new configuration
	// First, wait for all in-flight work to complete
	vkDeviceWaitIdle(this->impl->device);

	// Re-record all command buffers with the new configuration
	this->recordCommandBuffers();
}

void VulkanBackend::updateResamplingCurve(const float* curve, size_t length) {
	if (this->impl->resampleCurveBuffer == VK_NULL_HANDLE || length != static_cast<size_t>(this->impl->signalLength)) {
		return;
	}

	// Create staging buffer for upload
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	VkDeviceSize bufferSize = length * sizeof(float);

	createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	             stagingBuffer, stagingMemory);

	// Copy curve data to staging buffer
	void* mappedMemory;
	vkMapMemory(this->impl->device, stagingMemory, 0, bufferSize, 0, &mappedMemory);
	memcpy(mappedMemory, curve, bufferSize);
	vkUnmapMemory(this->impl->device, stagingMemory);

	// Copy from staging to device buffer using a temporary command buffer
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = this->impl->commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmdBuffer;
	vkAllocateCommandBuffers(this->impl->device, &allocInfo, &cmdBuffer);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(cmdBuffer, &beginInfo);

	VkBufferCopy copyRegion = {};
	copyRegion.size = bufferSize;
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->resampleCurveBuffer, 1, &copyRegion);

	vkEndCommandBuffer(cmdBuffer);

	// Submit and wait
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuffer;

	vkQueueSubmit(this->impl->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(this->impl->computeQueue);

	// Cleanup
	vkFreeCommandBuffers(this->impl->device, this->impl->commandPool, 1, &cmdBuffer);
	vkDestroyBuffer(this->impl->device, stagingBuffer, nullptr);
	vkFreeMemory(this->impl->device, stagingMemory, nullptr);
}

void VulkanBackend::updateDispersionCurve(const float* curve, size_t length) {
	if (this->impl->dispersionCurveBuffer == VK_NULL_HANDLE || length != static_cast<size_t>(this->impl->signalLength * 2)) {
		return;
	}

	// Create staging buffer for upload (complex data: 2 floats per element)
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	VkDeviceSize bufferSize = length * sizeof(float);

	createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	             stagingBuffer, stagingMemory);

	// Copy curve data to staging buffer
	void* mappedMemory;
	vkMapMemory(this->impl->device, stagingMemory, 0, bufferSize, 0, &mappedMemory);
	memcpy(mappedMemory, curve, bufferSize);
	vkUnmapMemory(this->impl->device, stagingMemory);

	// Copy from staging to device buffer using a temporary command buffer
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = this->impl->commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmdBuffer;
	vkAllocateCommandBuffers(this->impl->device, &allocInfo, &cmdBuffer);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(cmdBuffer, &beginInfo);

	VkBufferCopy copyRegion = {};
	copyRegion.size = bufferSize;
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->dispersionCurveBuffer, 1, &copyRegion);

	vkEndCommandBuffer(cmdBuffer);

	// Submit and wait
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuffer;

	vkQueueSubmit(this->impl->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(this->impl->computeQueue);

	// Cleanup
	vkFreeCommandBuffers(this->impl->device, this->impl->commandPool, 1, &cmdBuffer);
	vkDestroyBuffer(this->impl->device, stagingBuffer, nullptr);
	vkFreeMemory(this->impl->device, stagingMemory, nullptr);
}

void VulkanBackend::updateWindowCurve(const float* curve, size_t length) {
	if (this->impl->windowCurveBuffer == VK_NULL_HANDLE || length != static_cast<size_t>(this->impl->signalLength)) {
		return;
	}

	// Create staging buffer for upload
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	VkDeviceSize bufferSize = length * sizeof(float);

	createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	             stagingBuffer, stagingMemory);

	// Copy curve data to staging buffer
	void* mappedMemory;
	vkMapMemory(this->impl->device, stagingMemory, 0, bufferSize, 0, &mappedMemory);
	memcpy(mappedMemory, curve, bufferSize);
	vkUnmapMemory(this->impl->device, stagingMemory);

	// Copy from staging to device buffer using a temporary command buffer
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = this->impl->commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer cmdBuffer;
	vkAllocateCommandBuffers(this->impl->device, &allocInfo, &cmdBuffer);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(cmdBuffer, &beginInfo);

	VkBufferCopy copyRegion = {};
	copyRegion.size = bufferSize;
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->windowCurveBuffer, 1, &copyRegion);

	vkEndCommandBuffer(cmdBuffer);

	// Submit and wait
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuffer;

	vkQueueSubmit(this->impl->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(this->impl->computeQueue);

	// Cleanup
	vkFreeCommandBuffers(this->impl->device, this->impl->commandPool, 1, &cmdBuffer);
	vkDestroyBuffer(this->impl->device, stagingBuffer, nullptr);
	vkFreeMemory(this->impl->device, stagingMemory, nullptr);
}

// ============================================
// Buffer Management
// ============================================

IOBuffer& VulkanBackend::getInputBuffer(int index) {
	if (index < 0 || index >= this->impl->numInputBuffers) {
		throw std::out_of_range("Input buffer index out of range");
	}
	return this->impl->hostInputBuffers[index];
}

IOBuffer& VulkanBackend::getNextAvailableInputBuffer() {
	std::unique_lock<std::mutex> lock(this->impl->freeQueueMutex);

	// Block until a buffer is available
	while (this->impl->freeBuffersQueue.empty()) {
		this->impl->freeQueueCV.wait(lock);
	}

	IOBuffer* buffer = this->impl->freeBuffersQueue.front();
	this->impl->freeBuffersQueue.pop();
	return *buffer;
}

int VulkanBackend::getNumInputBuffers() const {
	return this->impl->numInputBuffers;
}

// ============================================
// Profile Management
// ============================================

void VulkanBackend::requestPostProcessBackgroundRecording() {
	this->impl->postProcessBackgroundRecordingRequested = true;
}

void VulkanBackend::setPostProcessBackgroundProfile(const float* background, size_t length) {
	this->impl->recordedPostProcessBackground.assign(background, background + length);
	// TODO: Upload to device buffer
}

const std::vector<float>& VulkanBackend::getPostProcessBackgroundProfile() const {
	return this->impl->recordedPostProcessBackground;
}

void VulkanBackend::requestFixedPatternNoiseDetermination() {
	this->impl->fixedPatternNoiseDetermined = false;
}

void VulkanBackend::setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) {
	this->impl->recordedFixedPatternNoise.assign(profileInterleaved, profileInterleaved + complexPairs * 2);
	this->impl->fixedPatternNoiseDetermined = true;
	// TODO: Upload to device buffer
}

const std::vector<float>& VulkanBackend::getFixedPatternNoiseProfile() const {
	return this->impl->recordedFixedPatternNoise;
}

// ============================================
// Individual Operations (Stubs for now)
// ============================================

std::vector<float> VulkanBackend::convertInput(
	const void* input,
	IOBuffer::DataType inputType,
	int bitDepth,
	int samples,
	bool applyBitshift
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::rollingAverageBackgroundRemoval(
	const float* input,
	int windowSize,
	int lineWidth,
	int numLines
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::kLinearization(
	const float* input,
	const float* resampleCurve,
	InterpolationMethod method,
	int lineWidth,
	int samples
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::windowing(
	const float* input,
	const float* windowCurve,
	int lineWidth,
	int samples
) {
	// Calculate buffer sizes
	size_t numSamples = static_cast<size_t>(lineWidth) * samples;
	size_t bufferSize = numSamples * sizeof(float);
	size_t windowSize = samples * sizeof(float);

	// Create temporary buffers for windowing operation
	VkBuffer inputBuffer = VK_NULL_HANDLE;
	VkDeviceMemory inputMemory = VK_NULL_HANDLE;
	VkBuffer windowBuffer = VK_NULL_HANDLE;
	VkDeviceMemory windowMemory = VK_NULL_HANDLE;
	VkBuffer outputBuffer = VK_NULL_HANDLE;
	VkDeviceMemory outputMemory = VK_NULL_HANDLE;

	try {
		// Create and populate input buffer
		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             inputBuffer, inputMemory);

		// Create and populate window buffer
		createBuffer(this->impl->device, this->impl->physicalDevice, windowSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             windowBuffer, windowMemory);

		// Create output buffer
		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             outputBuffer, outputMemory);

		// Create staging buffers for upload
		VkBuffer stagingInput = VK_NULL_HANDLE;
		VkDeviceMemory stagingInputMem = VK_NULL_HANDLE;
		VkBuffer stagingWindow = VK_NULL_HANDLE;
		VkDeviceMemory stagingWindowMem = VK_NULL_HANDLE;

		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             stagingInput, stagingInputMem);

		createBuffer(this->impl->device, this->impl->physicalDevice, windowSize,
		             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             stagingWindow, stagingWindowMem);

		// Upload input data
		void* mappedInput;
		vkMapMemory(this->impl->device, stagingInputMem, 0, bufferSize, 0, &mappedInput);
		memcpy(mappedInput, input, bufferSize);
		vkUnmapMemory(this->impl->device, stagingInputMem);

		// Upload window curve
		void* mappedWindow;
		vkMapMemory(this->impl->device, stagingWindowMem, 0, windowSize, 0, &mappedWindow);
		memcpy(mappedWindow, windowCurve, windowSize);
		vkUnmapMemory(this->impl->device, stagingWindowMem);

		// Create one-time command buffer for copy and compute
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = this->impl->commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer cmdBuffer;
		vkAllocateCommandBuffers(this->impl->device, &allocInfo, &cmdBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(cmdBuffer, &beginInfo);

		// Copy staging to device buffers
		VkBufferCopy inputCopyRegion = {};
		inputCopyRegion.size = bufferSize;
		vkCmdCopyBuffer(cmdBuffer, stagingInput, inputBuffer, 1, &inputCopyRegion);

		VkBufferCopy windowCopyRegion = {};
		windowCopyRegion.size = windowSize;
		vkCmdCopyBuffer(cmdBuffer, stagingWindow, windowBuffer, 1, &windowCopyRegion);

		// Pipeline barrier to ensure transfers complete before compute
		VkMemoryBarrier memoryBarrier = {};
		memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

		// Create descriptor set for this operation
		VkDescriptorSetAllocateInfo descAllocInfo = {};
		descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descAllocInfo.descriptorPool = this->impl->descriptorPool;
		descAllocInfo.descriptorSetCount = 1;
		descAllocInfo.pSetLayouts = &this->impl->windowingDescriptorSetLayout;

		VkDescriptorSet descriptorSet;
		checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &descAllocInfo, &descriptorSet));

		// Update descriptor set
		std::vector<VkWriteDescriptorSet> descriptorWrites(3);

		VkDescriptorBufferInfo inputBufferInfo = {};
		inputBufferInfo.buffer = inputBuffer;
		inputBufferInfo.offset = 0;
		inputBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &inputBufferInfo;

		VkDescriptorBufferInfo windowBufferInfo = {};
		windowBufferInfo.buffer = windowBuffer;
		windowBufferInfo.offset = 0;
		windowBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &windowBufferInfo;

		VkDescriptorBufferInfo outputBufferInfo = {};
		outputBufferInfo.buffer = outputBuffer;
		outputBufferInfo.offset = 0;
		outputBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = descriptorSet;
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[2].descriptorCount = 1;
		descriptorWrites[2].pBufferInfo = &outputBufferInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

		// Bind pipeline and descriptor set
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::Windowing));  // Windowing is pipeline index 2
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->windowingPipelineLayout,
		                        0, 1, &descriptorSet, 0, nullptr);

		// Set push constants
		uint32_t pushConstants[2] = {
			static_cast<uint32_t>(samples),
			static_cast<uint32_t>(numSamples)
		};
		vkCmdPushConstants(cmdBuffer, this->impl->windowingPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(pushConstants), pushConstants);

		// Dispatch compute shader
		uint32_t workgroups = (static_cast<uint32_t>(numSamples) + 127) / 128;
		vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

		// Pipeline barrier to ensure compute completes before transfer
		VkMemoryBarrier computeBarrier = {};
		computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		computeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

		// Create staging buffer for download
		VkBuffer stagingOutput = VK_NULL_HANDLE;
		VkDeviceMemory stagingOutputMem = VK_NULL_HANDLE;

		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             stagingOutput, stagingOutputMem);

		// Copy result to staging buffer
		VkBufferCopy outputCopyRegion = {};
		outputCopyRegion.size = bufferSize;
		vkCmdCopyBuffer(cmdBuffer, outputBuffer, stagingOutput, 1, &outputCopyRegion);

		vkEndCommandBuffer(cmdBuffer);

		// Submit and wait
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuffer;

		vkQueueSubmit(this->impl->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(this->impl->computeQueue);

		// Download result
		std::vector<float> result(numSamples);
		void* mappedOutput;
		vkMapMemory(this->impl->device, stagingOutputMem, 0, bufferSize, 0, &mappedOutput);
		memcpy(result.data(), mappedOutput, bufferSize);
		vkUnmapMemory(this->impl->device, stagingOutputMem);

		// Cleanup
		vkFreeCommandBuffers(this->impl->device, this->impl->commandPool, 1, &cmdBuffer);
		vkDestroyBuffer(this->impl->device, stagingInput, nullptr);
		vkFreeMemory(this->impl->device, stagingInputMem, nullptr);
		vkDestroyBuffer(this->impl->device, stagingWindow, nullptr);
		vkFreeMemory(this->impl->device, stagingWindowMem, nullptr);
		vkDestroyBuffer(this->impl->device, stagingOutput, nullptr);
		vkFreeMemory(this->impl->device, stagingOutputMem, nullptr);
		vkDestroyBuffer(this->impl->device, inputBuffer, nullptr);
		vkFreeMemory(this->impl->device, inputMemory, nullptr);
		vkDestroyBuffer(this->impl->device, windowBuffer, nullptr);
		vkFreeMemory(this->impl->device, windowMemory, nullptr);
		vkDestroyBuffer(this->impl->device, outputBuffer, nullptr);
		vkFreeMemory(this->impl->device, outputMemory, nullptr);

		return result;

	} catch (...) {
		// Cleanup on error
		if (inputBuffer != VK_NULL_HANDLE) vkDestroyBuffer(this->impl->device, inputBuffer, nullptr);
		if (inputMemory != VK_NULL_HANDLE) vkFreeMemory(this->impl->device, inputMemory, nullptr);
		if (windowBuffer != VK_NULL_HANDLE) vkDestroyBuffer(this->impl->device, windowBuffer, nullptr);
		if (windowMemory != VK_NULL_HANDLE) vkFreeMemory(this->impl->device, windowMemory, nullptr);
		if (outputBuffer != VK_NULL_HANDLE) vkDestroyBuffer(this->impl->device, outputBuffer, nullptr);
		if (outputMemory != VK_NULL_HANDLE) vkFreeMemory(this->impl->device, outputMemory, nullptr);
		throw;
	}
}

std::vector<float> VulkanBackend::dispersionCompensation(
	const float* input,
	const float* phaseComplex,
	int lineWidth,
	int samples
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::kLinearizationAndWindowing(
	const float* input,
	const float* resampleCurve,
	const float* windowCurve,
	InterpolationMethod method,
	int lineWidth,
	int samples
) {
	// TODO: Implement (fused kernel - not for initial implementation)
	return std::vector<float>();
}

std::vector<float> VulkanBackend::kLinearizationAndWindowingAndDispersion(
	const float* input,
	const float* resampleCurve,
	const float* windowCurve,
	const float* phaseComplex,
	InterpolationMethod method,
	int lineWidth,
	int samples
) {
	// TODO: Implement (fused kernel - not for initial implementation)
	return std::vector<float>();
}

std::vector<float> VulkanBackend::dispersionCompensationAndWindowing(
	const float* input,
	const float* phaseComplex,
	const float* windowCurve,
	int lineWidth,
	int samples
) {
	// TODO: Implement (fused kernel - not for initial implementation)
	return std::vector<float>();
}

std::vector<float> VulkanBackend::fft(const float* input, int lineWidth, int samples) {
	// TODO: Implement with VkFFT
	return std::vector<float>();
}

std::vector<float> VulkanBackend::ifft(const float* input, int lineWidth, int samples) {
	// TODO: Implement with VkFFT
	return std::vector<float>();
}

std::vector<float> VulkanBackend::getMinimumVarianceMean(
	const float* input,
	int width,
	int height,
	int segments
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::fixedPatternNoiseRemoval(
	const float* input,
	const float* meanALine,
	int lineWidth,
	int numLines
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::postProcessTruncate(
	const float* input,
	bool logScaling,
	float grayscaleMax,
	float grayscaleMin,
	float addend,
	float multiplicator,
	int lineWidth,
	int samples
) {
	// Calculate buffer sizes
	size_t numSamples = static_cast<size_t>(lineWidth) * samples;
	size_t bufferSize = numSamples * sizeof(float);

	// Create temporary buffers
	VkBuffer inputBuffer = VK_NULL_HANDLE;
	VkDeviceMemory inputMemory = VK_NULL_HANDLE;
	VkBuffer outputBuffer = VK_NULL_HANDLE;
	VkDeviceMemory outputMemory = VK_NULL_HANDLE;

	try {
		// Create input and output buffers
		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             inputBuffer, inputMemory);

		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             outputBuffer, outputMemory);

		// Create staging buffers
		VkBuffer stagingInput = VK_NULL_HANDLE;
		VkDeviceMemory stagingInputMem = VK_NULL_HANDLE;

		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             stagingInput, stagingInputMem);

		// Upload input data
		void* mappedInput;
		vkMapMemory(this->impl->device, stagingInputMem, 0, bufferSize, 0, &mappedInput);
		memcpy(mappedInput, input, bufferSize);
		vkUnmapMemory(this->impl->device, stagingInputMem);

		// Create command buffer
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = this->impl->commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer cmdBuffer;
		vkAllocateCommandBuffers(this->impl->device, &allocInfo, &cmdBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(cmdBuffer, &beginInfo);

		// Copy staging to device buffer
		VkBufferCopy inputCopyRegion = {};
		inputCopyRegion.size = bufferSize;
		vkCmdCopyBuffer(cmdBuffer, stagingInput, inputBuffer, 1, &inputCopyRegion);

		// Pipeline barrier
		VkMemoryBarrier memoryBarrier = {};
		memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

		// Create descriptor set
		VkDescriptorSetAllocateInfo descAllocInfo = {};
		descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		descAllocInfo.descriptorPool = this->impl->descriptorPool;
		descAllocInfo.descriptorSetCount = 1;
		descAllocInfo.pSetLayouts = &this->impl->postprocessDescriptorSetLayout;

		VkDescriptorSet descriptorSet;
		checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &descAllocInfo, &descriptorSet));

		// Update descriptor set
		std::vector<VkWriteDescriptorSet> descriptorWrites(2);

		VkDescriptorBufferInfo inputBufferInfo = {};
		inputBufferInfo.buffer = inputBuffer;
		inputBufferInfo.offset = 0;
		inputBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &inputBufferInfo;

		VkDescriptorBufferInfo outputBufferInfo = {};
		outputBufferInfo.buffer = outputBuffer;
		outputBufferInfo.offset = 0;
		outputBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &outputBufferInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

		// Bind pipeline and descriptor set
		vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::Postprocess));  // Postprocess is pipeline index 3
		vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->postprocessPipelineLayout,
		                        0, 1, &descriptorSet, 0, nullptr);

		// Set push constants
		struct PushConstants {
			uint32_t samplesPerBuffer;
			uint32_t logScaling;
			float grayscaleMax;
			float grayscaleMin;
			float addend;
			float multiplicator;
		} pushConstants;

		pushConstants.samplesPerBuffer = static_cast<uint32_t>(numSamples);
		pushConstants.logScaling = logScaling ? 1 : 0;
		pushConstants.grayscaleMax = grayscaleMax;
		pushConstants.grayscaleMin = grayscaleMin;
		pushConstants.addend = addend;
		pushConstants.multiplicator = multiplicator;

		vkCmdPushConstants(cmdBuffer, this->impl->postprocessPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(pushConstants), &pushConstants);

		// Dispatch compute shader
		uint32_t workgroups = (static_cast<uint32_t>(numSamples) + 127) / 128;
		vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

		// Pipeline barrier
		VkMemoryBarrier computeBarrier = {};
		computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		computeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(cmdBuffer,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

		// Create staging output buffer
		VkBuffer stagingOutput = VK_NULL_HANDLE;
		VkDeviceMemory stagingOutputMem = VK_NULL_HANDLE;

		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             stagingOutput, stagingOutputMem);

		// Copy result to staging buffer
		VkBufferCopy outputCopyRegion = {};
		outputCopyRegion.size = bufferSize;
		vkCmdCopyBuffer(cmdBuffer, outputBuffer, stagingOutput, 1, &outputCopyRegion);

		vkEndCommandBuffer(cmdBuffer);

		// Submit and wait
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuffer;

		vkQueueSubmit(this->impl->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(this->impl->computeQueue);

		// Download result
		std::vector<float> result(numSamples);
		void* mappedOutput;
		vkMapMemory(this->impl->device, stagingOutputMem, 0, bufferSize, 0, &mappedOutput);
		memcpy(result.data(), mappedOutput, bufferSize);
		vkUnmapMemory(this->impl->device, stagingOutputMem);

		// Cleanup
		vkFreeCommandBuffers(this->impl->device, this->impl->commandPool, 1, &cmdBuffer);
		vkDestroyBuffer(this->impl->device, stagingInput, nullptr);
		vkFreeMemory(this->impl->device, stagingInputMem, nullptr);
		vkDestroyBuffer(this->impl->device, stagingOutput, nullptr);
		vkFreeMemory(this->impl->device, stagingOutputMem, nullptr);
		vkDestroyBuffer(this->impl->device, inputBuffer, nullptr);
		vkFreeMemory(this->impl->device, inputMemory, nullptr);
		vkDestroyBuffer(this->impl->device, outputBuffer, nullptr);
		vkFreeMemory(this->impl->device, outputMemory, nullptr);

		return result;

	} catch (...) {
		// Cleanup on error
		if (inputBuffer != VK_NULL_HANDLE) vkDestroyBuffer(this->impl->device, inputBuffer, nullptr);
		if (inputMemory != VK_NULL_HANDLE) vkFreeMemory(this->impl->device, inputMemory, nullptr);
		if (outputBuffer != VK_NULL_HANDLE) vkDestroyBuffer(this->impl->device, outputBuffer, nullptr);
		if (outputMemory != VK_NULL_HANDLE) vkFreeMemory(this->impl->device, outputMemory, nullptr);
		throw;
	}
}

std::vector<float> VulkanBackend::bscanFlip(
	const float* input,
	int lineWidth,
	int linesPerBscan,
	int numBscans
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::sinusoidalScanCorrection(
	const float* input,
	const float* resampleCurve,
	int lineWidth,
	int linesPerBscan,
	int numBscans
) {
	// TODO: Implement
	return std::vector<float>();
}

std::vector<float> VulkanBackend::postProcessBackgroundSubtraction(
	const float* input,
	const float* backgroundLine,
	float weight,
	float offset,
	int lineWidth,
	int samples
) {
	// TODO: Implement
	return std::vector<float>();
}

// ============================================
// Helper Methods
// ============================================

void VulkanBackend::checkVulkanError(VkResult result, const char* context) {
	if (result != VK_SUCCESS) {
		std::stringstream ss;
		ss << "Vulkan error in " << context << ": " << result;
		throw std::runtime_error(ss.str());
	}
}

// Helper function to find memory type
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("Failed to find suitable memory type");
}

// Helper function to create buffer
void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size,
                  VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                  VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create buffer");
	}

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
		throw std::runtime_error("Failed to allocate buffer memory");
	}

	vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void VulkanBackend::allocateDeviceBuffers() {
	size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
	size_t complexSize = this->impl->samplesPerBuffer * sizeof(float) * 2;  // Complex float (2 floats per sample)
	// Output size is truncated to half signal length after FFT
	int outputSignalLength = this->impl->signalLength / 2;
	size_t outputSize = outputSignalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer * sizeof(float);

	// Allocate staging buffers (host-visible, for CPU-GPU transfer)
	this->impl->stagingInputBuffers.resize(this->impl->numCommandBuffers);
	this->impl->stagingInputMemory.resize(this->impl->numCommandBuffers);
	this->impl->stagingInputMapped.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		createBuffer(this->impl->device, this->impl->physicalDevice, inputSize,
		             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		             this->impl->stagingInputBuffers[i], this->impl->stagingInputMemory[i]);

		// Map staging buffer memory
		vkMapMemory(this->impl->device, this->impl->stagingInputMemory[i], 0, inputSize, 0, &this->impl->stagingInputMapped[i]);
	}

	// Allocate staging output buffers
	this->impl->stagingOutputBuffers.resize(this->impl->numCommandBuffers);
	this->impl->stagingOutputMemory.resize(this->impl->numCommandBuffers);
	this->impl->stagingOutputMapped.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		// Try to allocate with cached memory for fast CPU reads (60x faster than uncached!)
		// HOST_CACHED + HOST_COHERENT = best of both worlds: fast reads + automatic sync
		createBuffer(this->impl->device, this->impl->physicalDevice, outputSize,
		             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
		             this->impl->stagingOutputBuffers[i], this->impl->stagingOutputMemory[i]);

		// Map staging buffer memory
		vkMapMemory(this->impl->device, this->impl->stagingOutputMemory[i], 0, outputSize, 0, &this->impl->stagingOutputMapped[i]);
	}

	// Allocate device-local buffers (for computation)
	this->impl->deviceInputBuffers.resize(this->impl->numCommandBuffers);
	this->impl->deviceInputMemory.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		createBuffer(this->impl->device, this->impl->physicalDevice, inputSize,
		             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             this->impl->deviceInputBuffers[i], this->impl->deviceInputMemory[i]);
	}

	// FFT buffer (complex float)
	createBuffer(this->impl->device, this->impl->physicalDevice, complexSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->deviceFftBuffer, this->impl->deviceFftMemory);

	// Intermediate buffer for preprocessing ping-pong (same size as FFT buffer)
	createBuffer(this->impl->device, this->impl->physicalDevice, complexSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->deviceIntermediateBuffer, this->impl->deviceIntermediateMemory);

	// Processed buffer (output)
	createBuffer(this->impl->device, this->impl->physicalDevice, outputSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->deviceProcessedBuffer, this->impl->deviceProcessedMemory);

	// Curve buffers
	size_t curveSize = this->impl->signalLength * sizeof(float);

	createBuffer(this->impl->device, this->impl->physicalDevice, curveSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->resampleCurveBuffer, this->impl->resampleCurveMemory);

	createBuffer(this->impl->device, this->impl->physicalDevice, curveSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->windowCurveBuffer, this->impl->windowCurveMemory);

	createBuffer(this->impl->device, this->impl->physicalDevice, curveSize * 2,  // Complex
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->dispersionCurveBuffer, this->impl->dispersionCurveMemory);
}

void VulkanBackend::releaseDeviceBuffers() {
	// Unmap and destroy staging buffers
	for (size_t i = 0; i < this->impl->stagingInputBuffers.size(); ++i) {
		if (this->impl->stagingInputMemory[i] != VK_NULL_HANDLE) {
			vkUnmapMemory(this->impl->device, this->impl->stagingInputMemory[i]);
		}
		if (this->impl->stagingInputBuffers[i] != VK_NULL_HANDLE) {
			vkDestroyBuffer(this->impl->device, this->impl->stagingInputBuffers[i], nullptr);
		}
		if (this->impl->stagingInputMemory[i] != VK_NULL_HANDLE) {
			vkFreeMemory(this->impl->device, this->impl->stagingInputMemory[i], nullptr);
		}
	}

	for (size_t i = 0; i < this->impl->stagingOutputBuffers.size(); ++i) {
		if (this->impl->stagingOutputMemory[i] != VK_NULL_HANDLE) {
			vkUnmapMemory(this->impl->device, this->impl->stagingOutputMemory[i]);
		}
		if (this->impl->stagingOutputBuffers[i] != VK_NULL_HANDLE) {
			vkDestroyBuffer(this->impl->device, this->impl->stagingOutputBuffers[i], nullptr);
		}
		if (this->impl->stagingOutputMemory[i] != VK_NULL_HANDLE) {
			vkFreeMemory(this->impl->device, this->impl->stagingOutputMemory[i], nullptr);
		}
	}

	// Destroy device buffers
	for (size_t i = 0; i < this->impl->deviceInputBuffers.size(); ++i) {
		if (this->impl->deviceInputBuffers[i] != VK_NULL_HANDLE) {
			vkDestroyBuffer(this->impl->device, this->impl->deviceInputBuffers[i], nullptr);
		}
		if (this->impl->deviceInputMemory[i] != VK_NULL_HANDLE) {
			vkFreeMemory(this->impl->device, this->impl->deviceInputMemory[i], nullptr);
		}
	}

	if (this->impl->deviceFftBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->deviceFftBuffer, nullptr);
	}
	if (this->impl->deviceFftMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->deviceFftMemory, nullptr);
	}

	if (this->impl->deviceIntermediateBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->deviceIntermediateBuffer, nullptr);
	}
	if (this->impl->deviceIntermediateMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->deviceIntermediateMemory, nullptr);
	}

	if (this->impl->deviceProcessedBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->deviceProcessedBuffer, nullptr);
	}
	if (this->impl->deviceProcessedMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->deviceProcessedMemory, nullptr);
	}

	// Curve buffers
	if (this->impl->resampleCurveBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->resampleCurveBuffer, nullptr);
	}
	if (this->impl->resampleCurveMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->resampleCurveMemory, nullptr);
	}

	if (this->impl->windowCurveBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->windowCurveBuffer, nullptr);
	}
	if (this->impl->windowCurveMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->windowCurveMemory, nullptr);
	}

	if (this->impl->dispersionCurveBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->dispersionCurveBuffer, nullptr);
	}
	if (this->impl->dispersionCurveMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->dispersionCurveMemory, nullptr);
	}
}

void VulkanBackend::createCommandBuffersAndFences() {
	// Allocate command buffers
	this->impl->commandBuffers.resize(this->impl->numCommandBuffers);

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = this->impl->commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = this->impl->numCommandBuffers;

	checkVulkanErrors(vkAllocateCommandBuffers(this->impl->device, &allocInfo, this->impl->commandBuffers.data()));

	// Create fences
	this->impl->fences.resize(this->impl->numCommandBuffers);

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		checkVulkanErrors(vkCreateFence(this->impl->device, &fenceInfo, nullptr, &this->impl->fences[i]));
	}

	// Create dedicated fence for VkFFT
	checkVulkanErrors(vkCreateFence(this->impl->device, &fenceInfo, nullptr, &this->impl->fftFence));
}

void VulkanBackend::destroyCommandBuffersAndFences() {
	// Free command buffers (automatically freed when pool is destroyed)
	if (!this->impl->commandBuffers.empty() && this->impl->commandPool != VK_NULL_HANDLE) {
		vkFreeCommandBuffers(this->impl->device, this->impl->commandPool,
		                     static_cast<uint32_t>(this->impl->commandBuffers.size()),
		                     this->impl->commandBuffers.data());
		this->impl->commandBuffers.clear();
	}

	// Destroy fences
	for (auto& fence : this->impl->fences) {
		if (fence != VK_NULL_HANDLE) {
			vkDestroyFence(this->impl->device, fence, nullptr);
		}
	}
	this->impl->fences.clear();

	// Destroy FFT fence
	if (this->impl->fftFence != VK_NULL_HANDLE) {
		vkDestroyFence(this->impl->device, this->impl->fftFence, nullptr);
		this->impl->fftFence = VK_NULL_HANDLE;
	}
}

// ============================================
// Command Buffer Recording
// ============================================

void VulkanBackend::recordCommandBuffers() {
	// Pre-record all command buffers for maximum performance
	// This avoids the expensive vkBeginCommandBuffer/vkEndCommandBuffer overhead every frame

	for (int idx = 0; idx < this->impl->numCommandBuffers; ++idx) {
		VkCommandBuffer cmd = this->impl->commandBuffers[idx];

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;  // Reusable command buffer (not ONE_TIME_SUBMIT)

		checkVulkanErrors(vkBeginCommandBuffer(cmd, &beginInfo));

		// Copy from staging to device buffer
		size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
		VkBufferCopy copyRegion = {};
		copyRegion.size = inputSize;
		vkCmdCopyBuffer(cmd, this->impl->stagingInputBuffers[idx], this->impl->deviceInputBuffers[idx], 1, &copyRegion);

		// Memory barrier after input copy
		VkBufferMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.buffer = this->impl->deviceInputBuffers[idx];
		barrier.offset = 0;
		barrier.size = inputSize;

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0,
		                     0, nullptr,
		                     1, &barrier,
		                     0, nullptr);

		// NOTE: Rest of command recording will be added here
		// For now, just record input conversion as a test

		// Bind input conversion pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->getPipeline(Impl::PipelineIndex::InputConversion));

		// Bind descriptor set
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->impl->pipelineLayout,
		                        0, 1, &this->impl->descriptorSets[idx], 0, nullptr);

		// Push constants
		uint32_t pushConstants[3] = {
			static_cast<uint32_t>(this->impl->samplesPerBuffer),
			static_cast<uint32_t>(this->impl->config.dataParams.getBitDepth()),
			static_cast<uint32_t>(this->impl->bytesPerSample)
		};
		vkCmdPushConstants(cmd, this->impl->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(pushConstants), pushConstants);

		// Dispatch
		uint32_t numWorkgroups = (this->impl->samplesPerBuffer + 127) / 128;
		vkCmdDispatch(cmd, numWorkgroups, 1, 1);

		checkVulkanErrors(vkEndCommandBuffer(cmd));
	}
}

// ============================================
// Shader Compilation Helpers
// ============================================

// Helper function to find shader file in multiple possible locations
std::string findShaderPath(const std::string& relativePathFromRoot) {
	// Try multiple possible locations:
	std::vector<std::string> searchPaths = {
		relativePathFromRoot,                               // 1. From project root (current directory)
		"../" + relativePathFromRoot,                       // 2. One directory up
		"../../" + relativePathFromRoot,                    // 3. Two directories up
		"../../../" + relativePathFromRoot,                 // 4. Three directories up (e.g., from build/tests/Release)
	};

	for (const auto& path : searchPaths) {
		std::ifstream test(path);
		if (test.good()) {
			return path;
		}
	}

	// If not found in any location, return original path (will fail with helpful error)
	return relativePathFromRoot;
}

// Helper function to load shader source from file
std::string loadShaderSource(const std::string& filepath) {
	std::string actualPath = findShaderPath(filepath);
	std::ifstream file(actualPath);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open shader file: " + actualPath +
		                         " (searched from: " + filepath + ")");
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

// Helper function to compile GLSL to SPIR-V using shaderc
std::vector<uint32_t> compileGLSLToSPIRV(const std::string& source, const std::string& filename, shaderc_shader_kind kind) {
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;

	// Set optimization level
	options.SetOptimizationLevel(shaderc_optimization_level_performance);

	// Compile to SPIR-V
	shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(source, kind, filename.c_str(), options);

	if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
		std::stringstream ss;
		ss << "Shader compilation failed for " << filename << ":\n" << result.GetErrorMessage();
		throw std::runtime_error(ss.str());
	}

	return std::vector<uint32_t>(result.cbegin(), result.cend());
}

// Helper function to create VkShaderModule from SPIR-V
VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& spirv) {
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = spirv.size() * sizeof(uint32_t);
	createInfo.pCode = spirv.data();

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create shader module");
	}

	return shaderModule;
}

void VulkanBackend::createComputePipelines() {
	// ============================================
	// Create Descriptor Set Layout for Input Conversion Shader
	// ============================================
	// Layout: 2 storage buffers (binding 0: input, binding 1: output)

	std::vector<VkDescriptorSetLayoutBinding> bindings(2);

	// Binding 0: Input buffer (storage buffer)
	bindings[0].binding = 0;
	bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings[0].descriptorCount = 1;
	bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	bindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Output buffer (storage buffer)
	bindings[1].binding = 1;
	bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	bindings[1].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &layoutInfo, nullptr, &this->impl->descriptorSetLayout));

	// ============================================
	// Create Pipeline Layout with Push Constants
	// ============================================

	VkPushConstantRange pushConstantRange = {};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(uint32_t) * 3;  // samplesPerBuffer, inputBitDepth, bytesPerSample

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &this->impl->descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &pipelineLayoutInfo, nullptr, &this->impl->pipelineLayout));

	// ============================================
	// Load and Compile Input Conversion Shader
	// ============================================

	std::string shaderPath = "src/backends/vulkan/shaders/input_conversion.comp";
	std::string shaderSource = loadShaderSource(shaderPath);
	std::vector<uint32_t> spirv = compileGLSLToSPIRV(shaderSource, shaderPath, shaderc_compute_shader);

	VkShaderModule inputConversionShader = createShaderModule(this->impl->device, spirv);
	this->impl->shaderModules.push_back(inputConversionShader);

	// ============================================
	// Create Compute Pipeline for Input Conversion
	// ============================================

	VkPipelineShaderStageCreateInfo shaderStageInfo = {};
	shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStageInfo.module = inputConversionShader;
	shaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = shaderStageInfo;
	pipelineInfo.layout = this->impl->pipelineLayout;

	VkPipeline inputConversionPipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &inputConversionPipeline));
	this->impl->computePipelines.push_back(inputConversionPipeline);

	// ============================================
	// Create Truncate Shader Pipeline
	// ============================================

	// Create descriptor set layout for truncate (2 storage buffers: FFT input, processed output)
	std::vector<VkDescriptorSetLayoutBinding> truncateBindings(2);

	// Binding 0: FFT buffer (complex input)
	truncateBindings[0].binding = 0;
	truncateBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	truncateBindings[0].descriptorCount = 1;
	truncateBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	truncateBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Processed buffer (real output)
	truncateBindings[1].binding = 1;
	truncateBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	truncateBindings[1].descriptorCount = 1;
	truncateBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	truncateBindings[1].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo truncateLayoutInfo = {};
	truncateLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	truncateLayoutInfo.bindingCount = static_cast<uint32_t>(truncateBindings.size());
	truncateLayoutInfo.pBindings = truncateBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &truncateLayoutInfo, nullptr, &this->impl->truncateDescriptorSetLayout));

	// Create pipeline layout for truncate (different push constants)
	VkPushConstantRange truncatePushConstantRange = {};
	truncatePushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	truncatePushConstantRange.offset = 0;
	truncatePushConstantRange.size = sizeof(uint32_t) * 3;  // fullSignalLength, outputSignalLength, samplesPerBuffer

	VkPipelineLayoutCreateInfo truncatePipelineLayoutInfo = {};
	truncatePipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	truncatePipelineLayoutInfo.setLayoutCount = 1;
	truncatePipelineLayoutInfo.pSetLayouts = &this->impl->truncateDescriptorSetLayout;
	truncatePipelineLayoutInfo.pushConstantRangeCount = 1;
	truncatePipelineLayoutInfo.pPushConstantRanges = &truncatePushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &truncatePipelineLayoutInfo, nullptr, &this->impl->truncatePipelineLayout));

	// Load and compile truncate shader
	std::string truncateShaderPath = "src/backends/vulkan/shaders/truncate.comp";
	std::string truncateShaderSource = loadShaderSource(truncateShaderPath);
	std::vector<uint32_t> truncateSPIRV = compileGLSLToSPIRV(truncateShaderSource, truncateShaderPath, shaderc_compute_shader);

	VkShaderModule truncateShader = createShaderModule(this->impl->device, truncateSPIRV);
	this->impl->shaderModules.push_back(truncateShader);

	// Create truncate compute pipeline
	VkPipelineShaderStageCreateInfo truncateShaderStageInfo = {};
	truncateShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	truncateShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	truncateShaderStageInfo.module = truncateShader;
	truncateShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo truncatePipelineInfo = {};
	truncatePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	truncatePipelineInfo.stage = truncateShaderStageInfo;
	truncatePipelineInfo.layout = this->impl->truncatePipelineLayout;

	VkPipeline truncatePipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &truncatePipelineInfo, nullptr, &truncatePipeline));
	this->impl->computePipelines.push_back(truncatePipeline);

	// ============================================
	// Create Descriptor Pool
	// ============================================
	// Pool needs to allocate for all pipeline descriptor sets

	VkDescriptorPoolSize poolSize = {};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = static_cast<uint32_t>(this->impl->numCommandBuffers * 22);  // 2 (input conv) + 2 (truncate) + 3 (windowing) + 2 (postprocess) + 2 (DC removal) + 3 (klinear) + 3 (dispersion) + 5 (merged klinear+windowing+dispersion) per command buffer

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = static_cast<uint32_t>(this->impl->numCommandBuffers * 8);  // 8 descriptor sets per command buffer (including merged pipeline)
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // Allow individual descriptor sets to be freed

	checkVulkanErrors(vkCreateDescriptorPool(this->impl->device, &poolInfo, nullptr, &this->impl->descriptorPool));

	// ============================================
	// Allocate Descriptor Sets (one per command buffer)
	// ============================================

	std::vector<VkDescriptorSetLayout> layouts(this->impl->numCommandBuffers, this->impl->descriptorSetLayout);

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = this->impl->descriptorPool;
	allocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	allocInfo.pSetLayouts = layouts.data();

	this->impl->descriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &allocInfo, this->impl->descriptorSets.data()));

	// ============================================
	// Update Descriptor Sets with Buffer Bindings
	// ============================================

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> descriptorWrites(2);

		// Binding 0: Input buffer
		VkDescriptorBufferInfo inputBufferInfo = {};
		inputBufferInfo.buffer = this->impl->deviceInputBuffers[i];
		inputBufferInfo.offset = 0;
		inputBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = this->impl->descriptorSets[i];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &inputBufferInfo;

		// Binding 1: FFT buffer (output of input conversion)
		VkDescriptorBufferInfo outputBufferInfo = {};
		outputBufferInfo.buffer = this->impl->deviceFftBuffer;
		outputBufferInfo.offset = 0;
		outputBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = this->impl->descriptorSets[i];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &outputBufferInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}

	// ============================================
	// Allocate and Update Truncate Descriptor Sets
	// ============================================

	std::vector<VkDescriptorSetLayout> truncateLayouts(this->impl->numCommandBuffers, this->impl->truncateDescriptorSetLayout);

	VkDescriptorSetAllocateInfo truncateAllocInfo = {};
	truncateAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	truncateAllocInfo.descriptorPool = this->impl->descriptorPool;
	truncateAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	truncateAllocInfo.pSetLayouts = truncateLayouts.data();

	this->impl->truncateDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &truncateAllocInfo, this->impl->truncateDescriptorSets.data()));

	// Update truncate descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> truncateDescriptorWrites(2);

		// Binding 0: FFT buffer (complex input)
		VkDescriptorBufferInfo fftBufferInfo = {};
		fftBufferInfo.buffer = this->impl->deviceFftBuffer;
		fftBufferInfo.offset = 0;
		fftBufferInfo.range = VK_WHOLE_SIZE;

		truncateDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		truncateDescriptorWrites[0].dstSet = this->impl->truncateDescriptorSets[i];
		truncateDescriptorWrites[0].dstBinding = 0;
		truncateDescriptorWrites[0].dstArrayElement = 0;
		truncateDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		truncateDescriptorWrites[0].descriptorCount = 1;
		truncateDescriptorWrites[0].pBufferInfo = &fftBufferInfo;

		// Binding 1: Processed buffer (real output)
		VkDescriptorBufferInfo processedBufferInfo = {};
		processedBufferInfo.buffer = this->impl->deviceProcessedBuffer;
		processedBufferInfo.offset = 0;
		processedBufferInfo.range = VK_WHOLE_SIZE;

		truncateDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		truncateDescriptorWrites[1].dstSet = this->impl->truncateDescriptorSets[i];
		truncateDescriptorWrites[1].dstBinding = 1;
		truncateDescriptorWrites[1].dstArrayElement = 0;
		truncateDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		truncateDescriptorWrites[1].descriptorCount = 1;
		truncateDescriptorWrites[1].pBufferInfo = &processedBufferInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(truncateDescriptorWrites.size()), truncateDescriptorWrites.data(), 0, nullptr);
	}

	// ============================================
	// ============================================

	// Create descriptor set layout for windowing (3 storage buffers: input, window curve, output)
	std::vector<VkDescriptorSetLayoutBinding> windowingBindings(3);

	// Binding 0: Input buffer
	windowingBindings[0].binding = 0;
	windowingBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	windowingBindings[0].descriptorCount = 1;
	windowingBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	windowingBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Window curve buffer
	windowingBindings[1].binding = 1;
	windowingBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	windowingBindings[1].descriptorCount = 1;
	windowingBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	windowingBindings[1].pImmutableSamplers = nullptr;

	// Binding 2: Output buffer
	windowingBindings[2].binding = 2;
	windowingBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	windowingBindings[2].descriptorCount = 1;
	windowingBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	windowingBindings[2].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo windowingLayoutInfo = {};
	windowingLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	windowingLayoutInfo.bindingCount = static_cast<uint32_t>(windowingBindings.size());
	windowingLayoutInfo.pBindings = windowingBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &windowingLayoutInfo, nullptr, &this->impl->windowingDescriptorSetLayout));

	// Create pipeline layout for windowing
	VkPushConstantRange windowingPushConstantRange = {};
	windowingPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	windowingPushConstantRange.offset = 0;
	windowingPushConstantRange.size = sizeof(uint32_t) * 2;  // signalLength, samplesPerBuffer

	VkPipelineLayoutCreateInfo windowingPipelineLayoutInfo = {};
	windowingPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	windowingPipelineLayoutInfo.setLayoutCount = 1;
	windowingPipelineLayoutInfo.pSetLayouts = &this->impl->windowingDescriptorSetLayout;
	windowingPipelineLayoutInfo.pushConstantRangeCount = 1;
	windowingPipelineLayoutInfo.pPushConstantRanges = &windowingPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &windowingPipelineLayoutInfo, nullptr, &this->impl->windowingPipelineLayout));

	// Load and compile windowing shader
	std::string windowingShaderPath = "src/backends/vulkan/shaders/windowing.comp";
	std::string windowingShaderSource = loadShaderSource(windowingShaderPath);
	std::vector<uint32_t> windowingSPIRV = compileGLSLToSPIRV(windowingShaderSource, windowingShaderPath, shaderc_compute_shader);

	VkShaderModule windowingShader = createShaderModule(this->impl->device, windowingSPIRV);
	this->impl->shaderModules.push_back(windowingShader);

	// Create windowing compute pipeline
	VkPipelineShaderStageCreateInfo windowingShaderStageInfo = {};
	windowingShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	windowingShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	windowingShaderStageInfo.module = windowingShader;
	windowingShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo windowingPipelineInfo = {};
	windowingPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	windowingPipelineInfo.stage = windowingShaderStageInfo;
	windowingPipelineInfo.layout = this->impl->windowingPipelineLayout;

	VkPipeline windowingPipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &windowingPipelineInfo, nullptr, &windowingPipeline));
	this->impl->computePipelines.push_back(windowingPipeline);

	// Allocate and Update Windowing Descriptor Sets
	std::vector<VkDescriptorSetLayout> windowingLayouts(this->impl->numCommandBuffers, this->impl->windowingDescriptorSetLayout);

	VkDescriptorSetAllocateInfo windowingAllocInfo = {};
	windowingAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	windowingAllocInfo.descriptorPool = this->impl->descriptorPool;
	windowingAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	windowingAllocInfo.pSetLayouts = windowingLayouts.data();

	this->impl->windowingDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &windowingAllocInfo, this->impl->windowingDescriptorSets.data()));

	// Update windowing descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> windowingDescriptorWrites(3);

		// Binding 0: Input buffer (Intermediate buffer - from k-linearization)
		VkDescriptorBufferInfo windowingInputInfo = {};
		windowingInputInfo.buffer = this->impl->deviceIntermediateBuffer;
		windowingInputInfo.offset = 0;
		windowingInputInfo.range = VK_WHOLE_SIZE;

		windowingDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		windowingDescriptorWrites[0].dstSet = this->impl->windowingDescriptorSets[i];
		windowingDescriptorWrites[0].dstBinding = 0;
		windowingDescriptorWrites[0].dstArrayElement = 0;
		windowingDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		windowingDescriptorWrites[0].descriptorCount = 1;
		windowingDescriptorWrites[0].pBufferInfo = &windowingInputInfo;

		// Binding 1: Window curve buffer
		VkDescriptorBufferInfo windowCurveInfo = {};
		windowCurveInfo.buffer = this->impl->windowCurveBuffer;
		windowCurveInfo.offset = 0;
		windowCurveInfo.range = VK_WHOLE_SIZE;

		windowingDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		windowingDescriptorWrites[1].dstSet = this->impl->windowingDescriptorSets[i];
		windowingDescriptorWrites[1].dstBinding = 1;
		windowingDescriptorWrites[1].dstArrayElement = 0;
		windowingDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		windowingDescriptorWrites[1].descriptorCount = 1;
		windowingDescriptorWrites[1].pBufferInfo = &windowCurveInfo;

		// Binding 2: Output buffer (FFT buffer)
		VkDescriptorBufferInfo windowingOutputInfo = {};
		windowingOutputInfo.buffer = this->impl->deviceFftBuffer;
		windowingOutputInfo.offset = 0;
		windowingOutputInfo.range = VK_WHOLE_SIZE;

		windowingDescriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		windowingDescriptorWrites[2].dstSet = this->impl->windowingDescriptorSets[i];
		windowingDescriptorWrites[2].dstBinding = 2;
		windowingDescriptorWrites[2].dstArrayElement = 0;
		windowingDescriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		windowingDescriptorWrites[2].descriptorCount = 1;
		windowingDescriptorWrites[2].pBufferInfo = &windowingOutputInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(windowingDescriptorWrites.size()), windowingDescriptorWrites.data(), 0, nullptr);
	}

	// ============================================
	// Create Post-Processing Shader Pipeline
	// ============================================

	// Create descriptor set layout for post-processing (2 storage buffers: input, output)
	std::vector<VkDescriptorSetLayoutBinding> postprocessBindings(2);

	// Binding 0: Input buffer (magnitude data from truncate)
	postprocessBindings[0].binding = 0;
	postprocessBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	postprocessBindings[0].descriptorCount = 1;
	postprocessBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	postprocessBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Output buffer (scaled data)
	postprocessBindings[1].binding = 1;
	postprocessBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	postprocessBindings[1].descriptorCount = 1;
	postprocessBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	postprocessBindings[1].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo postprocessLayoutInfo = {};
	postprocessLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	postprocessLayoutInfo.bindingCount = static_cast<uint32_t>(postprocessBindings.size());
	postprocessLayoutInfo.pBindings = postprocessBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &postprocessLayoutInfo, nullptr, &this->impl->postprocessDescriptorSetLayout));

	// Create pipeline layout for post-processing
	VkPushConstantRange postprocessPushConstantRange = {};
	postprocessPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	postprocessPushConstantRange.offset = 0;
	postprocessPushConstantRange.size = sizeof(uint32_t) * 2 + sizeof(float) * 4;  // samplesPerBuffer, logScaling, grayscaleMax, grayscaleMin, addend, multiplicator

	VkPipelineLayoutCreateInfo postprocessPipelineLayoutInfo = {};
	postprocessPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	postprocessPipelineLayoutInfo.setLayoutCount = 1;
	postprocessPipelineLayoutInfo.pSetLayouts = &this->impl->postprocessDescriptorSetLayout;
	postprocessPipelineLayoutInfo.pushConstantRangeCount = 1;
	postprocessPipelineLayoutInfo.pPushConstantRanges = &postprocessPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &postprocessPipelineLayoutInfo, nullptr, &this->impl->postprocessPipelineLayout));

	// Load and compile post-processing shader
	std::string postprocessShaderPath = "src/backends/vulkan/shaders/postprocess.comp";
	std::string postprocessShaderSource = loadShaderSource(postprocessShaderPath);
	std::vector<uint32_t> postprocessSPIRV = compileGLSLToSPIRV(postprocessShaderSource, postprocessShaderPath, shaderc_compute_shader);

	VkShaderModule postprocessShader = createShaderModule(this->impl->device, postprocessSPIRV);
	this->impl->shaderModules.push_back(postprocessShader);

	// Create post-processing compute pipeline
	VkPipelineShaderStageCreateInfo postprocessShaderStageInfo = {};
	postprocessShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	postprocessShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	postprocessShaderStageInfo.module = postprocessShader;
	postprocessShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo postprocessPipelineInfo = {};
	postprocessPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	postprocessPipelineInfo.stage = postprocessShaderStageInfo;
	postprocessPipelineInfo.layout = this->impl->postprocessPipelineLayout;

	VkPipeline postprocessPipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &postprocessPipelineInfo, nullptr, &postprocessPipeline));
	this->impl->computePipelines.push_back(postprocessPipeline);
	// Allocate and Update Postprocess Descriptor Sets
	// ============================================

	std::vector<VkDescriptorSetLayout> postprocessLayouts(this->impl->numCommandBuffers, this->impl->postprocessDescriptorSetLayout);

	VkDescriptorSetAllocateInfo postprocessAllocInfo = {};
	postprocessAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	postprocessAllocInfo.descriptorPool = this->impl->descriptorPool;
	postprocessAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	postprocessAllocInfo.pSetLayouts = postprocessLayouts.data();

	this->impl->postprocessDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &postprocessAllocInfo, this->impl->postprocessDescriptorSets.data()));

	// Update postprocess descriptor sets (both input and output point to same buffer for in-place processing)
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> postprocessDescriptorWrites(2);

		// Binding 0: Processed buffer (input - magnitude data from truncate)
		VkDescriptorBufferInfo inputBufferInfo = {};
		inputBufferInfo.buffer = this->impl->deviceProcessedBuffer;
		inputBufferInfo.offset = 0;
		inputBufferInfo.range = VK_WHOLE_SIZE;

		postprocessDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		postprocessDescriptorWrites[0].dstSet = this->impl->postprocessDescriptorSets[i];
		postprocessDescriptorWrites[0].dstBinding = 0;
		postprocessDescriptorWrites[0].dstArrayElement = 0;
		postprocessDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		postprocessDescriptorWrites[0].descriptorCount = 1;
		postprocessDescriptorWrites[0].pBufferInfo = &inputBufferInfo;

		// Binding 1: Processed buffer (output - same buffer, in-place)
		VkDescriptorBufferInfo outputBufferInfo = {};
		outputBufferInfo.buffer = this->impl->deviceProcessedBuffer;
		outputBufferInfo.offset = 0;
		outputBufferInfo.range = VK_WHOLE_SIZE;

		postprocessDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		postprocessDescriptorWrites[1].dstSet = this->impl->postprocessDescriptorSets[i];
		postprocessDescriptorWrites[1].dstBinding = 1;
		postprocessDescriptorWrites[1].dstArrayElement = 0;
		postprocessDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		postprocessDescriptorWrites[1].descriptorCount = 1;
		postprocessDescriptorWrites[1].pBufferInfo = &outputBufferInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(postprocessDescriptorWrites.size()), postprocessDescriptorWrites.data(), 0, nullptr);
	}


	// ============================================
	// Create Windowing Shader Pipeline
	// ============================================

	// ============================================
	// Create DC Removal Shader Pipeline
	// ============================================

	// Create descriptor set layout for DC removal (2 storage buffers: input, output)
	std::vector<VkDescriptorSetLayoutBinding> dcRemovalBindings(2);

	// Binding 0: Input buffer (complex data)
	dcRemovalBindings[0].binding = 0;
	dcRemovalBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	dcRemovalBindings[0].descriptorCount = 1;
	dcRemovalBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	dcRemovalBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Output buffer (complex data with DC removed)
	dcRemovalBindings[1].binding = 1;
	dcRemovalBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	dcRemovalBindings[1].descriptorCount = 1;
	dcRemovalBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	dcRemovalBindings[1].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo dcRemovalLayoutInfo = {};
	dcRemovalLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dcRemovalLayoutInfo.bindingCount = static_cast<uint32_t>(dcRemovalBindings.size());
	dcRemovalLayoutInfo.pBindings = dcRemovalBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &dcRemovalLayoutInfo, nullptr, &this->impl->dcRemovalDescriptorSetLayout));

	// Create pipeline layout for DC removal
	VkPushConstantRange dcRemovalPushConstantRange = {};
	dcRemovalPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	dcRemovalPushConstantRange.offset = 0;
	dcRemovalPushConstantRange.size = sizeof(uint32_t) * 4;  // rollingAverageWindowSize, signalLength, ascansPerBscan, samplesPerBuffer

	VkPipelineLayoutCreateInfo dcRemovalPipelineLayoutInfo = {};
	dcRemovalPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	dcRemovalPipelineLayoutInfo.setLayoutCount = 1;
	dcRemovalPipelineLayoutInfo.pSetLayouts = &this->impl->dcRemovalDescriptorSetLayout;
	dcRemovalPipelineLayoutInfo.pushConstantRangeCount = 1;
	dcRemovalPipelineLayoutInfo.pPushConstantRanges = &dcRemovalPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &dcRemovalPipelineLayoutInfo, nullptr, &this->impl->dcRemovalPipelineLayout));

	// Load and compile DC removal shader
	std::string dcRemovalShaderPath = "src/backends/vulkan/shaders/dc_removal.comp";
	std::string dcRemovalShaderSource = loadShaderSource(dcRemovalShaderPath);
	std::vector<uint32_t> dcRemovalSPIRV = compileGLSLToSPIRV(dcRemovalShaderSource, dcRemovalShaderPath, shaderc_compute_shader);

	VkShaderModule dcRemovalShader = createShaderModule(this->impl->device, dcRemovalSPIRV);
	this->impl->shaderModules.push_back(dcRemovalShader);

	// Create DC removal compute pipeline
	VkPipelineShaderStageCreateInfo dcRemovalShaderStageInfo = {};
	dcRemovalShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	dcRemovalShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	dcRemovalShaderStageInfo.module = dcRemovalShader;
	dcRemovalShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo dcRemovalPipelineInfo = {};
	dcRemovalPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	dcRemovalPipelineInfo.stage = dcRemovalShaderStageInfo;
	dcRemovalPipelineInfo.layout = this->impl->dcRemovalPipelineLayout;

	VkPipeline dcRemovalPipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &dcRemovalPipelineInfo, nullptr, &dcRemovalPipeline));
	this->impl->computePipelines.push_back(dcRemovalPipeline);

	// Allocate and Update DC Removal Descriptor Sets
	std::vector<VkDescriptorSetLayout> dcRemovalLayouts(this->impl->numCommandBuffers, this->impl->dcRemovalDescriptorSetLayout);

	VkDescriptorSetAllocateInfo dcRemovalAllocInfo = {};
	dcRemovalAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dcRemovalAllocInfo.descriptorPool = this->impl->descriptorPool;
	dcRemovalAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	dcRemovalAllocInfo.pSetLayouts = dcRemovalLayouts.data();

	this->impl->dcRemovalDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &dcRemovalAllocInfo, this->impl->dcRemovalDescriptorSets.data()));

	// Update DC removal descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		VkDescriptorBufferInfo dcRemovalInputBufferInfo = {};
		dcRemovalInputBufferInfo.buffer = this->impl->deviceFftBuffer;
		dcRemovalInputBufferInfo.offset = 0;
		dcRemovalInputBufferInfo.range = this->impl->fftBufferSize;

		VkDescriptorBufferInfo dcRemovalOutputBufferInfo = {};
		dcRemovalOutputBufferInfo.buffer = this->impl->deviceIntermediateBuffer;  // Write to separate buffer
		dcRemovalOutputBufferInfo.offset = 0;
		dcRemovalOutputBufferInfo.range = VK_WHOLE_SIZE;

		std::vector<VkWriteDescriptorSet> dcRemovalDescriptorWrites(2);

		// Binding 0: Input buffer
		dcRemovalDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dcRemovalDescriptorWrites[0].dstSet = this->impl->dcRemovalDescriptorSets[i];
		dcRemovalDescriptorWrites[0].dstBinding = 0;
		dcRemovalDescriptorWrites[0].dstArrayElement = 0;
		dcRemovalDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		dcRemovalDescriptorWrites[0].descriptorCount = 1;
		dcRemovalDescriptorWrites[0].pBufferInfo = &dcRemovalInputBufferInfo;

		// Binding 1: Output buffer
		dcRemovalDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dcRemovalDescriptorWrites[1].dstSet = this->impl->dcRemovalDescriptorSets[i];
		dcRemovalDescriptorWrites[1].dstBinding = 1;
		dcRemovalDescriptorWrites[1].dstArrayElement = 0;
		dcRemovalDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		dcRemovalDescriptorWrites[1].descriptorCount = 1;
		dcRemovalDescriptorWrites[1].pBufferInfo = &dcRemovalOutputBufferInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(dcRemovalDescriptorWrites.size()), dcRemovalDescriptorWrites.data(), 0, nullptr);
	}

	// ============================================
	// Create K-Linearization Shader Pipeline
	// ============================================

	// Create descriptor set layout for K-linearization (3 storage buffers: input, resample curve, output)
	std::vector<VkDescriptorSetLayoutBinding> klinearBindings(3);

	// Binding 0: Input buffer (complex data)
	klinearBindings[0].binding = 0;
	klinearBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearBindings[0].descriptorCount = 1;
	klinearBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Resample curve buffer
	klinearBindings[1].binding = 1;
	klinearBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearBindings[1].descriptorCount = 1;
	klinearBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearBindings[1].pImmutableSamplers = nullptr;

	// Binding 2: Output buffer (k-linearized complex data)
	klinearBindings[2].binding = 2;
	klinearBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearBindings[2].descriptorCount = 1;
	klinearBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearBindings[2].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo klinearLayoutInfo = {};
	klinearLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	klinearLayoutInfo.bindingCount = static_cast<uint32_t>(klinearBindings.size());
	klinearLayoutInfo.pBindings = klinearBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &klinearLayoutInfo, nullptr, &this->impl->klinearizationDescriptorSetLayout));

	// Create pipeline layout for K-linearization
	VkPushConstantRange klinearPushConstantRange = {};
	klinearPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearPushConstantRange.offset = 0;
	klinearPushConstantRange.size = sizeof(uint32_t) * 2;  // signalLength, samplesPerBuffer

	VkPipelineLayoutCreateInfo klinearPipelineLayoutInfo = {};
	klinearPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	klinearPipelineLayoutInfo.setLayoutCount = 1;
	klinearPipelineLayoutInfo.pSetLayouts = &this->impl->klinearizationDescriptorSetLayout;
	klinearPipelineLayoutInfo.pushConstantRangeCount = 1;
	klinearPipelineLayoutInfo.pPushConstantRanges = &klinearPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &klinearPipelineLayoutInfo, nullptr, &this->impl->klinearizationPipelineLayout));

	// Load and compile K-linearization shader
	std::string klinearShaderPath = "src/backends/vulkan/shaders/klinearization.comp";
	std::string klinearShaderSource = loadShaderSource(klinearShaderPath);
	std::vector<uint32_t> klinearSPIRV = compileGLSLToSPIRV(klinearShaderSource, klinearShaderPath, shaderc_compute_shader);

	VkShaderModule klinearShader = createShaderModule(this->impl->device, klinearSPIRV);
	this->impl->shaderModules.push_back(klinearShader);

	// Create K-linearization compute pipeline
	VkPipelineShaderStageCreateInfo klinearShaderStageInfo = {};
	klinearShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	klinearShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearShaderStageInfo.module = klinearShader;
	klinearShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo klinearPipelineInfo = {};
	klinearPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	klinearPipelineInfo.stage = klinearShaderStageInfo;
	klinearPipelineInfo.layout = this->impl->klinearizationPipelineLayout;

	VkPipeline klinearPipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &klinearPipelineInfo, nullptr, &klinearPipeline));
	this->impl->computePipelines.push_back(klinearPipeline);

	// Allocate and Update K-linearization Descriptor Sets
	std::vector<VkDescriptorSetLayout> klinearLayouts(this->impl->numCommandBuffers, this->impl->klinearizationDescriptorSetLayout);

	VkDescriptorSetAllocateInfo klinearAllocInfo = {};
	klinearAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	klinearAllocInfo.descriptorPool = this->impl->descriptorPool;
	klinearAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	klinearAllocInfo.pSetLayouts = klinearLayouts.data();

	this->impl->klinearizationDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &klinearAllocInfo, this->impl->klinearizationDescriptorSets.data()));

	// Update k-linearization descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> klinearDescriptorWrites(3);

		// Binding 0: Input buffer (FFT buffer)
		VkDescriptorBufferInfo klinearInputInfo = {};
		klinearInputInfo.buffer = this->impl->deviceFftBuffer;
		klinearInputInfo.offset = 0;
		klinearInputInfo.range = VK_WHOLE_SIZE;

		klinearDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearDescriptorWrites[0].dstSet = this->impl->klinearizationDescriptorSets[i];
		klinearDescriptorWrites[0].dstBinding = 0;
		klinearDescriptorWrites[0].dstArrayElement = 0;
		klinearDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearDescriptorWrites[0].descriptorCount = 1;
		klinearDescriptorWrites[0].pBufferInfo = &klinearInputInfo;

		// Binding 1: Resample curve buffer
		VkDescriptorBufferInfo resampleCurveInfo = {};
		resampleCurveInfo.buffer = this->impl->resampleCurveBuffer;
		resampleCurveInfo.offset = 0;
		resampleCurveInfo.range = VK_WHOLE_SIZE;

		klinearDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearDescriptorWrites[1].dstSet = this->impl->klinearizationDescriptorSets[i];
		klinearDescriptorWrites[1].dstBinding = 1;
		klinearDescriptorWrites[1].dstArrayElement = 0;
		klinearDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearDescriptorWrites[1].descriptorCount = 1;
		klinearDescriptorWrites[1].pBufferInfo = &resampleCurveInfo;

		// Binding 2: Output buffer (Intermediate buffer)
		VkDescriptorBufferInfo klinearOutputInfo = {};
		klinearOutputInfo.buffer = this->impl->deviceIntermediateBuffer;
		klinearOutputInfo.offset = 0;
		klinearOutputInfo.range = VK_WHOLE_SIZE;

		klinearDescriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearDescriptorWrites[2].dstSet = this->impl->klinearizationDescriptorSets[i];
		klinearDescriptorWrites[2].dstBinding = 2;
		klinearDescriptorWrites[2].dstArrayElement = 0;
		klinearDescriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearDescriptorWrites[2].descriptorCount = 1;
		klinearDescriptorWrites[2].pBufferInfo = &klinearOutputInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(klinearDescriptorWrites.size()), klinearDescriptorWrites.data(), 0, nullptr);
	}

	// ============================================
	// Create Dispersion Compensation Shader Pipeline
	// ============================================

	// Create descriptor set layout for dispersion compensation (3 storage buffers: input, phase complex, output)
	std::vector<VkDescriptorSetLayoutBinding> dispersionBindings(3);

	// Binding 0: Input buffer (complex data)
	dispersionBindings[0].binding = 0;
	dispersionBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	dispersionBindings[0].descriptorCount = 1;
	dispersionBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	dispersionBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Phase complex buffer
	dispersionBindings[1].binding = 1;
	dispersionBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	dispersionBindings[1].descriptorCount = 1;
	dispersionBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	dispersionBindings[1].pImmutableSamplers = nullptr;

	// Binding 2: Output buffer (dispersion compensated complex data)
	dispersionBindings[2].binding = 2;
	dispersionBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	dispersionBindings[2].descriptorCount = 1;
	dispersionBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	dispersionBindings[2].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo dispersionLayoutInfo = {};
	dispersionLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dispersionLayoutInfo.bindingCount = static_cast<uint32_t>(dispersionBindings.size());
	dispersionLayoutInfo.pBindings = dispersionBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &dispersionLayoutInfo, nullptr, &this->impl->dispersionDescriptorSetLayout));

	// Create pipeline layout for dispersion compensation
	VkPushConstantRange dispersionPushConstantRange = {};
	dispersionPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	dispersionPushConstantRange.offset = 0;
	dispersionPushConstantRange.size = sizeof(uint32_t) * 2;  // signalLength, samplesPerBuffer

	VkPipelineLayoutCreateInfo dispersionPipelineLayoutInfo = {};
	dispersionPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	dispersionPipelineLayoutInfo.setLayoutCount = 1;
	dispersionPipelineLayoutInfo.pSetLayouts = &this->impl->dispersionDescriptorSetLayout;
	dispersionPipelineLayoutInfo.pushConstantRangeCount = 1;
	dispersionPipelineLayoutInfo.pPushConstantRanges = &dispersionPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &dispersionPipelineLayoutInfo, nullptr, &this->impl->dispersionPipelineLayout));

	// Load and compile dispersion compensation shader
	std::string dispersionShaderPath = "src/backends/vulkan/shaders/dispersion.comp";
	std::string dispersionShaderSource = loadShaderSource(dispersionShaderPath);
	std::vector<uint32_t> dispersionSPIRV = compileGLSLToSPIRV(dispersionShaderSource, dispersionShaderPath, shaderc_compute_shader);

	VkShaderModule dispersionShader = createShaderModule(this->impl->device, dispersionSPIRV);
	this->impl->shaderModules.push_back(dispersionShader);

	// Create dispersion compensation compute pipeline
	VkPipelineShaderStageCreateInfo dispersionShaderStageInfo = {};
	dispersionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	dispersionShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	dispersionShaderStageInfo.module = dispersionShader;
	dispersionShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo dispersionPipelineInfo = {};
	dispersionPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	dispersionPipelineInfo.stage = dispersionShaderStageInfo;
	dispersionPipelineInfo.layout = this->impl->dispersionPipelineLayout;

	VkPipeline dispersionPipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &dispersionPipelineInfo, nullptr, &dispersionPipeline));
	this->impl->computePipelines.push_back(dispersionPipeline);

	// Allocate and Update Dispersion Descriptor Sets
	std::vector<VkDescriptorSetLayout> dispersionLayouts(this->impl->numCommandBuffers, this->impl->dispersionDescriptorSetLayout);

	VkDescriptorSetAllocateInfo dispersionAllocInfo = {};
	dispersionAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dispersionAllocInfo.descriptorPool = this->impl->descriptorPool;
	dispersionAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	dispersionAllocInfo.pSetLayouts = dispersionLayouts.data();

	this->impl->dispersionDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &dispersionAllocInfo, this->impl->dispersionDescriptorSets.data()));

	// Update dispersion descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> dispersionDescriptorWrites(3);

		// Binding 0: Input buffer (FFT buffer - from k-linearization or windowing)
		VkDescriptorBufferInfo dispersionInputInfo = {};
		dispersionInputInfo.buffer = this->impl->deviceFftBuffer;
		dispersionInputInfo.offset = 0;
		dispersionInputInfo.range = VK_WHOLE_SIZE;

		dispersionDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dispersionDescriptorWrites[0].dstSet = this->impl->dispersionDescriptorSets[i];
		dispersionDescriptorWrites[0].dstBinding = 0;
		dispersionDescriptorWrites[0].dstArrayElement = 0;
		dispersionDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		dispersionDescriptorWrites[0].descriptorCount = 1;
		dispersionDescriptorWrites[0].pBufferInfo = &dispersionInputInfo;

		// Binding 1: Phase complex buffer (dispersion curve)
		VkDescriptorBufferInfo phaseComplexInfo = {};
		phaseComplexInfo.buffer = this->impl->dispersionCurveBuffer;
		phaseComplexInfo.offset = 0;
		phaseComplexInfo.range = VK_WHOLE_SIZE;

		dispersionDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dispersionDescriptorWrites[1].dstSet = this->impl->dispersionDescriptorSets[i];
		dispersionDescriptorWrites[1].dstBinding = 1;
		dispersionDescriptorWrites[1].dstArrayElement = 0;
		dispersionDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		dispersionDescriptorWrites[1].descriptorCount = 1;
		dispersionDescriptorWrites[1].pBufferInfo = &phaseComplexInfo;

		// Binding 2: Output buffer (Intermediate buffer)
		VkDescriptorBufferInfo dispersionOutputInfo = {};
		dispersionOutputInfo.buffer = this->impl->deviceIntermediateBuffer;
		dispersionOutputInfo.offset = 0;
		dispersionOutputInfo.range = VK_WHOLE_SIZE;

		dispersionDescriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dispersionDescriptorWrites[2].dstSet = this->impl->dispersionDescriptorSets[i];
		dispersionDescriptorWrites[2].dstBinding = 2;
		dispersionDescriptorWrites[2].dstArrayElement = 0;
		dispersionDescriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		dispersionDescriptorWrites[2].descriptorCount = 1;
		dispersionDescriptorWrites[2].pBufferInfo = &dispersionOutputInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(dispersionDescriptorWrites.size()), dispersionDescriptorWrites.data(), 0, nullptr);
	}

	// ============================================
	// Create Merged K-Linearization+Windowing+Dispersion Shader Pipeline (Cubic)
	// ============================================

	// Create descriptor set layout for merged k-linearization+windowing+dispersion (5 storage buffers)
	std::vector<VkDescriptorSetLayoutBinding> klinearCubicWindowingDispersionBindings(5);

	// Binding 0: Input buffer (complex data from FFT buffer)
	klinearCubicWindowingDispersionBindings[0].binding = 0;
	klinearCubicWindowingDispersionBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearCubicWindowingDispersionBindings[0].descriptorCount = 1;
	klinearCubicWindowingDispersionBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearCubicWindowingDispersionBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Resample curve buffer
	klinearCubicWindowingDispersionBindings[1].binding = 1;
	klinearCubicWindowingDispersionBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearCubicWindowingDispersionBindings[1].descriptorCount = 1;
	klinearCubicWindowingDispersionBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearCubicWindowingDispersionBindings[1].pImmutableSamplers = nullptr;

	// Binding 2: Window curve buffer
	klinearCubicWindowingDispersionBindings[2].binding = 2;
	klinearCubicWindowingDispersionBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearCubicWindowingDispersionBindings[2].descriptorCount = 1;
	klinearCubicWindowingDispersionBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearCubicWindowingDispersionBindings[2].pImmutableSamplers = nullptr;

	// Binding 3: Dispersion phase buffer
	klinearCubicWindowingDispersionBindings[3].binding = 3;
	klinearCubicWindowingDispersionBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearCubicWindowingDispersionBindings[3].descriptorCount = 1;
	klinearCubicWindowingDispersionBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearCubicWindowingDispersionBindings[3].pImmutableSamplers = nullptr;

	// Binding 4: Output buffer (complex data to intermediate buffer)
	klinearCubicWindowingDispersionBindings[4].binding = 4;
	klinearCubicWindowingDispersionBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	klinearCubicWindowingDispersionBindings[4].descriptorCount = 1;
	klinearCubicWindowingDispersionBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearCubicWindowingDispersionBindings[4].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo klinearCubicWindowingDispersionLayoutInfo = {};
	klinearCubicWindowingDispersionLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	klinearCubicWindowingDispersionLayoutInfo.bindingCount = static_cast<uint32_t>(klinearCubicWindowingDispersionBindings.size());
	klinearCubicWindowingDispersionLayoutInfo.pBindings = klinearCubicWindowingDispersionBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &klinearCubicWindowingDispersionLayoutInfo, nullptr, &this->impl->klinearCubicWindowingDispersionDescriptorSetLayout));

	// Create pipeline layout for merged k-linearization+windowing+dispersion
	VkPushConstantRange klinearCubicWindowingDispersionPushConstantRange = {};
	klinearCubicWindowingDispersionPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearCubicWindowingDispersionPushConstantRange.offset = 0;
	klinearCubicWindowingDispersionPushConstantRange.size = sizeof(uint32_t) * 2;  // signalLength, samplesPerBuffer

	VkPipelineLayoutCreateInfo klinearCubicWindowingDispersionPipelineLayoutInfo = {};
	klinearCubicWindowingDispersionPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	klinearCubicWindowingDispersionPipelineLayoutInfo.setLayoutCount = 1;
	klinearCubicWindowingDispersionPipelineLayoutInfo.pSetLayouts = &this->impl->klinearCubicWindowingDispersionDescriptorSetLayout;
	klinearCubicWindowingDispersionPipelineLayoutInfo.pushConstantRangeCount = 1;
	klinearCubicWindowingDispersionPipelineLayoutInfo.pPushConstantRanges = &klinearCubicWindowingDispersionPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &klinearCubicWindowingDispersionPipelineLayoutInfo, nullptr, &this->impl->klinearCubicWindowingDispersionPipelineLayout));

	// Load and compile merged k-linearization+windowing+dispersion shader (cubic)
	std::string klinearCubicWindowingDispersionShaderPath = "src/backends/vulkan/shaders/klinearization_cubic_windowing_dispersion.comp";
	std::string klinearCubicWindowingDispersionShaderSource = loadShaderSource(klinearCubicWindowingDispersionShaderPath);
	std::vector<uint32_t> klinearCubicWindowingDispersionSPIRV = compileGLSLToSPIRV(klinearCubicWindowingDispersionShaderSource, klinearCubicWindowingDispersionShaderPath, shaderc_compute_shader);

	VkShaderModule klinearCubicWindowingDispersionShader = createShaderModule(this->impl->device, klinearCubicWindowingDispersionSPIRV);
	this->impl->shaderModules.push_back(klinearCubicWindowingDispersionShader);

	// Create merged k-linearization+windowing+dispersion compute pipeline
	VkPipelineShaderStageCreateInfo klinearCubicWindowingDispersionShaderStageInfo = {};
	klinearCubicWindowingDispersionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	klinearCubicWindowingDispersionShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearCubicWindowingDispersionShaderStageInfo.module = klinearCubicWindowingDispersionShader;
	klinearCubicWindowingDispersionShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo klinearCubicWindowingDispersionPipelineInfo = {};
	klinearCubicWindowingDispersionPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	klinearCubicWindowingDispersionPipelineInfo.stage = klinearCubicWindowingDispersionShaderStageInfo;
	klinearCubicWindowingDispersionPipelineInfo.layout = this->impl->klinearCubicWindowingDispersionPipelineLayout;

	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &klinearCubicWindowingDispersionPipelineInfo, nullptr, &this->impl->klinearCubicWindowingDispersionPipeline));
	this->impl->computePipelines.push_back(this->impl->klinearCubicWindowingDispersionPipeline);

	// Load and compile merged k-linearization+windowing+dispersion shader (linear)
	std::string klinearLinearWindowingDispersionShaderPath = "src/backends/vulkan/shaders/klinearization_linear_windowing_dispersion.comp";
	std::string klinearLinearWindowingDispersionShaderSource = loadShaderSource(klinearLinearWindowingDispersionShaderPath);
	std::vector<uint32_t> klinearLinearWindowingDispersionSPIRV = compileGLSLToSPIRV(klinearLinearWindowingDispersionShaderSource, klinearLinearWindowingDispersionShaderPath, shaderc_compute_shader);

	VkShaderModule klinearLinearWindowingDispersionShader = createShaderModule(this->impl->device, klinearLinearWindowingDispersionSPIRV);
	this->impl->shaderModules.push_back(klinearLinearWindowingDispersionShader);

	VkPipelineShaderStageCreateInfo klinearLinearWindowingDispersionShaderStageInfo = {};
	klinearLinearWindowingDispersionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	klinearLinearWindowingDispersionShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearLinearWindowingDispersionShaderStageInfo.module = klinearLinearWindowingDispersionShader;
	klinearLinearWindowingDispersionShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo klinearLinearWindowingDispersionPipelineInfo = {};
	klinearLinearWindowingDispersionPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	klinearLinearWindowingDispersionPipelineInfo.stage = klinearLinearWindowingDispersionShaderStageInfo;
	klinearLinearWindowingDispersionPipelineInfo.layout = this->impl->klinearCubicWindowingDispersionPipelineLayout;

	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &klinearLinearWindowingDispersionPipelineInfo, nullptr, &this->impl->klinearLinearWindowingDispersionPipeline));
	this->impl->computePipelines.push_back(this->impl->klinearLinearWindowingDispersionPipeline);

	// Load and compile merged k-linearization+windowing+dispersion shader (lanczos)
	std::string klinearLanczosWindowingDispersionShaderPath = "src/backends/vulkan/shaders/klinearization_lanczos_windowing_dispersion.comp";
	std::string klinearLanczosWindowingDispersionShaderSource = loadShaderSource(klinearLanczosWindowingDispersionShaderPath);
	std::vector<uint32_t> klinearLanczosWindowingDispersionSPIRV = compileGLSLToSPIRV(klinearLanczosWindowingDispersionShaderSource, klinearLanczosWindowingDispersionShaderPath, shaderc_compute_shader);

	VkShaderModule klinearLanczosWindowingDispersionShader = createShaderModule(this->impl->device, klinearLanczosWindowingDispersionSPIRV);
	this->impl->shaderModules.push_back(klinearLanczosWindowingDispersionShader);

	VkPipelineShaderStageCreateInfo klinearLanczosWindowingDispersionShaderStageInfo = {};
	klinearLanczosWindowingDispersionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	klinearLanczosWindowingDispersionShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	klinearLanczosWindowingDispersionShaderStageInfo.module = klinearLanczosWindowingDispersionShader;
	klinearLanczosWindowingDispersionShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo klinearLanczosWindowingDispersionPipelineInfo = {};
	klinearLanczosWindowingDispersionPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	klinearLanczosWindowingDispersionPipelineInfo.stage = klinearLanczosWindowingDispersionShaderStageInfo;
	klinearLanczosWindowingDispersionPipelineInfo.layout = this->impl->klinearCubicWindowingDispersionPipelineLayout;

	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &klinearLanczosWindowingDispersionPipelineInfo, nullptr, &this->impl->klinearLanczosWindowingDispersionPipeline));
	this->impl->computePipelines.push_back(this->impl->klinearLanczosWindowingDispersionPipeline);

	// Allocate and Update Merged K-Linearization+Windowing+Dispersion Descriptor Sets
	// (Shared by all 3 interpolation variants: cubic, linear, lanczos)
	std::vector<VkDescriptorSetLayout> klinearCubicWindowingDispersionLayouts(this->impl->numCommandBuffers, this->impl->klinearCubicWindowingDispersionDescriptorSetLayout);

	VkDescriptorSetAllocateInfo klinearCubicWindowingDispersionAllocInfo = {};
	klinearCubicWindowingDispersionAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	klinearCubicWindowingDispersionAllocInfo.descriptorPool = this->impl->descriptorPool;
	klinearCubicWindowingDispersionAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	klinearCubicWindowingDispersionAllocInfo.pSetLayouts = klinearCubicWindowingDispersionLayouts.data();

	this->impl->klinearCubicWindowingDispersionDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &klinearCubicWindowingDispersionAllocInfo, this->impl->klinearCubicWindowingDispersionDescriptorSets.data()));

	// Update merged k-linearization+windowing+dispersion descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> klinearCubicWindowingDispersionDescriptorWrites(5);

		// Binding 0: Input buffer (FFT buffer)
		VkDescriptorBufferInfo klinearCubicWindowingDispersionInputInfo = {};
		klinearCubicWindowingDispersionInputInfo.buffer = this->impl->deviceFftBuffer;
		klinearCubicWindowingDispersionInputInfo.offset = 0;
		klinearCubicWindowingDispersionInputInfo.range = VK_WHOLE_SIZE;

		klinearCubicWindowingDispersionDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearCubicWindowingDispersionDescriptorWrites[0].dstSet = this->impl->klinearCubicWindowingDispersionDescriptorSets[i];
		klinearCubicWindowingDispersionDescriptorWrites[0].dstBinding = 0;
		klinearCubicWindowingDispersionDescriptorWrites[0].dstArrayElement = 0;
		klinearCubicWindowingDispersionDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearCubicWindowingDispersionDescriptorWrites[0].descriptorCount = 1;
		klinearCubicWindowingDispersionDescriptorWrites[0].pBufferInfo = &klinearCubicWindowingDispersionInputInfo;

		// Binding 1: Resample curve buffer
		VkDescriptorBufferInfo resampleCurveInfo = {};
		resampleCurveInfo.buffer = this->impl->resampleCurveBuffer;
		resampleCurveInfo.offset = 0;
		resampleCurveInfo.range = VK_WHOLE_SIZE;

		klinearCubicWindowingDispersionDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearCubicWindowingDispersionDescriptorWrites[1].dstSet = this->impl->klinearCubicWindowingDispersionDescriptorSets[i];
		klinearCubicWindowingDispersionDescriptorWrites[1].dstBinding = 1;
		klinearCubicWindowingDispersionDescriptorWrites[1].dstArrayElement = 0;
		klinearCubicWindowingDispersionDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearCubicWindowingDispersionDescriptorWrites[1].descriptorCount = 1;
		klinearCubicWindowingDispersionDescriptorWrites[1].pBufferInfo = &resampleCurveInfo;

		// Binding 2: Window curve buffer
		VkDescriptorBufferInfo windowCurveInfo = {};
		windowCurveInfo.buffer = this->impl->windowCurveBuffer;
		windowCurveInfo.offset = 0;
		windowCurveInfo.range = VK_WHOLE_SIZE;

		klinearCubicWindowingDispersionDescriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearCubicWindowingDispersionDescriptorWrites[2].dstSet = this->impl->klinearCubicWindowingDispersionDescriptorSets[i];
		klinearCubicWindowingDispersionDescriptorWrites[2].dstBinding = 2;
		klinearCubicWindowingDispersionDescriptorWrites[2].dstArrayElement = 0;
		klinearCubicWindowingDispersionDescriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearCubicWindowingDispersionDescriptorWrites[2].descriptorCount = 1;
		klinearCubicWindowingDispersionDescriptorWrites[2].pBufferInfo = &windowCurveInfo;

		// Binding 3: Dispersion phase buffer
		VkDescriptorBufferInfo dispersionPhaseInfo = {};
		dispersionPhaseInfo.buffer = this->impl->dispersionCurveBuffer;
		dispersionPhaseInfo.offset = 0;
		dispersionPhaseInfo.range = VK_WHOLE_SIZE;

		klinearCubicWindowingDispersionDescriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearCubicWindowingDispersionDescriptorWrites[3].dstSet = this->impl->klinearCubicWindowingDispersionDescriptorSets[i];
		klinearCubicWindowingDispersionDescriptorWrites[3].dstBinding = 3;
		klinearCubicWindowingDispersionDescriptorWrites[3].dstArrayElement = 0;
		klinearCubicWindowingDispersionDescriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearCubicWindowingDispersionDescriptorWrites[3].descriptorCount = 1;
		klinearCubicWindowingDispersionDescriptorWrites[3].pBufferInfo = &dispersionPhaseInfo;

		// Binding 4: Output buffer (Intermediate buffer)
		VkDescriptorBufferInfo klinearCubicWindowingDispersionOutputInfo = {};
		klinearCubicWindowingDispersionOutputInfo.buffer = this->impl->deviceIntermediateBuffer;
		klinearCubicWindowingDispersionOutputInfo.offset = 0;
		klinearCubicWindowingDispersionOutputInfo.range = VK_WHOLE_SIZE;

		klinearCubicWindowingDispersionDescriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		klinearCubicWindowingDispersionDescriptorWrites[4].dstSet = this->impl->klinearCubicWindowingDispersionDescriptorSets[i];
		klinearCubicWindowingDispersionDescriptorWrites[4].dstBinding = 4;
		klinearCubicWindowingDispersionDescriptorWrites[4].dstArrayElement = 0;
		klinearCubicWindowingDispersionDescriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		klinearCubicWindowingDispersionDescriptorWrites[4].descriptorCount = 1;
		klinearCubicWindowingDispersionDescriptorWrites[4].pBufferInfo = &klinearCubicWindowingDispersionOutputInfo;

		vkUpdateDescriptorSets(this->impl->device, static_cast<uint32_t>(klinearCubicWindowingDispersionDescriptorWrites.size()), klinearCubicWindowingDispersionDescriptorWrites.data(), 0, nullptr);
	}

	// ============================================
	// ============================================
	// NOTE: K-Linearization+Windowing (no dispersion) merged shader (and similar ones) are not used
	// We only use the full 3-operation merge for now
	// ============================================
	// 	// ============================================


	// ============================================
	// Validate Pipeline Creation
	// ============================================

	// Verify that the correct number of pipelines were created
	// This validation ensures the PipelineIndex enum stays in sync with the actual pipeline creation
	if (this->impl->computePipelines.size() != static_cast<size_t>(Impl::PipelineIndex::Count)) {
		throw std::runtime_error("Pipeline count mismatch in createComputePipelines(): expected " +
			std::to_string(static_cast<size_t>(Impl::PipelineIndex::Count)) +
			" pipelines but created " + std::to_string(this->impl->computePipelines.size()) +
			". The PipelineIndex enum must be updated to match the pipeline creation order.");
	}
}

void VulkanBackend::destroyComputePipelines() {
	// Destroy pipelines
	for (auto& pipeline : this->impl->computePipelines) {
		if (pipeline != VK_NULL_HANDLE) {
			vkDestroyPipeline(this->impl->device, pipeline, nullptr);
		}
	}
	this->impl->computePipelines.clear();

	// Destroy shader modules
	for (auto& shaderModule : this->impl->shaderModules) {
		if (shaderModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(this->impl->device, shaderModule, nullptr);
		}
	}
	this->impl->shaderModules.clear();

	// Destroy descriptor pool
	if (this->impl->descriptorPool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(this->impl->device, this->impl->descriptorPool, nullptr);
		this->impl->descriptorPool = VK_NULL_HANDLE;
	}

	// Destroy descriptor set layout
	if (this->impl->descriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->descriptorSetLayout, nullptr);
		this->impl->descriptorSetLayout = VK_NULL_HANDLE;
	}

	// Destroy pipeline layout
	if (this->impl->pipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->pipelineLayout, nullptr);
		this->impl->pipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy truncate pipeline resources
	if (this->impl->truncateDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->truncateDescriptorSetLayout, nullptr);
		this->impl->truncateDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->truncatePipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->truncatePipelineLayout, nullptr);
		this->impl->truncatePipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy windowing pipeline resources
	if (this->impl->windowingDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->windowingDescriptorSetLayout, nullptr);
		this->impl->windowingDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->windowingPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->windowingPipelineLayout, nullptr);
		this->impl->windowingPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy post-processing pipeline resources
	if (this->impl->postprocessDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->postprocessDescriptorSetLayout, nullptr);
		this->impl->postprocessDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->postprocessPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->postprocessPipelineLayout, nullptr);
		this->impl->postprocessPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy DC removal pipeline resources
	if (this->impl->dcRemovalDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->dcRemovalDescriptorSetLayout, nullptr);
		this->impl->dcRemovalDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->dcRemovalPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->dcRemovalPipelineLayout, nullptr);
		this->impl->dcRemovalPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy K-linearization pipeline resources
	if (this->impl->klinearizationDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->klinearizationDescriptorSetLayout, nullptr);
		this->impl->klinearizationDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->klinearizationPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->klinearizationPipelineLayout, nullptr);
		this->impl->klinearizationPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy dispersion compensation pipeline resources
	if (this->impl->dispersionDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->dispersionDescriptorSetLayout, nullptr);
		this->impl->dispersionDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->dispersionPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->dispersionPipelineLayout, nullptr);
		this->impl->dispersionPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy merged k-linearization+windowing+dispersion pipeline resources
	if (this->impl->klinearCubicWindowingDispersionDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->klinearCubicWindowingDispersionDescriptorSetLayout, nullptr);
		this->impl->klinearCubicWindowingDispersionDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->klinearCubicWindowingDispersionPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->klinearCubicWindowingDispersionPipelineLayout, nullptr);
		this->impl->klinearCubicWindowingDispersionPipelineLayout = VK_NULL_HANDLE;
	}

}

// ============================================
// Static Device Management Methods
// ============================================

std::vector<VulkanDeviceInfo> VulkanBackend::getAvailableDevices() {
	// todo
	return std::vector<VulkanDeviceInfo>();
}

bool VulkanBackend::setDevice(int deviceId) {
	// todo
	return false;
}

int VulkanBackend::getCurrentDevice() {
	// todo
	return 0;
}

bool VulkanBackend::isDeviceAvailable(int deviceId) {
	// todo
	return false;
}

VulkanDeviceInfo VulkanBackend::getDeviceInfo(int deviceId) {
	// todo
	return VulkanDeviceInfo();
}

} // namespace ope
