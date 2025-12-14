#include "vulkan_backend.h"

//#define VULKAN_DEBUG_LOGGING  // Uncomment to enable debug output


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
#include <array>
#include <shaderc/shaderc.hpp>
#include <glslang_c_interface.h>

// Helper macro for checking Vulkan errors
#define checkVulkanErrors(call) \
	do { \
		VkResult err = call; \
		if (err != VK_SUCCESS) { \
			std::stringstream ss; \
			ss << "Vulkan error at " << __FILE__ << ":" << __LINE__ << " - code: " << err; \
			if (err == VK_ERROR_DEVICE_LOST) { \
				std::cerr << "!!! GPU DEVICE LOST (VK_ERROR_DEVICE_LOST) - likely shader OOB/invalid memory access !!!" << std::endl; \
			} \
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

// Unified debug macro (controls logging + validation + debug utils extension)
//#define VULKAN_DEBUG  // Uncomment to enable debug output and Vulkan validation
#ifdef VULKAN_DEBUG
	#include <iostream>
	#define VKDBG_LOG(x) do { std::cout << x << std::endl; } while (0)
#else
	#define VKDBG_LOG(x) do { } while (0)
#endif

namespace ope {

// ============================================
// Static Variables
// ============================================

// glslang is process-wide, not per-instance
static bool s_glslangInitialized = false;
static std::mutex s_glslangMutex;

// ============================================
// Compute Shader Configuration
// ============================================

// ============================================
// Debug Callback for Validation
// ============================================

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData
) {
	(void)messageSeverity;
	(void)messageType;
	(void)pUserData;

	std::cerr << "[Vulkan Validation] " << pCallbackData->pMessage << std::endl;
	std::cerr.flush();
	return VK_FALSE;
}

static VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator,
	VkDebugUtilsMessengerEXT* pMessenger
) {
	auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (!fn) return VK_ERROR_EXTENSION_NOT_PRESENT;
	return fn(instance, pCreateInfo, pAllocator, pMessenger);
}

static void DestroyDebugUtilsMessengerEXT(
	VkInstance instance,
	VkDebugUtilsMessengerEXT messenger,
	const VkAllocationCallbacks* pAllocator
) {
	auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (fn) fn(instance, messenger, pAllocator);
}

// ============================================
// Compute Shader Configuration
// ============================================

// Workgroup size for all compute shaders
// Change this value to test different workgroup sizes (64, 128, 256, 512, etc.)
// Must be a power of 2 and within device limits (typically max 1024)
static constexpr uint32_t VULKAN_WORKGROUP_SIZE = 128;

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
		DcRemoval = 1,             // Rolling average background removal
		FpnDetermination = 2,      // Fixed pattern noise determination
		// NOTE: Universal pre-FFT pipeline variants are stored separately in universalPipelines[] array
		// NOTE: Universal post-FFT pipeline variants are stored separately in universalPostFFTPipelines[] array
		//       (replaces old Truncate + Postprocess pipelines)

		Count = 3  // Total number of pipelines (universal variants not included)
	};

	// Helper function to get pipeline with type-safe indexing
	inline VkPipeline getPipeline(PipelineIndex idx) const {
		return this->computePipelines[static_cast<size_t>(idx)];
	}

	// Configuration
	ProcessorConfiguration config;

	// Vulkan parameters
	int numCommandBuffers = 3;
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
	VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device = VK_NULL_HANDLE;
	VkQueue computeQueue = VK_NULL_HANDLE;
	uint32_t queueFamilyIndex = 0;

	// Transfer queue (for async GPU transfers)
	VkQueue transferQueue = VK_NULL_HANDLE;
	uint32_t transferQueueFamilyIndex = 0;
	VkCommandPool transferCommandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> transferCommandBuffers;
	bool useSeparateTransferQueue = false;
	std::vector<VkSemaphore> transferToComputeSemaphores;  // Signal from H2D transfer, wait on compute

	// D2H Transfer queue (for bidirectional DMA)
	VkQueue d2hTransferQueue = VK_NULL_HANDLE;
	VkCommandPool d2hTransferCommandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> d2hTransferCommandBuffers;
	std::vector<VkSemaphore> computeToD2hSemaphores;  // Signal from compute, wait on D2H transfer
	bool useBidirectionalTransfer = false;  // True if 2 transfer queues available

	// Timeline semaphore for ordered output transfers (ensures buffer N completes before buffer N+1)
	VkSemaphore outputOrderingSemaphore = VK_NULL_HANDLE;
	uint64_t nextOutputSignalValue = 1;  // Monotonically increasing value for timeline semaphore
	std::vector<uint64_t> lastTimelineValuePerCB;  // Last timeline value used by each command buffer
	std::vector<uint64_t> stagingLastWriteValue;  // Track last timeline value that wrote to each staging buffer

	// Debug utils (CRITICAL: gate on debugUtilsEnabled - vkGetDeviceProcAddr can return non-null even when extension disabled)
	bool debugUtilsEnabled = false;  // Set true when VK_EXT_debug_utils is enabled at instance creation
	PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT_fn = nullptr;
	PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT_fn = nullptr;
	PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT_fn = nullptr;

	// Command buffers and synchronization
	VkCommandPool commandPool = VK_NULL_HANDLE;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkFence> fences;
	// Note: No round-robin index needed - zero-copy input determines CB from buffer pointer
	bool commandBuffersValid = false;  // Track if command buffers need re-recording

	// Input buffer management (queue-based, thread-safe)
	// Zero-copy: numInputBuffers == numCommandBuffers (each input buffer is backed by staging)
	int numInputBuffers = numCommandBuffers;
	std::vector<IOBuffer> hostInputBuffers;
	std::queue<IOBuffer*> freeBuffersQueue;
	std::mutex freeQueueMutex;
	std::condition_variable freeQueueCV;

	// Output buffer management
	int numOutputBuffers = 0;  // 0 = auto (numCommandBuffers * 2)

	// Staging buffers (host-visible, one per command buffer)
	std::vector<VkBuffer> stagingInputBuffers;
	std::vector<VkDeviceMemory> stagingInputMemory;
	std::vector<void*> stagingInputMapped;

	std::vector<VkBuffer> stagingOutputBuffers;
	std::vector<VkDeviceMemory> stagingOutputMemory;
	std::vector<void*> stagingOutputMapped;

	// Free staging output buffer pool (decoupled from command buffer slots)
	std::queue<int> freeStagingOutputQueue;
	std::mutex freeStagingOutputMutex;
	std::condition_variable freeStagingOutputCV;
	std::unique_ptr<std::atomic<bool>[]> stagingInUse;  // Double-free guard (atomics can't go in vector)
	int numStagingBuffers = 0;  // Track size of stagingInUse array
	bool stagingOutputIsCoherent = false;  // Track if staging memory has HOST_COHERENT property

	// Device buffers (device-local)
	std::vector<VkBuffer> deviceInputBuffers;
	std::vector<VkDeviceMemory> deviceInputMemory;

	// Per-command-buffer processing buffers (eliminates data races, enables parallel execution)
	std::vector<VkBuffer> deviceFftBuffers;
	std::vector<VkDeviceMemory> deviceFftMemory;

	std::vector<VkBuffer> deviceIntermediateBuffers;  // For preprocessing ping-pong
	std::vector<VkDeviceMemory> deviceIntermediateMemory;

	std::vector<VkBuffer> deviceProcessedBuffers;
	std::vector<VkDeviceMemory> deviceProcessedMemory;

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
	VkBuffer postProcBackgroundStagingBuffer = VK_NULL_HANDLE;  // For copying profile back to host
	VkDeviceMemory postProcBackgroundStagingMemory = VK_NULL_HANDLE;
	void* postProcBackgroundStagingMapped = nullptr;  // Mapped pointer for readback
	bool postProcessBackgroundRecordingRequested = false;
	bool hasValidBackgroundProfile = false;  // Track whether background profile has been set
	std::atomic<bool> needRerecordAfterBgCapture{false};  // Trigger CB re-record after bg capture completes
	std::vector<float> recordedPostProcessBackground;

	// Compute pipelines (will be created later)
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> descriptorSets;

	// DC removal pipeline resources
	VkPipelineLayout dcRemovalPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout dcRemovalDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> dcRemovalDescriptorSets;

	// Universal pre-FFT processing pipeline resources
	VkPipelineLayout universalPreFFTPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout universalPreFFTDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> universalPreFFTDescriptorSets;

	// Universal pipeline variants (with different specialization constants)
	// Linear index: useIntermediateBuffer * 3 + interpolation_method
	//   useIntermediateBuffer: 0=read from fftBuffer, 1=read from intermediateBuffer (after DC removal)
	//   interpolation_method: 0=cubic, 1=linear, 2=lanczos
	// DC removal is now a separate pass
	VkPipeline universalPipelines[6];  // [useIntermediate][interpolation] flattened

	// Universal post-FFT processing pipeline resources
	VkPipelineLayout universalPostFFTPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout universalPostFFTDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<std::array<VkDescriptorSet, 2>> universalPostFFTDescriptorSets;  // [cmdBuf][variant]: 0=FFT input, 1=Intermediate input

	// Universal post-FFT pipeline variants (with different specialization constants)
	// Linear index: enableFixedPatternNoise * 2 + logScaling
	//   enableFixedPatternNoise: 0=disabled, 1=enabled
	//   logScaling: 0=linear, 1=log
	VkPipeline universalPostFFTPipelines[4];  // [FPN][logScaling] flattened

	// Fixed pattern noise determination pipeline resources
	VkPipelineLayout fpnDeterminationPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout fpnDeterminationDescriptorSetLayout = VK_NULL_HANDLE;
	std::vector<std::array<VkDescriptorSet, 2>> fpnDeterminationDescriptorSets;  // [cmdBuf][variant]: 0=FFT input, 1=Intermediate input

	// Background subtraction pipeline resources
	VkPipelineLayout backgroundSubtractionPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout backgroundSubtractionDescriptorSetLayout = VK_NULL_HANDLE;
	VkPipeline backgroundSubtractionPipeline = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> backgroundSubtractionDescriptorSets;

	// Background recording pipeline resources
	VkPipelineLayout backgroundRecordingPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout backgroundRecordingDescriptorSetLayout = VK_NULL_HANDLE;
	VkPipeline backgroundRecordingPipeline = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> backgroundRecordingDescriptorSets;

	// Shader modules (will be created later)
	std::vector<VkShaderModule> shaderModules;
	std::vector<VkPipeline> computePipelines;

	// VkFFT
	VkFFTConfiguration fftConfig;
	std::vector<VkFFTApplication> vkfftApps;  // One per command buffer slot (fixes descriptor invalidation)
	uint64_t fftBufferSize = 0;  // Must persist for VkFFT lifetime
	VkFence fftFence = VK_NULL_HANDLE;  // Dedicated fence for VkFFT operations

	// Output buffers for callback (one per command buffer)
	std::vector<IOBuffer> outputBuffers;

	// Async completion tracking (similar to CUDA stream callbacks)
	struct PendingWork {
		VkFence fence;
		int commandBufferIdx;
		IOBuffer* outputBuffer;
		IOBuffer* inputBuffer;   // Zero-copy: return to free queue after callback
		size_t outputSize;
		int outputSignalLength;
		uint64_t timelineValue;  // Timeline semaphore value to wait for
		uint64_t bufferId;       // Buffer ID to restore before callback (output buffer is shared)
		int stagingBufferIdx;    // Track which staging buffer this work uses (decoupled from commandBufferIdx)
	};
	std::queue<PendingWork> pendingWorkQueue;
	std::mutex pendingWorkMutex;
	std::condition_variable pendingWorkCV;
	std::thread completionThread;
	std::atomic<bool> completionThreadRunning{false};

	// Shutdown synchronization (prevents race conditions during cleanup)
	std::atomic<bool> shuttingDown{false};      // Prevents new work submission during shutdown
	std::atomic<int> pendingWorkCount{0};       // Tracks in-flight work for graceful draining
	std::atomic<bool> cleanupDone{false};       // Idempotency guard: prevents double-destroy crashes
	std::mutex submitMutex;                     // Protects ALL Vulkan handle access and lifetime (critical for shutdown safety)

	// Diagnostic sequence tracking
	std::atomic<uint32_t> diagSeq{0};

	// Callback
	std::function<void(const IOBuffer&)> callback;

	// Input buffers waiting for consumer release (indexed by command buffer index)
	// Used to defer returning input buffers until OutputBufferManager releases output
	std::vector<IOBuffer*> pendingInputBufferRelease;

	Impl() = default;

	~Impl() {
		// Cleanup is handled in cleanup() method
	}

	// Completion thread function (runs asynchronously, similar to CUDA stream callbacks)
	void completionThreadFunc() {
		while (this->completionThreadRunning.load(std::memory_order_acquire)) {
			PendingWork work;
			{
				std::unique_lock<std::mutex> lock(this->pendingWorkMutex);
				// Wait for work or shutdown signal
				this->pendingWorkCV.wait(lock, [this] {
					return !this->pendingWorkQueue.empty() || !this->completionThreadRunning.load(std::memory_order_acquire);
				});

				if (!this->completionThreadRunning.load(std::memory_order_acquire) && this->pendingWorkQueue.empty()) {
					break;  // Shutdown
				}

				if (this->pendingWorkQueue.empty()) {
					continue;  // Spurious wakeup
				}

				work = this->pendingWorkQueue.front();
				this->pendingWorkQueue.pop();
			}

			// Wait for timeline semaphore value (ensures callbacks are invoked in submission order)
			// This replaces fence waiting to avoid fence reuse race conditions
			VkSemaphoreWaitInfo waitInfo = {};
			waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
			waitInfo.pNext = nullptr;
			waitInfo.flags = 0;
			waitInfo.semaphoreCount = 1;
			waitInfo.pSemaphores = &this->outputOrderingSemaphore;
			waitInfo.pValues = &work.timelineValue;

			// Wait for timeline semaphore with TIMEOUT + shutdown check (robust against DEVICE_LOST and logic bugs)
			// Poll with timeout to allow shutdown cancellation
			bool workCompleted = false;
			for (;;) {
				// Check shutdown flag BEFORE waiting
				if (this->shuttingDown.load(std::memory_order_acquire)) {
					// Cancelled during shutdown - clean up work item
					this->pendingWorkCount.fetch_sub(1, std::memory_order_release);
					break;  // Skip to next iteration (thread will exit)
				}

				// Wait with 10ms timeout (in nanoseconds)
				VkResult result = vkWaitSemaphores(this->device, &waitInfo, 10'000'000ULL);

				if (result == VK_SUCCESS) {
					// Normal completion - process callback
					workCompleted = true;
					break;  // Exit wait loop
				} else if (result == VK_TIMEOUT) {
					// Timeout - loop and check shutdown again
					continue;
				} else {
					// Error (VK_ERROR_DEVICE_LOST, etc.)
					if (result == VK_ERROR_DEVICE_LOST) {
						std::cerr << "!!! VK_ERROR_DEVICE_LOST on vkWaitSemaphores !!!" << std::endl;
					}
					std::cerr << "Timeline semaphore wait failed: " << result << std::endl;
					this->pendingWorkCount.fetch_sub(1, std::memory_order_release);
					break;  // Skip callback, continue to next work item
				}
			}

			// Skip callback if cancelled or error
			if (!workCompleted || this->shuttingDown.load(std::memory_order_acquire)) {
				continue;  // Thread will exit on next loop check
			}


			// ============================================
			// Release Input Buffer Immediately
			// ============================================
			// CRITICAL: With decoupled staging buffer pool, input buffer can be released immediately
			// after GPU completion, allowing command buffer slot reuse independent of consumer speed
			{
				std::lock_guard<std::mutex> lock(this->freeQueueMutex);
				this->freeBuffersQueue.push(work.inputBuffer);
			}
			this->freeQueueCV.notify_one();

			// ============================================
			// Memory Coherency (Non-Coherent Memory Support)
			// ============================================
			// If staging memory is not coherent, invalidate mapped memory before consumers read
			if (!this->stagingOutputIsCoherent) {
				VkMappedMemoryRange range = {};
				range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
				range.memory = this->stagingOutputMemory[work.stagingBufferIdx];
				range.offset = 0;
				range.size = work.outputSize;
				vkInvalidateMappedMemoryRanges(this->device, 1, &range);
			}

			// If background recording was requested, copy the recorded profile from staging buffer
			if (this->postProcessBackgroundRecordingRequested) {
				size_t bgProfileSize = work.outputSignalLength;  // Number of floats in background profile
				this->recordedPostProcessBackground.resize(bgProfileSize);
				std::memcpy(this->recordedPostProcessBackground.data(),
				            this->postProcBackgroundStagingMapped,
				            bgProfileSize * sizeof(float));

				// Mark as valid and clear the request flag
				this->hasValidBackgroundProfile = true;
				this->postProcessBackgroundRecordingRequested = false;

				// Trigger re-record of command buffers on next process() call
				// This removes the background recording dispatch (now only subtraction will run)
				this->needRerecordAfterBgCapture.store(true, std::memory_order_release);
			}

			// ============================================
			// Invoke Callback (Output Now Safe to Read)
			// ============================================
			// Restore buffer ID and invoke callback if registered
			// Buffer ID must be restored because output buffer is shared between frames
			work.outputBuffer->setBufferId(work.bufferId);
			if (this->callback) {
				this->callback(*work.outputBuffer);
			}

			// Decrement pending work counter (work completed)
			this->pendingWorkCount.fetch_sub(1, std::memory_order_release);
		}
	}

	// Record all command buffers with current configuration
	// Called when commandBuffersValid is false (first run or after config change)
	void recordAllCommandBuffers();

	// Record D2H transfer command buffer (dynamic, supports decoupled staging buffer pool)
	// Records a single D2H transfer command buffer to copy from deviceProcessedBuffers[commandBufferIdx]
	// to stagingOutputBuffers[stagingBufferIdx]
	void recordD2hTransferCommandBuffer(int commandBufferIdx, int stagingBufferIdx, size_t outputSize);

	// Debug utils helpers
	void initDebugUtils();
	void nameObject(VkObjectType type, uint64_t handle, const char* name);
	bool beginLabel(VkCommandBuffer cmd, const char* text);
	void endLabel(VkCommandBuffer cmd, bool began);

	// Buffer selection helper (ensures VkFFT init and recording use same buffer choice)
	VkBuffer* getFftDataBufferForSlot(int i, const ProcessorConfiguration& cfg);

	// Diagnostic helpers
	void updateDescriptorSetsTagged(const char* tag, uint32_t writeCount, const VkWriteDescriptorSet* writes);
	void logRecordPoint(const char* tag);
	void logPoolOp(const char* op, VkDescriptorPool pool);
};

// ============================================
// Buffer Selection Helper
// ============================================

VkBuffer* VulkanBackend::Impl::getFftDataBufferForSlot(int i, const ProcessorConfiguration& cfg)
{
	// CRITICAL: This must match the actual buffer that FFT reads/writes in recordSingleCommandBuffer()
	// If DC removal is enabled: data is in fftBuffer after preprocessing
	// If DC removal is disabled: data is in intermediateBuffer after preprocessing
	return cfg.processingParams.dcRemoval.enabled ? &this->deviceFftBuffers[i] : &this->deviceIntermediateBuffers[i];
}

// ============================================
// Command Buffer Recording
// ============================================

void VulkanBackend::Impl::recordAllCommandBuffers() {
	vkDeviceWaitIdle(this->device);  // Wait for all GPU work to complete before re-recording

	// Calculate input size (used in all command buffers)
	size_t inputSize = this->samplesPerBuffer * this->bytesPerSample;
	uint32_t numWorkgroups = (this->samplesPerBuffer + VULKAN_WORKGROUP_SIZE - 1) / VULKAN_WORKGROUP_SIZE;

	// Diagnostic lambda for logging descriptor set binds
	auto logBindDS = [&](const char* tag, VkCommandBuffer cmd, VkDescriptorSet set)
	{
		uint32_t seq = this->diagSeq.fetch_add(1);
		VKDBG_LOG("[" << seq << "] [DS BIND] " << tag
			<< " cmd=" << std::hex << (uint64_t)cmd
			<< " set=" << (uint64_t)set
			<< std::dec);
	};

	// Loop through and record ALL command buffers (not just current frame's buffer)
	for (int idx = 0; idx < this->numCommandBuffers; idx++) {
		// ============================================
		// Record Transfer Command Buffer (Input Transfer)
		// ============================================

		VkCommandBuffer transferCmd = this->transferCommandBuffers[idx];

		// Reset command buffer before re-recording (required by Vulkan spec)
		vkResetCommandBuffer(transferCmd, 0);

		VkCommandBufferBeginInfo transferBeginInfo = {};
		transferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		transferBeginInfo.flags = 0;  // Reusable command buffer

		checkVulkanErrors(vkBeginCommandBuffer(transferCmd, &transferBeginInfo));

		// Copy from staging to device buffer
		VkBufferCopy copyRegion = {};
		copyRegion.size = inputSize;

		// Balanced label for H2D copy
		char labelName[128];
		snprintf(labelName, sizeof(labelName),
		         "H2D copy idx=%d dst=deviceInput[%d] src=stagingInput[%d] size=%zu offset=%zu",
		         idx, idx, idx, (size_t)copyRegion.size, (size_t)copyRegion.dstOffset);
		bool began = this->beginLabel(transferCmd, labelName);
		vkCmdCopyBuffer(transferCmd, this->stagingInputBuffers[idx], this->deviceInputBuffers[idx], 1, &copyRegion);
		this->endLabel(transferCmd, began);

		// Release barrier (transfer queue releases ownership to compute queue)
		VkBufferMemoryBarrier releaseBarrier = {};
		releaseBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		releaseBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		releaseBarrier.dstAccessMask = 0;  // No access on transfer queue side
		releaseBarrier.buffer = this->deviceInputBuffers[idx];
		releaseBarrier.offset = 0;
		releaseBarrier.size = inputSize;

		if (this->useSeparateTransferQueue) {
			// Queue family ownership transfer
			releaseBarrier.srcQueueFamilyIndex = this->transferQueueFamilyIndex;
			releaseBarrier.dstQueueFamilyIndex = this->queueFamilyIndex;
		} else {
			// Same queue, no ownership transfer
			releaseBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			releaseBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		}

		vkCmdPipelineBarrier(transferCmd,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		                     0,
		                     0, nullptr,
		                     1, &releaseBarrier,
		                     0, nullptr);

		checkVulkanErrors(vkEndCommandBuffer(transferCmd));

		// ============================================
		// Record Compute Command Buffer (All Compute + Output Transfer)
		// ============================================

		VkCommandBuffer cmd = this->commandBuffers[idx];

		// Reset command buffer before re-recording (required by Vulkan spec)
		vkResetCommandBuffer(cmd, 0);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;  // Reusable command buffer

		checkVulkanErrors(vkBeginCommandBuffer(cmd, &beginInfo));

		// Acquire barrier (compute queue acquires ownership from transfer queue)
		VkBufferMemoryBarrier acquireBarrier = {};
		acquireBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		acquireBarrier.srcAccessMask = 0;  // No access on compute queue side yet
		acquireBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		acquireBarrier.buffer = this->deviceInputBuffers[idx];
		acquireBarrier.offset = 0;
		acquireBarrier.size = inputSize;

		if (this->useSeparateTransferQueue) {
			// Queue family ownership transfer
			acquireBarrier.srcQueueFamilyIndex = this->transferQueueFamilyIndex;
			acquireBarrier.dstQueueFamilyIndex = this->queueFamilyIndex;
		} else {
			// Same queue, no ownership transfer
			acquireBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			acquireBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		}

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0,
		                     0, nullptr,
		                     1, &acquireBarrier,
		                     0, nullptr);

	// ============================================
	// Dispatch Input Conversion Shader
	// ============================================

	// Bind the input conversion pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->getPipeline(Impl::PipelineIndex::InputConversion));

	// Bind descriptor set for this command buffer
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->pipelineLayout,
	                        0, 1, &this->descriptorSets[idx], 0, nullptr);
	logBindDS("InputConversion", cmd, this->descriptorSets[idx]);

	// Push constants: samplesPerBuffer, inputBitDepth, bytesPerSample
	uint32_t pushConstants[3] = {
		static_cast<uint32_t>(this->samplesPerBuffer),
		static_cast<uint32_t>(this->config.dataParams.getBitDepth()),
		static_cast<uint32_t>(this->bytesPerSample)
	};
	vkCmdPushConstants(cmd, this->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(pushConstants), pushConstants);

	// Dispatch compute shader
	vkCmdDispatch(cmd, numWorkgroups, 1, 1);

	// Barrier after input conversion
	VkBufferMemoryBarrier preprocessBarrier = {};
	preprocessBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	preprocessBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	preprocessBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	preprocessBarrier.buffer = this->deviceFftBuffers[idx];
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

	const ProcessorConfiguration& config = this->config;
	bool dataInFftBuffer = true;  // Track final data location

	// Determine which operations are enabled
	bool dcRemoval = config.processingParams.dcRemoval.enabled;
	InterpolationMethod interpMethod = config.processingParams.resampling.method;

	// Step 1: DC Removal (if enabled) - separate pass
	if (dcRemoval) {
		// Bind DC removal pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->getPipeline(Impl::PipelineIndex::DcRemoval));

		// Bind DC removal descriptor set
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->dcRemovalPipelineLayout,
		                        0, 1, &this->dcRemovalDescriptorSets[idx], 0, nullptr);
		logBindDS("DCRemoval", cmd, this->dcRemovalDescriptorSets[idx]);

		// Push constants: rollingAverageWindowSize, signalLength, ascansPerBscan, samplesPerBuffer
		uint32_t dcPushConstants[4] = {
			static_cast<uint32_t>(config.processingParams.dcRemoval.windowSize),
			static_cast<uint32_t>(this->signalLength),
			static_cast<uint32_t>(this->ascansPerBscan),
			static_cast<uint32_t>(this->samplesPerBuffer)
		};
		vkCmdPushConstants(cmd, this->dcRemovalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(dcPushConstants), dcPushConstants);

		// Dispatch DC removal shader
		vkCmdDispatch(cmd, numWorkgroups, 1, 1);

		// Barrier after DC removal
		preprocessBarrier.buffer = this->deviceIntermediateBuffers[idx];
		preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0, 0, nullptr, 1, &preprocessBarrier, 0, nullptr);

		dataInFftBuffer = false;  // DC removal outputs to intermediateBuffer
	}

	// Step 2-4: Use Universal Pre-FFT Shader (K-Linearization + Windowing + Dispersion)
	// This shader combines k-linear + windowing + dispersion in one optimized pass
	// When features are disabled, identity/neutral curves are used (see curve initialization)
	{
		// Select the appropriate pipeline variant
		int useIntermediate = dcRemoval ? 1 : 0;  // Read from intermediateBuffer if DC was applied
		int interpIdx = (interpMethod == InterpolationMethod::CUBIC) ? 0 :
		                (interpMethod == InterpolationMethod::LINEAR) ? 1 : 2;
		int pipelineIdx = useIntermediate * 3 + interpIdx;

		VkPipeline universalPipeline = this->universalPipelines[pipelineIdx];

		// Bind universal pre-FFT pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, universalPipeline);

		// Bind universal pre-FFT descriptor set
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->universalPreFFTPipelineLayout,
		                        0, 1, &this->universalPreFFTDescriptorSets[idx], 0, nullptr);
		logBindDS("UniversalPreFFT", cmd, this->universalPreFFTDescriptorSets[idx]);

		// Push constants: signalLength, samplesPerBuffer
		uint32_t universalPushConstants[2] = {
			static_cast<uint32_t>(this->signalLength),
			static_cast<uint32_t>(this->samplesPerBuffer)
		};
		vkCmdPushConstants(cmd, this->universalPreFFTPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(universalPushConstants), universalPushConstants);

		// Dispatch universal pre-FFT shader
		vkCmdDispatch(cmd, numWorkgroups, 1, 1);

		// Barrier after universal pre-FFT operation (also serves as FFT input barrier)
		// Output buffer depends on input: if reading from intermediate (DC enabled), writes to fft
		preprocessBarrier.buffer = dcRemoval ? this->deviceFftBuffers[idx] : this->deviceIntermediateBuffers[idx];
		preprocessBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		preprocessBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;  // VkFFT needs both
		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0, 0, nullptr, 1, &preprocessBarrier, 0, nullptr);

		// Universal shader output location (ping-pong to avoid read-write hazard):
		// DC enabled: reads from intermediateBuffer, writes to fftBuffer
		// DC disabled: reads from fftBuffer, writes to intermediateBuffer
		dataInFftBuffer = dcRemoval;  // true if DC enabled (data in fftBuffer), false otherwise
	}


	// Select FFT buffer dynamically based on where data is (eliminates unnecessary buffer copy)
	VkBuffer* fftBuffer = dataInFftBuffer ? &this->deviceFftBuffers[idx] : &this->deviceIntermediateBuffers[idx];

	// Note: barrier before FFT is already handled by the universal pre-FFT barrier above
	// (both are on the same buffer, so the second barrier would be redundant)

	// ============================================
	// Execute VkFFT (Inverse FFT)
	// ============================================

	VkFFTLaunchParams fftLaunchParams = {};
	fftLaunchParams.commandBuffer = &cmd;
	// NOTE: Do NOT override buffer here - it's already set correctly during per-slot VkFFT app init
	// The buffer selection logic in getFftDataBufferForSlot() matches the dataInFftBuffer logic above

	checkVkFFTErrors(VkFFTAppend(&this->vkfftApps[idx], 1, &fftLaunchParams));  // +1 = inverse FFT, per-slot app

	// Barrier after FFT (wait for FFT to complete before post-FFT processing)
	VkBufferMemoryBarrier fftOutputBarrier = {};
	fftOutputBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	fftOutputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	fftOutputBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	fftOutputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	fftOutputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	fftOutputBarrier.buffer = *fftBuffer;  // Use same dynamic buffer
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
	// Fixed Pattern Noise Determination (if needed)
	// ============================================

	// Calculate required and available A-scans for FPN determination
	int requiredAscans = this->config.processingParams.fixedPatternNoise.bscanAverageCount * this->ascansPerBscan;
	int availableAscans = this->ascansPerBscan * this->bscansPerBuffer;

	// Warn if FPN is enabled but not enough A-scans available (once per recording)
	static bool fpnWarningShown = false;
	if (this->config.processingParams.fixedPatternNoise.enabled &&
	    !this->fixedPatternNoiseDetermined &&
	    requiredAscans > availableAscans &&
	    !fpnWarningShown) {
		std::cerr << "[VulkanBackend] Warning: FPN determination requires " << requiredAscans
		          << " A-scans but only " << availableAscans << " available per buffer" << std::endl;
		fpnWarningShown = true;
	}

	if (this->config.processingParams.fixedPatternNoise.enabled &&
	    !this->fixedPatternNoiseDetermined &&
	    requiredAscans <= availableAscans) {
		// Dispatch FPN determination shader to compute mean A-line
		// This happens once when FPN is first requested, using the current frame's data

		// Bind FPN determination pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->getPipeline(Impl::PipelineIndex::FpnDetermination));

		// Bind FPN determination descriptor set (select variant based on which buffer was used for FFT)
		int fpnDescriptorVariant = dataInFftBuffer ? 0 : 1;  // 0=FFT buffer, 1=Intermediate buffer
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->fpnDeterminationPipelineLayout,
		                        0, 1, &this->fpnDeterminationDescriptorSets[idx][fpnDescriptorVariant], 0, nullptr);
		logBindDS("FPNDetermination", cmd, this->fpnDeterminationDescriptorSets[idx][fpnDescriptorVariant]);

		// Push constants: width, height, segments, stride, outputSignalLength
		struct FpnDeterminationPushConstants {
			uint32_t width;         // outputSignalLength (samples per A-scan after truncation)
			uint32_t height;        // Number of A-scans to use for FPN (bscanAverageCount * ascansPerBscan)
			uint32_t segments;      // Number of segments for minimum variance calculation
			uint32_t stride;        // fullSignalLength (stride between A-scans in input)
			uint32_t outputSignalLength;  // Same as width
		} fpnPush;

		int outputSignalLength = this->signalLength / 2; // todo: when truncate step becomes optional, outputSignalLength needs to be obtained differently
		fpnPush.width = static_cast<uint32_t>(outputSignalLength);
		fpnPush.height = static_cast<uint32_t>(requiredAscans);  // Use bscanAverageCount * ascansPerBscan (like CUDA/OpenCL)
		fpnPush.segments = 8;  // FIXED_PATTERN_NOISE_REMOVAL_SEGMENTS constant from CUDA
		fpnPush.stride = static_cast<uint32_t>(this->signalLength);
		fpnPush.outputSignalLength = static_cast<uint32_t>(outputSignalLength);

		vkCmdPushConstants(cmd, this->fpnDeterminationPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(fpnPush), &fpnPush);

		// Dispatch FPN determination shader (one thread per sample in the output A-scan)
		uint32_t fpnWorkgroups = (static_cast<uint32_t>(outputSignalLength) + VULKAN_WORKGROUP_SIZE - 1) / VULKAN_WORKGROUP_SIZE;
		vkCmdDispatch(cmd, fpnWorkgroups, 1, 1);

		// Barrier: wait for FPN profile to be written before using it
		VkBufferMemoryBarrier fpnBarrier = {};
		fpnBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		fpnBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		fpnBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		fpnBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		fpnBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		fpnBarrier.buffer = this->meanALineBuffer;
		fpnBarrier.offset = 0;
		fpnBarrier.size = VK_WHOLE_SIZE;

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     0,
		                     0, nullptr,
		                     1, &fpnBarrier,
		                     0, nullptr);

		// Mark as determined for subsequent frames
		this->fixedPatternNoiseDetermined = true;
	}

	// ============================================
	// Dispatch Universal Post-FFT Shader
	// Merges: Fixed Pattern Noise Removal + Magnitude + Log/Linear Scaling + Normalization
	// ============================================

	// Select the appropriate universal post-FFT pipeline variant
	bool fpnEnabled = this->config.processingParams.fixedPatternNoise.enabled && this->fixedPatternNoiseDetermined;
	bool logScaling = this->config.processingParams.intensity.logScale;
	int fpnIdx = fpnEnabled ? 1 : 0;
	int logIdx = logScaling ? 1 : 0;
	int postFFTPipelineIdx = fpnIdx * 2 + logIdx;  // Linear index: 0-3

	VkPipeline universalPostFFTPipeline = this->universalPostFFTPipelines[postFFTPipelineIdx];

	// Bind universal post-FFT pipeline
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, universalPostFFTPipeline);

	// Bind universal post-FFT descriptor set (select variant based on which buffer was used for FFT)
	int postFFTDescriptorVariant = dataInFftBuffer ? 0 : 1;  // 0=FFT buffer, 1=Intermediate buffer
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->universalPostFFTPipelineLayout,
	                        0, 1, &this->universalPostFFTDescriptorSets[idx][postFFTDescriptorVariant], 0, nullptr);
	logBindDS("UniversalPostFFT", cmd, this->universalPostFFTDescriptorSets[idx][postFFTDescriptorVariant]);

	// Push constants: fullSignalLength, outputSignalLength, samplesPerBuffer, grayscaleMax, grayscaleMin, addend, multiplicator
	int outputSignalLength = this->signalLength / 2;
	size_t truncatedSamples = outputSignalLength * this->ascansPerBscan * this->bscansPerBuffer;

	struct UniversalPostFFTPushConstants {
		uint32_t fullSignalLength;
		uint32_t outputSignalLength;
		uint32_t samplesPerBuffer;
		float grayscaleMax;
		float grayscaleMin;
		float addend;
		float multiplicator;
	} universalPostFFTPush;

	universalPostFFTPush.fullSignalLength = static_cast<uint32_t>(this->signalLength);
	universalPostFFTPush.outputSignalLength = static_cast<uint32_t>(outputSignalLength);
	universalPostFFTPush.samplesPerBuffer = static_cast<uint32_t>(this->samplesPerBuffer);
	universalPostFFTPush.grayscaleMax = this->config.processingParams.intensity.rangeMax;
	universalPostFFTPush.grayscaleMin = this->config.processingParams.intensity.rangeMin;
	universalPostFFTPush.addend = this->config.processingParams.intensity.postOffset;
	universalPostFFTPush.multiplicator = this->config.processingParams.intensity.preScale;

	vkCmdPushConstants(cmd, this->universalPostFFTPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
	                   0, sizeof(universalPostFFTPush), &universalPostFFTPush);

	// Dispatch universal post-FFT shader
	uint32_t universalPostFFTWorkgroups = (this->samplesPerBuffer + VULKAN_WORKGROUP_SIZE - 1) / VULKAN_WORKGROUP_SIZE;
	vkCmdDispatch(cmd, universalPostFFTWorkgroups, 1, 1);

	// Barrier after universal post-FFT (wait for writes to complete before next stage)
	VkBufferMemoryBarrier postFFTBarrier = {};
	postFFTBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	postFFTBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	postFFTBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;  // Background subtraction reads and writes
	postFFTBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	postFFTBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	postFFTBarrier.buffer = this->deviceProcessedBuffers[idx];
	postFFTBarrier.offset = 0;
	postFFTBarrier.size = VK_WHOLE_SIZE;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     0,
	                     0, nullptr,
	                     1, &postFFTBarrier,
	                     0, nullptr);

	// ============================================
	// Background Recording (if requested)
	// ============================================

	if (this->postProcessBackgroundRecordingRequested) {
		// Bind background recording pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->backgroundRecordingPipeline);

		// Bind background recording descriptor set
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->backgroundRecordingPipelineLayout,
		                        0, 1, &this->backgroundRecordingDescriptorSets[idx], 0, nullptr);
		logBindDS("BackgroundRecording", cmd, this->backgroundRecordingDescriptorSets[idx]);

		// Push constants: samplesPerAscan, ascansPerBuffer
		struct BackgroundRecordingPushConstants {
			uint32_t samplesPerAscan;
			uint32_t ascansPerBuffer;
		} bgRecPush;

		bgRecPush.samplesPerAscan = static_cast<uint32_t>(outputSignalLength);
		bgRecPush.ascansPerBuffer = static_cast<uint32_t>(this->ascansPerBscan * this->bscansPerBuffer);

		vkCmdPushConstants(cmd, this->backgroundRecordingPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(bgRecPush), &bgRecPush);

		// Dispatch background recording shader (one thread per sample in background profile)
		uint32_t bgRecWorkgroups = (bgRecPush.samplesPerAscan + VULKAN_WORKGROUP_SIZE - 1) / VULKAN_WORKGROUP_SIZE;
		vkCmdDispatch(cmd, bgRecWorkgroups, 1, 1);

		// Barrier after background recording (wait for writes to complete before subtraction or copy)
		VkBufferMemoryBarrier bgRecBarrier = {};
		bgRecBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bgRecBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bgRecBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
		bgRecBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bgRecBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bgRecBarrier.buffer = this->postProcBackgroundBuffer;
		bgRecBarrier.offset = 0;
		bgRecBarrier.size = VK_WHOLE_SIZE;

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     0,
		                     0, nullptr,
		                     1, &bgRecBarrier,
		                     0, nullptr);

		// Copy recorded background profile from device to staging buffer for host readback
		VkBufferCopy bgCopyRegion = {};
		bgCopyRegion.size = static_cast<VkDeviceSize>(outputSignalLength * sizeof(float));

		// Balanced label for background copy
		char bgLabelName[128];
		snprintf(bgLabelName, sizeof(bgLabelName),
		         "Background copy dst=postProcBackgroundStaging src=postProcBackground size=%zu",
		         (size_t)bgCopyRegion.size);
		bool bgBegan = this->beginLabel(cmd, bgLabelName);
		vkCmdCopyBuffer(cmd, this->postProcBackgroundBuffer, this->postProcBackgroundStagingBuffer, 1, &bgCopyRegion);
		this->endLabel(cmd, bgBegan);

		// Barrier after copy (ensure copy completes before host reads)
		VkBufferMemoryBarrier bgCopyBarrier = {};
		bgCopyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bgCopyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		bgCopyBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		bgCopyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bgCopyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bgCopyBarrier.buffer = this->postProcBackgroundStagingBuffer;
		bgCopyBarrier.offset = 0;
		bgCopyBarrier.size = VK_WHOLE_SIZE;

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     VK_PIPELINE_STAGE_HOST_BIT,
		                     0,
		                     0, nullptr,
		                     1, &bgCopyBarrier,
		                     0, nullptr);
	}

	// ============================================
	// Background Subtraction (Post-Processing)
	// ============================================

	// Apply subtraction if we have a valid profile OR if we just recorded one (same behavior as CUDA)
	if (this->config.processingParams.background.enabled &&
	    (this->hasValidBackgroundProfile || this->postProcessBackgroundRecordingRequested)) {
		// Bind background subtraction pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->backgroundSubtractionPipeline);

		// Bind background subtraction descriptor set
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->backgroundSubtractionPipelineLayout,
		                        0, 1, &this->backgroundSubtractionDescriptorSets[idx], 0, nullptr);
		logBindDS("BackgroundSubtraction", cmd, this->backgroundSubtractionDescriptorSets[idx]);

		// Push constants: backgroundWeight, backgroundOffset, samplesPerAscan, samplesPerBuffer
		struct BackgroundSubtractionPushConstants {
			float backgroundWeight;
			float backgroundOffset;
			uint32_t samplesPerAscan;
			uint32_t samplesPerBuffer;
		} bgPush;

		bgPush.backgroundWeight = this->config.processingParams.background.weight;
		bgPush.backgroundOffset = this->config.processingParams.background.offset;
		bgPush.samplesPerAscan = static_cast<uint32_t>(outputSignalLength);
		bgPush.samplesPerBuffer = static_cast<uint32_t>(outputSignalLength * this->ascansPerBscan * this->bscansPerBuffer);

		vkCmdPushConstants(cmd, this->backgroundSubtractionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
		                   0, sizeof(bgPush), &bgPush);

		// Dispatch background subtraction shader
		uint32_t bgWorkgroups = (bgPush.samplesPerBuffer + VULKAN_WORKGROUP_SIZE - 1) / VULKAN_WORKGROUP_SIZE;
		vkCmdDispatch(cmd, bgWorkgroups, 1, 1);
	}

	// Release barrier for D2H on separate queue (shared by both paths)
	// Only set queue family indices when actually transferring between different families
	VkBufferMemoryBarrier d2hReleaseBarrier = {};
	d2hReleaseBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	d2hReleaseBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	d2hReleaseBarrier.dstAccessMask = 0;  // Release barrier
	if (this->useSeparateTransferQueue) {
		d2hReleaseBarrier.srcQueueFamilyIndex = this->queueFamilyIndex;
		d2hReleaseBarrier.dstQueueFamilyIndex = this->transferQueueFamilyIndex;
	} else {
		d2hReleaseBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		d2hReleaseBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	}
	d2hReleaseBarrier.buffer = this->deviceProcessedBuffers[idx];
	d2hReleaseBarrier.offset = 0;
	d2hReleaseBarrier.size = VK_WHOLE_SIZE;

	vkCmdPipelineBarrier(cmd,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
	                     0,
	                     0, nullptr,
	                     1, &d2hReleaseBarrier,
	                     0, nullptr);

	// D2H transfer is now handled dynamically in process() with recordD2hTransferCommandBuffer()

	checkVulkanErrors(vkEndCommandBuffer(cmd));
	}  // End of loop through all command buffers

	// D2H command buffers are now recorded dynamically in process() to support decoupled staging buffer pool

	this->commandBuffersValid = true;
}

// Dynamic D2H command buffer recording (supports decoupled staging buffer pool)
// Records a single D2H transfer command buffer to copy from deviceProcessedBuffers[commandBufferIdx]
// to stagingOutputBuffers[stagingBufferIdx]
void VulkanBackend::Impl::recordD2hTransferCommandBuffer(int commandBufferIdx, int stagingBufferIdx, size_t outputSize) {
	// Bounds check
	size_t outputSignalLength = this->signalLength / 2;
	size_t stagingCapacity = outputSignalLength * this->ascansPerBscan * this->bscansPerBuffer * sizeof(float);
	if (outputSize > stagingCapacity) {
		throw std::runtime_error("Output size exceeds staging buffer capacity");
	}

	// Verify 4-byte alignment (floats are always aligned, but verify anyway)
	if (outputSize % 4 != 0) {
		throw std::runtime_error("Output size not 4-byte aligned: " + std::to_string(outputSize));
	}

	VkCommandBuffer d2hCmd = this->d2hTransferCommandBuffers[commandBufferIdx];

	// Reset command buffer before re-recording (required by Vulkan spec)
	// SAFETY: This is safe because command buffer slot reuse gates ensure the GPU is done with
	// the previous submission before we call this function (timeline semaphore signaled)
	vkResetCommandBuffer(d2hCmd, 0);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  // Re-recorded every time

	checkVulkanErrors(vkBeginCommandBuffer(d2hCmd, &beginInfo));

	// Acquire barrier (for queue family ownership transfer) only if different queue families
	// useSeparateTransferQueue means transfer queue is in a DIFFERENT family from compute
	// useBidirectionalTransfer can be true even with same-family multi-queue (no ownership transfer needed)
	if (this->useSeparateTransferQueue) {
		VkBufferMemoryBarrier acquireBarrier = {};
		acquireBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		acquireBarrier.srcAccessMask = 0;  // Acquire barrier
		acquireBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		acquireBarrier.srcQueueFamilyIndex = this->queueFamilyIndex;  // From compute
		acquireBarrier.dstQueueFamilyIndex = this->transferQueueFamilyIndex;  // To D2H queue
		acquireBarrier.buffer = this->deviceProcessedBuffers[commandBufferIdx];
		acquireBarrier.offset = 0;
		acquireBarrier.size = VK_WHOLE_SIZE;

		vkCmdPipelineBarrier(d2hCmd,
		                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                     VK_PIPELINE_STAGE_TRANSFER_BIT,
		                     0,
		                     0, nullptr,
		                     1, &acquireBarrier,
		                     0, nullptr);
	}

	// D2H copy: device processed buffer → staging output buffer
	// KEY CHANGE: stagingBufferIdx is now decoupled from commandBufferIdx
	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = outputSize;  // Dynamic size (validated above)

	// Balanced label for D2H copy
	char labelName[128];
	snprintf(labelName, sizeof(labelName),
	         "D2H copy cmdIdx=%d staging=%d src=deviceProcessed[%d] size=%zu",
	         commandBufferIdx, stagingBufferIdx, commandBufferIdx, (size_t)copyRegion.size);
	bool began = this->beginLabel(d2hCmd, labelName);
	vkCmdCopyBuffer(d2hCmd,
	                this->deviceProcessedBuffers[commandBufferIdx],  // Source: per-CB device buffer
	                this->stagingOutputBuffers[stagingBufferIdx],    // Dest: from free pool
	                1, &copyRegion);
	this->endLabel(d2hCmd, began);

	checkVulkanErrors(vkEndCommandBuffer(d2hCmd));
}

// ============================================
// Debug Utils Helpers
// ============================================

void VulkanBackend::Impl::initDebugUtils() {
	// CRITICAL: Only load if extension was actually enabled
	if (!this->debugUtilsEnabled)
		return;

	this->vkSetDebugUtilsObjectNameEXT_fn = (PFN_vkSetDebugUtilsObjectNameEXT)
	    vkGetDeviceProcAddr(this->device, "vkSetDebugUtilsObjectNameEXT");
	this->vkCmdBeginDebugUtilsLabelEXT_fn = (PFN_vkCmdBeginDebugUtilsLabelEXT)
	    vkGetDeviceProcAddr(this->device, "vkCmdBeginDebugUtilsLabelEXT");
	this->vkCmdEndDebugUtilsLabelEXT_fn = (PFN_vkCmdEndDebugUtilsLabelEXT)
	    vkGetDeviceProcAddr(this->device, "vkCmdEndDebugUtilsLabelEXT");
}

void VulkanBackend::Impl::nameObject(VkObjectType type, uint64_t handle, const char* name) {
	if (!this->debugUtilsEnabled || !this->vkSetDebugUtilsObjectNameEXT_fn || handle == 0 || !name)
		return;

	VkDebugUtilsObjectNameInfoEXT info{};
	info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
	info.objectType = type;
	info.objectHandle = handle;
	info.pObjectName = name;

	this->vkSetDebugUtilsObjectNameEXT_fn(this->device, &info);
}

// CRITICAL: Balanced labeling - track whether Begin was called
bool VulkanBackend::Impl::beginLabel(VkCommandBuffer cmd, const char* text) {
	if (!this->debugUtilsEnabled || !this->vkCmdBeginDebugUtilsLabelEXT_fn || !cmd || !text)
		return false;

	VkDebugUtilsLabelEXT label{};
	label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pLabelName = text;

	this->vkCmdBeginDebugUtilsLabelEXT_fn(cmd, &label);
	return true;
}

void VulkanBackend::Impl::endLabel(VkCommandBuffer cmd, bool began) {
	if (!began)
		return;
	if (!this->debugUtilsEnabled || !this->vkCmdEndDebugUtilsLabelEXT_fn || !cmd)
		return;

	this->vkCmdEndDebugUtilsLabelEXT_fn(cmd);
}

// ============================================
// Diagnostic Helpers
// ============================================

void VulkanBackend::Impl::updateDescriptorSetsTagged(
	const char* tag,
	uint32_t writeCount,
	const VkWriteDescriptorSet* writes)
{
#ifdef VULKAN_DEBUG_LOGGING
	uint32_t seq = diagSeq.fetch_add(1);
	std::cout << "[" << seq << "] [DS UPDATE] " << tag
		<< " writeCount=" << writeCount;

	for (uint32_t i = 0; i < writeCount; ++i)
	{
		std::cout << " set[" << i << "]="
			<< std::hex << (uint64_t)writes[i].dstSet << std::dec;
	}

	std::cout << std::endl;
#endif
	vkUpdateDescriptorSets(device, writeCount, writes, 0, nullptr);
}

void VulkanBackend::Impl::logRecordPoint(const char* tag)
{
	uint32_t seq = diagSeq.fetch_add(1);
	VKDBG_LOG("[" << seq << "] [RECORD] " << tag);
}

void VulkanBackend::Impl::logPoolOp(const char* op, VkDescriptorPool pool)
{
	uint32_t seq = diagSeq.fetch_add(1);
	VKDBG_LOG("[" << seq << "] [POOL " << op << "] "
		<< std::hex << (uint64_t)pool << std::dec);
}

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

void VulkanBackend::setNumOutputBuffers(int count) {
	if (this->impl->vulkanInitialized) {
		throw std::runtime_error("Cannot change number of output buffers after initialization");
	}
	if (count < 0) {
		throw std::invalid_argument("Number of output buffers must be >= 0 (0 = auto)");
	}
	this->impl->numOutputBuffers = count;
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
	appInfo.apiVersion = VK_API_VERSION_1_2;  // Use Vulkan 1.2 for timeline semaphores

	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pApplicationInfo = &appInfo;

	// Conditional validation layer and debug utils extension setup
	std::vector<const char*> layers;
	std::vector<const char*> exts;
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
	VkValidationFeaturesEXT validationFeatures{};
	VkValidationFeatureEnableEXT enables[] = {
		VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
		VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT
	};
	bool willEnableValidation = false;

#ifdef VULKAN_DEBUG
	// Enumerate available layers
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		// Check if validation layer is available
		bool hasValidation = false;
		for (const auto& layer : availableLayers) {
			if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
				hasValidation = true;
				break;
			}
		}

		if (hasValidation) {
			layers.push_back("VK_LAYER_KHRONOS_validation");
		}

		// Enumerate and check debug utils extension
		uint32_t extCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
		std::vector<VkExtensionProperties> availableExts(extCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availableExts.data());

		bool hasDebugUtils = false;
		for (const auto& ext : availableExts) {
			if (strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
				hasDebugUtils = true;
				break;
			}
		}

		if (hasDebugUtils) {
			exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		// Only set up debug structs if we have both validation layer and debug utils
		willEnableValidation = hasValidation && hasDebugUtils;

		if (willEnableValidation) {
			// Setup debug messenger (will catch messages during vkCreateInstance too)
			debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			debugCreateInfo.messageSeverity =
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			debugCreateInfo.messageType =
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			debugCreateInfo.pfnUserCallback = debugCallback;

			// Enable synchronization validation (critical for catching semaphore/queue bugs!)
			validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
			validationFeatures.enabledValidationFeatureCount = 2;
			validationFeatures.pEnabledValidationFeatures = enables;

			// Chain pNext so we catch messages during vkCreateInstance
			validationFeatures.pNext = &debugCreateInfo;
			instanceCreateInfo.pNext = &validationFeatures;
		} else {
			instanceCreateInfo.pNext = nullptr;
			if (!hasValidation) {
				std::cerr << "WARNING: Vulkan validation layer not available - running without validation" << std::endl;
			}
			if (!hasDebugUtils) {
				std::cerr << "WARNING: Vulkan debug utils extension not available" << std::endl;
			}
		}
#else
	instanceCreateInfo.pNext = nullptr;
#endif

	instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
	instanceCreateInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
	instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(exts.size());
	instanceCreateInfo.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

	// Create instance
	checkVulkanErrors(vkCreateInstance(&instanceCreateInfo, nullptr, &this->impl->instance));

	// Create debug messenger if validation is enabled
#ifdef VULKAN_DEBUG
	if (willEnableValidation) {
			VkResult messengerResult = CreateDebugUtilsMessengerEXT(this->impl->instance, &debugCreateInfo, nullptr, &this->impl->debugMessenger);
			if (messengerResult == VK_SUCCESS) {
				VKDBG_LOG("Vulkan validation layers + sync validation enabled");
				this->impl->debugUtilsEnabled = true;  // Enable debug utils naming and labeling
			} else {
				VKDBG_LOG("Vulkan validation enabled (messenger creation failed)");
			}
		}
#endif

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

	// Find compute and transfer queue families
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(this->impl->physicalDevice, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(this->impl->physicalDevice, &queueFamilyCount, queueFamilies.data());

	uint32_t computeFamily = UINT32_MAX;
	uint32_t transferFamily = UINT32_MAX;

	// Look for dedicated transfer queue (TRANSFER but not COMPUTE/GRAPHICS)
	// This is rare but exists on some GPUs (e.g., NVIDIA has dedicated DMA queue)
	for (uint32_t i = 0; i < queueFamilyCount; i++) {
		VkQueueFlags flags = queueFamilies[i].queueFlags;

		if ((flags & VK_QUEUE_TRANSFER_BIT) &&
			!(flags & VK_QUEUE_COMPUTE_BIT) &&
			!(flags & VK_QUEUE_GRAPHICS_BIT)) {
			transferFamily = i;
		}

		if (flags & VK_QUEUE_COMPUTE_BIT) {
			computeFamily = i;
		}
	}

	if (computeFamily == UINT32_MAX) {
		throw std::runtime_error("No compute queue family found");
	}

	this->impl->queueFamilyIndex = computeFamily;
	this->impl->transferQueueFamilyIndex = (transferFamily != UINT32_MAX) ? transferFamily : computeFamily;
	this->impl->useSeparateTransferQueue = (transferFamily != UINT32_MAX);

	// Determine how many queues are available for parallelism
	uint32_t computeFamilyQueueCount = queueFamilies[computeFamily].queueCount;
	uint32_t transferFamilyQueueCount = (transferFamily != UINT32_MAX) ? queueFamilies[transferFamily].queueCount : 0;

	// Check if bidirectional DMA is available:
	// Option 1: Dedicated transfer family with 2+ queues
	// Option 2: Compute family with 3+ queues (compute + H2D + D2H)
	// Option 3: Compute family with 2+ queues (compute + shared transfer)
	if (this->impl->useSeparateTransferQueue && transferFamilyQueueCount >= 2) {
		this->impl->useBidirectionalTransfer = true;
	} else if (!this->impl->useSeparateTransferQueue && computeFamilyQueueCount >= 2) {
		// Use multiple queues from compute family for transfer parallelism
		this->impl->useBidirectionalTransfer = (computeFamilyQueueCount >= 3);
	} else {
		this->impl->useBidirectionalTransfer = false;
	}

	// Create logical device
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::vector<float> multiQueuePriorities = {1.0f, 1.0f, 1.0f};  // Up to 3 queues

	// Compute queue(s) - request multiple if no separate transfer family
	VkDeviceQueueCreateInfo computeQueueInfo = {};
	computeQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	computeQueueInfo.queueFamilyIndex = this->impl->queueFamilyIndex;
	computeQueueInfo.pQueuePriorities = multiQueuePriorities.data();

	if (this->impl->useSeparateTransferQueue) {
		// Dedicated transfer family exists - only need 1 compute queue
		computeQueueInfo.queueCount = 1;
	} else {
		// No dedicated transfer family - request multiple queues from compute family
		// Request up to 3 queues: compute, H2D transfer, D2H transfer
		computeQueueInfo.queueCount = std::min(computeFamilyQueueCount, 3u);
	}
	queueCreateInfos.push_back(computeQueueInfo);

	// Transfer queue(s) (only if different family)
	if (this->impl->useSeparateTransferQueue) {
		VkDeviceQueueCreateInfo transferQueueInfo = {};
		transferQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		transferQueueInfo.queueFamilyIndex = this->impl->transferQueueFamilyIndex;
		transferQueueInfo.pQueuePriorities = multiQueuePriorities.data();
		// Request up to 2 queues for bidirectional DMA (H2D + D2H)
		transferQueueInfo.queueCount = std::min(transferFamilyQueueCount, 2u);
		queueCreateInfos.push_back(transferQueueInfo);
	}

	// Query Vulkan 1.2 features and enable timeline semaphores
	VkPhysicalDeviceVulkan12Features features12 = {};
	features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

	VkPhysicalDeviceFeatures2 features2 = {};
	features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features2.pNext = &features12;

	vkGetPhysicalDeviceFeatures2(this->impl->physicalDevice, &features2);

	if (!features12.timelineSemaphore) {
		throw std::runtime_error("Device does not support timeline semaphores (Vulkan 1.2 required)");
	}

	// Enable timeline semaphores
	features12.timelineSemaphore = VK_TRUE;

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext = &features12;
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	deviceCreateInfo.pEnabledFeatures = &features2.features;

	checkVulkanErrors(vkCreateDevice(this->impl->physicalDevice, &deviceCreateInfo, nullptr, &this->impl->device));

	// Initialize debug utils function pointers (if enabled)
	this->impl->initDebugUtils();

	// Get queues. strategy depends on whether we have a dedicated transfer family
	if (this->impl->useSeparateTransferQueue) {
		// Dedicated transfer family exists
		vkGetDeviceQueue(this->impl->device, this->impl->queueFamilyIndex, 0, &this->impl->computeQueue);
		vkGetDeviceQueue(this->impl->device, this->impl->transferQueueFamilyIndex, 0, &this->impl->transferQueue);
		
		if (this->impl->useBidirectionalTransfer) {
			vkGetDeviceQueue(this->impl->device, this->impl->transferQueueFamilyIndex, 1, &this->impl->d2hTransferQueue);
		//	std::cout << "[VulkanBackend] Bidirectional DMA: dedicated transfer family with 2 queues" << std::endl;
		} else {
			this->impl->d2hTransferQueue = this->impl->transferQueue;
		//	std::cout << "[VulkanBackend] DMA: dedicated transfer family with 1 queue" << std::endl;
		}
	} else {
		// No dedicated transfer family. use multiple queues from compute family
		vkGetDeviceQueue(this->impl->device, this->impl->queueFamilyIndex, 0, &this->impl->computeQueue);

		if (computeFamilyQueueCount >= 2) {
			// Use queue 1 for H2D transfers (separate from compute)
			vkGetDeviceQueue(this->impl->device, this->impl->queueFamilyIndex, 1, &this->impl->transferQueue);

			if (computeFamilyQueueCount >= 3) {
				// Use queue 2 for D2H transfers (full bidirectional)
				vkGetDeviceQueue(this->impl->device, this->impl->queueFamilyIndex, 2, &this->impl->d2hTransferQueue);
				VKDBG_LOG("[VulkanBackend] Bidirectional DMA: 3 queues from compute family (compute + H2D + D2H)");
			} else {
				// Only 2 queues - share queue 1 for both H2D and D2H
				this->impl->d2hTransferQueue = this->impl->transferQueue;
				VKDBG_LOG("[VulkanBackend] DMA: 2 queues from compute family (compute + transfers)");
			}
		} else {
			// Only 1 queue - everything serialized (worst case)
			this->impl->transferQueue = this->impl->computeQueue;
			this->impl->d2hTransferQueue = this->impl->computeQueue;
			VKDBG_LOG("[VulkanBackend] Warning: Only 1 queue available - all operations serialized");
		}
	}

	// Create compute command pool
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = this->impl->queueFamilyIndex;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	checkVulkanErrors(vkCreateCommandPool(this->impl->device, &poolInfo, nullptr, &this->impl->commandPool));

	// Create transfer command pool
	VkCommandPoolCreateInfo transferPoolInfo = {};
	transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	transferPoolInfo.queueFamilyIndex = this->impl->transferQueueFamilyIndex;
	transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	checkVulkanErrors(vkCreateCommandPool(this->impl->device, &transferPoolInfo, nullptr, &this->impl->transferCommandPool));

	// Create D2H transfer command pool
	// D2H uses transfer family if available, otherwise compute family (same as H2D in that case)
	VkCommandPoolCreateInfo d2hPoolInfo = {};
	d2hPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	d2hPoolInfo.queueFamilyIndex = this->impl->transferQueueFamilyIndex;
	d2hPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	checkVulkanErrors(vkCreateCommandPool(this->impl->device, &d2hPoolInfo, nullptr, &this->impl->d2hTransferCommandPool));

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

	// Provide FFT buffer for VkFFT initialization (actual buffer selected at launch time for per-CB execution)
	this->impl->fftConfig.buffer = &this->impl->deviceFftBuffers[0];
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

	// Target threads per block for VkFFT FFT kernel
	this->impl->fftConfig.aimThreads = VULKAN_WORKGROUP_SIZE;

	// Number of shared memory banks (NVIDIA has 32)
	this->impl->fftConfig.numSharedBanks = 32;

	// Bandwidth boost optimization //todo: test if different value has impact on performance
	this->impl->fftConfig.performBandwidthBoost = 0;

	// Initialize VkFFT apps (one per command buffer slot to avoid descriptor invalidation)
	this->impl->vkfftApps.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i)
	{
		// Use helper to get correct buffer (ensures consistency with recording)
		VkBuffer* fftBuffer = this->impl->getFftDataBufferForSlot(i, this->impl->config);

		// Create VkFFT config for this slot
		VkFFTConfiguration fftCfg = this->impl->fftConfig;  // Copy base config
		fftCfg.buffer = fftBuffer;  // Set per-slot buffer

		// Zero-initialize the app structure before passing to VkFFT
		memset(&this->impl->vkfftApps[i], 0, sizeof(VkFFTApplication));

		checkVkFFTErrors(initializeVkFFT(&this->impl->vkfftApps[i], fftCfg));
	}

	// Initialize curve buffers with identity values (required for universal shader)
	// IMPORTANT: Must be done BEFORE createComputePipelines() so descriptor sets can reference these buffers
	// Resampling identity curve: [0.0, 1.0, 2.0, ..., signalLength-1]
	std::vector<float> identityResampleCurve(this->impl->signalLength);
	for (int i = 0; i < this->impl->signalLength; ++i) {
		identityResampleCurve[i] = static_cast<float>(i);
	}
	this->updateResamplingCurve(identityResampleCurve.data(), identityResampleCurve.size());

	// Windowing identity curve: all 1.0
	std::vector<float> identityWindowCurve(this->impl->signalLength, 1.0f);
	this->updateWindowCurve(identityWindowCurve.data(), identityWindowCurve.size());

	// Dispersion identity phase: all (1.0, 0.0) - identity for complex multiplication
	std::vector<float> identityDispersionCurve(this->impl->signalLength * 2);
	for (int i = 0; i < this->impl->signalLength; ++i) {
		identityDispersionCurve[i * 2 + 0] = 1.0f;  // Real part
		identityDispersionCurve[i * 2 + 1] = 0.0f;  // Imaginary part
	}
	this->updateDispersionCurve(identityDispersionCurve.data(), identityDispersionCurve.size());

	// Create compute pipelines and shaders (AFTER VkFFT so it gets its descriptor pool first)
	// AND after curve buffers are created so descriptor sets can reference them
	this->createComputePipelines();

	// Setup input buffers using zero-copy (direct access to staging buffers)
	// Each IOBuffer points directly to stagingInputMapped memory, eliminating memcpy in process()
	size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
	this->impl->hostInputBuffers.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		this->impl->hostInputBuffers[i].setBackendIndex(i);
		this->impl->hostInputBuffers[i].setDataType(config.dataParams.inputDataType);
		this->impl->hostInputBuffers[i].setExternalMemory(this->impl->stagingInputMapped[i], inputSize);
		this->impl->freeBuffersQueue.push(&this->impl->hostInputBuffers[i]);
	}

	// Allocate output buffers (decoupled from command buffer slots)
	// Output is truncated to half signal length after FFT
	int outputSignalLength = this->impl->signalLength / 2;
	size_t outputSamplesPerBuffer = outputSignalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer;
	size_t outputSize = outputSamplesPerBuffer * sizeof(float);  // Output is always float

	// Determine actual number of output buffers (like CUDA backend)
	int actualNumOutputBuffers = (this->impl->numOutputBuffers > 0)
		? this->impl->numOutputBuffers
		: (this->impl->numCommandBuffers * 2);

	this->impl->outputBuffers.resize(actualNumOutputBuffers);
	this->impl->pendingInputBufferRelease.resize(actualNumOutputBuffers, nullptr);

	for (int i = 0; i < actualNumOutputBuffers; ++i) {
		this->impl->outputBuffers[i].setBackendIndex(i);  // i = stagingBufferIdx (NOT commandBufferIdx)
		this->impl->outputBuffers[i].setDataType(IOBuffer::DataType::FLOAT32);
		// Memory will be set dynamically in process() after acquiring staging buffer
	}

	// Pre-record all command buffers for maximum performance
	this->impl->logRecordPoint("from_initialize");
	this->recordCommandBuffers();

	// Load recorded profiles from configuration (if backend was switched)
	if (config.hasCustomPostProcessBackgroundProfile()) {
		const std::vector<float>& profileVec = config.getBackgroundProfile();
		// Use setPostProcessBackgroundProfile to upload and configure
		this->setPostProcessBackgroundProfile(profileVec.data(), profileVec.size());
	}

	if (config.hasCustomFixedPatternNoiseProfile()) {
		const std::vector<float>& profileVec = config.getFixedPatternNoiseProfile();
		size_t complexPairs = profileVec.size() / 2;
		// Use setFixedPatternNoiseProfile to upload and configure
		this->setFixedPatternNoiseProfile(profileVec.data(), complexPairs);
	}

	// Start async completion thread (handles fence polling and callbacks)
	this->impl->completionThreadRunning = true;
	this->impl->completionThread = std::thread(&Impl::completionThreadFunc, this->impl.get());

	this->impl->vulkanInitialized = true;
}

void VulkanBackend::cleanup() {
	// Atomic exchange: returns OLD value and sets to true
	// If OLD was true, cleanup already ran -> return immediately
	// IMPORTANT: Guard BEFORE taking any mutex so second call returns immediately
	if (this->impl->cleanupDone.exchange(true, std::memory_order_acq_rel)) {
		return;
	}

	VKDBG_LOG("[CLEANUP] Starting cleanup...");
	if (!this->impl->vulkanInitialized) {
		return;
	}

	// Signal shutdown and wake ALL blocked threads
	VKDBG_LOG("[CLEANUP] Setting shuttingDown flag, pendingWorkCount=" << this->impl->pendingWorkCount.load());
	this->impl->shuttingDown.store(true, std::memory_order_release);
	this->impl->completionThreadRunning.store(false, std::memory_order_release);  // Prevent new waits
	this->impl->freeQueueCV.notify_all();      // Wake threads in getNextAvailableInputBuffer()
	this->impl->pendingWorkCV.notify_all();    // Wake completion thread

	// Lock submitMutex and wait for GPU (BEFORE joining thread!)
	VKDBG_LOG("[CLEANUP] Acquiring submitMutex and waiting for GPU...");
	{
		std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

		// Wait for GPU to finish all submitted work (signals all timeline values)
		VkResult waitResult = vkDeviceWaitIdle(this->impl->device);
		if (waitResult != VK_SUCCESS) {
			std::cerr << "[CLEANUP] vkDeviceWaitIdle failed with error " << waitResult;
			if (waitResult == VK_ERROR_DEVICE_LOST) {
				std::cerr << " (VK_ERROR_DEVICE_LOST - GPU crashed)";
			}
			std::cerr << " - continuing with cleanup..." << std::endl;
			std::cerr.flush();
			// Continue with cleanup anyway. must still clean up CPU resources
		}
	}  // Release submitMutex - allows completion thread to proceed

	// NOW stop completion thread safely
	// All timeline values are signaled, OR thread will exit via timeout + shuttingDown check
	VKDBG_LOG("[CLEANUP] Stopping completion thread...");
	if (this->impl->completionThread.joinable()) {
		this->impl->pendingWorkCV.notify_all();  // Final wake
		this->impl->completionThread.join();     // Won't hang (timeout-based wait)
	}

	// Drain pending work and destroy resources
	VKDBG_LOG("[CLEANUP] Draining pending work and destroying resources...");
	{
		std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

		// Drain pending work queue
		// After vkDeviceWaitIdle(), GPU work is complete. We can now safely:
		// 1. Return input buffers to free queue (no longer needed by GPU)
		// 2. Cancel work deterministically (completion thread already stopped)
		{
			std::lock_guard<std::mutex> workLock(this->impl->pendingWorkMutex);
			while (!this->impl->pendingWorkQueue.empty()) {
				VulkanBackend::Impl::PendingWork work = this->impl->pendingWorkQueue.front();
				this->impl->pendingWorkQueue.pop();

				// Return input buffer to free queue (cancel work)
				// Note: Don't fire callback during shutdown. output may be incomplete/invalid
				if (work.inputBuffer) {
					std::lock_guard<std::mutex> freeLock(this->impl->freeQueueMutex);
					this->impl->freeBuffersQueue.push(work.inputBuffer);
					this->impl->freeQueueCV.notify_one();
				}

				// Decrement pending work counter
				this->impl->pendingWorkCount.fetch_sub(1, std::memory_order_release);
			}
		}

		// Optional diagnostic: Check pendingWorkCount drained
		if (this->impl->pendingWorkCount.load(std::memory_order_acquire) != 0) {
			std::cerr << "WARNING: Cleanup completed with pendingWorkCount = "
			          << this->impl->pendingWorkCount.load() << std::endl;
			std::cerr.flush();
		}

		// Destroy resources while still holding submitMutex
		// This prevents any thread from entering process() and touching Vulkan handles during destruction
		VKDBG_LOG("[CLEANUP] Destroying Vulkan resources...");

		// Destroy VkFFT apps (one per command buffer slot)
		for (size_t i = 0; i < this->impl->vkfftApps.size(); ++i)
		{
			deleteVkFFT(&this->impl->vkfftApps[i]);
		}
		this->impl->vkfftApps.clear();

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

		// Destroy compute command pool
		if (this->impl->commandPool != VK_NULL_HANDLE) {
			vkDestroyCommandPool(this->impl->device, this->impl->commandPool, nullptr);
			this->impl->commandPool = VK_NULL_HANDLE;
		}

		// Destroy transfer command pool
		if (this->impl->transferCommandPool != VK_NULL_HANDLE) {
			vkDestroyCommandPool(this->impl->device, this->impl->transferCommandPool, nullptr);
			this->impl->transferCommandPool = VK_NULL_HANDLE;
		}

		// Destroy D2H transfer command pool
		if (this->impl->d2hTransferCommandPool != VK_NULL_HANDLE) {
			vkDestroyCommandPool(this->impl->device, this->impl->d2hTransferCommandPool, nullptr);
			this->impl->d2hTransferCommandPool = VK_NULL_HANDLE;
		}

		// Destroy device
		if (this->impl->device != VK_NULL_HANDLE) {
			vkDestroyDevice(this->impl->device, nullptr);
			this->impl->device = VK_NULL_HANDLE;
		}

		// Destroy debug messenger (only if debug mode enabled)
#ifdef VULKAN_DEBUG
		if (this->impl->debugMessenger != VK_NULL_HANDLE) {
				DestroyDebugUtilsMessengerEXT(this->impl->instance, this->impl->debugMessenger, nullptr);
				this->impl->debugMessenger = VK_NULL_HANDLE;
			}
#endif

		// Destroy instance
		if (this->impl->instance != VK_NULL_HANDLE) {
			vkDestroyInstance(this->impl->instance, nullptr);
			this->impl->instance = VK_NULL_HANDLE;
		}

		// Set vulkanInitialized to false while still holding lock (part of lifetime domain - Round 5)
		this->impl->vulkanInitialized = false;

	}  // submitMutex released here - all resources destroyed

	VKDBG_LOG("[CLEANUP] Cleanup complete.");
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

	// Early check (before acquiring expensive lock)
	if (this->impl->shuttingDown.load(std::memory_order_acquire)) {
		throw std::runtime_error("Backend is shutting down");
	}

	// ============================================
	// CRITICAL: Acquire submitMutex BEFORE accessing ANY Vulkan handles
	// This lock protects ALL Vulkan handle usage, not just submissions
	// ============================================
	std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

	// CRITICAL: Re-check shuttingDown AND vulkanInitialized AFTER lock (prevents TOCTOU race)
	if (this->impl->shuttingDown.load(std::memory_order_acquire)) {
		throw std::runtime_error("Backend is shutting down");
	}
	if (!this->impl->vulkanInitialized) {
		throw std::runtime_error("Backend was cleaned up during lock acquisition");
	}

	// Now safe to access Vulkan handles - cleanup() holds submitMutex during destruction

	// Get the backend index that was set during buffer initialization
	// Each input buffer has a fixed relationship to a command buffer
	int idx = input.getBackendIndex();

	VkCommandBuffer cmd = this->impl->commandBuffers[idx];
	VkFence fence = this->impl->fences[idx];

	// Zero-copy: Synchronization (fence wait + reset, timeline semaphore wait)
	// is done in getNextAvailableInputBuffer() before user writes to buffer.
	// User's data is already in staging buffer - no memcpy needed.

	// ============================================
	// Acquire Free Staging Output Buffer
	// ============================================
	// Wait for a free staging buffer from the decoupled pool (like CUDA backend)
	int stagingBufferIdx;
	{
		std::unique_lock<std::mutex> lock(this->impl->freeStagingOutputMutex);
		this->impl->freeStagingOutputCV.wait(lock, [this]() {
			return this->impl->shuttingDown.load(std::memory_order_acquire) ||
			       !this->impl->freeStagingOutputQueue.empty();
		});

		if (this->impl->shuttingDown.load(std::memory_order_acquire)) {
			throw std::runtime_error("Backend shutting down");
		}

		stagingBufferIdx = this->impl->freeStagingOutputQueue.front();
		this->impl->freeStagingOutputQueue.pop();

		// Mark as in-use (double-acquire guard)
		if (this->impl->stagingInUse[stagingBufferIdx].exchange(true, std::memory_order_acq_rel)) {
			throw std::runtime_error("Double-acquire of staging buffer " + std::to_string(stagingBufferIdx));
		}
	}

	// Get output buffer and configure it to point to acquired staging memory
	// Note: Don't set bufferId here - it would race with consumer threads reading it
	// The completion thread restores bufferId from work.bufferId before callback
	IOBuffer* outputBuf = &this->impl->outputBuffers[stagingBufferIdx];
	int outputSignalLength = this->impl->signalLength / 2;
	size_t outputSize = outputSignalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer * sizeof(float);
	outputBuf->setBackendIndex(stagingBufferIdx);  // Ensure index matches staging buffer
	outputBuf->setExternalMemory(this->impl->stagingOutputMapped[stagingBufferIdx], outputSize);

	// Record all command buffers if needed (first call, after config change, or after bg capture)
	// Re-recording and submission are both protected by submitMutex
	bool needRerecord = this->impl->needRerecordAfterBgCapture.exchange(false, std::memory_order_acq_rel);
	if (!this->impl->commandBuffersValid || needRerecord) {
		this->impl->recordAllCommandBuffers();
		cmd = this->impl->commandBuffers[idx];  // Restore cmd to current frame buffer
	}

	// ============================================
	// Submit Transfer and Compute Command Buffers
	// ============================================

	// Submit transfer command buffer first (signals semaphore when complete)
	VkCommandBuffer transferCmd = this->impl->transferCommandBuffers[idx];
	VkSemaphore transferCompleteSemaphore = this->impl->transferToComputeSemaphores[idx];

	// Diagnostic: Check if we're about to submit H2D before previous slot completed
	uint64_t slotWaitValue = this->impl->stagingLastWriteValue[idx];
#ifdef VULKAN_DEBUG
	uint64_t currentTimelineValue = 0;
	vkGetSemaphoreCounterValue(this->impl->device, this->impl->outputOrderingSemaphore, &currentTimelineValue);

	VKDBG_LOG("[H2D DIAGNOSTIC] idx=" << idx
	          << " slotWaitValue=" << slotWaitValue
	          << " currentTimeline=" << currentTimelineValue
	          << " needWait=" << (slotWaitValue > 0)
	          << " SAFE=" << (currentTimelineValue >= slotWaitValue || slotWaitValue == 0));
#endif

	// Wait on previous "slot complete" value for this idx (recorded when D2H for this idx was submitted)
	bool needSlotWait = (slotWaitValue > 0);

	VKDBG_LOG("[H2D DEBUG] idx=" << idx
	          << " slotWaitValue=" << slotWaitValue
	          << " needWait=" << needSlotWait);

	VkTimelineSemaphoreSubmitInfo h2dTimelineInfo{};
	VkSubmitInfo transferSubmit{};
	transferSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore h2dWaitSemaphores[1] = { this->impl->outputOrderingSemaphore };
	VkPipelineStageFlags h2dWaitStages[1] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
	uint64_t h2dWaitValues[1] = { slotWaitValue };

	if (needSlotWait) {
		h2dTimelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		h2dTimelineInfo.waitSemaphoreValueCount = 1;
		h2dTimelineInfo.pWaitSemaphoreValues = h2dWaitValues;
		h2dTimelineInfo.signalSemaphoreValueCount = 0;
		h2dTimelineInfo.pSignalSemaphoreValues = nullptr;

		transferSubmit.pNext = &h2dTimelineInfo;
		transferSubmit.waitSemaphoreCount = 1;
		transferSubmit.pWaitSemaphores = h2dWaitSemaphores;
		transferSubmit.pWaitDstStageMask = h2dWaitStages;
	} else {
		transferSubmit.pNext = nullptr;
		transferSubmit.waitSemaphoreCount = 0;
		transferSubmit.pWaitSemaphores = nullptr;
		transferSubmit.pWaitDstStageMask = nullptr;
	}

	transferSubmit.commandBufferCount = 1;
	transferSubmit.pCommandBuffers = &transferCmd;

	transferSubmit.signalSemaphoreCount = 1;
	transferSubmit.pSignalSemaphores = &transferCompleteSemaphore;

	checkVulkanErrors(vkQueueSubmit(this->impl->transferQueue, 1, &transferSubmit, VK_NULL_HANDLE));
	VKDBG_LOG("[H2D SUBMIT] idx=" << idx
	          << " cmd=" << transferCmd
	          << " src=stagingInput[" << idx << "]=" << this->impl->stagingInputBuffers[idx]
	          << " dst=deviceInput[" << idx << "]=" << this->impl->deviceInputBuffers[idx]);

	// Submit compute command buffer. signals computeToD2hSemaphore when done
	// Compute only waits on H2D transfer
	uint64_t signalValue = this->impl->nextOutputSignalValue;

	// Compute signals computeToD2hSemaphore (binary) to trigger D2H transfer
	VkSemaphore computeToD2hSemaphore = this->impl->computeToD2hSemaphores[idx];
	VkPipelineStageFlags computeWaitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkSubmitInfo computeSubmit = {};
	computeSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	computeSubmit.waitSemaphoreCount = 1;
	computeSubmit.pWaitSemaphores = &transferCompleteSemaphore;
	computeSubmit.pWaitDstStageMask = &computeWaitStage;
	computeSubmit.commandBufferCount = 1;
	computeSubmit.pCommandBuffers = &cmd;
	computeSubmit.signalSemaphoreCount = 1;
	computeSubmit.pSignalSemaphores = &computeToD2hSemaphore;

	checkVulkanErrors(vkQueueSubmit(this->impl->computeQueue, 1, &computeSubmit, VK_NULL_HANDLE));
	VKDBG_LOG("[COMPUTE SUBMIT] idx=" << idx
	          << " cmd=" << cmd
	          << " signalValue=" << signalValue
	          << " src=deviceInput[" << idx << "]=" << this->impl->deviceInputBuffers[idx]
	          << " dst=deviceProcessed[" << idx << "]=" << this->impl->deviceProcessedBuffers[idx]);

	// ============================================
	// Record D2H Transfer Command Buffer Dynamically
	// ============================================
	// Record the D2H command buffer to copy from deviceProcessed[idx] to stagingOutput[stagingBufferIdx]
	// This is safe because command buffer slot reuse gates ensure GPU is done with previous submission
	this->impl->recordD2hTransferCommandBuffer(idx, stagingBufferIdx, outputSize);
	VkCommandBuffer d2hCmd = this->impl->d2hTransferCommandBuffers[idx];

	// Submit D2H transfer command buffer. signals timeline semaphore for ordered completion
	// D2H waits on: compute complete
	// D2H signals: timeline semaphore (for completion thread ordering)

	// Determine if we need to wait for previous write to this staging buffer
	uint64_t stagingReuseWaitValue = this->impl->stagingLastWriteValue[stagingBufferIdx];  // Use stagingBufferIdx, not idx
	bool needStagingWait = (stagingReuseWaitValue > 0);
	VKDBG_LOG("[D2H DEBUG] cmdIdx=" << idx << " stagingIdx=" << stagingBufferIdx << " signal=" << signalValue << " waitValue=" << stagingReuseWaitValue << " needWait=" << needStagingWait);

	// Setup wait values: [binary_semaphore_value, timeline_wait_value (if needed)]
	uint64_t d2hWaitValues[2] = {0, stagingReuseWaitValue};
	uint64_t d2hSignalValues[1] = {signalValue};  // Timeline semaphore signal value

	// Setup wait semaphores: [computeToD2hSemaphore, outputOrderingSemaphore (if needed)]
	VkSemaphore d2hWaitSemaphores[2] = {
		computeToD2hSemaphore,
		this->impl->outputOrderingSemaphore
	};

	// Setup wait stages
	VkPipelineStageFlags d2hWaitStages[2] = {
		VK_PIPELINE_STAGE_TRANSFER_BIT,  // For compute→D2H dependency
		VK_PIPELINE_STAGE_TRANSFER_BIT   // For staging buffer reuse protection
	};

	VkTimelineSemaphoreSubmitInfo d2hTimelineInfo = {};
	d2hTimelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
	d2hTimelineInfo.pNext = nullptr;
	d2hTimelineInfo.waitSemaphoreValueCount = needStagingWait ? 2 : 1;
	d2hTimelineInfo.pWaitSemaphoreValues = d2hWaitValues;
	d2hTimelineInfo.signalSemaphoreValueCount = 1;
	d2hTimelineInfo.pSignalSemaphoreValues = d2hSignalValues;

	VkSubmitInfo d2hSubmit = {};
	d2hSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	d2hSubmit.pNext = &d2hTimelineInfo;
	d2hSubmit.waitSemaphoreCount = needStagingWait ? 2 : 1;
	d2hSubmit.pWaitSemaphores = d2hWaitSemaphores;
	d2hSubmit.pWaitDstStageMask = d2hWaitStages;
	d2hSubmit.commandBufferCount = 1;
	d2hSubmit.pCommandBuffers = &d2hCmd;
	d2hSubmit.signalSemaphoreCount = 1;
	d2hSubmit.pSignalSemaphores = &this->impl->outputOrderingSemaphore;

	// Increment pending work counter BEFORE submission (critical for shutdown draining)
	this->impl->pendingWorkCount.fetch_add(1, std::memory_order_release);

	VkResult submitResult = vkQueueSubmit(this->impl->d2hTransferQueue, 1, &d2hSubmit, fence);
	VKDBG_LOG("[D2H SUBMIT] cmdIdx=" << idx
	          << " stagingIdx=" << stagingBufferIdx
	          << " cmd=" << d2hCmd
	          << " signalValue=" << signalValue
	          << " waitValue=" << stagingReuseWaitValue
	          << " needWait=" << needStagingWait
	          << " src=deviceProcessed[" << idx << "]=" << this->impl->deviceProcessedBuffers[idx]
	          << " dst=stagingOutput[" << stagingBufferIdx << "]=" << this->impl->stagingOutputBuffers[stagingBufferIdx]);
	if (submitResult == VK_ERROR_DEVICE_LOST) {
		std::cerr << "!!! VK_ERROR_DEVICE_LOST on vkQueueSubmit - GPU crashed (likely shader OOB) !!!" << std::endl;
	}
	checkVulkanErrors(submitResult);

	this->impl->nextOutputSignalValue++;

	// Update per-buffer reuse guard: this staging buffer was last written at signalValue
	this->impl->stagingLastWriteValue[stagingBufferIdx] = signalValue;  // Use stagingBufferIdx, not idx

	// Queue work for async completion (completion thread will wait on timeline semaphore and invoke callback)
	// This enables true async overlap: process() returns immediately, allowing next frame to start
	{
		std::lock_guard<std::mutex> lock(this->impl->pendingWorkMutex);
		this->impl->pendingWorkQueue.push({
			fence,
			idx,                   // Command buffer index (for fence/timeline tracking)
			outputBuf,
			&input,                // Zero-copy: return input buffer to free queue after callback
			outputSize,
			outputSignalLength,
			signalValue,           // Timeline semaphore value for this frame
			input.getBufferId(),   // Save buffer ID (output buffer is shared between frames)
			stagingBufferIdx       // NEW: track which staging buffer this work uses (decoupled from commandBufferIdx)
		});
	}
	this->impl->pendingWorkCV.notify_one();  // Wake completion thread

	// Track which timeline value was used by this CB (for slot reuse protection)
	this->impl->lastTimelineValuePerCB[idx] = signalValue;
}

// ============================================
// Configuration Updates
// ============================================

void VulkanBackend::updateConfig(const ProcessorConfiguration& config) {
	// === STEP 1: Detect which changes invalidate command buffers ===
	// Currently all config changes invalidate CBs (see todo below)
	// In the future, this should check specific fields (FFT size, DC removal, etc.)
	bool cbInvalidatedNow = true;  // TODO: make this conditional once selective invalidation is implemented

	// === STEP 2: Update impl->config ===
	this->impl->config = config;

	// === STEP 3: Update impl->fftConfig if VkFFT-relevant fields changed ===
	// Currently fftConfig is set during initialize() and doesn't change at runtime
	// If future config changes affect VkFFTConfiguration fields (FFT size, precision, etc.),
	// update impl->fftConfig here before rebuilding VkFFT apps

	// === STEP 4: Rebuild VkFFT apps if command buffers were invalidated ===
	if (cbInvalidatedNow && !this->impl->vkfftApps.empty())
	{
		// GPU-safe teardown: Wait for GPU to finish using current VkFFT apps
		std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

		// Wait for all in-flight work to complete
		// NOTE: vkDeviceWaitIdle() stalls entire device - if updateConfig() is called frequently,
		// consider fence-based waiting (wait only for in-flight command buffers)
		VkResult waitResult = vkDeviceWaitIdle(this->impl->device);
		if (waitResult != VK_SUCCESS) {
			std::cerr << "[updateConfig] WARNING: vkDeviceWaitIdle failed with code " << waitResult << std::endl;
			// Continue with cleanup - resources must be destroyed
		}

		// Destroy all VkFFT apps (safe after vkDeviceWaitIdle)
		for (size_t i = 0; i < this->impl->vkfftApps.size(); ++i)
		{
			deleteVkFFT(&this->impl->vkfftApps[i]);
		}
		this->impl->vkfftApps.clear();

		// Re-create with new buffer selection using UPDATED config
		this->impl->vkfftApps.resize(this->impl->numCommandBuffers);

		int initializedCount = 0;  // Track for cleanup robustness

		for (int i = 0; i < this->impl->numCommandBuffers; ++i)
		{
			// Use helper to ensure consistent buffer selection (same logic as recording)
			VkBuffer* fftBuffer = this->impl->getFftDataBufferForSlot(i, this->impl->config);

			// Copy impl->fftConfig and override buffer per-slot
			VkFFTConfiguration fftCfg = this->impl->fftConfig;
			fftCfg.buffer = fftBuffer;  // Per-slot buffer from helper

			// Zero-initialize before passing to VkFFT
			memset(&this->impl->vkfftApps[i], 0, sizeof(VkFFTApplication));

			try {
				checkVkFFTErrors(initializeVkFFT(&this->impl->vkfftApps[i], fftCfg));
				initializedCount++;
			} catch (...) {
				// Cleanup robustness: If initializeVkFFT() fails midway,
				// destroy already-initialized apps before re-throwing
				for (int j = 0; j < initializedCount; ++j) {
					deleteVkFFT(&this->impl->vkfftApps[j]);
				}
				this->impl->vkfftApps.clear();
				throw;
			}
		}
	}

	// === STEP 5: Mark command buffers as invalid ===
	// Invalidate command buffers to force re-recording with new configuration
	// Next process() call will re-record all command buffers
	//todo: this can be improved to only re-record when relevant parameters change
	if (cbInvalidatedNow) {
		this->impl->commandBuffersValid = false;
	}
}

void VulkanBackend::updateResamplingCurve(const float* curve, size_t length) {
	// Acquire submitMutex to protect Vulkan handle access (command pool, queue submission)
	std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

	// Early exit during shutdown
	if (!this->impl->vulkanInitialized ||
	    this->impl->shuttingDown.load(std::memory_order_acquire)) {
		return;
	}

	// Create buffer if it doesn't exist
	if (this->impl->resampleCurveBuffer == VK_NULL_HANDLE && length == static_cast<size_t>(this->impl->signalLength)) {
		VkDeviceSize bufferSize = length * sizeof(float);
		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             this->impl->resampleCurveBuffer, this->impl->resampleCurveMemory);
	}

	if (this->impl->resampleCurveBuffer == VK_NULL_HANDLE || length != static_cast<size_t>(this->impl->signalLength)) {
		return;
	}

	// Ensure no in-flight work is using the curve buffer we're about to update
	// Curve updates are rare (user interaction, not per-frame), so vkDeviceWaitIdle is acceptable
	checkVulkanErrors(vkDeviceWaitIdle(this->impl->device));

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

	// Balanced label for resample curve upload
	char labelName[128];
	snprintf(labelName, sizeof(labelName),
	         "Resample curve upload dst=resampleCurveBuffer src=staging size=%zu",
	         (size_t)copyRegion.size);
	bool began = this->impl->beginLabel(cmdBuffer, labelName);
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->resampleCurveBuffer, 1, &copyRegion);
	this->impl->endLabel(cmdBuffer, began);

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
	// Acquire submitMutex to protect Vulkan handle access (command pool, queue submission)
	std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

	// Early exit during shutdown
	if (!this->impl->vulkanInitialized ||
	    this->impl->shuttingDown.load(std::memory_order_acquire)) {
		return;
	}

	// Create buffer if it doesn't exist
	if (this->impl->dispersionCurveBuffer == VK_NULL_HANDLE && length == static_cast<size_t>(this->impl->signalLength * 2)) {
		VkDeviceSize bufferSize = length * sizeof(float);
		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             this->impl->dispersionCurveBuffer, this->impl->dispersionCurveMemory);
	}

	if (this->impl->dispersionCurveBuffer == VK_NULL_HANDLE || length != static_cast<size_t>(this->impl->signalLength * 2)) {
		return;
	}

	// Ensure no in-flight work is using the curve buffer we're about to update
	// Curve updates are rare (user interaction, not per-frame), so vkDeviceWaitIdle is acceptable
	checkVulkanErrors(vkDeviceWaitIdle(this->impl->device));

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

	// Balanced label for dispersion curve upload
	char labelName[128];
	snprintf(labelName, sizeof(labelName),
	         "Dispersion curve upload dst=dispersionCurveBuffer src=staging size=%zu",
	         (size_t)copyRegion.size);
	bool began = this->impl->beginLabel(cmdBuffer, labelName);
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->dispersionCurveBuffer, 1, &copyRegion);
	this->impl->endLabel(cmdBuffer, began);

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
	// Acquire submitMutex to protect Vulkan handle access (command pool, queue submission)
	std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

	// Early exit during shutdown
	if (!this->impl->vulkanInitialized ||
	    this->impl->shuttingDown.load(std::memory_order_acquire)) {
		return;
	}

	// Create buffer if it doesn't exist
	if (this->impl->windowCurveBuffer == VK_NULL_HANDLE && length == static_cast<size_t>(this->impl->signalLength)) {
		VkDeviceSize bufferSize = length * sizeof(float);
		createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             this->impl->windowCurveBuffer, this->impl->windowCurveMemory);
	}

	if (this->impl->windowCurveBuffer == VK_NULL_HANDLE || length != static_cast<size_t>(this->impl->signalLength)) {
		return;
	}

	// Ensure no in-flight work is using the curve buffer we're about to update
	// Curve updates are rare (user interaction, not per-frame), so vkDeviceWaitIdle is acceptable
	checkVulkanErrors(vkDeviceWaitIdle(this->impl->device));

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

	// Balanced label for window curve upload
	char labelName[128];
	snprintf(labelName, sizeof(labelName),
	         "Window curve upload dst=windowCurveBuffer src=staging size=%zu",
	         (size_t)copyRegion.size);
	bool began = this->impl->beginLabel(cmdBuffer, labelName);
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->windowCurveBuffer, 1, &copyRegion);
	this->impl->endLabel(cmdBuffer, began);

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
	// Zero-copy: Input buffers are tied to staging buffers (one per command buffer)
	if (index < 0 || index >= this->impl->numCommandBuffers) {
		throw std::out_of_range("Input buffer index out of range");
	}
	return this->impl->hostInputBuffers[index];
}

IOBuffer& VulkanBackend::getNextAvailableInputBuffer() {
	IOBuffer* buffer = nullptr;
	{
		std::unique_lock<std::mutex> lock(this->impl->freeQueueMutex);

		// Block until a buffer is available OR shutdown is signaled
		this->impl->freeQueueCV.wait(lock, [this] {
			return !this->impl->freeBuffersQueue.empty() || this->impl->shuttingDown.load(std::memory_order_acquire);
		});

		// After waking, check if we woke due to shutdown
		if (this->impl->shuttingDown.load(std::memory_order_acquire)) {
			throw std::runtime_error("Backend is shutting down");
		}

		buffer = this->impl->freeBuffersQueue.front();
		this->impl->freeBuffersQueue.pop();
	}

	// Get input buffer index that was set during initialization
	int idx = buffer->getBackendIndex();

	VkFence fence = this->impl->fences[idx];

	// Wait for this CB to be available before writing to its input buffers
	// This ensures we don't overwrite input buffers while GPU is reading them
	checkVulkanErrors(vkWaitForFences(this->impl->device, 1, &fence, VK_TRUE, UINT64_MAX));

	checkVulkanErrors(vkResetFences(this->impl->device, 1, &fence));

	// Note: No timeline wait needed here. freeBuffersQueue gating already guarantees
	// completion thread has finished with this buffer (it only pushes after callback)

	return *buffer;
}

int VulkanBackend::getNumInputBuffers() const {
	// there is one input buffer per command buffer
	return this->impl->numCommandBuffers;
}

int VulkanBackend::getOutputBufferCount() const {
	return static_cast<int>(this->impl->outputBuffers.size());
}

void VulkanBackend::releaseOutputBuffer(IOBuffer* buffer) {
	if (!buffer) return;

	// Get staging buffer index (decoupled from command buffer index)
	int stagingBufferIdx = buffer->getBackendIndex();
	if (stagingBufferIdx < 0 || stagingBufferIdx >= this->impl->numStagingBuffers) {
		return;
	}

	// ============================================
	// Return Staging Buffer to Free Queue
	// ============================================
	// NEW POLICY: Input buffers are released in completion thread, NOT here
	// This function ONLY returns the staging buffer to the free pool
	{
		std::lock_guard<std::mutex> lock(this->impl->freeStagingOutputMutex);

		// Safety: check if already released (double-free guard)
		if (!this->impl->stagingInUse[stagingBufferIdx].load(std::memory_order_acquire)) {
			std::cerr << "WARNING: Double release of staging buffer " << stagingBufferIdx << std::endl;
			return;
		}

		// Mark as not in use and return to free queue
		this->impl->stagingInUse[stagingBufferIdx].store(false, std::memory_order_release);
		this->impl->freeStagingOutputQueue.push(stagingBufferIdx);
	}
	this->impl->freeStagingOutputCV.notify_one();
}

// ============================================
// Profile Management
// ============================================

void VulkanBackend::requestPostProcessBackgroundRecording() {
	this->impl->postProcessBackgroundRecordingRequested = true;
	this->impl->commandBuffersValid = false;  // Force re-record on next process()
}

void VulkanBackend::setPostProcessBackgroundProfile(const float* background, size_t length) {
	// Acquire submitMutex to protect Vulkan handle access (command pool, queue submission)
	std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

	// Early exit during shutdown
	if (!this->impl->vulkanInitialized ||
	    this->impl->shuttingDown.load(std::memory_order_acquire)) {
		return;
	}

	if (!background || length == 0) {
		throw std::invalid_argument("Invalid background profile pointer or size");
	}

	// Expect length == outputSignalLength (signalLength/2)
	size_t expectedLength = static_cast<size_t>(this->impl->signalLength / 2);
	if (length != expectedLength) {
		throw std::invalid_argument("Invalid background profile size. Expected " + std::to_string(expectedLength) + " floats but got " + std::to_string(length));
	}

	// Store profile locally
	this->impl->recordedPostProcessBackground.assign(background, background + length);

	// Upload to GPU
	if (this->impl->postProcBackgroundBuffer == VK_NULL_HANDLE) {
		throw std::runtime_error("Background buffer not initialized");
	}

	// Create staging buffer for upload
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	VkDeviceSize bufferSize = length * sizeof(float);

	createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	             stagingBuffer, stagingMemory);

	// Copy profile data to staging buffer
	void* mappedMemory;
	vkMapMemory(this->impl->device, stagingMemory, 0, bufferSize, 0, &mappedMemory);
	memcpy(mappedMemory, background, bufferSize);
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

	// Balanced label for background profile upload
	char labelName[128];
	snprintf(labelName, sizeof(labelName),
	         "Background profile upload dst=postProcBackgroundBuffer src=staging size=%zu",
	         (size_t)copyRegion.size);
	bool began = this->impl->beginLabel(cmdBuffer, labelName);
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->postProcBackgroundBuffer, 1, &copyRegion);
	this->impl->endLabel(cmdBuffer, began);

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

	// Mark background profile as valid
	this->impl->hasValidBackgroundProfile = true;

	// Re-record command buffers to include background subtraction
	// First, wait for all in-flight work to complete
	vkDeviceWaitIdle(this->impl->device);

	// Re-record all command buffers with background subtraction enabled
	this->impl->logRecordPoint("from_setPostProcessBackgroundProfile");
	this->recordCommandBuffers();
}

const std::vector<float>& VulkanBackend::getPostProcessBackgroundProfile() const {
	return this->impl->recordedPostProcessBackground;
}

void VulkanBackend::requestFixedPatternNoiseDetermination() {
	this->impl->fixedPatternNoiseDetermined = false;
	this->impl->commandBuffersValid = false;  // Force re-record to include FPN determination shader
}

void VulkanBackend::setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) {
	// Acquire submitMutex to protect Vulkan handle access (command pool, queue submission)
	std::lock_guard<std::mutex> submitLock(this->impl->submitMutex);

	// Early exit during shutdown
	if (!this->impl->vulkanInitialized ||
	    this->impl->shuttingDown.load(std::memory_order_acquire)) {
		return;
	}

	if (!profileInterleaved || complexPairs == 0) {
		throw std::invalid_argument("Invalid fixed pattern noise profile pointer or size");
	}

	// Expect complexPairs == outputSignalLength (signalLength/2)
	size_t expectedPairs = static_cast<size_t>(this->impl->signalLength / 2);
	if (complexPairs != expectedPairs) {
		throw std::invalid_argument("Invalid fixed pattern noise profile size. Expected " + std::to_string(expectedPairs) + " complex pairs but got " + std::to_string(complexPairs));
	}

	// Store profile locally
	this->impl->recordedFixedPatternNoise.assign(profileInterleaved, profileInterleaved + complexPairs * 2);

	// Upload to GPU
	if (this->impl->meanALineBuffer == VK_NULL_HANDLE) {
		throw std::runtime_error("Mean A-line buffer not initialized");
	}

	// Create staging buffer for upload
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingMemory;
	VkDeviceSize bufferSize = complexPairs * 2 * sizeof(float);  // Interleaved complex data

	createBuffer(this->impl->device, this->impl->physicalDevice, bufferSize,
	             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	             stagingBuffer, stagingMemory);

	// Copy profile data to staging buffer
	void* mappedMemory;
	vkMapMemory(this->impl->device, stagingMemory, 0, bufferSize, 0, &mappedMemory);
	memcpy(mappedMemory, profileInterleaved, bufferSize);
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

	// Balanced label for mean A-line upload
	char labelName[128];
	snprintf(labelName, sizeof(labelName),
	         "Mean A-line upload dst=meanALineBuffer src=staging size=%zu",
	         (size_t)copyRegion.size);
	bool began = this->impl->beginLabel(cmdBuffer, labelName);
	vkCmdCopyBuffer(cmdBuffer, stagingBuffer, this->impl->meanALineBuffer, 1, &copyRegion);
	this->impl->endLabel(cmdBuffer, began);

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

	// Mark FPN as determined
	this->impl->fixedPatternNoiseDetermined = true;
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
	// TODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::rollingAverageBackgroundRemoval(
	const float* input,
	int windowSize,
	int lineWidth,
	int numLines
) {
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::kLinearization(
	const float* input,
	const float* resampleCurve,
	InterpolationMethod method,
	int lineWidth,
	int samples
) {
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::windowing(
	const float* input,
	const float* windowCurve,
	int lineWidth,
	int samples
) {
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::dispersionCompensation(
	const float* input,
	const float* phaseComplex,
	int lineWidth,
	int samples
) {
	// TTODO: remove from all backends
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
	// TTODO: remove from all backends
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
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::dispersionCompensationAndWindowing(
	const float* input,
	const float* phaseComplex,
	const float* windowCurve,
	int lineWidth,
	int samples
) {
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::fft(const float* input, int lineWidth, int samples) {
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::ifft(const float* input, int lineWidth, int samples) {
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::getMinimumVarianceMean(
	const float* input,
	int width,
	int height,
	int segments
) {
	// TTODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::fixedPatternNoiseRemoval(
	const float* input,
	const float* meanALine,
	int lineWidth,
	int numLines
) {
	// TTODO: remove from all backends
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
	//todo: remove
	return std::vector<float>();
}

std::vector<float> VulkanBackend::bscanFlip(
	const float* input,
	int lineWidth,
	int linesPerBscan,
	int numBscans
) {
	// TODO: remove from all backends
	return std::vector<float>();
}

std::vector<float> VulkanBackend::sinusoidalScanCorrection(
	const float* input,
	const float* resampleCurve,
	int lineWidth,
	int linesPerBscan,
	int numBscans
) {
	// TODO: remove from all backends
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
	// TODO: remove from all backends
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
		// Try to allocate with cached memory for fast CPU writes (uses cache hierarchy)
		// HOST_CACHED + HOST_COHERENT = fast writes + automatic sync
		// Fallback to non-cached if cached memory is not available on this platform
		try {
			createBuffer(this->impl->device, this->impl->physicalDevice, inputSize,
			             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
			             this->impl->stagingInputBuffers[i], this->impl->stagingInputMemory[i]);
		} catch (const std::runtime_error&) {
			// HOST_CACHED not available - fall back to non-cached
			createBuffer(this->impl->device, this->impl->physicalDevice, inputSize,
			             VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			             this->impl->stagingInputBuffers[i], this->impl->stagingInputMemory[i]);
		}

		// Map staging buffer memory
		vkMapMemory(this->impl->device, this->impl->stagingInputMemory[i], 0, inputSize, 0, &this->impl->stagingInputMapped[i]);
	}

	// Allocate staging output buffers (decoupled pool, larger than command buffer count)
	// Determine actual number of output buffers (like CUDA backend)
	int actualNumOutputBuffers = (this->impl->numOutputBuffers > 0)
		? this->impl->numOutputBuffers
		: (this->impl->numCommandBuffers * 2);

	this->impl->stagingOutputBuffers.resize(actualNumOutputBuffers);
	this->impl->stagingOutputMemory.resize(actualNumOutputBuffers);
	this->impl->stagingOutputMapped.resize(actualNumOutputBuffers);
	this->impl->stagingLastWriteValue.resize(actualNumOutputBuffers, 0);  // Initialize to 0 (no previous write)

	// Initialize stagingInUse atomic bools (unique_ptr array because atomics can't go in vector)
	this->impl->numStagingBuffers = actualNumOutputBuffers;
	this->impl->stagingInUse.reset(new std::atomic<bool>[actualNumOutputBuffers]);
	for (int i = 0; i < actualNumOutputBuffers; ++i) {
		this->impl->stagingInUse[i].store(false, std::memory_order_relaxed);
	}

	// Track whether we successfully allocated with coherent memory (for flexible platform support)
	bool allocatedCoherent = true;

	for (int i = 0; i < actualNumOutputBuffers; ++i) {
		// Try to allocate with cached + coherent memory for fast CPU reads (60x faster than uncached!)
		// HOST_CACHED + HOST_COHERENT = best of both worlds: fast reads + automatic sync
		// Fallback to non-coherent if coherent memory is not available on this platform
		bool bufferIsCoherent = false;
		try {
			createBuffer(this->impl->device, this->impl->physicalDevice, outputSize,
			             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
			             this->impl->stagingOutputBuffers[i], this->impl->stagingOutputMemory[i]);
			bufferIsCoherent = true;
		} catch (const std::runtime_error&) {
			// HOST_CACHED not available - try non-cached coherent
			try {
				createBuffer(this->impl->device, this->impl->physicalDevice, outputSize,
				             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				             this->impl->stagingOutputBuffers[i], this->impl->stagingOutputMemory[i]);
				bufferIsCoherent = true;
			} catch (const std::runtime_error&) {
				// HOST_COHERENT not available - fall back to non-coherent (requires manual invalidation)
				createBuffer(this->impl->device, this->impl->physicalDevice, outputSize,
				             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
				             this->impl->stagingOutputBuffers[i], this->impl->stagingOutputMemory[i]);
				bufferIsCoherent = false;
				allocatedCoherent = false;
			}
		}

		// Map staging buffer memory
		vkMapMemory(this->impl->device, this->impl->stagingOutputMemory[i], 0, outputSize, 0, &this->impl->stagingOutputMapped[i]);
	}

	// Track coherency status for completion thread (determines if vkInvalidateMappedMemoryRanges needed)
	this->impl->stagingOutputIsCoherent = allocatedCoherent;

	// Initialize free staging output buffer queue
	for (int i = 0; i < actualNumOutputBuffers; ++i) {
		this->impl->freeStagingOutputQueue.push(i);
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

	// Per-command-buffer processing buffers (eliminates data races, enables parallel execution like CUDA)
	this->impl->deviceFftBuffers.resize(this->impl->numCommandBuffers);
	this->impl->deviceFftMemory.resize(this->impl->numCommandBuffers);
	this->impl->deviceIntermediateBuffers.resize(this->impl->numCommandBuffers);
	this->impl->deviceIntermediateMemory.resize(this->impl->numCommandBuffers);
	this->impl->deviceProcessedBuffers.resize(this->impl->numCommandBuffers);
	this->impl->deviceProcessedMemory.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		// FFT buffer (complex float)
		createBuffer(this->impl->device, this->impl->physicalDevice, complexSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             this->impl->deviceFftBuffers[i], this->impl->deviceFftMemory[i]);

		// Intermediate buffer for preprocessing ping-pong (same size as FFT buffer)
		createBuffer(this->impl->device, this->impl->physicalDevice, complexSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             this->impl->deviceIntermediateBuffers[i], this->impl->deviceIntermediateMemory[i]);

		// Processed buffer (output)
		createBuffer(this->impl->device, this->impl->physicalDevice, outputSize,
		             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		             this->impl->deviceProcessedBuffers[i], this->impl->deviceProcessedMemory[i]);
	}

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

	// Mean A-line buffer for fixed pattern noise (complex float, outputSignalLength size)
	// outputSignalLength = signalLength / 2, complex float = 2 floats
	// So size = (signalLength / 2) * 2 * sizeof(float) = signalLength * sizeof(float) = curveSize
	size_t meanALineSize = curveSize;
	createBuffer(this->impl->device, this->impl->physicalDevice, meanALineSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->meanALineBuffer, this->impl->meanALineMemory);

	// Background buffer for post-processing background subtraction (float, outputSignalLength size)
	size_t backgroundSize = outputSignalLength * sizeof(float);
	createBuffer(this->impl->device, this->impl->physicalDevice, backgroundSize,
	             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	             this->impl->postProcBackgroundBuffer, this->impl->postProcBackgroundMemory);

	// Staging buffer for background profile readback (host-visible)
	createBuffer(this->impl->device, this->impl->physicalDevice, backgroundSize,
	             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	             this->impl->postProcBackgroundStagingBuffer, this->impl->postProcBackgroundStagingMemory);

	// Map staging buffer for persistent access
	vkMapMemory(this->impl->device, this->impl->postProcBackgroundStagingMemory, 0, backgroundSize, 0, &this->impl->postProcBackgroundStagingMapped);

	// Initialize background buffer to zeros (so background subtraction works even without recording)
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = this->impl->commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer initCmdBuffer;
	vkAllocateCommandBuffers(this->impl->device, &allocInfo, &initCmdBuffer);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(initCmdBuffer, &beginInfo);
	vkCmdFillBuffer(initCmdBuffer, this->impl->postProcBackgroundBuffer, 0, VK_WHOLE_SIZE, 0);  // Fill with zeros
	vkEndCommandBuffer(initCmdBuffer);

	// Submit and wait for initialization
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &initCmdBuffer;

	vkQueueSubmit(this->impl->computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(this->impl->computeQueue);

	// Name buffers for debug identification
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		char name[64];

		// Staging output buffers (D2H destination)
		snprintf(name, sizeof(name), "stagingOutput[%d]", i);
		this->impl->nameObject(VK_OBJECT_TYPE_BUFFER,
		                       (uint64_t)this->impl->stagingOutputBuffers[i], name);

		// Device processed buffers (compute output, D2H source)
		snprintf(name, sizeof(name), "deviceProcessed[%d]", i);
		this->impl->nameObject(VK_OBJECT_TYPE_BUFFER,
		                       (uint64_t)this->impl->deviceProcessedBuffers[i], name);

		// Staging input buffers (H2D source)
		snprintf(name, sizeof(name), "stagingInput[%d]", i);
		this->impl->nameObject(VK_OBJECT_TYPE_BUFFER,
		                       (uint64_t)this->impl->stagingInputBuffers[i], name);

		// Device input buffers (H2D destination, compute source)
		snprintf(name, sizeof(name), "deviceInput[%d]", i);
		this->impl->nameObject(VK_OBJECT_TYPE_BUFFER,
		                       (uint64_t)this->impl->deviceInputBuffers[i], name);
	}

	// Cleanup
	vkFreeCommandBuffers(this->impl->device, this->impl->commandPool, 1, &initCmdBuffer);
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

	// Destroy per-CB processing buffers (FFT, Intermediate, Processed)
	for (size_t i = 0; i < this->impl->deviceFftBuffers.size(); ++i) {
		if (this->impl->deviceFftBuffers[i] != VK_NULL_HANDLE) {
			vkDestroyBuffer(this->impl->device, this->impl->deviceFftBuffers[i], nullptr);
		}
		if (this->impl->deviceFftMemory[i] != VK_NULL_HANDLE) {
			vkFreeMemory(this->impl->device, this->impl->deviceFftMemory[i], nullptr);
		}
	}

	for (size_t i = 0; i < this->impl->deviceIntermediateBuffers.size(); ++i) {
		if (this->impl->deviceIntermediateBuffers[i] != VK_NULL_HANDLE) {
			vkDestroyBuffer(this->impl->device, this->impl->deviceIntermediateBuffers[i], nullptr);
		}
		if (this->impl->deviceIntermediateMemory[i] != VK_NULL_HANDLE) {
			vkFreeMemory(this->impl->device, this->impl->deviceIntermediateMemory[i], nullptr);
		}
	}

	for (size_t i = 0; i < this->impl->deviceProcessedBuffers.size(); ++i) {
		if (this->impl->deviceProcessedBuffers[i] != VK_NULL_HANDLE) {
			vkDestroyBuffer(this->impl->device, this->impl->deviceProcessedBuffers[i], nullptr);
		}
		if (this->impl->deviceProcessedMemory[i] != VK_NULL_HANDLE) {
			vkFreeMemory(this->impl->device, this->impl->deviceProcessedMemory[i], nullptr);
		}
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

	// Mean A-line buffer for fixed pattern noise
	if (this->impl->meanALineBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->meanALineBuffer, nullptr);
	}
	if (this->impl->meanALineMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->meanALineMemory, nullptr);
	}

	// Background buffer for post-processing background subtraction
	if (this->impl->postProcBackgroundBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->postProcBackgroundBuffer, nullptr);
	}
	if (this->impl->postProcBackgroundMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->postProcBackgroundMemory, nullptr);
	}

	// Unmap and cleanup staging buffer for background profile
	if (this->impl->postProcBackgroundStagingMapped != nullptr) {
		vkUnmapMemory(this->impl->device, this->impl->postProcBackgroundStagingMemory);
		this->impl->postProcBackgroundStagingMapped = nullptr;
	}
	if (this->impl->postProcBackgroundStagingBuffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(this->impl->device, this->impl->postProcBackgroundStagingBuffer, nullptr);
	}
	if (this->impl->postProcBackgroundStagingMemory != VK_NULL_HANDLE) {
		vkFreeMemory(this->impl->device, this->impl->postProcBackgroundStagingMemory, nullptr);
	}
}

void VulkanBackend::createCommandBuffersAndFences() {
	// Allocate compute command buffers
	this->impl->commandBuffers.resize(this->impl->numCommandBuffers);

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = this->impl->commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = this->impl->numCommandBuffers;

	checkVulkanErrors(vkAllocateCommandBuffers(this->impl->device, &allocInfo, this->impl->commandBuffers.data()));

	// Allocate transfer command buffers
	this->impl->transferCommandBuffers.resize(this->impl->numCommandBuffers);

	VkCommandBufferAllocateInfo transferAllocInfo = {};
	transferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	transferAllocInfo.commandPool = this->impl->transferCommandPool;
	transferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	transferAllocInfo.commandBufferCount = this->impl->numCommandBuffers;

	checkVulkanErrors(vkAllocateCommandBuffers(this->impl->device, &transferAllocInfo, this->impl->transferCommandBuffers.data()));

	// Allocate D2H transfer command buffers
	this->impl->d2hTransferCommandBuffers.resize(this->impl->numCommandBuffers);

	VkCommandBufferAllocateInfo d2hAllocInfo = {};
	d2hAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	d2hAllocInfo.commandPool = this->impl->d2hTransferCommandPool;
	d2hAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	d2hAllocInfo.commandBufferCount = this->impl->numCommandBuffers;

	checkVulkanErrors(vkAllocateCommandBuffers(this->impl->device, &d2hAllocInfo, this->impl->d2hTransferCommandBuffers.data()));

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

	// Create semaphores for transfer→compute synchronization
	this->impl->transferToComputeSemaphores.resize(this->impl->numCommandBuffers);

	VkSemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		checkVulkanErrors(vkCreateSemaphore(this->impl->device, &semaphoreInfo, nullptr, &this->impl->transferToComputeSemaphores[i]));
	}

	// Create semaphores for compute to D2H synchronization
	this->impl->computeToD2hSemaphores.resize(this->impl->numCommandBuffers);

	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		checkVulkanErrors(vkCreateSemaphore(this->impl->device, &semaphoreInfo, nullptr, &this->impl->computeToD2hSemaphores[i]));
	}

	// Create timeline semaphore for ordered output transfers
	// See: https://docs.vulkan.org/samples/latest/samples/extensions/timeline_semaphore/README.html
	VkSemaphoreTypeCreateInfo timelineCreateInfo = {};
	timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
	timelineCreateInfo.pNext = nullptr;
	timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	timelineCreateInfo.initialValue = 0;

	VkSemaphoreCreateInfo timelineSemaphoreInfo = {};
	timelineSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	timelineSemaphoreInfo.pNext = &timelineCreateInfo;
	timelineSemaphoreInfo.flags = 0;

	checkVulkanErrors(vkCreateSemaphore(this->impl->device, &timelineSemaphoreInfo, nullptr,
	                                     &this->impl->outputOrderingSemaphore));
	this->impl->nextOutputSignalValue = 1;

	// Initialize per-CB timeline tracking (0 means no previous frame used this CB)
	this->impl->lastTimelineValuePerCB.resize(this->impl->numCommandBuffers, 0);

	// Name command buffers and semaphores for debug identification
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		char name[64];

		snprintf(name, sizeof(name), "transferCmd[%d]", i);
		this->impl->nameObject(VK_OBJECT_TYPE_COMMAND_BUFFER,
		                       (uint64_t)this->impl->transferCommandBuffers[i], name);

		snprintf(name, sizeof(name), "computeCmd[%d]", i);
		this->impl->nameObject(VK_OBJECT_TYPE_COMMAND_BUFFER,
		                       (uint64_t)this->impl->commandBuffers[i], name);

		snprintf(name, sizeof(name), "d2hCmd[%d]", i);
		this->impl->nameObject(VK_OBJECT_TYPE_COMMAND_BUFFER,
		                       (uint64_t)this->impl->d2hTransferCommandBuffers[i], name);
	}

	// Name semaphores
	this->impl->nameObject(VK_OBJECT_TYPE_SEMAPHORE,
	                       (uint64_t)this->impl->outputOrderingSemaphore,
	                       "outputOrderingTimeline");
}

void VulkanBackend::destroyCommandBuffersAndFences() {
	// Free compute command buffers (automatically freed when pool is destroyed)
	if (!this->impl->commandBuffers.empty() && this->impl->commandPool != VK_NULL_HANDLE) {
		vkFreeCommandBuffers(this->impl->device, this->impl->commandPool,
		                     static_cast<uint32_t>(this->impl->commandBuffers.size()),
		                     this->impl->commandBuffers.data());
		this->impl->commandBuffers.clear();
	}

	// Free transfer command buffers
	if (!this->impl->transferCommandBuffers.empty() && this->impl->transferCommandPool != VK_NULL_HANDLE) {
		vkFreeCommandBuffers(this->impl->device, this->impl->transferCommandPool,
		                     static_cast<uint32_t>(this->impl->transferCommandBuffers.size()),
		                     this->impl->transferCommandBuffers.data());
		this->impl->transferCommandBuffers.clear();
	}

	// Free D2H transfer command buffers
	if (!this->impl->d2hTransferCommandBuffers.empty() && this->impl->d2hTransferCommandPool != VK_NULL_HANDLE) {
		vkFreeCommandBuffers(this->impl->device, this->impl->d2hTransferCommandPool,
		                     static_cast<uint32_t>(this->impl->d2hTransferCommandBuffers.size()),
		                     this->impl->d2hTransferCommandBuffers.data());
		this->impl->d2hTransferCommandBuffers.clear();
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

	// Destroy transfer→compute semaphores
	for (auto& semaphore : this->impl->transferToComputeSemaphores) {
		if (semaphore != VK_NULL_HANDLE) {
			vkDestroySemaphore(this->impl->device, semaphore, nullptr);
		}
	}
	this->impl->transferToComputeSemaphores.clear();

	// Destroy compute to D2H semaphores
	for (auto& semaphore : this->impl->computeToD2hSemaphores) {
		if (semaphore != VK_NULL_HANDLE) {
			vkDestroySemaphore(this->impl->device, semaphore, nullptr);
		}
	}
	this->impl->computeToD2hSemaphores.clear();

	// Destroy timeline semaphore for output ordering
	if (this->impl->outputOrderingSemaphore != VK_NULL_HANDLE) {
		vkDestroySemaphore(this->impl->device, this->impl->outputOrderingSemaphore, nullptr);
		this->impl->outputOrderingSemaphore = VK_NULL_HANDLE;
	}
}

// ============================================
// Command Buffer Recording
// ============================================

void VulkanBackend::recordCommandBuffers() {
	this->impl->recordAllCommandBuffers(); //todo: using pimpl pattern in the backend probably does not make sense. think about it, and come up with clean backend implementation 
}

// ============================================
// Shader Compilation Helpers
// ============================================

// Helper function to load shader source from file
std::string loadShaderSource(const std::string& filepath) {
	// Search paths relative to current working directory
	// Covers: executable directory, parent directories (development), and project root
	std::vector<std::string> searchPaths = {
		filepath,                                      // Next to executable (normal case)
		"../" + filepath,                              // One directory up
		"../../" + filepath,                           // Two directories up
		"../../../" + filepath,                        // Three directories up
		"tests/Release/" + filepath,                   // From build dir to test dir (Windows)
		"tests/" + filepath,                           // From build dir to test dir (Linux)
		"examples/Release/" + filepath,                // From build dir to examples dir (Windows)
		"examples/" + filepath,                        // From build dir to examples dir (Linux)
		"src/backends/vulkan/shaders/" + filepath.substr(filepath.find_last_of('/') + 1)  // Source tree fallback
	};

	// Try each path
	for (const auto& path : searchPaths) {
		std::ifstream file(path);
		if (file.is_open()) {
			std::stringstream buffer;
			buffer << file.rdbuf();
			return buffer.str();
		}
	}

	// Build error message with all attempted paths
	std::stringstream errorMsg;
	errorMsg << "Failed to open shader file: " << filepath << "\nSearched in:\n";
	for (const auto& path : searchPaths) {
		errorMsg << "  - " << path << "\n";
	}
	throw std::runtime_error(errorMsg.str());
}

// Helper function to compile GLSL to SPIR-V using shaderc
std::vector<uint32_t> compileGLSLToSPIRV(const std::string& source, const std::string& filename, shaderc_shader_kind kind) {
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;

	// Set optimization level
	options.SetOptimizationLevel(shaderc_optimization_level_performance);

	// Define workgroup size macro for all shaders
	options.AddMacroDefinition("WORKGROUP_SIZE", std::to_string(VULKAN_WORKGROUP_SIZE));

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

	std::string shaderPath = "shaders/input_conversion.comp";
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
	// Create Descriptor Pool
	// ============================================
	// Pool needs to allocate for all pipeline descriptor sets

	VkDescriptorPoolSize poolSize = {};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = static_cast<uint32_t>(this->impl->numCommandBuffers * 34);  // 2 (input conv) + 3 (windowing) + 2 (DC removal) + 3 (klinear) + 3 (dispersion) + 7 (universal pre-FFT) + 3 (universal post-FFT) + 2 (FPN determination) + 5 (merged klinear+windowing+dispersion) + 2 (background subtraction) + 2 (background recording) per command buffer

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = static_cast<uint32_t>(this->impl->numCommandBuffers * 12);  // 12 descriptor sets per command buffer (including merged, universal, FPN, background subtraction, and background recording pipelines)
	poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // Allow individual descriptor sets to be freed

	checkVulkanErrors(vkCreateDescriptorPool(this->impl->device, &poolInfo, nullptr, &this->impl->descriptorPool));
	this->impl->logPoolOp("CREATE", this->impl->descriptorPool);

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
		outputBufferInfo.buffer = this->impl->deviceFftBuffers[i];
		outputBufferInfo.offset = 0;
		outputBufferInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = this->impl->descriptorSets[i];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &outputBufferInfo;

		this->impl->updateDescriptorSetsTagged("InputConversion", static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data());
	}

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
	std::string dcRemovalShaderPath = "shaders/dc_removal.comp";
	std::string dcRemovalShaderSource = loadShaderSource(dcRemovalShaderPath);
	std::vector<uint32_t> dcRemovalSPIRV = compileGLSLToSPIRV(dcRemovalShaderSource, dcRemovalShaderPath, shaderc_compute_shader);

	VkShaderModule dcRemovalShader = createShaderModule(this->impl->device, dcRemovalSPIRV);
	this->impl->shaderModules.push_back(dcRemovalShader);

	// Calculate required shared memory size for DC removal
	// Shared memory needs to hold: localSize + 2 * maxWindowSize
	// maxWindowSize can be as large as signalLength
	uint32_t dcRemovalLocalSize = VULKAN_WORKGROUP_SIZE;  // From shader local_size_x
	uint32_t maxWindowSize = static_cast<uint32_t>(this->impl->signalLength);
	uint32_t requiredSharedMemSize = dcRemovalLocalSize + 2 * maxWindowSize;

	// Query device limits to ensure we don't exceed maximum shared memory size
	VkPhysicalDeviceProperties deviceProps;
	vkGetPhysicalDeviceProperties(this->impl->physicalDevice, &deviceProps);
	uint32_t maxSharedMemSize = deviceProps.limits.maxComputeSharedMemorySize / sizeof(float);  // Convert bytes to float count

	// Clamp to device limit if needed
	if (requiredSharedMemSize > maxSharedMemSize) {
		requiredSharedMemSize = maxSharedMemSize;
		std::cerr << "Warning: DC removal shared memory size clamped to device limit: "
		          << maxSharedMemSize << " floats (" << (maxSharedMemSize * sizeof(float)) << " bytes)" << std::endl;
	}

	// Set up specialization constant for shared memory size
	VkSpecializationMapEntry dcRemovalSpecEntry = {};
	dcRemovalSpecEntry.constantID = 0;  // Matches layout(constant_id = 0) in shader
	dcRemovalSpecEntry.offset = 0;
	dcRemovalSpecEntry.size = sizeof(uint32_t);

	VkSpecializationInfo dcRemovalSpecInfo = {};
	dcRemovalSpecInfo.mapEntryCount = 1;
	dcRemovalSpecInfo.pMapEntries = &dcRemovalSpecEntry;
	dcRemovalSpecInfo.dataSize = sizeof(uint32_t);
	dcRemovalSpecInfo.pData = &requiredSharedMemSize;

	// Create DC removal compute pipeline
	VkPipelineShaderStageCreateInfo dcRemovalShaderStageInfo = {};
	dcRemovalShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	dcRemovalShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	dcRemovalShaderStageInfo.module = dcRemovalShader;
	dcRemovalShaderStageInfo.pName = "main";
	dcRemovalShaderStageInfo.pSpecializationInfo = &dcRemovalSpecInfo;

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
		dcRemovalInputBufferInfo.buffer = this->impl->deviceFftBuffers[i];
		dcRemovalInputBufferInfo.offset = 0;
		dcRemovalInputBufferInfo.range = this->impl->fftBufferSize;

		VkDescriptorBufferInfo dcRemovalOutputBufferInfo = {};
		dcRemovalOutputBufferInfo.buffer = this->impl->deviceIntermediateBuffers[i];  // Write to separate buffer
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

		this->impl->updateDescriptorSetsTagged("DCRemoval", static_cast<uint32_t>(dcRemovalDescriptorWrites.size()), dcRemovalDescriptorWrites.data());
	}

	// ============================================
	// Create Fixed Pattern Noise Determination Shader Pipeline
	// ============================================

	// Create descriptor set layout for FPN determination (2 storage buffers: input, mean A-line output)
	std::vector<VkDescriptorSetLayoutBinding> fpnDeterminationBindings(2);

	// Binding 0: Input buffer (complex data after IFFT)
	fpnDeterminationBindings[0].binding = 0;
	fpnDeterminationBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	fpnDeterminationBindings[0].descriptorCount = 1;
	fpnDeterminationBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fpnDeterminationBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Mean A-line output buffer
	fpnDeterminationBindings[1].binding = 1;
	fpnDeterminationBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	fpnDeterminationBindings[1].descriptorCount = 1;
	fpnDeterminationBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fpnDeterminationBindings[1].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo fpnDeterminationLayoutInfo = {};
	fpnDeterminationLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	fpnDeterminationLayoutInfo.bindingCount = static_cast<uint32_t>(fpnDeterminationBindings.size());
	fpnDeterminationLayoutInfo.pBindings = fpnDeterminationBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &fpnDeterminationLayoutInfo, nullptr, &this->impl->fpnDeterminationDescriptorSetLayout));

	// Create pipeline layout for FPN determination
	VkPushConstantRange fpnDeterminationPushConstantRange = {};
	fpnDeterminationPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	fpnDeterminationPushConstantRange.offset = 0;
	fpnDeterminationPushConstantRange.size = sizeof(uint32_t) * 5;  // width, height, segments, stride, outputSignalLength

	VkPipelineLayoutCreateInfo fpnDeterminationPipelineLayoutInfo = {};
	fpnDeterminationPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	fpnDeterminationPipelineLayoutInfo.setLayoutCount = 1;
	fpnDeterminationPipelineLayoutInfo.pSetLayouts = &this->impl->fpnDeterminationDescriptorSetLayout;
	fpnDeterminationPipelineLayoutInfo.pushConstantRangeCount = 1;
	fpnDeterminationPipelineLayoutInfo.pPushConstantRanges = &fpnDeterminationPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &fpnDeterminationPipelineLayoutInfo, nullptr, &this->impl->fpnDeterminationPipelineLayout));

	// Load and compile FPN determination shader
	std::string fpnDeterminationShaderPath = "shaders/fixed_pattern_noise_determination.comp";
	std::string fpnDeterminationShaderSource = loadShaderSource(fpnDeterminationShaderPath);
	std::vector<uint32_t> fpnDeterminationSPIRV = compileGLSLToSPIRV(fpnDeterminationShaderSource, fpnDeterminationShaderPath, shaderc_compute_shader);

	VkShaderModule fpnDeterminationShader = createShaderModule(this->impl->device, fpnDeterminationSPIRV);
	this->impl->shaderModules.push_back(fpnDeterminationShader);

	// Create FPN determination compute pipeline
	VkPipelineShaderStageCreateInfo fpnDeterminationShaderStageInfo = {};
	fpnDeterminationShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fpnDeterminationShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	fpnDeterminationShaderStageInfo.module = fpnDeterminationShader;
	fpnDeterminationShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo fpnDeterminationPipelineInfo = {};
	fpnDeterminationPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	fpnDeterminationPipelineInfo.stage = fpnDeterminationShaderStageInfo;
	fpnDeterminationPipelineInfo.layout = this->impl->fpnDeterminationPipelineLayout;

	VkPipeline fpnDeterminationPipeline;
	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &fpnDeterminationPipelineInfo, nullptr, &fpnDeterminationPipeline));
	this->impl->computePipelines.push_back(fpnDeterminationPipeline);

	// ============================================
	// Create Background Subtraction Shader Pipeline
	// ============================================

	// Create descriptor set layout for background subtraction (2 storage buffers: data in/out, background profile)
	std::vector<VkDescriptorSetLayoutBinding> backgroundSubtractionBindings(2);

	// Binding 0: Data buffer (magnitude data, in-place processing)
	backgroundSubtractionBindings[0].binding = 0;
	backgroundSubtractionBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	backgroundSubtractionBindings[0].descriptorCount = 1;
	backgroundSubtractionBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	backgroundSubtractionBindings[0].pImmutableSamplers = nullptr;

	// Binding 1: Background profile buffer
	backgroundSubtractionBindings[1].binding = 1;
	backgroundSubtractionBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	backgroundSubtractionBindings[1].descriptorCount = 1;
	backgroundSubtractionBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	backgroundSubtractionBindings[1].pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutCreateInfo backgroundSubtractionLayoutInfo = {};
	backgroundSubtractionLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	backgroundSubtractionLayoutInfo.bindingCount = static_cast<uint32_t>(backgroundSubtractionBindings.size());
	backgroundSubtractionLayoutInfo.pBindings = backgroundSubtractionBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &backgroundSubtractionLayoutInfo, nullptr, &this->impl->backgroundSubtractionDescriptorSetLayout));

	// Create pipeline layout for background subtraction
	VkPushConstantRange backgroundSubtractionPushConstantRange = {};
	backgroundSubtractionPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	backgroundSubtractionPushConstantRange.offset = 0;
	backgroundSubtractionPushConstantRange.size = sizeof(float) * 2 + sizeof(uint32_t) * 2;  // backgroundWeight, backgroundOffset, samplesPerAscan, samplesPerBuffer

	VkPipelineLayoutCreateInfo backgroundSubtractionPipelineLayoutInfo = {};
	backgroundSubtractionPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	backgroundSubtractionPipelineLayoutInfo.setLayoutCount = 1;
	backgroundSubtractionPipelineLayoutInfo.pSetLayouts = &this->impl->backgroundSubtractionDescriptorSetLayout;
	backgroundSubtractionPipelineLayoutInfo.pushConstantRangeCount = 1;
	backgroundSubtractionPipelineLayoutInfo.pPushConstantRanges = &backgroundSubtractionPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &backgroundSubtractionPipelineLayoutInfo, nullptr, &this->impl->backgroundSubtractionPipelineLayout));

	// Load and compile background subtraction shader
	std::string backgroundSubtractionShaderPath = "shaders/background_subtraction.comp";
	std::string backgroundSubtractionShaderSource = loadShaderSource(backgroundSubtractionShaderPath);
	std::vector<uint32_t> backgroundSubtractionSPIRV = compileGLSLToSPIRV(backgroundSubtractionShaderSource, backgroundSubtractionShaderPath, shaderc_compute_shader);

	VkShaderModule backgroundSubtractionShader = createShaderModule(this->impl->device, backgroundSubtractionSPIRV);
	this->impl->shaderModules.push_back(backgroundSubtractionShader);

	// Create background subtraction compute pipeline
	VkPipelineShaderStageCreateInfo backgroundSubtractionShaderStageInfo = {};
	backgroundSubtractionShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	backgroundSubtractionShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	backgroundSubtractionShaderStageInfo.module = backgroundSubtractionShader;
	backgroundSubtractionShaderStageInfo.pName = "main";

	VkComputePipelineCreateInfo backgroundSubtractionPipelineInfo = {};
	backgroundSubtractionPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	backgroundSubtractionPipelineInfo.stage = backgroundSubtractionShaderStageInfo;
	backgroundSubtractionPipelineInfo.layout = this->impl->backgroundSubtractionPipelineLayout;

	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &backgroundSubtractionPipelineInfo, nullptr, &this->impl->backgroundSubtractionPipeline));

	// ============================================
	// Background Recording Pipeline
	// ============================================
	// Records background profile by averaging all A-scans

	// Create descriptor set layout for background recording (2 storage buffers)
	std::vector<VkDescriptorSetLayoutBinding> backgroundRecordingBindings(2);

	// Binding 0: Input buffer (magnitude data from deviceProcessedBuffer)
	backgroundRecordingBindings[0].binding = 0;
	backgroundRecordingBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	backgroundRecordingBindings[0].descriptorCount = 1;
	backgroundRecordingBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 1: Background buffer (output: averaged background profile)
	backgroundRecordingBindings[1].binding = 1;
	backgroundRecordingBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	backgroundRecordingBindings[1].descriptorCount = 1;
	backgroundRecordingBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo backgroundRecordingLayoutInfo = {};
	backgroundRecordingLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	backgroundRecordingLayoutInfo.bindingCount = static_cast<uint32_t>(backgroundRecordingBindings.size());
	backgroundRecordingLayoutInfo.pBindings = backgroundRecordingBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &backgroundRecordingLayoutInfo, nullptr, &this->impl->backgroundRecordingDescriptorSetLayout));

	// Create pipeline layout with push constants (samplesPerAscan, ascansPerBuffer)
	VkPushConstantRange backgroundRecordingPushConstantRange = {};
	backgroundRecordingPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	backgroundRecordingPushConstantRange.offset = 0;
	backgroundRecordingPushConstantRange.size = sizeof(uint32_t) * 2;  // samplesPerAscan, ascansPerBuffer

	VkPipelineLayoutCreateInfo backgroundRecordingPipelineLayoutInfo = {};
	backgroundRecordingPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	backgroundRecordingPipelineLayoutInfo.setLayoutCount = 1;
	backgroundRecordingPipelineLayoutInfo.pSetLayouts = &this->impl->backgroundRecordingDescriptorSetLayout;
	backgroundRecordingPipelineLayoutInfo.pushConstantRangeCount = 1;
	backgroundRecordingPipelineLayoutInfo.pPushConstantRanges = &backgroundRecordingPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &backgroundRecordingPipelineLayoutInfo, nullptr, &this->impl->backgroundRecordingPipelineLayout));

	// Load and compile shader
	std::string backgroundRecordingShaderPath = "shaders/get_background.comp";
	std::string backgroundRecordingShaderSource = loadShaderSource(backgroundRecordingShaderPath);
	std::vector<uint32_t> backgroundRecordingSPIRV = compileGLSLToSPIRV(backgroundRecordingShaderSource, backgroundRecordingShaderPath, shaderc_compute_shader);
	VkShaderModule backgroundRecordingShader = createShaderModule(this->impl->device, backgroundRecordingSPIRV);

	// Create compute pipeline
	VkComputePipelineCreateInfo backgroundRecordingPipelineInfo = {};
	backgroundRecordingPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	backgroundRecordingPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	backgroundRecordingPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	backgroundRecordingPipelineInfo.stage.module = backgroundRecordingShader;
	backgroundRecordingPipelineInfo.stage.pName = "main";
	backgroundRecordingPipelineInfo.layout = this->impl->backgroundRecordingPipelineLayout;

	checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &backgroundRecordingPipelineInfo, nullptr, &this->impl->backgroundRecordingPipeline));

	vkDestroyShaderModule(this->impl->device, backgroundRecordingShader, nullptr);

	// ============================================
	// Universal Pre-FFT Processing Shader
	// ============================================
	// This shader combines DC removal, k-linearization, windowing, and dispersion
	// Uses specialization constants for compile-time optimization
	// Supports dual input buffer selection to eliminate buffer copies

	// Create descriptor set layout for universal shader (7 bindings)
	// DC removal is now a separate pass, universal shader only does k-linear + windowing + dispersion
	// Needs dual output buffers to avoid read-write hazard
	std::vector<VkDescriptorSetLayoutBinding> universalPreFFTBindings(7);

	// Binding 0: Input buffer A (deviceFftBuffer)
	universalPreFFTBindings[0].binding = 0;
	universalPreFFTBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPreFFTBindings[0].descriptorCount = 1;
	universalPreFFTBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 1: Input buffer B (deviceIntermediateBuffer)
	universalPreFFTBindings[1].binding = 1;
	universalPreFFTBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPreFFTBindings[1].descriptorCount = 1;
	universalPreFFTBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 2: Resample curve buffer
	universalPreFFTBindings[2].binding = 2;
	universalPreFFTBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPreFFTBindings[2].descriptorCount = 1;
	universalPreFFTBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 3: Window curve buffer
	universalPreFFTBindings[3].binding = 3;
	universalPreFFTBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPreFFTBindings[3].descriptorCount = 1;
	universalPreFFTBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 4: Dispersion phase buffer
	universalPreFFTBindings[4].binding = 4;
	universalPreFFTBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPreFFTBindings[4].descriptorCount = 1;
	universalPreFFTBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 5: Output buffer A (deviceIntermediateBuffer)
	universalPreFFTBindings[5].binding = 5;
	universalPreFFTBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPreFFTBindings[5].descriptorCount = 1;
	universalPreFFTBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 6: Output buffer B (deviceFftBuffer)
	universalPreFFTBindings[6].binding = 6;
	universalPreFFTBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPreFFTBindings[6].descriptorCount = 1;
	universalPreFFTBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo universalPreFFTDescriptorSetLayoutInfo = {};
	universalPreFFTDescriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	universalPreFFTDescriptorSetLayoutInfo.bindingCount = static_cast<uint32_t>(universalPreFFTBindings.size());
	universalPreFFTDescriptorSetLayoutInfo.pBindings = universalPreFFTBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &universalPreFFTDescriptorSetLayoutInfo, nullptr, &this->impl->universalPreFFTDescriptorSetLayout));

	// Create pipeline layout for universal shader
	VkPushConstantRange universalPreFFTPushConstantRange = {};
	universalPreFFTPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	universalPreFFTPushConstantRange.offset = 0;
	universalPreFFTPushConstantRange.size = sizeof(uint32_t) * 2;  // signalLength, samplesPerBuffer

	VkPipelineLayoutCreateInfo universalPreFFTPipelineLayoutInfo = {};
	universalPreFFTPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	universalPreFFTPipelineLayoutInfo.setLayoutCount = 1;
	universalPreFFTPipelineLayoutInfo.pSetLayouts = &this->impl->universalPreFFTDescriptorSetLayout;
	universalPreFFTPipelineLayoutInfo.pushConstantRangeCount = 1;
	universalPreFFTPipelineLayoutInfo.pPushConstantRanges = &universalPreFFTPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &universalPreFFTPipelineLayoutInfo, nullptr, &this->impl->universalPreFFTPipelineLayout));

	// Load and compile universal pre-FFT shader
	std::string universalShaderPath = "shaders/universal_prefft_processing.comp";
	std::string universalShaderSource = loadShaderSource(universalShaderPath);
	std::vector<uint32_t> universalSPIRV = compileGLSLToSPIRV(universalShaderSource, universalShaderPath, shaderc_compute_shader);

	VkShaderModule universalShader = createShaderModule(this->impl->device, universalSPIRV);
	this->impl->shaderModules.push_back(universalShader);

	// Create pipeline variants with different specialization constants
	// We create variants for: interpolation (cubic/linear/lanczos) × input buffer (fft/intermediate)
	// Input buffer selection: 0=fftBuffer (no DC removal), 1=intermediateBuffer (after DC removal)
	for (int useIntermediate = 0; useIntermediate <= 1; ++useIntermediate) {
		for (int interpolation = 0; interpolation < 3; ++interpolation) {
			// Setup specialization constants
			struct SpecializationData {
				uint32_t enableKLinear;
				uint32_t enableWindowing;
				uint32_t enableDispersion;
				uint32_t interpolationMethod;
				uint32_t useIntermediateBuffer;
			} specData;

			specData.enableKLinear = 1;  // Always enable k-linearization
			specData.enableWindowing = 1;  // Always enable windowing
			specData.enableDispersion = 1;  // Always enable dispersion
			specData.interpolationMethod = interpolation;  // 0=cubic, 1=linear, 2=lanczos
			specData.useIntermediateBuffer = useIntermediate;  // 0=fftBuffer, 1=intermediateBuffer

			VkSpecializationMapEntry specEntries[5];
			for (int i = 0; i < 5; ++i) {
				specEntries[i].constantID = i;
				specEntries[i].offset = i * sizeof(uint32_t);
				specEntries[i].size = sizeof(uint32_t);
			}

			VkSpecializationInfo specInfo = {};
			specInfo.mapEntryCount = 5;
			specInfo.pMapEntries = specEntries;
			specInfo.dataSize = sizeof(SpecializationData);
			specInfo.pData = &specData;

			VkPipelineShaderStageCreateInfo shaderStageInfo = {};
			shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderStageInfo.module = universalShader;
			shaderStageInfo.pName = "main";
			shaderStageInfo.pSpecializationInfo = &specInfo;

			VkComputePipelineCreateInfo pipelineInfo = {};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipelineInfo.stage = shaderStageInfo;
			pipelineInfo.layout = this->impl->universalPreFFTPipelineLayout;

			int pipelineIdx = useIntermediate * 3 + interpolation;  // Linear index: 0-5
			checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &this->impl->universalPipelines[pipelineIdx]));
		}
	}

	// Allocate universal pre-FFT descriptor sets
	std::vector<VkDescriptorSetLayout> universalLayouts(this->impl->numCommandBuffers, this->impl->universalPreFFTDescriptorSetLayout);

	VkDescriptorSetAllocateInfo universalAllocInfo = {};
	universalAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	universalAllocInfo.descriptorPool = this->impl->descriptorPool;
	universalAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	universalAllocInfo.pSetLayouts = universalLayouts.data();

	this->impl->universalPreFFTDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &universalAllocInfo, this->impl->universalPreFFTDescriptorSets.data()));

	// Update universal pre-FFT descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> descriptorWrites(7);

		// Binding 0: Input buffer A (deviceFftBuffers[i])
		VkDescriptorBufferInfo inputBufferAInfo = {};
		inputBufferAInfo.buffer = this->impl->deviceFftBuffers[i];
		inputBufferAInfo.offset = 0;
		inputBufferAInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = this->impl->universalPreFFTDescriptorSets[i];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &inputBufferAInfo;

		// Binding 1: Input buffer B (deviceIntermediateBuffers[i])
		VkDescriptorBufferInfo inputBufferBInfo = {};
		inputBufferBInfo.buffer = this->impl->deviceIntermediateBuffers[i];
		inputBufferBInfo.offset = 0;
		inputBufferBInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = this->impl->universalPreFFTDescriptorSets[i];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &inputBufferBInfo;

		// Binding 2: Resample curve buffer
		VkDescriptorBufferInfo resampleInfo = {};
		resampleInfo.buffer = this->impl->resampleCurveBuffer;
		resampleInfo.offset = 0;
		resampleInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = this->impl->universalPreFFTDescriptorSets[i];
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[2].descriptorCount = 1;
		descriptorWrites[2].pBufferInfo = &resampleInfo;

		// Binding 3: Window curve buffer
		VkDescriptorBufferInfo windowInfo = {};
		windowInfo.buffer = this->impl->windowCurveBuffer;
		windowInfo.offset = 0;
		windowInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[3].dstSet = this->impl->universalPreFFTDescriptorSets[i];
		descriptorWrites[3].dstBinding = 3;
		descriptorWrites[3].dstArrayElement = 0;
		descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[3].descriptorCount = 1;
		descriptorWrites[3].pBufferInfo = &windowInfo;

		// Binding 4: Dispersion phase buffer
		VkDescriptorBufferInfo dispersionInfo = {};
		dispersionInfo.buffer = this->impl->dispersionCurveBuffer;
		dispersionInfo.offset = 0;
		dispersionInfo.range = VK_WHOLE_SIZE;

		descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[4].dstSet = this->impl->universalPreFFTDescriptorSets[i];
		descriptorWrites[4].dstBinding = 4;
		descriptorWrites[4].dstArrayElement = 0;
		descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[4].descriptorCount = 1;
		descriptorWrites[4].pBufferInfo = &dispersionInfo;

		// Binding 5: Output buffer A (deviceIntermediateBuffers[i])
		VkDescriptorBufferInfo outputInfoA = {};
		outputInfoA.buffer = this->impl->deviceIntermediateBuffers[i];
		outputInfoA.offset = 0;
		outputInfoA.range = VK_WHOLE_SIZE;

		descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[5].dstSet = this->impl->universalPreFFTDescriptorSets[i];
		descriptorWrites[5].dstBinding = 5;
		descriptorWrites[5].dstArrayElement = 0;
		descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[5].descriptorCount = 1;
		descriptorWrites[5].pBufferInfo = &outputInfoA;

		// Binding 6: Output buffer B (deviceFftBuffers[i])
		VkDescriptorBufferInfo outputInfoB = {};
		outputInfoB.buffer = this->impl->deviceFftBuffers[i];
		outputInfoB.offset = 0;
		outputInfoB.range = VK_WHOLE_SIZE;

		descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[6].dstSet = this->impl->universalPreFFTDescriptorSets[i];
		descriptorWrites[6].dstBinding = 6;
		descriptorWrites[6].dstArrayElement = 0;
		descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[6].descriptorCount = 1;
		descriptorWrites[6].pBufferInfo = &outputInfoB;

		this->impl->updateDescriptorSetsTagged("UniversalPreFFT", static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data());
	}

	// ============================================
	// Universal Post-FFT Processing Shader
	// ============================================
	// This shader merges: Fixed Pattern Noise Removal + Magnitude Calculation + Log/Linear Scaling + Normalization
	// Uses specialization constants for compile-time optimization
	// Replaces old truncate + postprocess pipelines

	// Create descriptor set layout for universal post-FFT shader (3 bindings)
	std::vector<VkDescriptorSetLayoutBinding> universalPostFFTBindings(3);

	// Binding 0: Input buffer (deviceFftBuffer, complex post-IFFT data)
	universalPostFFTBindings[0].binding = 0;
	universalPostFFTBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPostFFTBindings[0].descriptorCount = 1;
	universalPostFFTBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 1: Mean A-line buffer (for fixed pattern noise removal)
	universalPostFFTBindings[1].binding = 1;
	universalPostFFTBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPostFFTBindings[1].descriptorCount = 1;
	universalPostFFTBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	// Binding 2: Output buffer (deviceProcessedBuffer, real magnitude data)
	universalPostFFTBindings[2].binding = 2;
	universalPostFFTBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	universalPostFFTBindings[2].descriptorCount = 1;
	universalPostFFTBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo universalPostFFTDescriptorSetLayoutInfo = {};
	universalPostFFTDescriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	universalPostFFTDescriptorSetLayoutInfo.bindingCount = static_cast<uint32_t>(universalPostFFTBindings.size());
	universalPostFFTDescriptorSetLayoutInfo.pBindings = universalPostFFTBindings.data();

	checkVulkanErrors(vkCreateDescriptorSetLayout(this->impl->device, &universalPostFFTDescriptorSetLayoutInfo, nullptr, &this->impl->universalPostFFTDescriptorSetLayout));

	// Create pipeline layout for universal post-FFT shader
	VkPushConstantRange universalPostFFTPushConstantRange = {};
	universalPostFFTPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	universalPostFFTPushConstantRange.offset = 0;
	universalPostFFTPushConstantRange.size = sizeof(uint32_t) * 3 + sizeof(float) * 4;  // fullSignalLength, outputSignalLength, samplesPerBuffer, grayscaleMax, grayscaleMin, addend, multiplicator

	VkPipelineLayoutCreateInfo universalPostFFTPipelineLayoutInfo = {};
	universalPostFFTPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	universalPostFFTPipelineLayoutInfo.setLayoutCount = 1;
	universalPostFFTPipelineLayoutInfo.pSetLayouts = &this->impl->universalPostFFTDescriptorSetLayout;
	universalPostFFTPipelineLayoutInfo.pushConstantRangeCount = 1;
	universalPostFFTPipelineLayoutInfo.pPushConstantRanges = &universalPostFFTPushConstantRange;

	checkVulkanErrors(vkCreatePipelineLayout(this->impl->device, &universalPostFFTPipelineLayoutInfo, nullptr, &this->impl->universalPostFFTPipelineLayout));

	// Load and compile universal post-FFT shader
	std::string universalPostFFTShaderPath = "shaders/universal_postfft_processing.comp";
	std::string universalPostFFTShaderSource = loadShaderSource(universalPostFFTShaderPath);
	std::vector<uint32_t> universalPostFFTSPIRV = compileGLSLToSPIRV(universalPostFFTShaderSource, universalPostFFTShaderPath, shaderc_compute_shader);

	VkShaderModule universalPostFFTShader = createShaderModule(this->impl->device, universalPostFFTSPIRV);
	this->impl->shaderModules.push_back(universalPostFFTShader);

	// Create pipeline variants with different specialization constants
	// We create variants for: FPN (enabled/disabled) × log scaling (log/linear)
	for (int enableFPN = 0; enableFPN <= 1; ++enableFPN) {
		for (int logScaling = 0; logScaling <= 1; ++logScaling) {
			// Setup specialization constants
			struct SpecializationData {
				uint32_t enableFixedPatternNoise;
				uint32_t logScaling;
			} specData;

			specData.enableFixedPatternNoise = enableFPN;
			specData.logScaling = logScaling;

			VkSpecializationMapEntry specEntries[2];
			for (int i = 0; i < 2; ++i) {
				specEntries[i].constantID = i;
				specEntries[i].offset = i * sizeof(uint32_t);
				specEntries[i].size = sizeof(uint32_t);
			}

			VkSpecializationInfo specInfo = {};
			specInfo.mapEntryCount = 2;
			specInfo.pMapEntries = specEntries;
			specInfo.dataSize = sizeof(SpecializationData);
			specInfo.pData = &specData;

			VkPipelineShaderStageCreateInfo shaderStageInfo = {};
			shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			shaderStageInfo.module = universalPostFFTShader;
			shaderStageInfo.pName = "main";
			shaderStageInfo.pSpecializationInfo = &specInfo;

			VkComputePipelineCreateInfo pipelineInfo = {};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipelineInfo.stage = shaderStageInfo;
			pipelineInfo.layout = this->impl->universalPostFFTPipelineLayout;

			int pipelineIdx = enableFPN * 2 + logScaling;  // Linear index: 0-3
			checkVulkanErrors(vkCreateComputePipelines(this->impl->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &this->impl->universalPostFFTPipelines[pipelineIdx]));
		}
	}

	// Allocate universal post-FFT descriptor sets
	std::vector<VkDescriptorSetLayout> universalPostFFTLayouts(this->impl->numCommandBuffers, this->impl->universalPostFFTDescriptorSetLayout);

	VkDescriptorSetAllocateInfo universalPostFFTAllocInfo = {};
	universalPostFFTAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	universalPostFFTAllocInfo.descriptorPool = this->impl->descriptorPool;
	universalPostFFTAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	universalPostFFTAllocInfo.pSetLayouts = universalPostFFTLayouts.data();

	// Allocate 2 descriptor sets per command buffer (one for each input buffer variant)
	std::vector<VkDescriptorSetLayout> universalPostFFTLayoutsExpanded(this->impl->numCommandBuffers * 2, this->impl->universalPostFFTDescriptorSetLayout);

	VkDescriptorSetAllocateInfo universalPostFFTAllocInfoExpanded = {};
	universalPostFFTAllocInfoExpanded.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	universalPostFFTAllocInfoExpanded.descriptorPool = this->impl->descriptorPool;
	universalPostFFTAllocInfoExpanded.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers * 2);
	universalPostFFTAllocInfoExpanded.pSetLayouts = universalPostFFTLayoutsExpanded.data();

	std::vector<VkDescriptorSet> universalPostFFTDescriptorSetsFlat(this->impl->numCommandBuffers * 2);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &universalPostFFTAllocInfoExpanded, universalPostFFTDescriptorSetsFlat.data()));

	// Copy into 2D array structure
	this->impl->universalPostFFTDescriptorSets.resize(this->impl->numCommandBuffers);
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		this->impl->universalPostFFTDescriptorSets[i][0] = universalPostFFTDescriptorSetsFlat[i * 2 + 0];
		this->impl->universalPostFFTDescriptorSets[i][1] = universalPostFFTDescriptorSetsFlat[i * 2 + 1];
	}

	// Update universal post-FFT descriptor sets (2 variants per command buffer)
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		// Mean A-line buffer info (same for both variants)
		VkDescriptorBufferInfo meanALineInfo = {};
		meanALineInfo.buffer = this->impl->meanALineBuffer;
		meanALineInfo.offset = 0;
		meanALineInfo.range = VK_WHOLE_SIZE;

		// Output buffer info (same for both variants)
		VkDescriptorBufferInfo outputInfo = {};
		outputInfo.buffer = this->impl->deviceProcessedBuffers[i];
		outputInfo.offset = 0;
		outputInfo.range = VK_WHOLE_SIZE;

		// --- Variant 0: Input from deviceFftBuffers[i] ---
		{
			std::vector<VkWriteDescriptorSet> descriptorWrites(3);

			// Binding 0: Input buffer (deviceFftBuffers[i])
			VkDescriptorBufferInfo inputInfo = {};
			inputInfo.buffer = this->impl->deviceFftBuffers[i];
			inputInfo.offset = 0;
			inputInfo.range = VK_WHOLE_SIZE;

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = this->impl->universalPostFFTDescriptorSets[i][0];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &inputInfo;

			// Binding 1: Mean A-line buffer
			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = this->impl->universalPostFFTDescriptorSets[i][0];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &meanALineInfo;

			// Binding 2: Output buffer (deviceProcessedBuffer)
			descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet = this->impl->universalPostFFTDescriptorSets[i][0];
			descriptorWrites[2].dstBinding = 2;
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].pBufferInfo = &outputInfo;

			this->impl->updateDescriptorSetsTagged("UniversalPostFFT_variant0", static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data());
		}

		// --- Variant 1: Input from deviceIntermediateBuffers[i] ---
		{
			std::vector<VkWriteDescriptorSet> descriptorWrites(3);

			// Binding 0: Input buffer (deviceIntermediateBuffers[i])
			VkDescriptorBufferInfo inputInfo = {};
			inputInfo.buffer = this->impl->deviceIntermediateBuffers[i];
			inputInfo.offset = 0;
			inputInfo.range = VK_WHOLE_SIZE;

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = this->impl->universalPostFFTDescriptorSets[i][1];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &inputInfo;

			// Binding 1: Mean A-line buffer
			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = this->impl->universalPostFFTDescriptorSets[i][1];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &meanALineInfo;

			// Binding 2: Output buffer (deviceProcessedBuffer)
			descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet = this->impl->universalPostFFTDescriptorSets[i][1];
			descriptorWrites[2].dstBinding = 2;
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].pBufferInfo = &outputInfo;

			this->impl->updateDescriptorSetsTagged("UniversalPostFFT_variant1", static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data());
		}
	}

	// ============================================
	// Allocate and Update FPN Determination Descriptor Sets
	// ============================================

	std::vector<VkDescriptorSetLayout> fpnDeterminationLayouts(this->impl->numCommandBuffers, this->impl->fpnDeterminationDescriptorSetLayout);

	VkDescriptorSetAllocateInfo fpnDeterminationAllocInfo = {};
	fpnDeterminationAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	fpnDeterminationAllocInfo.descriptorPool = this->impl->descriptorPool;
	fpnDeterminationAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	fpnDeterminationAllocInfo.pSetLayouts = fpnDeterminationLayouts.data();

	// Allocate 2 descriptor sets per command buffer (one for each input buffer variant)
	std::vector<VkDescriptorSetLayout> fpnDeterminationLayoutsExpanded(this->impl->numCommandBuffers * 2, this->impl->fpnDeterminationDescriptorSetLayout);

	VkDescriptorSetAllocateInfo fpnDeterminationAllocInfoExpanded = {};
	fpnDeterminationAllocInfoExpanded.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	fpnDeterminationAllocInfoExpanded.descriptorPool = this->impl->descriptorPool;
	fpnDeterminationAllocInfoExpanded.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers * 2);
	fpnDeterminationAllocInfoExpanded.pSetLayouts = fpnDeterminationLayoutsExpanded.data();

	std::vector<VkDescriptorSet> fpnDeterminationDescriptorSetsFlat(this->impl->numCommandBuffers * 2);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &fpnDeterminationAllocInfoExpanded, fpnDeterminationDescriptorSetsFlat.data()));

	// Copy into 2D array structure
	this->impl->fpnDeterminationDescriptorSets.resize(this->impl->numCommandBuffers);
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		this->impl->fpnDeterminationDescriptorSets[i][0] = fpnDeterminationDescriptorSetsFlat[i * 2 + 0];
		this->impl->fpnDeterminationDescriptorSets[i][1] = fpnDeterminationDescriptorSetsFlat[i * 2 + 1];
	}

	// Update FPN determination descriptor sets (2 variants per command buffer)
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		// Mean A-line buffer info (same for both variants)
		VkDescriptorBufferInfo meanALineInfo = {};
		meanALineInfo.buffer = this->impl->meanALineBuffer;
		meanALineInfo.offset = 0;
		meanALineInfo.range = VK_WHOLE_SIZE;

		// --- Variant 0: Input from deviceFftBuffers[i] ---
		{
			std::vector<VkWriteDescriptorSet> fpnDescriptorWrites(2);

			// Binding 0: Input buffer (deviceFftBuffers[i])
			VkDescriptorBufferInfo inputInfo = {};
			inputInfo.buffer = this->impl->deviceFftBuffers[i];
			inputInfo.offset = 0;
			inputInfo.range = VK_WHOLE_SIZE;

			fpnDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			fpnDescriptorWrites[0].dstSet = this->impl->fpnDeterminationDescriptorSets[i][0];
			fpnDescriptorWrites[0].dstBinding = 0;
			fpnDescriptorWrites[0].dstArrayElement = 0;
			fpnDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			fpnDescriptorWrites[0].descriptorCount = 1;
			fpnDescriptorWrites[0].pBufferInfo = &inputInfo;

			// Binding 1: Mean A-line output buffer
			fpnDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			fpnDescriptorWrites[1].dstSet = this->impl->fpnDeterminationDescriptorSets[i][0];
			fpnDescriptorWrites[1].dstBinding = 1;
			fpnDescriptorWrites[1].dstArrayElement = 0;
			fpnDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			fpnDescriptorWrites[1].descriptorCount = 1;
			fpnDescriptorWrites[1].pBufferInfo = &meanALineInfo;

			this->impl->updateDescriptorSetsTagged("FPNDetermination_variant0", static_cast<uint32_t>(fpnDescriptorWrites.size()), fpnDescriptorWrites.data());
		}

		// --- Variant 1: Input from deviceIntermediateBuffers[i] ---
		{
			std::vector<VkWriteDescriptorSet> fpnDescriptorWrites(2);

			// Binding 0: Input buffer (deviceIntermediateBuffers[i])
			VkDescriptorBufferInfo inputInfo = {};
			inputInfo.buffer = this->impl->deviceIntermediateBuffers[i];
			inputInfo.offset = 0;
			inputInfo.range = VK_WHOLE_SIZE;

			fpnDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			fpnDescriptorWrites[0].dstSet = this->impl->fpnDeterminationDescriptorSets[i][1];
			fpnDescriptorWrites[0].dstBinding = 0;
			fpnDescriptorWrites[0].dstArrayElement = 0;
			fpnDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			fpnDescriptorWrites[0].descriptorCount = 1;
			fpnDescriptorWrites[0].pBufferInfo = &inputInfo;

			// Binding 1: Mean A-line output buffer
			fpnDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			fpnDescriptorWrites[1].dstSet = this->impl->fpnDeterminationDescriptorSets[i][1];
			fpnDescriptorWrites[1].dstBinding = 1;
			fpnDescriptorWrites[1].dstArrayElement = 0;
			fpnDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			fpnDescriptorWrites[1].descriptorCount = 1;
			fpnDescriptorWrites[1].pBufferInfo = &meanALineInfo;

			this->impl->updateDescriptorSetsTagged("FPNDetermination_variant1", static_cast<uint32_t>(fpnDescriptorWrites.size()), fpnDescriptorWrites.data());
		}
	}

	// ============================================
	// Allocate and Update Background Subtraction Descriptor Sets
	// ============================================

	std::vector<VkDescriptorSetLayout> backgroundSubtractionLayouts(this->impl->numCommandBuffers, this->impl->backgroundSubtractionDescriptorSetLayout);

	VkDescriptorSetAllocateInfo backgroundSubtractionAllocInfo = {};
	backgroundSubtractionAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	backgroundSubtractionAllocInfo.descriptorPool = this->impl->descriptorPool;
	backgroundSubtractionAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	backgroundSubtractionAllocInfo.pSetLayouts = backgroundSubtractionLayouts.data();

	this->impl->backgroundSubtractionDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &backgroundSubtractionAllocInfo, this->impl->backgroundSubtractionDescriptorSets.data()));

	// Update background subtraction descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> bgDescriptorWrites(2);

		// Binding 0: Data buffer (deviceProcessedBuffers[i] - magnitude data)
		VkDescriptorBufferInfo dataInfo = {};
		dataInfo.buffer = this->impl->deviceProcessedBuffers[i];
		dataInfo.offset = 0;
		dataInfo.range = VK_WHOLE_SIZE;

		bgDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		bgDescriptorWrites[0].dstSet = this->impl->backgroundSubtractionDescriptorSets[i];
		bgDescriptorWrites[0].dstBinding = 0;
		bgDescriptorWrites[0].dstArrayElement = 0;
		bgDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bgDescriptorWrites[0].descriptorCount = 1;
		bgDescriptorWrites[0].pBufferInfo = &dataInfo;

		// Binding 1: Background profile buffer
		VkDescriptorBufferInfo backgroundInfo = {};
		backgroundInfo.buffer = this->impl->postProcBackgroundBuffer;
		backgroundInfo.offset = 0;
		backgroundInfo.range = VK_WHOLE_SIZE;

		bgDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		bgDescriptorWrites[1].dstSet = this->impl->backgroundSubtractionDescriptorSets[i];
		bgDescriptorWrites[1].dstBinding = 1;
		bgDescriptorWrites[1].dstArrayElement = 0;
		bgDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bgDescriptorWrites[1].descriptorCount = 1;
		bgDescriptorWrites[1].pBufferInfo = &backgroundInfo;

		this->impl->updateDescriptorSetsTagged("BackgroundSubtraction", static_cast<uint32_t>(bgDescriptorWrites.size()), bgDescriptorWrites.data());
	}

	// ============================================
	// Allocate and Update Background Recording Descriptor Sets
	// ============================================

	std::vector<VkDescriptorSetLayout> backgroundRecordingLayouts(this->impl->numCommandBuffers, this->impl->backgroundRecordingDescriptorSetLayout);

	VkDescriptorSetAllocateInfo backgroundRecordingAllocInfo = {};
	backgroundRecordingAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	backgroundRecordingAllocInfo.descriptorPool = this->impl->descriptorPool;
	backgroundRecordingAllocInfo.descriptorSetCount = static_cast<uint32_t>(this->impl->numCommandBuffers);
	backgroundRecordingAllocInfo.pSetLayouts = backgroundRecordingLayouts.data();

	this->impl->backgroundRecordingDescriptorSets.resize(this->impl->numCommandBuffers);
	checkVulkanErrors(vkAllocateDescriptorSets(this->impl->device, &backgroundRecordingAllocInfo, this->impl->backgroundRecordingDescriptorSets.data()));

	// Update background recording descriptor sets
	for (int i = 0; i < this->impl->numCommandBuffers; ++i) {
		std::vector<VkWriteDescriptorSet> bgRecDescriptorWrites(2);

		// Binding 0: Input buffer (deviceProcessedBuffers[i] - magnitude data)
		VkDescriptorBufferInfo inputInfo = {};
		inputInfo.buffer = this->impl->deviceProcessedBuffers[i];
		inputInfo.offset = 0;
		inputInfo.range = VK_WHOLE_SIZE;

		bgRecDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		bgRecDescriptorWrites[0].dstSet = this->impl->backgroundRecordingDescriptorSets[i];
		bgRecDescriptorWrites[0].dstBinding = 0;
		bgRecDescriptorWrites[0].dstArrayElement = 0;
		bgRecDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bgRecDescriptorWrites[0].descriptorCount = 1;
		bgRecDescriptorWrites[0].pBufferInfo = &inputInfo;

		// Binding 1: Background buffer (output: averaged background profile)
		VkDescriptorBufferInfo backgroundInfo = {};
		backgroundInfo.buffer = this->impl->postProcBackgroundBuffer;
		backgroundInfo.offset = 0;
		backgroundInfo.range = VK_WHOLE_SIZE;

		bgRecDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		bgRecDescriptorWrites[1].dstSet = this->impl->backgroundRecordingDescriptorSets[i];
		bgRecDescriptorWrites[1].dstBinding = 1;
		bgRecDescriptorWrites[1].dstArrayElement = 0;
		bgRecDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bgRecDescriptorWrites[1].descriptorCount = 1;
		bgRecDescriptorWrites[1].pBufferInfo = &backgroundInfo;

		this->impl->updateDescriptorSetsTagged("BackgroundRecording", static_cast<uint32_t>(bgRecDescriptorWrites.size()), bgRecDescriptorWrites.data());
	}

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
		this->impl->logPoolOp("DESTROY", this->impl->descriptorPool);
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

	// Destroy DC removal pipeline resources
	if (this->impl->dcRemovalDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->dcRemovalDescriptorSetLayout, nullptr);
		this->impl->dcRemovalDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->dcRemovalPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->dcRemovalPipelineLayout, nullptr);
		this->impl->dcRemovalPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy FPN determination pipeline resources
	if (this->impl->fpnDeterminationDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->fpnDeterminationDescriptorSetLayout, nullptr);
		this->impl->fpnDeterminationDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->fpnDeterminationPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->fpnDeterminationPipelineLayout, nullptr);
		this->impl->fpnDeterminationPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy background subtraction pipeline resources
	if (this->impl->backgroundSubtractionPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(this->impl->device, this->impl->backgroundSubtractionPipeline, nullptr);
		this->impl->backgroundSubtractionPipeline = VK_NULL_HANDLE;
	}

	if (this->impl->backgroundSubtractionDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->backgroundSubtractionDescriptorSetLayout, nullptr);
		this->impl->backgroundSubtractionDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->backgroundSubtractionPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->backgroundSubtractionPipelineLayout, nullptr);
		this->impl->backgroundSubtractionPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy background recording pipeline resources
	if (this->impl->backgroundRecordingPipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(this->impl->device, this->impl->backgroundRecordingPipeline, nullptr);
		this->impl->backgroundRecordingPipeline = VK_NULL_HANDLE;
	}

	if (this->impl->backgroundRecordingDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->backgroundRecordingDescriptorSetLayout, nullptr);
		this->impl->backgroundRecordingDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->backgroundRecordingPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->backgroundRecordingPipelineLayout, nullptr);
		this->impl->backgroundRecordingPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy universal pre-FFT pipeline resources
	for (int i = 0; i < 6; ++i) {
		if (this->impl->universalPipelines[i] != VK_NULL_HANDLE) {
			vkDestroyPipeline(this->impl->device, this->impl->universalPipelines[i], nullptr);
			this->impl->universalPipelines[i] = VK_NULL_HANDLE;
		}
	}

	if (this->impl->universalPreFFTDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->universalPreFFTDescriptorSetLayout, nullptr);
		this->impl->universalPreFFTDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->universalPreFFTPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->universalPreFFTPipelineLayout, nullptr);
		this->impl->universalPreFFTPipelineLayout = VK_NULL_HANDLE;
	}

	// Destroy universal post-FFT pipeline resources
	for (int i = 0; i < 4; ++i) {
		if (this->impl->universalPostFFTPipelines[i] != VK_NULL_HANDLE) {
			vkDestroyPipeline(this->impl->device, this->impl->universalPostFFTPipelines[i], nullptr);
			this->impl->universalPostFFTPipelines[i] = VK_NULL_HANDLE;
		}
	}

	if (this->impl->universalPostFFTDescriptorSetLayout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(this->impl->device, this->impl->universalPostFFTDescriptorSetLayout, nullptr);
		this->impl->universalPostFFTDescriptorSetLayout = VK_NULL_HANDLE;
	}

	if (this->impl->universalPostFFTPipelineLayout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(this->impl->device, this->impl->universalPostFFTPipelineLayout, nullptr);
		this->impl->universalPostFFTPipelineLayout = VK_NULL_HANDLE;
	}

}

// ============================================
// Static Device Management Methods
// ============================================

std::vector<VulkanDeviceInfo> VulkanBackend::getAvailableDevices() {
	std::vector<VulkanDeviceInfo> devices;

	// Create a minimal temporary instance for device enumeration
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "OCTproEngine Device Query";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "OCTproEngine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = 0;
	createInfo.enabledLayerCount = 0;

	VkInstance tempInstance;
	VkResult result = vkCreateInstance(&createInfo, nullptr, &tempInstance);
	if (result != VK_SUCCESS) {
		// Failed to create instance - return empty list
		return devices;
	}

	// Enumerate physical devices
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(tempInstance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		vkDestroyInstance(tempInstance, nullptr);
		return devices;
	}

	std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
	vkEnumeratePhysicalDevices(tempInstance, &deviceCount, physicalDevices.data());

	// Query properties for each device
	for (uint32_t i = 0; i < deviceCount; i++) {
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevices[i], &deviceProperties);

		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevices[i], &memProperties);

		// Calculate total memory from all device-local heaps
		size_t totalMemory = 0;
		for (uint32_t j = 0; j < memProperties.memoryHeapCount; j++) {
			if (memProperties.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
				totalMemory += memProperties.memoryHeaps[j].size;
			}
		}

		VulkanDeviceInfo info;
		info.deviceId = static_cast<int>(i);
		info.name = std::string(deviceProperties.deviceName);
		info.totalMemory = totalMemory;
		info.freeMemory = totalMemory; // Vulkan doesn't provide free memory query
		info.apiVersionMajor = VK_VERSION_MAJOR(deviceProperties.apiVersion);
		info.apiVersionMinor = VK_VERSION_MINOR(deviceProperties.apiVersion);
		info.apiVersionPatch = VK_VERSION_PATCH(deviceProperties.apiVersion);
		info.driverVersion = deviceProperties.driverVersion;
		info.maxWorkGroupSize = deviceProperties.limits.maxComputeWorkGroupInvocations;
		info.maxComputeSharedMemorySize = deviceProperties.limits.maxComputeSharedMemorySize;
		info.isAvailable = true;

		devices.push_back(info);
	}

	vkDestroyInstance(tempInstance, nullptr);
	return devices;
}

} // namespace ope
