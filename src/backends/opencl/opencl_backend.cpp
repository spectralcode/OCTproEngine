#include "opencl_backend.h"
#include "opencl_kernels.h"

//	VkFFT backend selection: 3 = OpenCL
#define VKFFT_BACKEND 3
#include <vkFFT/vkFFT.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <map>
#include <thread>

//	Uncomment to enable profiling (adds overhead, use only for debugging)
//#define OPENCL_PROFILE_TIMING

#ifdef OPENCL_PROFILE_TIMING
#include <iomanip>
struct ProfilingData {
	double inputTransferMs = 0;
	double fftMs = 0;
	double kernelsMs = 0;
	double outputTransferMs = 0;
	double linearizationMs = 0;
	double windowingMs = 0;
	double dispersionMs = 0;
	double postProcessMs = 0;
	size_t inputBytes = 0;
	size_t outputBytes = 0;
	int frameCount = 0;
};
static ProfilingData g_profiling;
static std::mutex g_profilingMutex;

static double getEventDurationMs(cl_event event) {
	cl_ulong start, end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
	return (end - start) / 1000000.0;  //	Convert ns to ms
}
#endif

// Helper macro for checking OpenCL errors
#define checkOpenClErrors(call) \
	do { \
		cl_int err = call; \
		if (err != CL_SUCCESS) { \
			std::stringstream ss; \
			ss << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - code: " << err; \
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
// Implementation Structure
// ============================================

struct OpenClBackend::Impl {
	//	Configuration
	ProcessorConfiguration config;

	//	OpenCL parameters
	int platformId = 0;
	int deviceId = -1;
	bool preferGpu = true;
	int numCommandQueues = 3;
	size_t workGroupSize = 128;
	size_t maxLocalMemSize = 0;
	bool openclInitialized = false;

	//	OpenCL objects
	cl_platform_id platform = nullptr;
	cl_device_id device = nullptr;
	cl_context context = nullptr;
	std::vector<cl_command_queue> commandQueues;
	cl_program program = nullptr;


	//	OpenCL kernels
	cl_kernel kernelInputToComplex = nullptr;
	cl_kernel kernelInputToComplexBitshift = nullptr;
	cl_kernel kernelRollingAverage = nullptr;
	cl_kernel kernelKLinearization = nullptr;
	cl_kernel kernelKLinearizationCubic = nullptr;
	cl_kernel kernelKLinearizationLanczos = nullptr;
	cl_kernel kernelWindowing = nullptr;
	cl_kernel kernelKLinearizationAndWindowing = nullptr;
	cl_kernel kernelKLinearizationCubicAndWindowing = nullptr;
	cl_kernel kernelKLinearizationLanczosAndWindowing = nullptr;
	cl_kernel kernelDispersionCompensation = nullptr;
	cl_kernel kernelKLinearizationAndWindowingAndDispersion = nullptr;
	cl_kernel kernelKLinearizationCubicAndWindowingAndDispersion = nullptr;
	cl_kernel kernelKLinearizationLanczosAndWindowingAndDispersion = nullptr;
	cl_kernel kernelPostProcessTruncateLog = nullptr;
	cl_kernel kernelPostProcessTruncateLin = nullptr;
	cl_kernel kernelGetMinimumVarianceMean = nullptr;
	cl_kernel kernelMeanALineSubtraction = nullptr;
	cl_kernel kernelBscanFlip = nullptr;
	cl_kernel kernelFillSinusoidalScanCorrectionCurve = nullptr;
	cl_kernel kernelSinusoidalScanCorrection = nullptr;
	cl_kernel kernelGetPostProcessBackground = nullptr;
	cl_kernel kernelPostProcessBackgroundSubtraction = nullptr;

	//	Data dimensions
	int signalLength = 0;
	int ascansPerBscan = 0;
	int bscansPerBuffer = 0;
	int samplesPerBuffer = 0;
	int bytesPerSample = 0;

	//	Input buffer management (queue-based, thread-safe)
	int numInputBuffers = numCommandQueues;
	std::vector<IOBuffer> hostInputBuffers;
	std::queue<IOBuffer*> freeBuffersQueue;
	std::mutex freeQueueMutex;
	std::condition_variable freeQueueCV;

	//	Device buffers
	std::vector<cl_mem> d_inputBuffers;

	//	Processing buffers
	std::vector<cl_mem> d_fftBuffers;
	std::vector<cl_mem> d_inputLinearizedBuffers;
	std::vector<cl_mem> d_outputBuffers;

	//	Curve buffers
	cl_mem d_resampleCurve = nullptr;
	cl_mem d_windowCurve = nullptr;
	cl_mem d_phaseCartesian = nullptr;

	//	Fixed pattern noise removal
	cl_mem d_meanALine = nullptr;
	bool fixedPatternNoiseDetermined = false;
	std::vector<float> recordedFixedPatternNoise;

	//	Post-processing
	cl_mem d_postProcBackgroundLine = nullptr;
	std::vector<cl_mem> d_sinusoidalScanTmpBuffers;
	cl_mem d_sinusoidalResampleCurve = nullptr;
	bool postProcessBackgroundRecordingRequested = false;
	bool postProcessBackgroundUpdated = false;
	std::vector<float> recordedPostProcessBackground;

	//	VkFFT
	VkFFTConfiguration fftConfig;
	VkFFTApplication fftApp;
	bool fftDebugPrinted = false;

	//	Host output buffers (rotating pool for ordered callback delivery)
	std::vector<IOBuffer> hostOutputBuffers;
	std::atomic<int> currentOutputBuffer{0};

	//	Callback
	std::function<void(const IOBuffer&)> callback;

	//	Pre-allocated callback data pool
	struct CallbackData {
		Impl* impl;
		IOBuffer* inputBuffer;
		IOBuffer* outputBuffer;
		cl_event event;  // Track the associated event for cleanup
	};
	std::vector<CallbackData> callbackDataPool;
	std::atomic<int> nextCallbackIndex{0};

	// Callback ordering. delivers callbacks in buffer submission order
	std::mutex callbackQueueMutex;
	std::condition_variable callbackQueueCV;
	std::map<uint64_t, IOBuffer*> pendingCallbacks;  // bufferId -> outputBuffer
	uint64_t nextExpectedCallback = 0;
	std::thread callbackWorkerThread;
	std::atomic<bool> callbackWorkerRunning{false};

	// Semaphore to limit in-flight output buffers (prevents buffer reuse before callback delivery)
	std::mutex outputSemaphoreMutex;
	std::condition_variable outputSemaphoreCV;
	int availableOutputBuffers = 0;

	Impl() = default;

	~Impl() {
		//	Cleanup is handled in cleanup() method
	}
};

// ============================================
// Constructor / Destructor
// ============================================

OpenClBackend::OpenClBackend() : impl(std::make_unique<Impl>()) {
}

OpenClBackend::~OpenClBackend() {
	this->cleanup();
}

// ============================================
// Configuration Methods
// ============================================

void OpenClBackend::setNumInputBuffers(int count) {
	if (this->impl->openclInitialized) {
		throw std::runtime_error("Cannot change number of input buffers after initialization");
	}
	if (count < 1) {
		throw std::invalid_argument("Number of input buffers must be at least 1");
	}
	this->impl->numInputBuffers = count;
}

void OpenClBackend::setNumCommandQueues(int numQueues) {
	if (this->impl->openclInitialized) {
		throw std::runtime_error("Cannot change number of command queues after initialization");
	}
	this->impl->numCommandQueues = numQueues;
}

void OpenClBackend::setWorkGroupSize(int workGroupSize) {
	if (this->impl->openclInitialized) {
		throw std::runtime_error("Cannot change work group size after initialization");
	}
	this->impl->workGroupSize = workGroupSize;
}

void OpenClBackend::setPlatformId(int platformId) {
	if (this->impl->openclInitialized) {
		throw std::runtime_error("Cannot change platform ID after initialization");
	}
	this->impl->platformId = platformId;
}

void OpenClBackend::setDeviceId(int deviceId) {
	if (this->impl->openclInitialized) {
		throw std::runtime_error("Cannot change device ID after initialization");
	}
	this->impl->deviceId = deviceId;
}

void OpenClBackend::setPreferGpu(bool prefer) {
	if (this->impl->openclInitialized) {
		throw std::runtime_error("Cannot change preferGpu after initialization");
	}
	this->impl->preferGpu = prefer;
}

int OpenClBackend::getNumCommandQueues() const {
	return this->impl->numCommandQueues;
}

int OpenClBackend::getWorkGroupSize() const {
	return static_cast<int>(this->impl->workGroupSize);
}

int OpenClBackend::getCurrentPlatformId() const {
	return this->impl->platformId;
}

int OpenClBackend::getCurrentDeviceId() const {
	return this->impl->deviceId;
}

// ============================================
// Error Checking Helper
// ============================================

void OpenClBackend::checkOpenClError(cl_int error, const char* context) {
	if (error != CL_SUCCESS) {
		std::stringstream ss;
		ss << "OpenCL error in " << context << ": code " << error;
		throw std::runtime_error(ss.str());
	}
}

// ============================================
// Static Device Management Methods
// ============================================

std::vector<OpenClDeviceInfo> OpenClBackend::getAvailableDevices() {
	std::vector<OpenClDeviceInfo> devices;

	cl_uint numPlatforms;
	cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms == 0) {
		return devices;
	}

	std::vector<cl_platform_id> platforms(numPlatforms);
	checkOpenClErrors(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));

	for (cl_uint p = 0; p < numPlatforms; p++) {
		char platformName[256];
		checkOpenClErrors(clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr));

		cl_uint numDevices;
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
		if (err != CL_SUCCESS || numDevices == 0) {
			continue;
		}

		std::vector<cl_device_id> devs(numDevices);
		checkOpenClErrors(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devs.data(), nullptr));

		for (cl_uint d = 0; d < numDevices; d++) {
			OpenClDeviceInfo info;
			info.platformId = p;
			info.deviceId = d;
			info.platformName = platformName;

			char deviceName[256];
			checkOpenClErrors(clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr));
			info.deviceName = deviceName;

			checkOpenClErrors(clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(info.deviceType), &info.deviceType, nullptr));
			checkOpenClErrors(clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info.globalMemorySize), &info.globalMemorySize, nullptr));
			checkOpenClErrors(clGetDeviceInfo(devs[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(info.localMemorySize), &info.localMemorySize, nullptr));
			checkOpenClErrors(clGetDeviceInfo(devs[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info.maxWorkGroupSize), &info.maxWorkGroupSize, nullptr));
			checkOpenClErrors(clGetDeviceInfo(devs[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info.maxComputeUnits), &info.maxComputeUnits, nullptr));
			checkOpenClErrors(clGetDeviceInfo(devs[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(info.maxClockFrequency), &info.maxClockFrequency, nullptr));

			info.isAvailable = true;
			devices.push_back(info);
		}
	}

	return devices;
}

bool OpenClBackend::selectDevice(int platformId, int deviceId) {
	try {
		cl_uint numPlatforms;
		checkOpenClErrors(clGetPlatformIDs(0, nullptr, &numPlatforms));
		if (platformId >= static_cast<int>(numPlatforms)) {
			return false;
		}

		std::vector<cl_platform_id> platforms(numPlatforms);
		checkOpenClErrors(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));

		cl_uint numDevices;
		checkOpenClErrors(clGetDeviceIDs(platforms[platformId], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices));
		if (deviceId >= static_cast<int>(numDevices)) {
			return false;
		}

		return true;
	} catch (...) {
		return false;
	}
}

OpenClDeviceInfo OpenClBackend::getDeviceInfo(int platformId, int deviceId) {
	auto devices = getAvailableDevices();
	for (const auto& dev : devices) {
		if (dev.platformId == platformId && dev.deviceId == deviceId) {
			return dev;
		}
	}
	throw std::runtime_error("Device not found");
}

// ============================================
// Helper Methods
// ============================================

void OpenClBackend::createCommandQueues() {
	this->impl->commandQueues.resize(this->impl->numCommandQueues);
	for (int i = 0; i < this->impl->numCommandQueues; i++) {
		cl_int err;
		//	Use in-order queues (like CUDA streams) - operations execute sequentially
#ifdef OPENCL_PROFILE_TIMING
		this->impl->commandQueues[i] = clCreateCommandQueue(this->impl->context, this->impl->device, CL_QUEUE_PROFILING_ENABLE, &err);
#else
		this->impl->commandQueues[i] = clCreateCommandQueue(this->impl->context, this->impl->device, 0, &err);
#endif
		checkOpenClError(err, "clCreateCommandQueue");
	}
}

void OpenClBackend::destroyCommandQueues() {
	for (auto& queue : this->impl->commandQueues) {
		if (queue) {
			clReleaseCommandQueue(queue);
			queue = nullptr;
		}
	}
	this->impl->commandQueues.clear();
}

void OpenClBackend::loadAndBuildKernels() {
	//	Get kernel source
	const char* source = ope::opencl::getKernelSource();
	size_t sourceSize = strlen(source);

	//	Create program
	cl_int err;
	this->impl->program = clCreateProgramWithSource(this->impl->context, 1, &source, &sourceSize, &err);
	checkOpenClError(err, "clCreateProgramWithSource");

	//	Build program
	err = clBuildProgram(this->impl->program, 1, &this->impl->device, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		//	Get build log
		size_t logSize;
		clGetProgramBuildInfo(this->impl->program, this->impl->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
		std::vector<char> log(logSize);
		clGetProgramBuildInfo(this->impl->program, this->impl->device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
		std::cerr << "OpenCL build log:\n" << log.data() << std::endl;
		throw std::runtime_error("clBuildProgram failed");
	}

	//	Create kernels
	this->impl->kernelInputToComplex = clCreateKernel(this->impl->program, ope::opencl::KERNEL_INPUT_TO_COMPLEX, &err);
	checkOpenClError(err, "create inputToComplex kernel");

	this->impl->kernelInputToComplexBitshift = clCreateKernel(this->impl->program, ope::opencl::KERNEL_INPUT_TO_COMPLEX_BITSHIFT, &err);
	checkOpenClError(err, "create inputToComplexBitshift kernel");

	this->impl->kernelRollingAverage = clCreateKernel(this->impl->program, ope::opencl::KERNEL_ROLLING_AVERAGE_BACKGROUND_REMOVAL, &err);
	checkOpenClError(err, "create rollingAverage kernel");

	this->impl->kernelKLinearization = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION, &err);
	checkOpenClError(err, "create kLinearization kernel");

	this->impl->kernelKLinearizationCubic = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_CUBIC, &err);
	checkOpenClError(err, "create kLinearizationCubic kernel");

	this->impl->kernelKLinearizationLanczos = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_LANCZOS, &err);
	checkOpenClError(err, "create kLinearizationLanczos kernel");

	this->impl->kernelWindowing = clCreateKernel(this->impl->program, ope::opencl::KERNEL_WINDOWING, &err);
	checkOpenClError(err, "create windowing kernel");

	this->impl->kernelKLinearizationAndWindowing = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_AND_WINDOWING, &err);
	checkOpenClError(err, "create kLinearizationAndWindowing kernel");

	this->impl->kernelKLinearizationCubicAndWindowing = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_CUBIC_AND_WINDOWING, &err);
	checkOpenClError(err, "create kLinearizationCubicAndWindowing kernel");

	this->impl->kernelKLinearizationLanczosAndWindowing = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_LANCZOS_AND_WINDOWING, &err);
	checkOpenClError(err, "create kLinearizationLanczosAndWindowing kernel");

	this->impl->kernelDispersionCompensation = clCreateKernel(this->impl->program, ope::opencl::KERNEL_DISPERSION_COMPENSATION, &err);
	checkOpenClError(err, "create dispersionCompensation kernel");

	this->impl->kernelKLinearizationAndWindowingAndDispersion = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_AND_WINDOWING_AND_DISPERSION, &err);
	checkOpenClError(err, "create kLinearizationAndWindowingAndDispersion kernel");

	this->impl->kernelKLinearizationCubicAndWindowingAndDispersion = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_CUBIC_AND_WINDOWING_AND_DISPERSION, &err);
	checkOpenClError(err, "create kLinearizationCubicAndWindowingAndDispersion kernel");

	this->impl->kernelKLinearizationLanczosAndWindowingAndDispersion = clCreateKernel(this->impl->program, ope::opencl::KERNEL_KLINEARIZATION_LANCZOS_AND_WINDOWING_AND_DISPERSION, &err);
	checkOpenClError(err, "create kLinearizationLanczosAndWindowingAndDispersion kernel");

	this->impl->kernelPostProcessTruncateLog = clCreateKernel(this->impl->program, ope::opencl::KERNEL_POST_PROCESS_TRUNCATE_LOG, &err);
	checkOpenClError(err, "create postProcessTruncateLog kernel");

	this->impl->kernelPostProcessTruncateLin = clCreateKernel(this->impl->program, ope::opencl::KERNEL_POST_PROCESS_TRUNCATE_LIN, &err);
	checkOpenClError(err, "create postProcessTruncateLin kernel");

	this->impl->kernelGetMinimumVarianceMean = clCreateKernel(this->impl->program, ope::opencl::KERNEL_GET_MINIMUM_VARIANCE_MEAN, &err);
	checkOpenClError(err, "create getMinimumVarianceMean kernel");

	this->impl->kernelMeanALineSubtraction = clCreateKernel(this->impl->program, ope::opencl::KERNEL_MEAN_ALINE_SUBTRACTION, &err);
	checkOpenClError(err, "create meanALineSubtraction kernel");

	this->impl->kernelBscanFlip = clCreateKernel(this->impl->program, ope::opencl::KERNEL_BSCAN_FLIP, &err);
	checkOpenClError(err, "create bscanFlip kernel");

	this->impl->kernelFillSinusoidalScanCorrectionCurve = clCreateKernel(this->impl->program, ope::opencl::KERNEL_FILL_SINUSOIDAL_SCAN_CURVE, &err);
	checkOpenClError(err, "create fillSinusoidalScanCorrectionCurve kernel");

	this->impl->kernelSinusoidalScanCorrection = clCreateKernel(this->impl->program, ope::opencl::KERNEL_SINUSOIDAL_SCAN_CORRECTION, &err);
	checkOpenClError(err, "create sinusoidalScanCorrection kernel");

	this->impl->kernelGetPostProcessBackground = clCreateKernel(this->impl->program, ope::opencl::KERNEL_GET_POST_PROCESS_BACKGROUND, &err);
	checkOpenClError(err, "create getPostProcessBackground kernel");

	this->impl->kernelPostProcessBackgroundSubtraction = clCreateKernel(this->impl->program, ope::opencl::KERNEL_POST_PROCESS_BACKGROUND_SUBTRACTION, &err);
	checkOpenClError(err, "create postProcessBackgroundSubtraction kernel");
}

void OpenClBackend::releaseKernels() {
	if (this->impl->kernelInputToComplex) { clReleaseKernel(this->impl->kernelInputToComplex); this->impl->kernelInputToComplex = nullptr; }
	if (this->impl->kernelInputToComplexBitshift) { clReleaseKernel(this->impl->kernelInputToComplexBitshift); this->impl->kernelInputToComplexBitshift = nullptr; }
	if (this->impl->kernelRollingAverage) { clReleaseKernel(this->impl->kernelRollingAverage); this->impl->kernelRollingAverage = nullptr; }
	if (this->impl->kernelKLinearization) { clReleaseKernel(this->impl->kernelKLinearization); this->impl->kernelKLinearization = nullptr; }
	if (this->impl->kernelKLinearizationCubic) { clReleaseKernel(this->impl->kernelKLinearizationCubic); this->impl->kernelKLinearizationCubic = nullptr; }
	if (this->impl->kernelKLinearizationLanczos) { clReleaseKernel(this->impl->kernelKLinearizationLanczos); this->impl->kernelKLinearizationLanczos = nullptr; }
	if (this->impl->kernelWindowing) { clReleaseKernel(this->impl->kernelWindowing); this->impl->kernelWindowing = nullptr; }
	if (this->impl->kernelKLinearizationAndWindowing) { clReleaseKernel(this->impl->kernelKLinearizationAndWindowing); this->impl->kernelKLinearizationAndWindowing = nullptr; }
	if (this->impl->kernelKLinearizationCubicAndWindowing) { clReleaseKernel(this->impl->kernelKLinearizationCubicAndWindowing); this->impl->kernelKLinearizationCubicAndWindowing = nullptr; }
	if (this->impl->kernelKLinearizationLanczosAndWindowing) { clReleaseKernel(this->impl->kernelKLinearizationLanczosAndWindowing); this->impl->kernelKLinearizationLanczosAndWindowing = nullptr; }
	if (this->impl->kernelDispersionCompensation) { clReleaseKernel(this->impl->kernelDispersionCompensation); this->impl->kernelDispersionCompensation = nullptr; }
	if (this->impl->kernelKLinearizationAndWindowingAndDispersion) { clReleaseKernel(this->impl->kernelKLinearizationAndWindowingAndDispersion); this->impl->kernelKLinearizationAndWindowingAndDispersion = nullptr; }
	if (this->impl->kernelKLinearizationCubicAndWindowingAndDispersion) { clReleaseKernel(this->impl->kernelKLinearizationCubicAndWindowingAndDispersion); this->impl->kernelKLinearizationCubicAndWindowingAndDispersion = nullptr; }
	if (this->impl->kernelKLinearizationLanczosAndWindowingAndDispersion) { clReleaseKernel(this->impl->kernelKLinearizationLanczosAndWindowingAndDispersion); this->impl->kernelKLinearizationLanczosAndWindowingAndDispersion = nullptr; }
	if (this->impl->kernelPostProcessTruncateLog) { clReleaseKernel(this->impl->kernelPostProcessTruncateLog); this->impl->kernelPostProcessTruncateLog = nullptr; }
	if (this->impl->kernelPostProcessTruncateLin) { clReleaseKernel(this->impl->kernelPostProcessTruncateLin); this->impl->kernelPostProcessTruncateLin = nullptr; }
	if (this->impl->kernelGetMinimumVarianceMean) { clReleaseKernel(this->impl->kernelGetMinimumVarianceMean); this->impl->kernelGetMinimumVarianceMean = nullptr; }
	if (this->impl->kernelMeanALineSubtraction) { clReleaseKernel(this->impl->kernelMeanALineSubtraction); this->impl->kernelMeanALineSubtraction = nullptr; }
	if (this->impl->kernelBscanFlip) { clReleaseKernel(this->impl->kernelBscanFlip); this->impl->kernelBscanFlip = nullptr; }
	if (this->impl->kernelFillSinusoidalScanCorrectionCurve) { clReleaseKernel(this->impl->kernelFillSinusoidalScanCorrectionCurve); this->impl->kernelFillSinusoidalScanCorrectionCurve = nullptr; }
	if (this->impl->kernelSinusoidalScanCorrection) { clReleaseKernel(this->impl->kernelSinusoidalScanCorrection); this->impl->kernelSinusoidalScanCorrection = nullptr; }
	if (this->impl->kernelGetPostProcessBackground) { clReleaseKernel(this->impl->kernelGetPostProcessBackground); this->impl->kernelGetPostProcessBackground = nullptr; }
	if (this->impl->kernelPostProcessBackgroundSubtraction) { clReleaseKernel(this->impl->kernelPostProcessBackgroundSubtraction); this->impl->kernelPostProcessBackgroundSubtraction = nullptr; }

	if (this->impl->program) {
		clReleaseProgram(this->impl->program);
		this->impl->program = nullptr;
	}
}

void OpenClBackend::allocateDeviceBuffers() {
	cl_int err;
	size_t complexSize = this->impl->samplesPerBuffer * sizeof(float) * 2;  // float2

	//	Input buffers (one per command queue. backendIndex stores queue assignment)
	this->impl->d_inputBuffers.resize(this->impl->numCommandQueues);
	for (int i = 0; i < this->impl->numCommandQueues; i++) {
		size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
		this->impl->d_inputBuffers[i] = clCreateBuffer(this->impl->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, inputSize, nullptr, &err);
		checkOpenClError(err, "create input buffer");
	}

	//	FFT buffers (one per command queue for parallel processing)
	this->impl->d_fftBuffers.resize(this->impl->numCommandQueues);
	for (int i = 0; i < this->impl->numCommandQueues; i++) {
		this->impl->d_fftBuffers[i] = clCreateBuffer(this->impl->context, CL_MEM_READ_WRITE, complexSize, nullptr, &err);
		checkOpenClError(err, "create FFT buffer");
	}

	//	Linearization buffers (one per command queue)
	this->impl->d_inputLinearizedBuffers.resize(this->impl->numCommandQueues);
	for (int i = 0; i < this->impl->numCommandQueues; i++) {
		this->impl->d_inputLinearizedBuffers[i] = clCreateBuffer(this->impl->context, CL_MEM_READ_WRITE, complexSize, nullptr, &err);
		checkOpenClError(err, "create linearization buffer");
	}

	//	Output buffers (one per command queue)
	size_t outputSize = (this->impl->samplesPerBuffer / 2) * sizeof(float);
	this->impl->d_outputBuffers.resize(this->impl->numCommandQueues);
	for (int i = 0; i < this->impl->numCommandQueues; i++) {
		this->impl->d_outputBuffers[i] = clCreateBuffer(this->impl->context, CL_MEM_READ_WRITE, outputSize, nullptr, &err);
		checkOpenClError(err, "create output buffer");
	}

	//	Curve buffers
	size_t curveSize = this->impl->signalLength * sizeof(float);
	this->impl->d_resampleCurve = clCreateBuffer(this->impl->context, CL_MEM_READ_ONLY, curveSize, nullptr, &err);
	checkOpenClError(err, "create resample curve buffer");

	this->impl->d_windowCurve = clCreateBuffer(this->impl->context, CL_MEM_READ_ONLY, curveSize, nullptr, &err);
	checkOpenClError(err, "create window curve buffer");

	this->impl->d_phaseCartesian = clCreateBuffer(this->impl->context, CL_MEM_READ_WRITE, curveSize * 2, nullptr, &err);
	checkOpenClError(err, "create phase cartesian buffer");

	//	Fixed pattern noise
	this->impl->d_meanALine = clCreateBuffer(this->impl->context, CL_MEM_READ_WRITE, curveSize * 2, nullptr, &err);
	checkOpenClError(err, "create mean A-line buffer");

	//	Post-process background (initialize to zero)
	this->impl->d_postProcBackgroundLine = clCreateBuffer(this->impl->context, CL_MEM_READ_WRITE, curveSize, nullptr, &err);
	checkOpenClError(err, "create post-process background buffer");

	//	Initialize to zero to prevent issues if used before recording
	std::vector<float> zeros(this->impl->signalLength / 2, 0.0f);
	checkOpenClErrors(clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_postProcBackgroundLine, CL_TRUE, 0,
		zeros.size() * sizeof(float), zeros.data(), 0, nullptr, nullptr));

	//	Sinusoidal scan correction temp buffers (one per command queue for parallel processing)
	this->impl->d_sinusoidalScanTmpBuffers.resize(this->impl->numCommandQueues);
	for (int i = 0; i < this->impl->numCommandQueues; i++) {
		this->impl->d_sinusoidalScanTmpBuffers[i] = clCreateBuffer(this->impl->context, CL_MEM_READ_WRITE, outputSize, nullptr, &err);
		checkOpenClError(err, "create sinusoidal scan tmp buffer");
	}

	size_t sinusoidalCurveSize = this->impl->ascansPerBscan * sizeof(float);
	this->impl->d_sinusoidalResampleCurve = clCreateBuffer(this->impl->context, CL_MEM_READ_ONLY, sinusoidalCurveSize, nullptr, &err);
	checkOpenClError(err, "create sinusoidal resample curve buffer");

	//	Fill sinusoidal scan correction curve
	if (this->impl->d_sinusoidalResampleCurve != nullptr) {
		size_t globalWorkSize = this->impl->ascansPerBscan;
		checkOpenClErrors(clSetKernelArg(this->impl->kernelFillSinusoidalScanCorrectionCurve, 0, sizeof(cl_mem), &this->impl->d_sinusoidalResampleCurve));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelFillSinusoidalScanCorrectionCurve, 1, sizeof(int), &this->impl->ascansPerBscan));
		checkOpenClErrors(clEnqueueNDRangeKernel(this->impl->commandQueues[0], this->impl->kernelFillSinusoidalScanCorrectionCurve, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr));
		checkOpenClErrors(clFinish(this->impl->commandQueues[0]));
	}
}

void OpenClBackend::releaseDeviceBuffers() {
	for (auto& buf : this->impl->d_inputBuffers) {
		if (buf) { clReleaseMemObject(buf); buf = nullptr; }
	}
	this->impl->d_inputBuffers.clear();

	for (auto& buf : this->impl->d_fftBuffers) {
		if (buf) { clReleaseMemObject(buf); buf = nullptr; }
	}
	this->impl->d_fftBuffers.clear();

	for (auto& buf : this->impl->d_inputLinearizedBuffers) {
		if (buf) { clReleaseMemObject(buf); buf = nullptr; }
	}
	this->impl->d_inputLinearizedBuffers.clear();

	for (auto& buf : this->impl->d_outputBuffers) {
		if (buf) { clReleaseMemObject(buf); buf = nullptr; }
	}
	this->impl->d_outputBuffers.clear();

	for (auto& buf : this->impl->d_sinusoidalScanTmpBuffers) {
		if (buf) { clReleaseMemObject(buf); buf = nullptr; }
	}
	this->impl->d_sinusoidalScanTmpBuffers.clear();

	if (this->impl->d_resampleCurve) { clReleaseMemObject(this->impl->d_resampleCurve); this->impl->d_resampleCurve = nullptr; }
	if (this->impl->d_windowCurve) { clReleaseMemObject(this->impl->d_windowCurve); this->impl->d_windowCurve = nullptr; }
	if (this->impl->d_phaseCartesian) { clReleaseMemObject(this->impl->d_phaseCartesian); this->impl->d_phaseCartesian = nullptr; }
	if (this->impl->d_meanALine) { clReleaseMemObject(this->impl->d_meanALine); this->impl->d_meanALine = nullptr; }
	if (this->impl->d_postProcBackgroundLine) { clReleaseMemObject(this->impl->d_postProcBackgroundLine); this->impl->d_postProcBackgroundLine = nullptr; }
	if (this->impl->d_sinusoidalResampleCurve) { clReleaseMemObject(this->impl->d_sinusoidalResampleCurve); this->impl->d_sinusoidalResampleCurve = nullptr; }
}

void OpenClBackend::registerHostMemory() {
	//	OpenCL doesn't require explicit host memory registration like CUDA
	//	Memory transfers are handled by clEnqueueWriteBuffer/clEnqueueReadBuffer
}

void OpenClBackend::unregisterHostMemory() {
	//	No-op for OpenCL
}

// ============================================
// Lifecycle Methods
// ============================================

void OpenClBackend::initialize(const ProcessorConfiguration& config) {
	this->impl->config = config;

	//	Extract dimensions
	this->impl->signalLength = config.dataParams.signalLength;
	this->impl->ascansPerBscan = config.dataParams.ascansPerBscan;
	this->impl->bscansPerBuffer = config.dataParams.bscansPerBuffer;
	this->impl->samplesPerBuffer = this->impl->signalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer;
	this->impl->bytesPerSample = config.dataParams.getBytesPerSample();

	//	Get OpenCL platform
	cl_uint numPlatforms;
	checkOpenClErrors(clGetPlatformIDs(0, nullptr, &numPlatforms));
	if (numPlatforms == 0) {
		throw std::runtime_error("No OpenCL platforms found");
	}

	std::vector<cl_platform_id> platforms(numPlatforms);
	checkOpenClErrors(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));

	if (this->impl->platformId >= static_cast<int>(numPlatforms)) {
		throw std::runtime_error("Invalid platform ID");
	}
	this->impl->platform = platforms[this->impl->platformId];

	//	Get OpenCL device
	cl_uint numDevices;
	checkOpenClErrors(clGetDeviceIDs(this->impl->platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices));
	if (numDevices == 0) {
		throw std::runtime_error("No OpenCL devices found on platform");
	}

	std::vector<cl_device_id> devices(numDevices);
	checkOpenClErrors(clGetDeviceIDs(this->impl->platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr));

	//	Auto-select device if deviceId is -1
	if (this->impl->deviceId < 0) {
		cl_device_type preferredType = this->impl->preferGpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
		int selectedDevice = -1;

		//	First pass: find preferred device type
		for (cl_uint d = 0; d < numDevices; d++) {
			cl_device_type deviceType;
			checkOpenClErrors(clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			if (deviceType & preferredType) {
				selectedDevice = static_cast<int>(d);
				break;
			}
		}

		//	Fallback: use first available device
		if (selectedDevice < 0) {
			selectedDevice = 0;
		}

		this->impl->deviceId = selectedDevice;
	}

	if (this->impl->deviceId >= static_cast<int>(numDevices)) {
		throw std::runtime_error("Invalid device ID");
	}
	this->impl->device = devices[this->impl->deviceId];

	//	Query device's local memory size
	cl_ulong localMemSize;
	checkOpenClErrors(clGetDeviceInfo(this->impl->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr));
	this->impl->maxLocalMemSize = static_cast<size_t>(localMemSize);

	//	Create context
	cl_int err;
	this->impl->context = clCreateContext(nullptr, 1, &this->impl->device, nullptr, nullptr, &err);
	checkOpenClError(err, "clCreateContext");

	//	Create command queues
	this->createCommandQueues();

	//	Load and build kernels
	this->loadAndBuildKernels();

	//	Initialize VkFFT library
	memset(&this->impl->fftConfig, 0, sizeof(VkFFTConfiguration));
	memset(&this->impl->fftApp, 0, sizeof(VkFFTApplication));

	//	Configure VkFFT for 1D IFFT
	this->impl->fftConfig.FFTdim = 1;  //	1D FFT
	this->impl->fftConfig.size[0] = this->impl->signalLength;
	this->impl->fftConfig.size[1] = 1;
	this->impl->fftConfig.size[2] = 1;

	//	OpenCL backend specific configuration
	this->impl->fftConfig.platform = &this->impl->platform;
	this->impl->fftConfig.device = &this->impl->device;
	this->impl->fftConfig.context = &this->impl->context;
	this->impl->fftConfig.commandQueue = &this->impl->commandQueues[0];

	//	Verify OpenCL objects are valid
	if (!this->impl->platform || !this->impl->device || !this->impl->context || this->impl->commandQueues.empty()) {
		throw std::runtime_error("VkFFT: Invalid OpenCL objects (platform, device, context, or queue not initialized)");
	}

	//	Precision (single precision float)
	this->impl->fftConfig.doublePrecision = 0;

	//	Set number of batches
	this->impl->fftConfig.numberBatches = this->impl->ascansPerBscan * this->impl->bscansPerBuffer;

	//	Disable normalization to match cuFFT behavior (we handle it in post-processing) //todo: use this normalization and remove it from post-processing?
	this->impl->fftConfig.normalize = 0;

	//	Performance optimizations - set coalescedMemory based on vendor
	//	Get device vendor for vendor-specific optimizations
	char vendorName[256] = {0};
	checkOpenClErrors(clGetDeviceInfo(this->impl->device, CL_DEVICE_VENDOR, sizeof(vendorName), vendorName, nullptr));
	std::string vendor(vendorName);

	//	For Nvidia and AMD: 32 bytes, for Intel: 64 bytes (per VkFFT documentation)
	if (vendor.find("Intel") != std::string::npos) {
		this->impl->fftConfig.coalescedMemory = 64;
	} else {
		//	Nvidia, AMD, and others
		this->impl->fftConfig.coalescedMemory = 32;
	}

	//	Use automatic LUT selection (let VkFFT decide based on FFT size)
	//	-1 = off, 0 = auto, 1 = on
	this->impl->fftConfig.useLUT = 0;

	//	Target threads per block to match our kernel work group size
	this->impl->fftConfig.aimThreads = 128;

	//	Number of shared memory banks (NVIDIA has 32)
	this->impl->fftConfig.numSharedBanks = 32;

	//	Try bandwidth boost optimization for better memory coalescing
	//	This reduces coalesced number to get bigger sequences in one upload
	this->impl->fftConfig.performBandwidthBoost = 2;

	//	Allocate device buffers
	this->allocateDeviceBuffers();

	//	Initialize VkFFT application
	checkVkFFTErrors(initializeVkFFT(&this->impl->fftApp, this->impl->fftConfig));

	//	Allocate host input buffers
	size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
	this->impl->hostInputBuffers.resize(this->impl->numInputBuffers);

	for (int i = 0; i < this->impl->numInputBuffers; ++i) {
		if (!this->impl->hostInputBuffers[i].allocateMemory(inputSize)) {
			throw std::runtime_error("Failed to allocate input buffer " + std::to_string(i));
		}

		this->impl->hostInputBuffers[i].setDataType(config.dataParams.inputDataType);
		// Fixed mapping: each buffer maps to a specific command queue
		this->impl->hostInputBuffers[i].setBackendIndex(i % this->impl->numCommandQueues);

		//	Add to free queue
		this->impl->freeBuffersQueue.push(&this->impl->hostInputBuffers[i]);
	}

	//	Allocate host output buffers (one per command queue)
	size_t outputSize = (this->impl->samplesPerBuffer / 2) * sizeof(float);
	this->impl->hostOutputBuffers.resize(this->impl->numCommandQueues);
	for (int i = 0; i < this->impl->numCommandQueues; ++i) {
		if (!this->impl->hostOutputBuffers[i].allocateMemory(outputSize)) {
			throw std::runtime_error("Failed to allocate output buffer " + std::to_string(i));
		}
		this->impl->hostOutputBuffers[i].setDataType(IOBuffer::DataType::FLOAT32);
	}

	//	Pre-allocate callback data pool
	int poolSize = this->impl->numCommandQueues * 4;
	this->impl->callbackDataPool.resize(poolSize);
	for (auto& data : this->impl->callbackDataPool) {
		data.impl = this->impl.get();
	}

	//	Initialize output buffer semaphore (all buffers available at start)
	this->impl->availableOutputBuffers = this->impl->numCommandQueues;

	//load recorded profiles from configuration
	if (config.hasCustomPostProcessBackgroundProfile()) {
		const std::vector<float>& profileVec = config.getBackgroundProfile();
		this->impl->recordedPostProcessBackground = profileVec;
		//copy to device
		checkOpenClErrors(clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_postProcBackgroundLine, CL_TRUE, 0,
			profileVec.size() * sizeof(float), profileVec.data(), 0, nullptr, nullptr));
	}
	if (config.hasCustomFixedPatternNoiseProfile()) {
		const std::vector<float>& profileVec = config.getFixedPatternNoiseProfile();
		this->impl->recordedFixedPatternNoise = profileVec;
		size_t complexPairs = profileVec.size() / 2;
		//copy to device (need to pad to full signal length with zeros)
		std::vector<float> hostMeanInterleaved(this->impl->signalLength * 2, 0.0f);
		for (size_t i = 0; i < complexPairs; ++i) {
			hostMeanInterleaved[i*2] = profileVec[i*2];
			hostMeanInterleaved[i*2+1] = profileVec[i*2+1];
		}
		checkOpenClErrors(clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_meanALine, CL_TRUE, 0,
			this->impl->signalLength * 2 * sizeof(float), hostMeanInterleaved.data(), 0, nullptr, nullptr));
		this->impl->fixedPatternNoiseDetermined = true;
	}

	//	Start callback worker thread for ordered callback delivery
	//	Multiple command queues can complete out of order, this ensures callbacks are delivered in submission order
	this->impl->callbackWorkerRunning = true;
	this->impl->nextExpectedCallback = 0;
	this->impl->callbackWorkerThread = std::thread([this]() {
		while (this->impl->callbackWorkerRunning) {
			IOBuffer* bufferToDeliver = nullptr;

			{
				std::unique_lock<std::mutex> lock(this->impl->callbackQueueMutex);
				//	Wait until we have the next expected callback or shutdown
				this->impl->callbackQueueCV.wait(lock, [this]() {
					return !this->impl->callbackWorkerRunning ||
					       this->impl->pendingCallbacks.count(this->impl->nextExpectedCallback) > 0;
				});

				if (!this->impl->callbackWorkerRunning && this->impl->pendingCallbacks.empty()) {
					break;
				}

				//	Check if next expected callback is ready
				auto it = this->impl->pendingCallbacks.find(this->impl->nextExpectedCallback);
				if (it != this->impl->pendingCallbacks.end()) {
					bufferToDeliver = it->second;
					this->impl->pendingCallbacks.erase(it);
					this->impl->nextExpectedCallback++;
				}
			}

			//	Deliver callback outside the lock
			if (bufferToDeliver) {
				if (this->impl->callback) {
					this->impl->callback(*bufferToDeliver);
				}
				// NOTE: Do NOT release here. OutputBufferManager handles release via releaseOutputBuffer()
			}
		}

		//	Drain remaining callbacks on shutdown
		while (true) {
			IOBuffer* bufferToDeliver = nullptr;
			{
				std::lock_guard<std::mutex> lock(this->impl->callbackQueueMutex);
				auto it = this->impl->pendingCallbacks.find(this->impl->nextExpectedCallback);
				if (it == this->impl->pendingCallbacks.end()) {
					break;
				}
				bufferToDeliver = it->second;
				this->impl->pendingCallbacks.erase(it);
				this->impl->nextExpectedCallback++;
			}
			if (bufferToDeliver) {
				if (this->impl->callback) {
					this->impl->callback(*bufferToDeliver);
				}
				// NOTE: Do NOT release here - OutputBufferManager handles release via releaseOutputBuffer()
			}
		}
	});

	this->impl->openclInitialized = true;
}

void OpenClBackend::cleanup() {
	if (!this->impl->openclInitialized) {
		return;
	}

	//	Wait for all queues to finish before cleanup
	//	This ensures all event callbacks complete before we destroy resources
	for (auto& queue : this->impl->commandQueues) {
		if (queue) {
			clFinish(queue);
		}
	}

	//	Stop callback worker thread
	{
		std::lock_guard<std::mutex> lock(this->impl->callbackQueueMutex);
		this->impl->callbackWorkerRunning = false;
	}
	this->impl->callbackQueueCV.notify_all();
	if (this->impl->callbackWorkerThread.joinable()) {
		this->impl->callbackWorkerThread.join();
	}

	//	Release device buffers
	this->releaseDeviceBuffers();

	//	Unregister host memory
	this->unregisterHostMemory();

	//	Release kernels
	this->releaseKernels();

	//	Destroy VkFFT application
	deleteVkFFT(&this->impl->fftApp);

	//	Destroy command queues
	this->destroyCommandQueues();

	//	Release context
	if (this->impl->context) {
		clReleaseContext(this->impl->context);
		this->impl->context = nullptr;
	}

	this->impl->openclInitialized = false;
}

void OpenClBackend::setOutputCallback(std::function<void(const IOBuffer&)> callback) {
	this->impl->callback = callback;
}

void OpenClBackend::process(IOBuffer& input) {
	if (!this->impl->openclInitialized) {
		throw std::runtime_error("Backend not initialized");
	}

	// Get queue index from backendIndex
	const int queueIndex = input.getBackendIndex();
	if (queueIndex < 0 || queueIndex >= this->impl->numCommandQueues) {
		throw std::runtime_error("Invalid queue index: " + std::to_string(queueIndex));
	}

	cl_command_queue queue = this->impl->commandQueues[queueIndex];
	if (!queue) {
		throw std::runtime_error("Command queue is NULL! Queue index: " + std::to_string(queueIndex));
	}

	// Per-queue device buffers (all indexed by queueIndex)
	cl_mem d_input = this->impl->d_inputBuffers[queueIndex];
	cl_mem d_fftBuffer = this->impl->d_fftBuffers[queueIndex];
	cl_mem d_inputLinearized = this->impl->d_inputLinearizedBuffers[queueIndex];
	cl_mem d_outputBuffer = this->impl->d_outputBuffers[queueIndex];
	cl_mem d_sinusoidalScanTmpBuffer = this->impl->d_sinusoidalScanTmpBuffers[queueIndex];

	//	Wait for an output buffer to be available (prevents buffer reuse before callback delivery)
	{
		std::unique_lock<std::mutex> lock(this->impl->outputSemaphoreMutex);
		this->impl->outputSemaphoreCV.wait(lock, [this]() {
			return this->impl->availableOutputBuffers > 0;
		});
		this->impl->availableOutputBuffers--;
	}

	//	Select output buffer from rotating pool (not tied to queue index)
	int outputBufIdx = this->impl->currentOutputBuffer.fetch_add(1, std::memory_order_relaxed)
	                   % static_cast<int>(this->impl->hostOutputBuffers.size());
	IOBuffer* currentOutputBuf = &this->impl->hostOutputBuffers[outputBufIdx];

	//	Get buffer ID from input to propagate to output later
	uint64_t bufferId = input.getBufferId();

	//	Copy input to device (async for better pipelining)
	cl_event inputEvent;
	size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
	checkOpenClErrors(clEnqueueWriteBuffer(queue, d_input, CL_FALSE, 0,
		inputSize,
		input.getDataPointer(), 0, nullptr, &inputEvent));

#ifdef OPENCL_PROFILE_TIMING
	//	Track input transfer time
	clWaitForEvents(1, &inputEvent);
	double inputTransferMs = getEventDurationMs(inputEvent);
	{
		std::lock_guard<std::mutex> lock(g_profilingMutex);
		g_profiling.inputTransferMs += inputTransferMs;
		g_profiling.inputBytes += inputSize;
	}
#endif

	//	Return input buffer as soon as transfer completes (using pre-allocated pool)
	int inputCbIdx = this->impl->nextCallbackIndex.fetch_add(1, std::memory_order_relaxed) %
	                 static_cast<int>(this->impl->callbackDataPool.size());
	Impl::CallbackData* inputCbData = &this->impl->callbackDataPool[inputCbIdx];
	inputCbData->inputBuffer = &input;
	inputCbData->outputBuffer = nullptr;
	inputCbData->event = inputEvent;

	checkOpenClErrors(clSetEventCallback(inputEvent, CL_COMPLETE, returnBufferCallback, inputCbData));

	//	=== PROCESSING PIPELINE ===

	const ProcessorConfiguration& config = this->impl->config;
	const int signalLength = this->impl->signalLength;
	const int samplesPerBuffer = this->impl->samplesPerBuffer;
	const int ascansPerBscan = this->impl->ascansPerBscan;
	const int bscansPerBuffer = this->impl->bscansPerBuffer;

	size_t globalWorkSize = samplesPerBuffer;
	size_t localWorkSize = this->impl->workGroupSize;

	//	Round up global work size to be a multiple of local work size
	if (globalWorkSize % localWorkSize != 0) {
		globalWorkSize = ((globalWorkSize + localWorkSize - 1) / localWorkSize) * localWorkSize;
	}

	//	Step 1: Convert input to complex
	cl_kernel inputKernel = config.processingParams.input.bitshift ?
		this->impl->kernelInputToComplexBitshift : this->impl->kernelInputToComplex;

	checkOpenClErrors(clSetKernelArg(inputKernel, 0, sizeof(cl_mem), &d_fftBuffer));
	checkOpenClErrors(clSetKernelArg(inputKernel, 1, sizeof(cl_mem), &d_input));
	checkOpenClErrors(clSetKernelArg(inputKernel, 2, sizeof(int), &signalLength));
	checkOpenClErrors(clSetKernelArg(inputKernel, 3, sizeof(int), &signalLength));
	int bitDepth = config.dataParams.getBitDepth();
	checkOpenClErrors(clSetKernelArg(inputKernel, 4, sizeof(int), &bitDepth));
	checkOpenClErrors(clSetKernelArg(inputKernel, 5, sizeof(int), &samplesPerBuffer));
	checkOpenClErrors(clEnqueueNDRangeKernel(queue, inputKernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));

	//	Step 2: Rolling average background removal
	if (config.processingParams.dcRemoval.enabled) {
		int windowSize = config.processingParams.dcRemoval.windowSize;
		//	Allocate local memory: kernel needs local_size + 2 * rollingAverageWindowSize elements
		size_t localMemSize = (localWorkSize + 2 * windowSize) * sizeof(float);

		//	Check if local memory exceeds device limit and adjust work group size if needed
		size_t adjustedLocalWorkSize = localWorkSize;
		while (localMemSize > this->impl->maxLocalMemSize && adjustedLocalWorkSize > 1) {
			adjustedLocalWorkSize /= 2;
			localMemSize = (adjustedLocalWorkSize + 2 * windowSize) * sizeof(float);
		}

		//	If still too large, throw an error
		if (localMemSize > this->impl->maxLocalMemSize) {
			throw std::runtime_error("Rolling average window size too large for device local memory. "
				"Window size: " + std::to_string(windowSize) + ", "
				"Required local memory: " + std::to_string(localMemSize) + " bytes, "
				"Device maximum: " + std::to_string(this->impl->maxLocalMemSize) + " bytes");
		}

		//	Re-calculate globalWorkSize to be a multiple of adjustedLocalWorkSize
		size_t adjustedGlobalWorkSize = samplesPerBuffer;
		if (adjustedGlobalWorkSize % adjustedLocalWorkSize != 0) {
			adjustedGlobalWorkSize = ((adjustedGlobalWorkSize + adjustedLocalWorkSize - 1) / adjustedLocalWorkSize) * adjustedLocalWorkSize;
		}

		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 0, sizeof(cl_mem), &d_inputLinearized));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 1, sizeof(cl_mem), &d_fftBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 2, sizeof(int), &windowSize));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 4, sizeof(int), &ascansPerBscan));
		int pitch = signalLength * ascansPerBscan;
		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 5, sizeof(int), &pitch));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 6, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelRollingAverage, 7, localMemSize, nullptr));  // __local memory
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelRollingAverage, 1, nullptr, &adjustedGlobalWorkSize, &adjustedLocalWorkSize, 0, nullptr, nullptr));

		//	Swap pointers
		cl_mem tmpSwapPointer = d_inputLinearized;
		d_inputLinearized = d_fftBuffer;
		d_fftBuffer = tmpSwapPointer;
	}

	//	Step 3: K-linearization, windowing, and dispersion compensation
	cl_mem d_fftBuffer2 = d_fftBuffer;

	bool resampling = config.processingParams.resampling.enabled;
	bool windowing = config.processingParams.windowing.enabled;
	bool dispersion = config.processingParams.dispersion.enabled;
	InterpolationMethod interpMethod = config.processingParams.resampling.method;

	if (resampling && windowing && dispersion) {
		//	K-linearization + windowing + dispersion (most common case)
		cl_kernel kernel = (interpMethod == InterpolationMethod::CUBIC) ? this->impl->kernelKLinearizationCubicAndWindowingAndDispersion :
			(interpMethod == InterpolationMethod::LINEAR) ? this->impl->kernelKLinearizationAndWindowingAndDispersion :
			this->impl->kernelKLinearizationLanczosAndWindowingAndDispersion;

		checkOpenClErrors(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputLinearized));
		checkOpenClErrors(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_fftBuffer));
		checkOpenClErrors(clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->impl->d_resampleCurve));
		checkOpenClErrors(clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->impl->d_windowCurve));
		checkOpenClErrors(clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->impl->d_phaseCartesian));
		checkOpenClErrors(clSetKernelArg(kernel, 5, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(kernel, 6, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));

		d_fftBuffer2 = d_inputLinearized;
	} else if (resampling && windowing && !dispersion) {
		//	K-linearization + windowing
		cl_kernel kernel = (interpMethod == InterpolationMethod::CUBIC) ? this->impl->kernelKLinearizationCubicAndWindowing :
			(interpMethod == InterpolationMethod::LINEAR) ? this->impl->kernelKLinearizationAndWindowing :
			this->impl->kernelKLinearizationLanczosAndWindowing;

		checkOpenClErrors(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputLinearized));
		checkOpenClErrors(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_fftBuffer));
		checkOpenClErrors(clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->impl->d_resampleCurve));
		checkOpenClErrors(clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->impl->d_windowCurve));
		checkOpenClErrors(clSetKernelArg(kernel, 4, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(kernel, 5, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));

		d_fftBuffer2 = d_inputLinearized;
	} else if (resampling && !windowing && !dispersion) {
		//	Just k-linearization
		cl_kernel kernel = (interpMethod == InterpolationMethod::CUBIC) ? this->impl->kernelKLinearizationCubic :
			(interpMethod == InterpolationMethod::LINEAR) ? this->impl->kernelKLinearization :
			this->impl->kernelKLinearizationLanczos;

		checkOpenClErrors(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputLinearized));
		checkOpenClErrors(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_fftBuffer));
		checkOpenClErrors(clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->impl->d_resampleCurve));
		checkOpenClErrors(clSetKernelArg(kernel, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(kernel, 4, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));

		d_fftBuffer2 = d_inputLinearized;
	} else if (!resampling && windowing && dispersion) {
		//	Dispersion + windowing
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 0, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 1, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 2, sizeof(cl_mem), &this->impl->d_phaseCartesian));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 4, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelDispersionCompensation, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));

		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 0, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 1, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 2, sizeof(cl_mem), &this->impl->d_windowCurve));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 4, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelWindowing, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));
	} else if (!resampling && windowing && !dispersion) {
		//	Just windowing
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 0, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 1, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 2, sizeof(cl_mem), &this->impl->d_windowCurve));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelWindowing, 4, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelWindowing, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));
	} else if (!resampling && !windowing && dispersion) {
		//	Just dispersion
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 0, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 1, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 2, sizeof(cl_mem), &this->impl->d_phaseCartesian));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 4, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelDispersionCompensation, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));
	} else if (resampling && !windowing && dispersion) {
		//	K-linearization + dispersion (rarely used)
		cl_kernel kernel = (interpMethod == InterpolationMethod::CUBIC) ? this->impl->kernelKLinearizationCubic :
			(interpMethod == InterpolationMethod::LINEAR) ? this->impl->kernelKLinearization :
			this->impl->kernelKLinearizationLanczos;

		checkOpenClErrors(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_inputLinearized));
		checkOpenClErrors(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_fftBuffer));
		checkOpenClErrors(clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->impl->d_resampleCurve));
		checkOpenClErrors(clSetKernelArg(kernel, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(kernel, 4, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));

		d_fftBuffer2 = d_inputLinearized;

		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 0, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 1, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 2, sizeof(cl_mem), &this->impl->d_phaseCartesian));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 3, sizeof(int), &signalLength));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelDispersionCompensation, 4, sizeof(int), &samplesPerBuffer));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelDispersionCompensation, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr));
	}

	//	Step 4: IFFT using VkFFT
	VkFFTLaunchParams launchParams = {};
	launchParams.commandQueue = &queue;
	launchParams.buffer = &d_fftBuffer2;  //	Specify buffer dynamically for this FFT operation


#ifdef OPENCL_PROFILE_TIMING
	//	VkFFT doesn't directly provide events, so we enqueue a marker after the FFT
	cl_event fftStartEvent, fftEndEvent;
	checkOpenClErrors(clEnqueueMarkerWithWaitList(queue, 0, nullptr, &fftStartEvent));
	checkVkFFTErrors(VkFFTAppend(&this->impl->fftApp, 1, &launchParams));  //	+1 = inverse/backward FFT (VkFFT sign convention)
	checkOpenClErrors(clEnqueueMarkerWithWaitList(queue, 0, nullptr, &fftEndEvent));
	clWaitForEvents(1, &fftEndEvent);
	double fftMs = getEventDurationMs(fftEndEvent) - getEventDurationMs(fftStartEvent);
	clReleaseEvent(fftStartEvent);
	clReleaseEvent(fftEndEvent);
	{
		std::lock_guard<std::mutex> lock(g_profilingMutex);
		g_profiling.fftMs += fftMs;
	}
#else
	checkVkFFTErrors(VkFFTAppend(&this->impl->fftApp, 1, &launchParams));
#endif

	//	Step 5: Fixed-pattern noise removal
	if (config.processingParams.fixedPatternNoise.enabled) {
		int width = signalLength;
		int height = config.processingParams.fixedPatternNoise.bscanAverageCount * ascansPerBscan;

		if ((!config.processingParams.fixedPatternNoise.continuous &&
			!this->impl->fixedPatternNoiseDetermined) ||
			config.processingParams.fixedPatternNoise.continuous) {

			int ascansInBuffer = samplesPerBuffer / signalLength;
			if (height <= ascansInBuffer) {
				const int segments = 8;  // FIXED_PATTERN_NOISE_REMOVAL_SEGMENTS (matches CUDA)
				size_t globalWorkSizeMean = width;
				//	Round up to be a multiple of local work size
				if (globalWorkSizeMean % localWorkSize != 0) {
					globalWorkSizeMean = ((globalWorkSizeMean + localWorkSize - 1) / localWorkSize) * localWorkSize;
				}
				checkOpenClErrors(clSetKernelArg(this->impl->kernelGetMinimumVarianceMean, 0, sizeof(cl_mem), &this->impl->d_meanALine));
				checkOpenClErrors(clSetKernelArg(this->impl->kernelGetMinimumVarianceMean, 1, sizeof(cl_mem), &d_fftBuffer2));
				checkOpenClErrors(clSetKernelArg(this->impl->kernelGetMinimumVarianceMean, 2, sizeof(int), &width));
				checkOpenClErrors(clSetKernelArg(this->impl->kernelGetMinimumVarianceMean, 3, sizeof(int), &height));
				checkOpenClErrors(clSetKernelArg(this->impl->kernelGetMinimumVarianceMean, 4, sizeof(int), &segments));
				checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelGetMinimumVarianceMean, 1, nullptr, &globalWorkSizeMean, &localWorkSize, 0, nullptr, nullptr));

				//copy fixed pattern noise profile to host and sync to configuration
				int positivePairs = signalLength / 2;
				std::vector<float> hostMeanInterleaved(signalLength * 2);
				checkOpenClErrors(clEnqueueReadBuffer(queue, this->impl->d_meanALine, CL_TRUE, 0,
					signalLength * 2 * sizeof(float), hostMeanInterleaved.data(), 0, nullptr, nullptr));

				this->impl->recordedFixedPatternNoise.resize(positivePairs * 2);
				for (int i = 0; i < positivePairs; ++i) {
					this->impl->recordedFixedPatternNoise[i*2] = hostMeanInterleaved[i*2];
					this->impl->recordedFixedPatternNoise[i*2+1] = hostMeanInterleaved[i*2+1];
				}

				//sync to configuration
				this->impl->config.setFixedPatternNoiseProfile(
					this->impl->recordedFixedPatternNoise
				);

				this->impl->fixedPatternNoiseDetermined = true;
			}
		}

		int width2 = width / 2;
		int samplesPerBuffer2 = samplesPerBuffer / 2;
		size_t globalWorkSize2 = samplesPerBuffer2;
		//	Round up to be a multiple of local work size
		if (globalWorkSize2 % localWorkSize != 0) {
			globalWorkSize2 = ((globalWorkSize2 + localWorkSize - 1) / localWorkSize) * localWorkSize;
		}
		checkOpenClErrors(clSetKernelArg(this->impl->kernelMeanALineSubtraction, 0, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelMeanALineSubtraction, 1, sizeof(cl_mem), &this->impl->d_meanALine));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelMeanALineSubtraction, 2, sizeof(int), &width2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelMeanALineSubtraction, 3, sizeof(int), &samplesPerBuffer2));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelMeanALineSubtraction, 1, nullptr, &globalWorkSize2, &localWorkSize, 0, nullptr, nullptr));
	}

	//	Step 6: Post-process truncate (magnitude, log scaling, copy to output)
	cl_mem d_currBuffer = d_outputBuffer;
	size_t globalWorkSize2 = samplesPerBuffer / 2;
		//	Round up to be a multiple of local work size
		if (globalWorkSize2 % localWorkSize != 0) {
			globalWorkSize2 = ((globalWorkSize2 + localWorkSize - 1) / localWorkSize) * localWorkSize;
		}

	if (config.processingParams.intensity.logScale) {
		int signalLength2 = signalLength / 2;
		int samplesPerBuffer2 = samplesPerBuffer / 2;
		float maxVal = config.processingParams.intensity.rangeMax;
		float minVal = config.processingParams.intensity.rangeMin;
		float addend = config.processingParams.intensity.postOffset;
		float multiplicator = config.processingParams.intensity.preScale;

		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 0, sizeof(cl_mem), &d_currBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 1, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 2, sizeof(float), &maxVal));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 3, sizeof(float), &minVal));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 4, sizeof(float), &addend));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 5, sizeof(float), &multiplicator));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 6, sizeof(int), &signalLength));  // width_in = full complex A-scan length
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 7, sizeof(int), &signalLength2));  // width_out = output A-scan length
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLog, 8, sizeof(int), &samplesPerBuffer2));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelPostProcessTruncateLog, 1, nullptr, &globalWorkSize2, &localWorkSize, 0, nullptr, nullptr));
	} else {
		int signalLength2 = signalLength / 2;
		int samplesPerBuffer2 = samplesPerBuffer / 2;
		float maxVal = config.processingParams.intensity.rangeMax;
		float minVal = config.processingParams.intensity.rangeMin;
		float addend = config.processingParams.intensity.postOffset;
		float multiplicator = config.processingParams.intensity.preScale;

		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 0, sizeof(cl_mem), &d_currBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 1, sizeof(cl_mem), &d_fftBuffer2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 2, sizeof(float), &maxVal));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 3, sizeof(float), &minVal));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 4, sizeof(float), &addend));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 5, sizeof(float), &multiplicator));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 6, sizeof(int), &signalLength));  // width_in = full complex A-scan length
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 7, sizeof(int), &signalLength2));  // width_out = output A-scan length
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessTruncateLin, 8, sizeof(int), &samplesPerBuffer2));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelPostProcessTruncateLin, 1, nullptr, &globalWorkSize2, &localWorkSize, 0, nullptr, nullptr));
	}

	//	Step 7: B-scan flip
	if (config.processingParams.geometry.alternatingBscanFlip) {
		int signalLength2 = signalLength / 2;
		int pitch = (signalLength * ascansPerBscan) / 2;
		int samplesPerBuffer4 = samplesPerBuffer / 4;

		checkOpenClErrors(clSetKernelArg(this->impl->kernelBscanFlip, 0, sizeof(cl_mem), &d_currBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelBscanFlip, 1, sizeof(cl_mem), &d_currBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelBscanFlip, 2, sizeof(int), &signalLength2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelBscanFlip, 3, sizeof(int), &ascansPerBscan));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelBscanFlip, 4, sizeof(int), &pitch));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelBscanFlip, 5, sizeof(int), &samplesPerBuffer4));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelBscanFlip, 1, nullptr, &globalWorkSize2, &localWorkSize, 0, nullptr, nullptr));
	}

	//	Step 8: Sinusoidal scan correction
	if (config.processingParams.geometry.sinusoidalCorrection &&
		d_sinusoidalScanTmpBuffer != nullptr) {
		size_t copySize = sizeof(float) * samplesPerBuffer / 2;
		checkOpenClErrors(clEnqueueCopyBuffer(queue, d_currBuffer, d_sinusoidalScanTmpBuffer,
			0, 0, copySize, 0, nullptr, nullptr));

		int signalLength2 = signalLength / 2;
		int samplesPerBuffer2 = samplesPerBuffer / 2;
		checkOpenClErrors(clSetKernelArg(this->impl->kernelSinusoidalScanCorrection, 0, sizeof(cl_mem), &d_currBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelSinusoidalScanCorrection, 1, sizeof(cl_mem), &d_sinusoidalScanTmpBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelSinusoidalScanCorrection, 2, sizeof(cl_mem), &this->impl->d_sinusoidalResampleCurve));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelSinusoidalScanCorrection, 3, sizeof(int), &signalLength2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelSinusoidalScanCorrection, 4, sizeof(int), &ascansPerBscan));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelSinusoidalScanCorrection, 5, sizeof(int), &bscansPerBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelSinusoidalScanCorrection, 6, sizeof(int), &samplesPerBuffer2));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelSinusoidalScanCorrection, 1, nullptr, &globalWorkSize2, &localWorkSize, 0, nullptr, nullptr));
	}

	//	Step 9: Post-process background removal
	if (config.processingParams.background.enabled) {
		//	Record background if requested
		if (this->impl->postProcessBackgroundRecordingRequested) {
			int signalLength2 = signalLength / 2;
			checkOpenClErrors(clSetKernelArg(this->impl->kernelGetPostProcessBackground, 0, sizeof(cl_mem), &this->impl->d_postProcBackgroundLine));
			checkOpenClErrors(clSetKernelArg(this->impl->kernelGetPostProcessBackground, 1, sizeof(cl_mem), &d_currBuffer));
			checkOpenClErrors(clSetKernelArg(this->impl->kernelGetPostProcessBackground, 2, sizeof(int), &signalLength2));
			checkOpenClErrors(clSetKernelArg(this->impl->kernelGetPostProcessBackground, 3, sizeof(int), &ascansPerBscan));
			checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelGetPostProcessBackground, 1, nullptr, &globalWorkSize2, &localWorkSize, 0, nullptr, nullptr));

			//	Copy to host
			size_t bgSize = signalLength / 2;
			this->impl->recordedPostProcessBackground.resize(bgSize);
			checkOpenClErrors(clEnqueueReadBuffer(queue, this->impl->d_postProcBackgroundLine, CL_TRUE, 0,
				bgSize * sizeof(float), this->impl->recordedPostProcessBackground.data(), 0, nullptr, nullptr));

			//sync recorded profile to configuration
			this->impl->config.setBackgroundProfile(
				this->impl->recordedPostProcessBackground
			);

			this->impl->postProcessBackgroundRecordingRequested = false;
		}

		//	Update background if user provided new one
		if (this->impl->postProcessBackgroundUpdated) {
			//	Background was already copied to device in setPostProcessBackground()
			this->impl->postProcessBackgroundUpdated = false;
		}

		//	Apply background removal
		int signalLength2 = signalLength / 2;
		int samplesPerBuffer2 = samplesPerBuffer / 2;
		float bgWeight = config.processingParams.background.weight;
		float bgOffset = config.processingParams.background.offset;

		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessBackgroundSubtraction, 0, sizeof(cl_mem), &d_currBuffer));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessBackgroundSubtraction, 1, sizeof(cl_mem), &this->impl->d_postProcBackgroundLine));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessBackgroundSubtraction, 2, sizeof(float), &bgWeight));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessBackgroundSubtraction, 3, sizeof(float), &bgOffset));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessBackgroundSubtraction, 4, sizeof(int), &signalLength2));
		checkOpenClErrors(clSetKernelArg(this->impl->kernelPostProcessBackgroundSubtraction, 5, sizeof(int), &samplesPerBuffer2));
		checkOpenClErrors(clEnqueueNDRangeKernel(queue, this->impl->kernelPostProcessBackgroundSubtraction, 1, nullptr, &globalWorkSize2, &localWorkSize, 0, nullptr, nullptr));
	}

	//	Step 10: Copy result to host output buffer asynchronously
	currentOutputBuf->setBufferId(bufferId);
	size_t outputSize = (samplesPerBuffer / 2) * sizeof(float);

	//	Verify buffer is valid
	if (!d_currBuffer) {
		throw std::runtime_error("d_currBuffer is NULL before readback");
	}

	//	Copy result back (async with event to enable callback)
	cl_event outputEvent;
	checkOpenClErrors(clEnqueueReadBuffer(queue, d_currBuffer, CL_FALSE, 0, outputSize,
		currentOutputBuf->getDataPointer(), 0, nullptr, &outputEvent));

#ifdef OPENCL_PROFILE_TIMING
	//	Track output transfer time
	clWaitForEvents(1, &outputEvent);
	double outputTransferMs = getEventDurationMs(outputEvent);
	{
		std::lock_guard<std::mutex> lock(g_profilingMutex);
		g_profiling.outputTransferMs += outputTransferMs;
		g_profiling.outputBytes += outputSize;
		g_profiling.frameCount++;

		//	Print profiling stats every 100 frames
		if (g_profiling.frameCount % 100 == 0) {
			double avgInputMs = g_profiling.inputTransferMs / g_profiling.frameCount;
			double avgOutputMs = g_profiling.outputTransferMs / g_profiling.frameCount;
			double avgFftMs = g_profiling.fftMs / g_profiling.frameCount;
			double avgKernelsMs = g_profiling.kernelsMs / g_profiling.frameCount;
			double avgLinearMs = g_profiling.linearizationMs / g_profiling.frameCount;
			double avgWindowMs = g_profiling.windowingMs / g_profiling.frameCount;
			double avgDispersionMs = g_profiling.dispersionMs / g_profiling.frameCount;
			double avgPostProcMs = g_profiling.postProcessMs / g_profiling.frameCount;
			double avgTotalMs = avgInputMs + avgOutputMs + avgFftMs + avgKernelsMs + avgLinearMs + avgWindowMs + avgDispersionMs + avgPostProcMs;

			double inputGBps = (g_profiling.inputBytes / (g_profiling.inputTransferMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
			double outputGBps = (g_profiling.outputBytes / (g_profiling.outputTransferMs / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

			std::cout << "\n=== OpenCL Profiling (avg over " << g_profiling.frameCount << " frames) ===" << std::endl;
			std::cout << std::fixed << std::setprecision(3);
			std::cout << "  Input transfer:  " << avgInputMs << " ms (" << inputGBps << " GB/s)" << std::endl;
			std::cout << "  Linearization:   " << avgLinearMs << " ms" << std::endl;
			std::cout << "  Windowing:       " << avgWindowMs << " ms" << std::endl;
			std::cout << "  FFT operations:  " << avgFftMs << " ms" << std::endl;
			std::cout << "  Dispersion comp: " << avgDispersionMs << " ms" << std::endl;
			std::cout << "  Post-process:    " << avgPostProcMs << " ms" << std::endl;
			std::cout << "  Other kernels:   " << avgKernelsMs << " ms" << std::endl;
			std::cout << "  Output transfer: " << avgOutputMs << " ms (" << outputGBps << " GB/s)" << std::endl;
			std::cout << "  Total GPU time:  " << avgTotalMs << " ms" << std::endl;
			std::cout << "=============================================\n" << std::endl;
		}
	}
#endif

	//	Create callback data for user output callback (using pre-allocated pool)
	int outputCbIdx = this->impl->nextCallbackIndex.fetch_add(1, std::memory_order_relaxed) %
	                  static_cast<int>(this->impl->callbackDataPool.size());
	Impl::CallbackData* outputCbData = &this->impl->callbackDataPool[outputCbIdx];
	outputCbData->inputBuffer = nullptr;
	outputCbData->outputBuffer = currentOutputBuf;
	outputCbData->event = outputEvent;

	//	Register callback to trigger user callback when output transfer completes
	checkOpenClErrors(clSetEventCallback(outputEvent, CL_COMPLETE, outputCallback, outputCbData));
}


void OpenClBackend::updateConfig(const ProcessorConfiguration& config) {
	this->impl->config = config;
}

void OpenClBackend::updateResamplingCurve(const float* curve, size_t length) {
	if (!this->impl->openclInitialized) {
		return;
	}

	cl_int err = clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_resampleCurve, CL_TRUE, 0,
		length * sizeof(float), curve, 0, nullptr, nullptr);
	checkOpenClError(err, "update resampling curve");
}

void OpenClBackend::updateDispersionCurve(const float* curve, size_t length) {
	if (!this->impl->openclInitialized) {
		return;
	}

	cl_int err = clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_phaseCartesian, CL_TRUE, 0,
		length * sizeof(float), curve, 0, nullptr, nullptr);
	checkOpenClError(err, "update dispersion curve");
}

void OpenClBackend::updateWindowCurve(const float* curve, size_t length) {
	if (!this->impl->openclInitialized) {
		return;
	}

	cl_int err = clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_windowCurve, CL_TRUE, 0,
		length * sizeof(float), curve, 0, nullptr, nullptr);
	checkOpenClError(err, "update window curve");
}

IOBuffer& OpenClBackend::getInputBuffer(int index) {
	if (index < 0 || index >= this->impl->numInputBuffers) {
		throw std::out_of_range("Input buffer index out of range");
	}
	return this->impl->hostInputBuffers[index];
}

IOBuffer& OpenClBackend::getNextAvailableInputBuffer() {
	std::unique_lock<std::mutex> lock(this->impl->freeQueueMutex);
	this->impl->freeQueueCV.wait(lock, [this]() { return !this->impl->freeBuffersQueue.empty(); });

	IOBuffer* buffer = this->impl->freeBuffersQueue.front();
	this->impl->freeBuffersQueue.pop();

	return *buffer;
}

int OpenClBackend::getNumInputBuffers() const {
	return this->impl->numInputBuffers;
}

int OpenClBackend::getOutputBufferCount() const {
	return static_cast<int>(this->impl->hostOutputBuffers.size());
}

void OpenClBackend::releaseOutputBuffer(IOBuffer* buffer) {
	(void)buffer;
	{
		std::lock_guard<std::mutex> lock(this->impl->outputSemaphoreMutex);
		this->impl->availableOutputBuffers++;
	}
	this->impl->outputSemaphoreCV.notify_one();
}

void OpenClBackend::requestPostProcessBackgroundRecording() {
	this->impl->postProcessBackgroundRecordingRequested = true;
}

void OpenClBackend::setPostProcessBackgroundProfile(const float* background, size_t length) {
	if (!this->impl->openclInitialized) {
		return;
	}

	cl_int err = clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_postProcBackgroundLine, CL_TRUE, 0,
		length * sizeof(float), background, 0, nullptr, nullptr);
	checkOpenClError(err, "set post-process background");

	this->impl->recordedPostProcessBackground.assign(background, background + length);
	this->impl->postProcessBackgroundUpdated = true;
}

const std::vector<float>& OpenClBackend::getPostProcessBackgroundProfile() const {
	return this->impl->recordedPostProcessBackground;
}

void OpenClBackend::requestFixedPatternNoiseDetermination() {
	this->impl->fixedPatternNoiseDetermined = false;
}

void OpenClBackend::setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) {
	if (!this->impl->openclInitialized) {
		return;
	}

	cl_int err = clEnqueueWriteBuffer(this->impl->commandQueues[0], this->impl->d_meanALine, CL_TRUE, 0,
		complexPairs * 2 * sizeof(float), profileInterleaved, 0, nullptr, nullptr);
	checkOpenClError(err, "set fixed pattern noise profile");

	this->impl->recordedFixedPatternNoise.assign(profileInterleaved, profileInterleaved + complexPairs * 2);
	this->impl->fixedPatternNoiseDetermined = true;
}

const std::vector<float>& OpenClBackend::getFixedPatternNoiseProfile() const {
	return this->impl->recordedFixedPatternNoise;
}

//	Individual test methods. not needed, will be removed from all backends in future
std::vector<float> OpenClBackend::convertInput(const void* input, IOBuffer::DataType inputType, int bitDepth, int samples, bool applyBitshift) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::rollingAverageBackgroundRemoval(const float* input, int windowSize, int lineWidth, int numLines) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::kLinearization(const float* input, const float* resampleCurve, InterpolationMethod method, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::windowing(const float* input, const float* windowCurve, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::dispersionCompensation(const float* input, const float* phaseComplex, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::kLinearizationAndWindowing(const float* input, const float* resampleCurve, const float* windowCurve, InterpolationMethod method, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::kLinearizationAndWindowingAndDispersion(const float* input, const float* resampleCurve, const float* windowCurve, const float* phaseComplex, InterpolationMethod method, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::dispersionCompensationAndWindowing(const float* input, const float* phaseComplex, const float* windowCurve, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::fft(const float* input, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::ifft(const float* input, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::getMinimumVarianceMean(const float* input, int width, int height, int segments) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::fixedPatternNoiseRemoval(const float* input, const float* meanALine, int lineWidth, int numLines) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::postProcessTruncate(const float* input, bool logScaling, float grayscaleMax, float grayscaleMin, float addend, float multiplicator, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::bscanFlip(const float* input, int lineWidth, int linesPerBscan, int numBscans) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::sinusoidalScanCorrection(const float* input, const float* resampleCurve, int lineWidth, int linesPerBscan, int numBscans) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

std::vector<float> OpenClBackend::postProcessBackgroundSubtraction(const float* input, const float* backgroundLine, float weight, float offset, int lineWidth, int samples) {
	throw std::runtime_error("Individual test methods not implemented for OpenCL backend");
}

void CL_CALLBACK OpenClBackend::returnBufferCallback(cl_event event, cl_int status, void* userData) {
	if (status != CL_COMPLETE) {
		return;
	}

	Impl::CallbackData* data = static_cast<Impl::CallbackData*>(userData);
	if (!data || !data->impl || !data->inputBuffer) {
		return;
	}

	//	Return input buffer to free queue (like CUDA does)
	{
		std::lock_guard<std::mutex> lock(data->impl->freeQueueMutex);
		data->impl->freeBuffersQueue.push(data->inputBuffer);
		data->impl->freeQueueCV.notify_one();
	}

	//	Release the event (no delete needed - using pool)
	clReleaseEvent(event);
}

void CL_CALLBACK OpenClBackend::outputCallback(cl_event event, cl_int status, void* userData) {
	if (status != CL_COMPLETE) {
		return;
	}

	Impl::CallbackData* data = static_cast<Impl::CallbackData*>(userData);
	if (!data || !data->impl || !data->outputBuffer) {
		return;
	}

	//	Queue the callback for ordered delivery (non-blocking)
	uint64_t bufferId = data->outputBuffer->getBufferId();
	{
		std::lock_guard<std::mutex> lock(data->impl->callbackQueueMutex);
		data->impl->pendingCallbacks[bufferId] = data->outputBuffer;
	}
	data->impl->callbackQueueCV.notify_one();

	//	Release the event
	clReleaseEvent(event);
}

} // namespace ope
