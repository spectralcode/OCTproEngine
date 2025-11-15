#include "cuda_backend.h"
#include "cuda_kernels.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Helper macro for checking CUDA errors
#define checkCudaErrors(call) \
	do { \
		cudaError_t err = call; \
		if (err != cudaSuccess) { \
			std::stringstream ss; \
			ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
			   << cudaGetErrorString(err); \
			throw std::runtime_error(ss.str()); \
		} \
	} while(0)

#define checkCufftErrors(call) \
	do { \
		cufftResult err = call; \
		if (err != CUFFT_SUCCESS) { \
			std::stringstream ss; \
			ss << "cuFFT error at " << __FILE__ << ":" << __LINE__ << " - code: " << err; \
			throw std::runtime_error(ss.str()); \
		} \
	} while(0)

namespace ope {

// ============================================
// Implementation Structure
// ============================================

struct CudaBackend::Impl {
	// Configuration
	ProcessorConfiguration config;
	
	// CUDA parameters
	int numStreams = 8;  // Default from original code
	int blockSize = 128;  // Default from original code
	int gridSize = 0;
	int deviceId = 0;
	bool cudaInitialized = false;
	
	// Data dimensions
	int signalLength = 0;
	int ascansPerBscan = 0;
	int bscansPerBuffer = 0;
	int samplesPerBuffer = 0;
	int bytesPerSample = 0;
	
	// CUDA streams and events
	std::vector<cudaStream_t> streams;
	cudaStream_t userRequestStream;
	cudaEvent_t syncEvent;
	int currentStream = 0;
	
	// Input buffer management (queue-based, thread-safe)
	int numInputBuffers = 2;  // Default 2, user configurable
	std::vector<IOBuffer> hostInputBuffers;
	std::queue<IOBuffer*> freeBuffersQueue;
	std::mutex freeQueueMutex;
	std::condition_variable freeQueueCV;
	
	// Device buffers (one per stream for overlapping)
	std::vector<void*> d_inputBuffers;
	int currentBuffer = 0;
	
	// Processing buffers
	cufftComplex* d_fftBuffer = nullptr;
	cufftComplex* d_inputLinearized = nullptr;
	float* d_processedBuffer = nullptr;
	
	// Curve buffers
	float* d_resampleCurve = nullptr;
	float* d_windowCurve = nullptr;
	float* d_dispersionCurve = nullptr;
	cufftComplex* d_phaseCartesian = nullptr;
	float* d_sinusoidalResampleCurve = nullptr;
	
	// Fixed pattern noise removal
	cufftComplex* d_meanALine = nullptr;
	bool fixedPatternNoiseDetermined = false;
	
	// Post-processing
	float* d_postProcBackgroundLine = nullptr;
	float* d_sinusoidalScanTmpBuffer = nullptr;
	bool postProcessBackgroundRecordingRequested = false;
	bool postProcessBackgroundUpdated = false;
	std::vector<float> recordedPostProcessBackground;
	
	// cuFFT plan
	cufftHandle fftPlan = 0;
	
	// Output buffers for callback (ping-pong)
	IOBuffer outputBuffer1;
	IOBuffer outputBuffer2;
	int currentOutputBuffer = 0;
	
	// Callback
	std::function<void(const IOBuffer&)> callback;
	
	// Pre-allocated callback data pool (NO heap allocations!)
	struct CallbackData {
		Impl* impl;
		IOBuffer* inputBuffer;      // For return buffer callback
		IOBuffer* outputBuffer;     // For output callback
	};
	std::vector<CallbackData> callbackDataPool;
	std::atomic<int> nextCallbackIndex{0};
	
	Impl() = default;
	
	~Impl() {
		// Cleanup is handled in cleanup() method
	}
};

// ============================================
// Constructor / Destructor
// ============================================

CudaBackend::CudaBackend() : impl(std::make_unique<Impl>()) {
	// Get current CUDA device
	checkCudaErrors(cudaGetDevice(&this->impl->deviceId));
}

CudaBackend::~CudaBackend() {
	this->cleanup();
}

// ============================================
// Configuration Methods (before initialize)
// ============================================

void CudaBackend::setNumInputBuffers(int count) {
	if (this->impl->cudaInitialized) {
		throw std::runtime_error("Cannot change number of input buffers after initialization");
	}
	if (count < 1) {
		throw std::invalid_argument("Number of input buffers must be at least 1");
	}
	this->impl->numInputBuffers = count;
}

void CudaBackend::setNumStreams(int numStreams) {
	if (this->impl->cudaInitialized) {
		throw std::runtime_error("Cannot change number of streams after initialization");
	}
	this->impl->numStreams = numStreams;
}

void CudaBackend::setBlockSize(int blockSize) {
	if (this->impl->cudaInitialized) {
		throw std::runtime_error("Cannot change block size after initialization");
	}
	this->impl->blockSize = blockSize;
}

void CudaBackend::setDeviceId(int deviceId) {
	if (this->impl->cudaInitialized) {
		throw std::runtime_error("Cannot change device after initialization");
	}
	this->impl->deviceId = deviceId;
}

// ============================================
// Lifecycle Methods
// ============================================

void CudaBackend::initialize(const ProcessorConfiguration& config) {
	this->impl->config = config;
	
	// Set CUDA device
	checkCudaErrors(cudaSetDevice(this->impl->deviceId));
	
	// Extract dimensions
	this->impl->signalLength = config.dataParams.signalLength;
	this->impl->ascansPerBscan = config.dataParams.ascansPerBscan;
	this->impl->bscansPerBuffer = config.dataParams.bscansPerBuffer;
	this->impl->samplesPerBuffer = this->impl->signalLength * this->impl->ascansPerBscan * this->impl->bscansPerBuffer;
	this->impl->bytesPerSample = config.dataParams.getBytesPerSample();
	
	// Calculate grid size
	this->impl->gridSize = this->impl->samplesPerBuffer / this->impl->blockSize;
	
	// Pre-allocate callback data pool (NO heap allocations during processing!)
	// Size: numStreams Ã— 2 callbacks per stream Ã— 2 for safety = numStreams Ã— 4
	int poolSize = this->impl->numStreams * 4;
	this->impl->callbackDataPool.resize(poolSize);
	for (auto& data : this->impl->callbackDataPool) {
		data.impl = this->impl.get();
	}
	
	// Create streams and events
	this->createStreamsAndEvents();
	
	// Allocate device buffers
	this->allocateDeviceBuffers();
	
	// Allocate and register host input buffers
	size_t inputSize = this->impl->samplesPerBuffer * this->impl->bytesPerSample;
	this->impl->hostInputBuffers.resize(this->impl->numInputBuffers);
	
	for (int i = 0; i < this->impl->numInputBuffers; ++i) {
		// Allocate memory using IOBuffer's normal allocation
		if (!this->impl->hostInputBuffers[i].allocateMemory(inputSize)) {
			throw std::runtime_error("Failed to allocate input buffer " + std::to_string(i));
		}
		
#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
		// Jetson with zero-copy: IOBuffer already allocated with cudaHostAlloc + cudaHostAllocMapped
		// No need to register - memory is already mapped
		// Note: Use cudaHostGetDevicePointer() when accessing on device side
#else
		// Desktop: Register the allocated memory with CUDA for fast PCIe transfers
		void* ptr = this->impl->hostInputBuffers[i].getDataPointer();
		cudaError_t err = cudaHostRegister(ptr, inputSize, cudaHostRegisterPortable);
		if (err != cudaSuccess) {
			// Warning only - continue with unregistered memory
			fprintf(stderr, "Warning: cudaHostRegister failed for input buffer %d: %s\n", 
			        i, cudaGetErrorString(err));
			fprintf(stderr, "Continuing with unregistered memory (may be slower)\n");
		}
#endif
		
		// Set data type from configuration (user-specified)
		this->impl->hostInputBuffers[i].setDataType(config.dataParams.inputDataType);
		
		// Add to free queue
		this->impl->freeBuffersQueue.push(&this->impl->hostInputBuffers[i]);
	}
	
	// Allocate output buffers
	size_t outputSize = (this->impl->samplesPerBuffer / 2) * sizeof(float);
	if (!this->impl->outputBuffer1.allocateMemory(outputSize) ||
		!this->impl->outputBuffer2.allocateMemory(outputSize)) {
		throw std::runtime_error("Failed to allocate output buffers");
	}
	this->impl->outputBuffer1.setDataType(IOBuffer::DataType::FLOAT32);
	this->impl->outputBuffer2.setDataType(IOBuffer::DataType::FLOAT32);
	
#ifndef __aarch64__
	// Register output buffers for fast host-to-device transfers (Desktop only)
	void* outPtr1 = this->impl->outputBuffer1.getDataPointer();
	void* outPtr2 = this->impl->outputBuffer2.getDataPointer();
	
	cudaError_t err1 = cudaHostRegister(outPtr1, outputSize, cudaHostRegisterPortable);
	if (err1 != cudaSuccess) {
		fprintf(stderr, "Warning: cudaHostRegister failed for output buffer 1: %s\n", 
		        cudaGetErrorString(err1));
	}
	
	cudaError_t err2 = cudaHostRegister(outPtr2, outputSize, cudaHostRegisterPortable);
	if (err2 != cudaSuccess) {
		fprintf(stderr, "Warning: cudaHostRegister failed for output buffer 2: %s\n", 
		        cudaGetErrorString(err2));
	}
#endif
	
	// Create cuFFT plan
	checkCufftErrors(cufftPlan1d(
		&this->impl->fftPlan,
		this->impl->signalLength,
		CUFFT_C2C,
		this->impl->ascansPerBscan * this->impl->bscansPerBuffer
	));
	
	
	// Fill sinusoidal scan correction curve
	if (this->impl->d_sinusoidalResampleCurve != nullptr) {
		cuda_kernels::fillSinusoidalScanCorrectionCurve<<<this->impl->ascansPerBscan, 1, 0, this->impl->streams[0]>>>(
			this->impl->d_sinusoidalResampleCurve,
			this->impl->ascansPerBscan
		);
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}
	
	// Reset fixed pattern noise state
	this->impl->fixedPatternNoiseDetermined = false;
	
	this->impl->cudaInitialized = true;
}

void CudaBackend::cleanup() {
	if (!this->impl->cudaInitialized) {
		return;
	}
	
	// Set device
	checkCudaErrors(cudaSetDevice(this->impl->deviceId));
	
	// Wait for all streams to complete
	for (auto& stream : this->impl->streams) {
		cudaStreamSynchronize(stream);
	}
	
	// Unregister and release host input buffers
	for (auto& buffer : this->impl->hostInputBuffers) {
		void* ptr = buffer.getDataPointer();
		if (ptr) {
#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
			// Jetson with zero-copy: Memory was allocated with cudaHostAlloc
			// IOBuffer will handle cleanup via cudaFreeHost (see iobuffer.cpp)
			// No unregistration needed
#else
			// Desktop: Unregister the memory from CUDA before releasing it
			cudaError_t err = cudaHostUnregister(ptr);
			if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
				fprintf(stderr, "Warning: cudaHostUnregister failed for input buffer: %s\n", 
				        cudaGetErrorString(err));
			}
#endif
		}
		// Release the memory through IOBuffer
		buffer.releaseMemory();
	}
	this->impl->hostInputBuffers.clear();
	
	// Clear queue
	while (!this->impl->freeBuffersQueue.empty()) {
		this->impl->freeBuffersQueue.pop();
	}
	
#ifndef __aarch64__
	// Unregister output buffers (Desktop only)
	void* outPtr1 = this->impl->outputBuffer1.getDataPointer();
	void* outPtr2 = this->impl->outputBuffer2.getDataPointer();
	
	if (outPtr1) {
		cudaError_t err = cudaHostUnregister(outPtr1);
		if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
			fprintf(stderr, "Warning: cudaHostUnregister failed for output buffer 1: %s\n", 
			        cudaGetErrorString(err));
		}
	}
	
	if (outPtr2) {
		cudaError_t err = cudaHostUnregister(outPtr2);
		if (err != cudaSuccess && err != cudaErrorHostMemoryNotRegistered) {
			fprintf(stderr, "Warning: cudaHostUnregister failed for output buffer 2: %s\n", 
			        cudaGetErrorString(err));
		}
	}
#endif
	
	// Release device buffers
	this->releaseDeviceBuffers();
	
	// Destroy cuFFT plan
	if (this->impl->fftPlan) {
		cufftDestroy(this->impl->fftPlan);
		this->impl->fftPlan = 0;
	}
	
	// Destroy streams and events
	this->destroyStreamsAndEvents();
	
	// Release output buffers
	this->impl->outputBuffer1.releaseMemory();
	this->impl->outputBuffer2.releaseMemory();
	
	this->impl->cudaInitialized = false;
}


// ============================================
// Input Buffer Management
// ============================================

IOBuffer& CudaBackend::getInputBuffer(int index) {
	if (index < 0 || index >= this->impl->numInputBuffers) {
		throw std::out_of_range("Input buffer index out of range");
	}
	return this->impl->hostInputBuffers[index];
}

IOBuffer& CudaBackend::getNextAvailableInputBuffer() {
	std::unique_lock<std::mutex> lock(this->impl->freeQueueMutex);
	
	// Wait for a free buffer (blocks if all buffers in use)
	while (this->impl->freeBuffersQueue.empty()) {
		this->impl->freeQueueCV.wait(lock);
	}
	
	IOBuffer* buffer = this->impl->freeBuffersQueue.front();
	this->impl->freeBuffersQueue.pop();
	
	return *buffer;
}

int CudaBackend::getNumInputBuffers() const {
	return this->impl->numInputBuffers;
}

// ============================================
// Callback Management
// ============================================

void CudaBackend::setOutputCallback(std::function<void(const IOBuffer&)> callback) {
	this->impl->callback = callback;
}

void CUDART_CB CudaBackend::returnBufferCallback(void* userData) {
	Impl::CallbackData* data = static_cast<Impl::CallbackData*>(userData);
	
	if (data && data->impl && data->inputBuffer) {
		// Return buffer to free queue
		{
			std::lock_guard<std::mutex> lock(data->impl->freeQueueMutex);
			data->impl->freeBuffersQueue.push(data->inputBuffer);
		}
		data->impl->freeQueueCV.notify_one();  // Wake up waiting threads
	}
	
	// NO delete! Memory is pre-allocated
}

void CUDART_CB CudaBackend::outputCallback(void* userData) {
	Impl::CallbackData* data = static_cast<Impl::CallbackData*>(userData);
	
	if (data && data->impl && data->impl->callback && data->outputBuffer) {
		data->impl->callback(*data->outputBuffer);
	}
	
	// NO delete! Memory is pre-allocated
}

// ============================================
// Main Processing Pipeline
// ============================================

void CudaBackend::process(IOBuffer& input) {
	if (!this->impl->cudaInitialized) {
		throw std::runtime_error("CUDA backend not initialized");
	}
	
	// Round-robin stream selection
	this->impl->currentStream = (this->impl->currentStream + 1) % this->impl->numStreams;
	cudaStream_t stream = this->impl->streams[this->impl->currentStream];
	
	// Round-robin device buffer selection
	this->impl->currentBuffer = (this->impl->currentBuffer + 1) % static_cast<int>(this->impl->d_inputBuffers.size());
	
	// Select output buffer (ping-pong)
	this->impl->currentOutputBuffer = (this->impl->currentOutputBuffer + 1) % 2;
	IOBuffer* currentOutputBuf = (this->impl->currentOutputBuffer == 0) ? 
		&this->impl->outputBuffer1 : &this->impl->outputBuffer2;
	
	// Copy input to device
	void* d_input = nullptr;
	
#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
	// Zero-copy: Get device pointer directly to host memory
	checkCudaErrors(cudaHostGetDevicePointer(&d_input, input.getDataPointer(), 0));
#else
	// Regular path: Async copy to device
	d_input = this->impl->d_inputBuffers[this->impl->currentBuffer];
	checkCudaErrors(cudaMemcpyAsync(
		d_input,
		input.getDataPointer(),
		this->impl->samplesPerBuffer * this->impl->bytesPerSample,
		cudaMemcpyHostToDevice,
		stream
	));
	
	// Get pre-allocated callback data for returning buffer after memcpy
	int idx = this->impl->nextCallbackIndex.fetch_add(1, std::memory_order_relaxed) % 
	          static_cast<int>(this->impl->callbackDataPool.size());
	Impl::CallbackData* returnData = &this->impl->callbackDataPool[idx];
	returnData->inputBuffer = &input;
	
	// Register callback to return buffer after memcpy completes
	checkCudaErrors(cudaLaunchHostFunc(stream, returnBufferCallback, returnData));
#endif
	
	// === PROCESSING PIPELINE ===
	
	const ProcessorConfiguration& config = this->impl->config;
	const int signalLength = this->impl->signalLength;
	const int samplesPerBuffer = this->impl->samplesPerBuffer;
	const int gridSize = this->impl->gridSize;
	const int blockSize = this->impl->blockSize;
	const int ascansPerBscan = this->impl->ascansPerBscan;
	const int bscansPerBuffer = this->impl->bscansPerBuffer;
	
	// Step 1: Convert input to cufftComplex
	if (config.dataParams.bitshift) {
		cuda_kernels::inputToCufftComplex_and_bitshift<<<gridSize, blockSize, 0, stream>>>(
			this->impl->d_fftBuffer,
			d_input,
			signalLength,
			signalLength,
			config.dataParams.getBitDepth(),
			samplesPerBuffer
		);
	} else {
		cuda_kernels::inputToCufftComplex<<<gridSize, blockSize, 0, stream>>>(
			this->impl->d_fftBuffer,
			d_input,
			signalLength,
			signalLength,
			config.dataParams.getBitDepth(),
			samplesPerBuffer
		);
	}
	
#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
	// Zero-copy: Return buffer after inputToCufftComplex completes
	// (input buffer no longer needed after conversion to d_fftBuffer)
	int idx = this->impl->nextCallbackIndex.fetch_add(1, std::memory_order_relaxed) % 
	          static_cast<int>(this->impl->callbackDataPool.size());
	Impl::CallbackData* returnData = &this->impl->callbackDataPool[idx];
	returnData->inputBuffer = &input;
	checkCudaErrors(cudaLaunchHostFunc(stream, returnBufferCallback, returnData));
#else
	// Synchronization for data acquisition pacing / Zero-copy buffer return
	//todo: review. probably can be removed, synchronization is done via input buffer callbacks
	cudaEventRecord(this->impl->syncEvent, stream);
	cudaEventSynchronize(this->impl->syncEvent);
#endif
	
	// Step 2: Rolling average background removal
	if (config.backgroundRemovalParams.enabled) {
		int sharedMemSize = (blockSize + 2 * config.backgroundRemovalParams.rollingAverageWindowSize) * sizeof(float);
		cuda_kernels::rollingAverageBackgroundRemoval<<<gridSize, blockSize, sharedMemSize, stream>>>(
			this->impl->d_inputLinearized,
			this->impl->d_fftBuffer,
			config.backgroundRemovalParams.rollingAverageWindowSize,
			signalLength,
			ascansPerBscan,
			signalLength * ascansPerBscan,
			samplesPerBuffer
		);
		// Swap pointers
		cufftComplex* tmpSwapPointer = this->impl->d_inputLinearized;
		this->impl->d_inputLinearized = this->impl->d_fftBuffer;
		this->impl->d_fftBuffer = tmpSwapPointer;
	}
	
	// Step 3: K-linearization, windowing, and dispersion compensation
	cufftComplex* d_fftBuffer2 = this->impl->d_fftBuffer;
	
	// Determine which fused kernel to use based on enabled features
	bool resampling = config.resamplingParams.enabled;
	bool windowing = config.windowingParams.enabled;
	bool dispersion = config.dispersionParams.enabled;
	InterpolationMethod interpMethod = config.resamplingParams.interpolationMethod;
	
	if (this->impl->d_inputLinearized != nullptr && resampling && windowing && dispersion) {
		// K-linearization + windowing + dispersion (most common case)
		if (interpMethod == InterpolationMethod::CUBIC) {
			cuda_kernels::klinearizationCubicAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer, this->impl->d_resampleCurve,
				this->impl->d_windowCurve, this->impl->d_phaseCartesian, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LINEAR) {
			cuda_kernels::klinearizationAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer, this->impl->d_resampleCurve,
				this->impl->d_windowCurve, this->impl->d_phaseCartesian, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LANCZOS) {
			cuda_kernels::klinearizationLanczosAndWindowingAndDispersionCompensation<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer, this->impl->d_resampleCurve,
				this->impl->d_windowCurve, this->impl->d_phaseCartesian, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = this->impl->d_inputLinearized;
	} else if (this->impl->d_inputLinearized != nullptr && resampling && windowing && !dispersion) {
		// K-linearization + windowing
		if (interpMethod == InterpolationMethod::CUBIC) {
			cuda_kernels::klinearizationCubicAndWindowing<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer, this->impl->d_resampleCurve,
				this->impl->d_windowCurve, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LINEAR) {
			cuda_kernels::klinearizationAndWindowing<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer, this->impl->d_resampleCurve,
				this->impl->d_windowCurve, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LANCZOS) {
			cuda_kernels::klinearizationLanczosAndWindowing<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer, this->impl->d_resampleCurve,
				this->impl->d_windowCurve, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = this->impl->d_inputLinearized;
	} else if (!resampling && windowing && dispersion) {
		// Dispersion + windowing
		cuda_kernels::dispersionCompensationAndWindowing<<<gridSize, blockSize, 0, stream>>>(
			d_fftBuffer2, d_fftBuffer2, this->impl->d_phaseCartesian,
			this->impl->d_windowCurve, signalLength, samplesPerBuffer);
	} else if (this->impl->d_inputLinearized != nullptr && resampling && !windowing && !dispersion) {
		// Just k-linearization
		if (interpMethod == InterpolationMethod::CUBIC) {
			cuda_kernels::klinearizationCubic<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer,
				this->impl->d_resampleCurve, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LINEAR) {
			cuda_kernels::klinearization<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer,
				this->impl->d_resampleCurve, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LANCZOS) {
			cuda_kernels::klinearizationLanczos<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer,
				this->impl->d_resampleCurve, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = this->impl->d_inputLinearized;
	} else if (!resampling && windowing && !dispersion) {
		// Just windowing
		cuda_kernels::windowing<<<gridSize, blockSize, 0, stream>>>(
			d_fftBuffer2, d_fftBuffer2, this->impl->d_windowCurve, signalLength, samplesPerBuffer);
	} else if (!resampling && !windowing && dispersion) {
		// Just dispersion
		cuda_kernels::dispersionCompensation<<<gridSize, blockSize, 0, stream>>>(
			d_fftBuffer2, d_fftBuffer2, this->impl->d_phaseCartesian, signalLength, samplesPerBuffer);
	} else if (this->impl->d_inputLinearized != nullptr && resampling && !windowing && dispersion) {
		// K-linearization + dispersion (rarely used, so not optimized with fused kernel)
		if (interpMethod == InterpolationMethod::CUBIC) {
			cuda_kernels::klinearizationCubic<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer,
				this->impl->d_resampleCurve, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LINEAR) {
			cuda_kernels::klinearization<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer,
				this->impl->d_resampleCurve, signalLength, samplesPerBuffer);
		} else if (interpMethod == InterpolationMethod::LANCZOS) {
			cuda_kernels::klinearizationLanczos<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_inputLinearized, this->impl->d_fftBuffer,
				this->impl->d_resampleCurve, signalLength, samplesPerBuffer);
		}
		d_fftBuffer2 = this->impl->d_inputLinearized;
		cuda_kernels::dispersionCompensation<<<gridSize, blockSize, 0, stream>>>(
			d_fftBuffer2, d_fftBuffer2, this->impl->d_phaseCartesian, signalLength, samplesPerBuffer);
	}
	
	// Step 4: IFFT
	cufftSetStream(this->impl->fftPlan, stream);
	checkCufftErrors(cufftExecC2C(this->impl->fftPlan, d_fftBuffer2, d_fftBuffer2, CUFFT_INVERSE));
	
	// Step 5: Fixed-pattern noise removal
	if (config.postProcessingParams.fixedPatternNoiseRemoval) {
		int width = signalLength;
		int height = config.postProcessingParams.bscansForNoiseDetermination * ascansPerBscan;
		
		if ((!config.postProcessingParams.continuousFixedPatternNoiseDetermination && 
			 !this->impl->fixedPatternNoiseDetermined) ||
			config.postProcessingParams.continuousFixedPatternNoiseDetermination) {
			cuda_kernels::getMinimumVarianceMean<<<gridSize, blockSize, 0, stream>>>(
				this->impl->d_meanALine, d_fftBuffer2, width, height,
				FIXED_PATTERN_NOISE_REMOVAL_SEGMENTS);
			this->impl->fixedPatternNoiseDetermined = true;
		}
		
		cuda_kernels::meanALineSubtraction<<<gridSize/2, blockSize, 0, stream>>>(
			d_fftBuffer2, this->impl->d_meanALine, width/2, samplesPerBuffer/2);
	}
	
	// Step 6: Post-process truncate (magnitude, log scaling, copy to output)
	float* d_currBuffer = this->impl->d_processedBuffer;
	
	if (config.postProcessingParams.logScaling) {
		cuda_kernels::postProcessTruncateLog<<<gridSize/2, blockSize, 0, stream>>>(
			d_currBuffer, d_fftBuffer2, signalLength / 2, samplesPerBuffer, 0,
			config.postProcessingParams.grayscaleMax,
			config.postProcessingParams.grayscaleMin,
			config.postProcessingParams.addend,
			config.postProcessingParams.multiplicator);
	} else {
		cuda_kernels::postProcessTruncateLin<<<gridSize/2, blockSize, 0, stream>>>(
			d_currBuffer, d_fftBuffer2, signalLength / 2, samplesPerBuffer,
			config.postProcessingParams.grayscaleMax,
			config.postProcessingParams.grayscaleMin,
			config.postProcessingParams.addend,
			config.postProcessingParams.multiplicator);
	}
	
	// Step 7: B-scan flip
	if (config.postProcessingParams.bscanFlip) {
		cuda_kernels::cuda_bscanFlip<<<gridSize/2, blockSize, 0, stream>>>(
			d_currBuffer, d_currBuffer, signalLength / 2, ascansPerBscan,
			(signalLength * ascansPerBscan) / 2, samplesPerBuffer / 4);
	}
	
	// Step 8: Sinusoidal scan correction
	if (config.postProcessingParams.sinusoidalScanCorrection && 
		this->impl->d_sinusoidalScanTmpBuffer != nullptr) {
		checkCudaErrors(cudaMemcpyAsync(
			this->impl->d_sinusoidalScanTmpBuffer, d_currBuffer,
			sizeof(float) * samplesPerBuffer / 2,
			cudaMemcpyDeviceToDevice, stream));
		cuda_kernels::sinusoidalScanCorrection<<<gridSize/2, blockSize, 0, stream>>>(
			d_currBuffer, this->impl->d_sinusoidalScanTmpBuffer,
			this->impl->d_sinusoidalResampleCurve, signalLength / 2,
			ascansPerBscan, bscansPerBuffer, samplesPerBuffer / 2);
	}
	
	// Step 9: Post-process background removal
	if (config.postProcessingParams.backgroundRemoval) {
		// Record background if requested
		if (this->impl->postProcessBackgroundRecordingRequested) {
			cuda_kernels::getPostProcessBackground<<<gridSize/2, blockSize, 0, stream>>>(
				this->impl->d_postProcBackgroundLine, d_currBuffer,
				signalLength / 2, ascansPerBscan);
			
			// Copy to host
			size_t bgSize = signalLength / 2;
			this->impl->recordedPostProcessBackground.resize(bgSize);
			checkCudaErrors(cudaMemcpyAsync(
				this->impl->recordedPostProcessBackground.data(),
				this->impl->d_postProcBackgroundLine,
				bgSize * sizeof(float),
				cudaMemcpyDeviceToHost,
				stream));
			
			this->impl->postProcessBackgroundRecordingRequested = false;
		}

		
		// Update background if user provided new one
		if (this->impl->postProcessBackgroundUpdated) {
			// Background was already copied to device in setPostProcessBackground()
			this->impl->postProcessBackgroundUpdated = false;
		}
		
		// Apply background removal
		cuda_kernels::postProcessBackgroundSubtraction<<<gridSize/2, blockSize, 0, stream>>>(
			d_currBuffer, this->impl->d_postProcBackgroundLine,
			config.postProcessingParams.backgroundWeight,
			config.postProcessingParams.backgroundOffset,
			signalLength / 2, samplesPerBuffer / 2);
	}
	
	// Step 10: Copy result to host output buffer asynchronously
	size_t outputSize = (samplesPerBuffer / 2) * sizeof(float);
	checkCudaErrors(cudaMemcpyAsync(
		currentOutputBuf->getDataPointer(),
		d_currBuffer,
		outputSize,
		cudaMemcpyDeviceToHost,
		stream
	));
	
	// Step 11: Register callback to be called when stream completes
	if (this->impl->callback) {
		// Get pre-allocated callback data for output callback
		int idx = this->impl->nextCallbackIndex.fetch_add(1, std::memory_order_relaxed) % 
		          static_cast<int>(this->impl->callbackDataPool.size());
		Impl::CallbackData* callbackData = &this->impl->callbackDataPool[idx];
		callbackData->outputBuffer = currentOutputBuf;
		
		checkCudaErrors(cudaLaunchHostFunc(stream, outputCallback, callbackData));
	}
	
	// Check for errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::stringstream ss;
		ss << "CUDA error in process(): " << cudaGetErrorString(err);
		throw std::runtime_error(ss.str());
	}
}

// ============================================
// Configuration Updates
// ============================================

void CudaBackend::updateConfig(const ProcessorConfiguration& config) {
	this->impl->config = config;
	
}

void CudaBackend::updateResamplingCurve(const float* curve, size_t length) {
	if (this->impl->d_resampleCurve == nullptr || length != static_cast<size_t>(this->impl->signalLength)) {
		return;
	}
	
	checkCudaErrors(cudaMemcpyAsync(
		this->impl->d_resampleCurve,
		curve,
		length * sizeof(float),
		cudaMemcpyHostToDevice,
		this->impl->userRequestStream
	));
	checkCudaErrors(cudaStreamSynchronize(this->impl->userRequestStream));
}

void CudaBackend::updateDispersionCurve(const float* curve, size_t length) {
    if (this->impl->d_phaseCartesian == nullptr) {
        return;
    }

    checkCudaErrors(cudaMemcpyAsync(
        this->impl->d_phaseCartesian,
        curve,
        length * sizeof(float),
        cudaMemcpyHostToDevice,
        this->impl->userRequestStream
    ));
    
    checkCudaErrors(cudaStreamSynchronize(this->impl->userRequestStream));
}

void CudaBackend::updateWindowCurve(const float* curve, size_t length) {
	if (this->impl->d_windowCurve == nullptr || length != static_cast<size_t>(this->impl->signalLength)) {
		return;
	}
	
	checkCudaErrors(cudaMemcpyAsync(
		this->impl->d_windowCurve,
		curve,
		length * sizeof(float),
		cudaMemcpyHostToDevice,
		this->impl->userRequestStream
	));
	checkCudaErrors(cudaStreamSynchronize(this->impl->userRequestStream));
}

// ============================================
// Post-Process Background Methods
// ============================================

void CudaBackend::requestPostProcessBackgroundRecording() {
	this->impl->postProcessBackgroundRecordingRequested = true;
}

void CudaBackend::setPostProcessBackgroundProfile(const float* background, size_t length) {
	if (!background) {
		throw std::invalid_argument("Background curve data is null");
	}
	
	size_t expectedSize = static_cast<size_t>(this->impl->signalLength / 2);
	if (this->impl->d_postProcBackgroundLine == nullptr || length != expectedSize) {
		throw std::invalid_argument("Invalid background buffer size. Expected " + 
		                            std::to_string(expectedSize) + " but got " + 
		                            std::to_string(length));
	}
	
	// Copy to device
	checkCudaErrors(cudaMemcpyAsync(
		this->impl->d_postProcBackgroundLine,
		background,
		length * sizeof(float),
		cudaMemcpyHostToDevice,
		this->impl->userRequestStream
	));
	checkCudaErrors(cudaStreamSynchronize(this->impl->userRequestStream));
	
	// Update host copy
	this->impl->recordedPostProcessBackground.assign(background, background + length);
	this->impl->postProcessBackgroundUpdated = true;
}

const std::vector<float>& CudaBackend::getPostProcessBackgroundProfile() const {
	return this->impl->recordedPostProcessBackground;
}	


// ============================================
// CUDA-Specific Getters
// ============================================

int CudaBackend::getNumStreams() const {
	return this->impl->numStreams;
}

int CudaBackend::getBlockSize() const {
	return this->impl->blockSize;
}

int CudaBackend::getCurrentDeviceId() const {
	return this->impl->deviceId;
}

// ============================================
// Static GPU Management Methods
// ============================================

std::vector<GpuDeviceInfo> CudaBackend::getAvailableDevices() {
	std::vector<GpuDeviceInfo> devices;
	
	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	
	if (error != cudaSuccess || deviceCount == 0) {
		return devices;  // Return empty vector if no devices
	}
	
	for (int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			GpuDeviceInfo info;
			info.deviceId = i;
			info.name = prop.name;
			info.totalMemory = prop.totalGlobalMem;
			info.computeCapabilityMajor = prop.major;
			info.computeCapabilityMinor = prop.minor;
			info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
			info.multiProcessorCount = prop.multiProcessorCount;
			info.isAvailable = true;
			
			// Get free memory
			size_t free, total;
			cudaSetDevice(i);
			if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
				info.freeMemory = free;
			} else {
				info.freeMemory = 0;
			}
			
			devices.push_back(info);
		}
	}
	
	return devices;
}

bool CudaBackend::setDevice(int deviceId) {
	return cudaSetDevice(deviceId) == cudaSuccess;
}

int CudaBackend::getCurrentDevice() {
	int device = -1;
	cudaGetDevice(&device);
	return device;
}

bool CudaBackend::isDeviceAvailable(int deviceId) {
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		return false;
	}
	return deviceId >= 0 && deviceId < deviceCount;
}

GpuDeviceInfo CudaBackend::getDeviceInfo(int deviceId) {
	GpuDeviceInfo info;
	info.deviceId = deviceId;
	info.isAvailable = false;
	
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
		info.name = prop.name;
		info.totalMemory = prop.totalGlobalMem;
		info.computeCapabilityMajor = prop.major;
		info.computeCapabilityMinor = prop.minor;
		info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
		info.multiProcessorCount = prop.multiProcessorCount;
		info.isAvailable = true;
		
		// Get free memory
		size_t free, total;
		int currentDevice;
		cudaGetDevice(&currentDevice);
		cudaSetDevice(deviceId);
		if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
			info.freeMemory = free;
		}
		cudaSetDevice(currentDevice);
	}
	
	return info;
}

// ============================================
// Helper Methods - Memory Management
// ============================================

void CudaBackend::allocateDeviceBuffers() {
	// Allocate device input buffers (one per stream for overlap, unless using zero-copy)
#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
	this->impl->d_inputBuffers.resize(this->impl->numStreams);
	for (int i = 0; i < this->impl->numStreams; ++i) {
		checkCudaErrors(cudaMalloc(&this->impl->d_inputBuffers[i], 
			this->impl->bytesPerSample * this->impl->samplesPerBuffer));
	}
#endif
	
	// Allocate processing buffers
	checkCudaErrors(cudaMalloc(&this->impl->d_fftBuffer, 
		sizeof(cufftComplex) * this->impl->samplesPerBuffer));
	
	checkCudaErrors(cudaMalloc(&this->impl->d_inputLinearized, 
		sizeof(cufftComplex) * this->impl->samplesPerBuffer));
	
	checkCudaErrors(cudaMalloc(&this->impl->d_processedBuffer, 
		sizeof(float) * this->impl->samplesPerBuffer / 2));
	
	// Allocate curve buffers
	checkCudaErrors(cudaMalloc(&this->impl->d_resampleCurve, 
		sizeof(float) * this->impl->signalLength));
	
	checkCudaErrors(cudaMalloc(&this->impl->d_windowCurve, 
		sizeof(float) * this->impl->signalLength));
	
	checkCudaErrors(cudaMalloc(&this->impl->d_dispersionCurve, 
		sizeof(float) * this->impl->signalLength));
	
	checkCudaErrors(cudaMalloc(&this->impl->d_phaseCartesian, 
		sizeof(cufftComplex) * this->impl->signalLength));
	
	checkCudaErrors(cudaMalloc(&this->impl->d_sinusoidalResampleCurve, 
		sizeof(float) * this->impl->ascansPerBscan));
	
	// Allocate fixed pattern noise buffer
	checkCudaErrors(cudaMalloc(&this->impl->d_meanALine, 
		sizeof(cufftComplex) * this->impl->signalLength));
	
	// Allocate post-processing buffers
	checkCudaErrors(cudaMalloc(&this->impl->d_postProcBackgroundLine, 
		sizeof(float) * this->impl->signalLength / 2));
	
	checkCudaErrors(cudaMalloc(&this->impl->d_sinusoidalScanTmpBuffer, 
		sizeof(float) * this->impl->samplesPerBuffer / 2));
}

void CudaBackend::releaseDeviceBuffers() {
	// Free device input buffers
	for (void* buf : this->impl->d_inputBuffers) {
		if (buf) cudaFree(buf);
	}
	this->impl->d_inputBuffers.clear();
	
	// Free processing buffers
	if (this->impl->d_fftBuffer) { cudaFree(this->impl->d_fftBuffer); this->impl->d_fftBuffer = nullptr; }
	if (this->impl->d_inputLinearized) { cudaFree(this->impl->d_inputLinearized); this->impl->d_inputLinearized = nullptr; }
	if (this->impl->d_processedBuffer) { cudaFree(this->impl->d_processedBuffer); this->impl->d_processedBuffer = nullptr; }
	
	// Free curve buffers
	if (this->impl->d_resampleCurve) { cudaFree(this->impl->d_resampleCurve); this->impl->d_resampleCurve = nullptr; }
	if (this->impl->d_windowCurve) { cudaFree(this->impl->d_windowCurve); this->impl->d_windowCurve = nullptr; }
	if (this->impl->d_dispersionCurve) { cudaFree(this->impl->d_dispersionCurve); this->impl->d_dispersionCurve = nullptr; }
	if (this->impl->d_phaseCartesian) { cudaFree(this->impl->d_phaseCartesian); this->impl->d_phaseCartesian = nullptr; }
	if (this->impl->d_sinusoidalResampleCurve) { cudaFree(this->impl->d_sinusoidalResampleCurve); this->impl->d_sinusoidalResampleCurve = nullptr; }
	
	// Free fixed pattern noise buffer
	if (this->impl->d_meanALine) { cudaFree(this->impl->d_meanALine); this->impl->d_meanALine = nullptr; }
	
	// Free post-processing buffers
	if (this->impl->d_postProcBackgroundLine) { cudaFree(this->impl->d_postProcBackgroundLine); this->impl->d_postProcBackgroundLine = nullptr; }
	if (this->impl->d_sinusoidalScanTmpBuffer) { cudaFree(this->impl->d_sinusoidalScanTmpBuffer); this->impl->d_sinusoidalScanTmpBuffer = nullptr; }
}

void CudaBackend::createStreamsAndEvents() {
	this->impl->streams.resize(this->impl->numStreams);
	for (int i = 0; i < this->impl->numStreams; ++i) {
		checkCudaErrors(cudaStreamCreate(&this->impl->streams[i]));
	}
	checkCudaErrors(cudaStreamCreate(&this->impl->userRequestStream));
	checkCudaErrors(cudaEventCreate(&this->impl->syncEvent));
}

void CudaBackend::destroyStreamsAndEvents() {
	for (cudaStream_t stream : this->impl->streams) {
		if (stream) cudaStreamDestroy(stream);
	}
	this->impl->streams.clear();
	
	if (this->impl->userRequestStream) {
		cudaStreamDestroy(this->impl->userRequestStream);
		this->impl->userRequestStream = nullptr;
	}
	
	if (this->impl->syncEvent) {
		cudaEventDestroy(this->impl->syncEvent);
		this->impl->syncEvent = nullptr;
	}
}

// ============================================
// Individual Operations (for testing)
// ============================================

std::vector<float> CudaBackend::convertInput(
	const void* input,
	IOBuffer::DataType inputType,
	int bitDepth,
	int samples,
	bool applyBitshift
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::rollingAverageBackgroundRemoval(
	const float* input,
	int windowSize,
	int lineWidth,
	int numLines
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::kLinearization(
	const float* input,
	const float* resampleCurve,
	InterpolationMethod method,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::windowing(
	const float* input,
	const float* windowCurve,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::dispersionCompensation(
	const float* input,
	const float* phaseComplex,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::kLinearizationAndWindowing(
	const float* input,
	const float* resampleCurve,
	const float* windowCurve,
	InterpolationMethod method,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::kLinearizationAndWindowingAndDispersion(
	const float* input,
	const float* resampleCurve,
	const float* windowCurve,
	const float* phaseComplex,
	InterpolationMethod method,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::dispersionCompensationAndWindowing(
	const float* input,
	const float* phaseComplex,
	const float* windowCurve,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::fft(const float* input, int lineWidth, int samples) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::ifft(const float* input, int lineWidth, int samples) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::getMinimumVarianceMean(
	const float* input,
	int width,
	int height,
	int segments
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::fixedPatternNoiseRemoval(
	const float* input,
	const float* meanALine,
	int lineWidth,
	int numLines
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::postProcessTruncate(
	const float* input,
	bool logScaling,
	float grayscaleMax,
	float grayscaleMin,
	float addend,
	float multiplicator,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::bscanFlip(
	const float* input,
	int lineWidth,
	int linesPerBscan,
	int numBscans
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::sinusoidalScanCorrection(
	const float* input,
	const float* resampleCurve,
	int lineWidth,
	int linesPerBscan,
	int numBscans
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

std::vector<float> CudaBackend::postProcessBackgroundSubtraction(
	const float* input,
	const float* backgroundLine,
	float weight,
	float offset,
	int lineWidth,
	int samples
) {
	// TODO: Implement as needed for testing
	return std::vector<float>();
}

} // namespace ope
