#include "cpu_backend.h"
#include "cpu_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ope {

// Internal implementation
struct CpuBackend::Impl {
	ProcessorConfiguration config;
	std::function<void(const IOBuffer&)> callback;
	
	// Output buffers (ping-pong)
	IOBuffer outputBuffer1;
	IOBuffer outputBuffer2;
	int currentOutputBuffer;
	
	// Input buffer management
	std::vector<IOBuffer> hostInputBuffers;
	std::queue<IOBuffer*> freeBuffersQueue;
	std::mutex freeQueueMutex;
	std::condition_variable freeQueueCV;
	int numInputBuffers;
	
	// FFTW plans and buffers
	fftwf_complex* fftIn;
	fftwf_complex* fftOut;
	fftwf_plan fftPlan;
	
	// Curves
	std::vector<float> resampleCurve;
	std::vector<std::complex<float>> dispersionPhaseComplex;
	std::vector<float> windowCurve;
	
	// Processing thread
	std::thread processingThread;
	bool stopProcessing;
	
	Impl() 
		: currentOutputBuffer(0)
		, fftIn(nullptr)
		, fftOut(nullptr)
		, fftPlan(nullptr)
		, stopProcessing(false)
		, numInputBuffers(2)  // Default: 2 buffers (ping-pong)
	{}
	
	~Impl() {
		if (this->fftPlan) {
			fftwf_destroy_plan(this->fftPlan);
		}
		if (this->fftIn) {
			fftwf_free(this->fftIn);
		}
		if (this->fftOut) {
			fftwf_free(this->fftOut);
		}
	}
};

CpuBackend::CpuBackend() : impl(std::make_unique<Impl>()) {
}

CpuBackend::~CpuBackend() {
	this->cleanup();
}

void CpuBackend::initialize(const ProcessorConfiguration& config) {
	this->impl->config = config;
	
	int signalLength = config.dataParams.signalLength;
	
	// Allocate FFTW buffers
	this->impl->fftIn = fftwf_alloc_complex(signalLength);
	this->impl->fftOut = fftwf_alloc_complex(signalLength);
	
	if (!this->impl->fftIn || !this->impl->fftOut) {
		throw std::runtime_error("Failed to allocate FFTW buffers");
	}
	
	// Create FFTW plan for IFFT
	this->impl->fftPlan = fftwf_plan_dft_1d(
		signalLength,
		this->impl->fftIn,
		this->impl->fftOut,
		FFTW_BACKWARD,
		FFTW_ESTIMATE
	);
	
	if (!this->impl->fftPlan) {
		throw std::runtime_error("Failed to create FFTW plan");
	}
	
	// Allocate output buffers
	size_t outputSize = (config.dataParams.samplesPerBuffer / 2) * sizeof(float);
	
	if (!this->impl->outputBuffer1.allocateMemory(outputSize) ||
		!this->impl->outputBuffer2.allocateMemory(outputSize)) {
		throw std::runtime_error("Failed to allocate output buffers");
	}
	
	this->impl->outputBuffer1.setDataType(IOBuffer::DataType::FLOAT32);
	this->impl->outputBuffer2.setDataType(IOBuffer::DataType::FLOAT32);
	
	// Allocate input buffers
	size_t inputSize = config.dataParams.samplesPerBuffer * sizeof(float);
	this->impl->hostInputBuffers.resize(this->impl->numInputBuffers);
	
	for (int i = 0; i < this->impl->numInputBuffers; ++i) {
		if (!this->impl->hostInputBuffers[i].allocateMemory(inputSize)) {
			throw std::runtime_error("Failed to allocate input buffer " + std::to_string(i));
		}
		this->impl->hostInputBuffers[i].setDataType(IOBuffer::DataType::FLOAT32);
		this->impl->freeBuffersQueue.push(&this->impl->hostInputBuffers[i]);
	}

	// Initialize curves
	this->impl->resampleCurve.resize(signalLength);
	this->impl->dispersionPhaseComplex.resize(signalLength);
	this->impl->windowCurve.resize(signalLength);
}

void CpuBackend::cleanup() {
	this->impl->stopProcessing = true;
	if (this->impl->processingThread.joinable()) {
		this->impl->processingThread.join();
	}
	
	this->impl->outputBuffer1.releaseMemory();
	this->impl->outputBuffer2.releaseMemory();
	
	// Release input buffers
	for (auto& buffer : this->impl->hostInputBuffers) {
		buffer.releaseMemory();
	}
	this->impl->hostInputBuffers.clear();
	
	// Clear the queue
	while (!this->impl->freeBuffersQueue.empty()) {
		this->impl->freeBuffersQueue.pop();
	}
}

void CpuBackend::setOutputCallback(std::function<void(const IOBuffer&)> callback) {
	this->impl->callback = callback;
}

void CpuBackend::process(IOBuffer& input) {
	// Switch output buffer (ping-pong)
	this->impl->currentOutputBuffer = (this->impl->currentOutputBuffer + 1) % 2;
	IOBuffer& output = (this->impl->currentOutputBuffer == 0) 
		? this->impl->outputBuffer1 
		: this->impl->outputBuffer2;
	
	// Get configuration
	const ProcessorConfiguration& config = this->impl->config;
	const int signalLength = config.dataParams.signalLength;
	const int samplesPerBuffer = config.dataParams.samplesPerBuffer;
	const int numAscans = samplesPerBuffer / signalLength;
	const int outputSamplesPerAscan = signalLength / 2;
	
	// Convert input to complex array
	std::vector<std::complex<float>> complexData;
	cpu_kernels::convertInputData<float>(
		input.getDataPointer(),
		samplesPerBuffer,
		config.dataParams.getBitDepth(),
		complexData
	);
	
	// Get output pointer
	float* outputPtr = static_cast<float*>(output.getDataPointer());
	
	// Process each A-scan
	for (int ascanIdx = 0; ascanIdx < numAscans; ++ascanIdx) {
		// Extract A-scan
		std::vector<std::complex<float>> ascan(signalLength);
		int startIdx = ascanIdx * signalLength;
		std::copy(complexData.begin() + startIdx, 
		          complexData.begin() + startIdx + signalLength,
		          ascan.begin());
		
		// Apply processing pipeline using kernels
		if (config.backgroundRemovalParams.enabled) {
			cpu_kernels::rollingAverageDCRemoval<float>(
				ascan,
				config.backgroundRemovalParams.rollingAverageWindowSize
			);
		}
		
		//k-linearization
		if (config.resamplingParams.enabled) {
			std::vector<std::complex<float>> resampled;
			
			InterpolationMethod method = config.resamplingParams.interpolationMethod;
			if (method == InterpolationMethod::CUBIC) {
				cpu_kernels::kLinearizationCubic<float>(
					ascan,
					this->impl->resampleCurve,
					resampled
				);
			} else if (method == InterpolationMethod::LINEAR) {
				cpu_kernels::kLinearizationLinear<float>(
					ascan,
					this->impl->resampleCurve,
					resampled
				);
			} else if (method == InterpolationMethod::LANCZOS) {
				cpu_kernels::kLinearizationLanczos<float>(
					ascan,
					this->impl->resampleCurve,
					resampled
				);
			}
			
			ascan = std::move(resampled);
		}
		
		//dispersion compensation
		if (config.dispersionParams.enabled) {
			cpu_kernels::dispersionCompensation<float>(
				ascan,
				this->impl->dispersionPhaseComplex
			);
		}
		
		//windowing
		if (config.windowingParams.enabled) {
			cpu_kernels::applyWindow<float>(
				ascan,
				this->impl->windowCurve
			);
		}
		
		//IFFT
		std::vector<std::complex<float>> ifftOutput;
		cpu_kernels::computeIFFT<float>(
			ascan,
			ifftOutput,
			this->impl->fftPlan,
			this->impl->fftIn,
			this->impl->fftOut
		);
		
		//magnitude calculation, grayscale conversion, truncation 
		std::vector<float> processedAscan;
		if (config.postProcessingParams.logScaling) {
			cpu_kernels::logScaleAndTruncate<float>(
				ifftOutput,
				processedAscan,
				config.postProcessingParams.multiplicator,
				config.postProcessingParams.grayscaleMin,
				config.postProcessingParams.grayscaleMax,
				config.postProcessingParams.addend,
				(config.postProcessingParams.grayscaleMin == config.postProcessingParams.grayscaleMax)
			);
		} else {
			cpu_kernels::linearScaleAndTrucate<float>(
				ifftOutput,
				processedAscan,
				config.postProcessingParams.multiplicator,
				config.postProcessingParams.grayscaleMin,
				config.postProcessingParams.grayscaleMax,
				config.postProcessingParams.addend
			);
		}
		
		// Copy to output
		int outputStartIdx = ascanIdx * outputSamplesPerAscan;
		std::copy(processedAscan.begin(),
				processedAscan.end(),
				outputPtr + outputStartIdx);
	}
	
	// Call callback when done
	if (this->impl->callback) {
		this->impl->callback(output);
	}
	
	// Return the input buffer to the free queue
	// For CPU backend, processing is synchronous so we can return immediately
	{
		std::lock_guard<std::mutex> lock(this->impl->freeQueueMutex);
		this->impl->freeBuffersQueue.push(&input);
		this->impl->freeQueueCV.notify_one();
	}
}

void CpuBackend::updateConfig(const ProcessorConfiguration& config) {
	this->impl->config = config;

}

void CpuBackend::updateResamplingCurve(const float* curve, size_t length) {
	if (!curve && length > 0) {
		throw std::runtime_error("Invalid window curve pointer");
	}
	this->impl->resampleCurve.assign(curve, curve + length);
}

void CpuBackend::updateDispersionCurve(const float* curve, size_t length) {
	// Curve is interleaved real/imag
	this->impl->dispersionPhaseComplex.resize(length/2);
	for (size_t i = 0; i < length/2; ++i) {
		this->impl->dispersionPhaseComplex[i] = std::complex<float>(
			curve[i * 2],     // real
			curve[i * 2 + 1]  // imag
		);
	}
}

void CpuBackend::updateWindowCurve(const float* curve, size_t length) {
	if (!curve && length > 0) {
		throw std::runtime_error("Invalid window curve pointer");
	}
	this->impl->windowCurve.assign(curve, curve + length);
}

// ============================================
// Individual operations (adapted from your code)
// ============================================

std::vector<float> CpuBackend::convertInput(
	const void* input,
	IOBuffer::DataType inputType,
	int bitDepth,
	int samples,
	bool applyBitshift)
{
	// Use kernel to convert to complex, then extract real part
	std::vector<std::complex<float>> complexOutput;
	cpu_kernels::convertInputData<float>(input, samples, bitDepth, complexOutput);
	
	std::vector<float> output(samples);
	for (int i = 0; i < samples; ++i) {
		output[i] = complexOutput[i].real();
	}
	
	return output;
}

std::vector<float> CpuBackend::kLinearization(
	const float* input,
	const float* resampleCurve,
	InterpolationMethod method,
	int lineWidth,
	int samples)
{
	// Convert input to complex for kernel processing
	std::vector<std::complex<float>> complexInput(lineWidth);
	for (int i = 0; i < lineWidth; ++i) {
		complexInput[i] = std::complex<float>(input[i], 0.0f);
	}
	
	// Convert resample curve to vector
	std::vector<float> resampleVec(resampleCurve, resampleCurve + lineWidth);
	
	// Use kernel for k-linearization
	std::vector<std::complex<float>> complexOutput;
	cpu_kernels::kLinearizationCubic<float>(complexInput, resampleVec, complexOutput);
	
	// Extract real part
	std::vector<float> output(complexOutput.size());
	for (size_t i = 0; i < complexOutput.size(); ++i) {
		output[i] = complexOutput[i].real();
	}
	
	return output;
}

std::vector<float> CpuBackend::windowing(
	const float* input,
	const float* windowCurve,
	int lineWidth,
	int samples)
{
	// Convert input to complex
	std::vector<std::complex<float>> complexData(samples);
	for (int i = 0; i < samples; ++i) {
		complexData[i] = std::complex<float>(input[i], 0.0f);
	}
	
	// Convert window curve to vector
	std::vector<float> windowVec(windowCurve, windowCurve + lineWidth);
	
	// Apply window using kernel
	cpu_kernels::applyWindow<float>(complexData, windowVec);
	
	// Extract real part
	std::vector<float> output(samples);
	for (int i = 0; i < samples; ++i) {
		output[i] = complexData[i].real();
	}
	
	return output;
}

std::vector<float> CpuBackend::dispersionCompensation(
	const float* input,
	const float* phaseComplex,
	int lineWidth,
	int samples)
{
	// Convert input to complex
	std::vector<std::complex<float>> complexData(samples);
	for (int i = 0; i < samples; ++i) {
		complexData[i] = std::complex<float>(input[i], 0.0f);
	}
	
	// Convert phaseComplex (interleaved real/imag) to vector
	std::vector<std::complex<float>> phaseVec(lineWidth);
	for (int i = 0; i < lineWidth; ++i) {
		phaseVec[i] = std::complex<float>(phaseComplex[i * 2], phaseComplex[i * 2 + 1]);
	}
	
	// Apply dispersion using kernel
	cpu_kernels::dispersionCompensation<float>(complexData, phaseVec);
	
	// Extract real part
	std::vector<float> output(samples);
	for (int i = 0; i < samples; ++i) {
		output[i] = complexData[i].real();
	}
	
	return output;
}

std::vector<float> CpuBackend::fft(const float* input, int lineWidth, int samples) {
	// Stub for now
	return std::vector<float>(samples);
}

std::vector<float> CpuBackend::ifft(const float* input, int lineWidth, int samples) {
	// Convert input to complex
	std::vector<std::complex<float>> complexInput(lineWidth);
	for (int i = 0; i < lineWidth; ++i) {
		complexInput[i] = std::complex<float>(input[i], 0.0f);
	}
	
	// Compute IFFT using kernel
	std::vector<std::complex<float>> complexOutput;
	cpu_kernels::computeIFFT<float>(
		complexInput,
		complexOutput,
		this->impl->fftPlan,
		this->impl->fftIn,
		this->impl->fftOut
	);
	
	// Return magnitude
	std::vector<float> output(lineWidth);
	for (int i = 0; i < lineWidth; ++i) {
		float real = complexOutput[i].real();
		float imag = complexOutput[i].imag();
		output[i] = std::sqrt(real * real + imag * imag);
	}
	
	return output;
}

// ============================================
// Stubs for other operations (implement as needed)
// ============================================

std::vector<float> CpuBackend::rollingAverageBackgroundRemoval(const float* input, int windowSize, int lineWidth, int numLines) {
	return std::vector<float>(lineWidth * numLines);  // Stub
}

std::vector<float> CpuBackend::kLinearizationAndWindowing(const float* input, const float* resampleCurve, const float* windowCurve, InterpolationMethod method, int lineWidth, int samples) {
	return std::vector<float>(samples);  // Stub
}

std::vector<float> CpuBackend::kLinearizationAndWindowingAndDispersion(const float* input, const float* resampleCurve, const float* windowCurve, const float* phaseComplex, InterpolationMethod method, int lineWidth, int samples) {
	return std::vector<float>(samples);  // Stub
}

std::vector<float> CpuBackend::dispersionCompensationAndWindowing(const float* input, const float* phaseComplex, const float* windowCurve, int lineWidth, int samples) {
	return std::vector<float>(samples);  // Stub
}

std::vector<float> CpuBackend::getMinimumVarianceMean(const float* input, int width, int height, int segments) {
	return std::vector<float>(width);  // Stub
}

std::vector<float> CpuBackend::fixedPatternNoiseRemoval(const float* input, const float* meanALine, int lineWidth, int numLines) {
	return std::vector<float>(lineWidth * numLines);  // Stub
}

std::vector<float> CpuBackend::postProcessTruncate(const float* input, bool logScaling, float grayscaleMax, float grayscaleMin, float addend, float multiplicator, int lineWidth, int samples) {
	return std::vector<float>(samples);  // Stub
}

std::vector<float> CpuBackend::bscanFlip(const float* input, int lineWidth, int linesPerBscan, int numBscans) {
	return std::vector<float>(lineWidth * linesPerBscan * numBscans);  // Stub
}

std::vector<float> CpuBackend::sinusoidalScanCorrection(const float* input, const float* resampleCurve, int lineWidth, int linesPerBscan, int numBscans) {
	return std::vector<float>(lineWidth * linesPerBscan * numBscans);  // Stub
}

std::vector<float> CpuBackend::postProcessBackgroundRemoval(const float* input, const float* backgroundLine, float weight, float offset, int lineWidth, int samples) {
	return std::vector<float>(samples);  // Stub
}

// ============================================
// BUFFER MANAGEMENT
// ============================================

IOBuffer& CpuBackend::getInputBuffer(int index) {
	if (index < 0 || index >= this->impl->numInputBuffers) {
		throw std::runtime_error("Buffer index out of range");
	}
	return this->impl->hostInputBuffers[index];
}

IOBuffer& CpuBackend::getNextAvailableInputBuffer() {
	std::unique_lock<std::mutex> lock(this->impl->freeQueueMutex);
	
	// Wait until a buffer is available
	while (this->impl->freeBuffersQueue.empty()) {
		this->impl->freeQueueCV.wait(lock);
	}
	
	// Get the next free buffer
	IOBuffer* buffer = this->impl->freeBuffersQueue.front();
	this->impl->freeBuffersQueue.pop();
	
	return *buffer;
}

int CpuBackend::getNumInputBuffers() const {
	return this->impl->numInputBuffers;
}

float CpuBackend::cubicHermiteInterpolation(float y0, float y1, float y2, float y3, float t) {
	return cpu_kernels::cubicHermiteInterpolation<float>(y0, y1, y2, y3, t);
}

float CpuBackend::clamp(float value, float low, float high) {
	return cpu_kernels::clamp<float>(value, low, high);
}

} // namespace ope