#include "cpu_backend.h"
#include "cpu_kernels.h"
#define POCKETFFT_CACHE_SIZE 8
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>


namespace ope {


// ============================================
// Internal implementation
// ============================================

struct CpuBackend::Impl {
	ProcessorConfiguration config;
	std::function<void(const IOBuffer&)> callback;
	
	// Output buffers (ping-pong)
	IOBuffer outputBuffer1;
	IOBuffer outputBuffer2;
	int currentOutputBuffer;

	// Semaphore to limit in-flight output buffers (prevents buffer reuse before consumer release)
	std::mutex outputSemaphoreMutex;
	std::condition_variable outputSemaphoreCV;
	int availableOutputBuffers = 2;  // 2 ping-pong buffers
	
	// Input buffer management
	std::vector<uint8_t> processingBuffer; 
	std::vector<IOBuffer> hostInputBuffers;
	std::queue<IOBuffer*> freeBuffersQueue;
	std::mutex freeQueueMutex;
	std::condition_variable freeQueueCV;
	int numInputBuffers;
	
	// Work queue for async processing
	std::queue<IOBuffer*> workQueue;
	std::mutex workQueueMutex;
	std::condition_variable workQueueCV;
	
	// Processing thread
	std::thread processingThread;
	std::atomic<bool> stopProcessing;
	
	// Reuse the transform metadata and PocketFFT's cached plans across A-scans.
	pocketfft::shape_t fftShape;
	const pocketfft::shape_t fftAxes{0};
	const pocketfft::stride_t fftStride{sizeof(std::complex<float>)};

	// Storage for per-buffer IFFT outputs when post-processing requires whole-buffer context
	std::vector<std::vector<std::complex<float>>> allIfftOutputs;
	
	// Accumulated IFFT outputs for FPN determination across buffers
	std::vector<std::vector<std::complex<float>>> accumulatedIfftOutputs;
	int accumulatedAscanCount = 0;
	
	// Curves
	std::vector<float> resampleCurve;
	std::vector<std::complex<float>> dispersionPhaseComplex;
	std::vector<float> windowCurve;
	std::vector<float> postProcessBackgroundProfile;

	// Fixed-pattern noise storage (interleaved real/imag)
	std::vector<float> recordedFixedPatternNoise;
	bool fixedPatternNoiseDeterminationRequested = false;

	bool postProcessBackgroundRecordingRequested;

	// Background frame (line-field OCT) state. Guarded by processingStateMutex together
	// with config: the worker thread holds the lock for a whole buffer while public
	// setters may replace these vectors from another thread
	mutable std::mutex processingStateMutex;
	std::vector<float> backgroundFrame;              // samplesPerBscan floats, valid when non-empty
	std::vector<float> smoothedBackgroundFrame;      // cached smoothed copy of backgroundFrame
	bool smoothedFrameDirty = true;
	std::vector<float> backgroundFrameAccumulator;
	int backgroundFrameBscansRecorded = 0;
	int backgroundFrameBscansTarget = 0;     // latched at requestBackgroundFrameRecording()
	bool backgroundFrameRecordingInProgress = false;

	// Per-buffer spectral averages for post-FFT frame correction (one value per A-scan)
	std::vector<float> liveSpectralAverages;

	// Buffer-level preparation for line-field OCT features (must run before the per-A-scan
	// loop: EMA folds ALL B-scans of the buffer into the background first, then the FINAL
	// background is subtracted from the entire buffer - matching the CUDA/OCTproZ semantics)
	void prepareBackgroundFrame(const void* inputData);

	// Temporary buffers for per-A-scan processing
	std::vector<std::complex<float>> spectrum;
	std::vector<std::complex<float>> linearizedSpectrum;
	std::vector<std::complex<float>> ifftOutput;
	std::vector<float> processedAscan;

	void computeIFFT() {
		pocketfft::c2c(fftShape, fftStride, fftStride, fftAxes, pocketfft::BACKWARD,
			linearizedSpectrum.data(), ifftOutput.data(), 1.0f, 1);
	}

	// Compute Fixed-pattern noise helper
	void computeFixedPatternNoiseIfRequested(const ProcessorConfiguration& config, int signalLength, int totalAscans);
	
	Impl() 
		: currentOutputBuffer(0)
		, stopProcessing(false)
		, numInputBuffers(2)  // default: 2 buffers (ping-pong)
		, postProcessBackgroundRecordingRequested(false)
	{}
	
	~Impl() {
		this->stopProcessing = true;
		if (this->processingThread.joinable()) {
			this->workQueueCV.notify_one();
			this->processingThread.join();
		}
	}


	void recordPostProcessBackground(const float* processedData, int samplesPerAscan, int totalAscans) {
		this->postProcessBackgroundProfile.resize(samplesPerAscan, 0.0f);

		//average all A-scans to create background prfile
		for (int i = 0; i < samplesPerAscan; ++i) {
			float sum = 0.0f;
			for (int ascan = 0; ascan < totalAscans; ++ascan) {
				sum += processedData[ascan * samplesPerAscan + i];
			}
			this->postProcessBackgroundProfile[i] = sum / static_cast<float>(totalAscans);
		}

		//sync recorded profile to configuration
		if (!this->postProcessBackgroundProfile.empty()) {
			this->config.setBackgroundProfile(this->postProcessBackgroundProfile);
		}

		this->postProcessBackgroundRecordingRequested = false;
	}	
	
	void processingThreadFunc() {
		while (!this->stopProcessing) {
			// Get work from queue
			IOBuffer* inputBuffer = nullptr;
			{
				std::unique_lock<std::mutex> lock(this->workQueueMutex);
				this->workQueueCV.wait(lock, [this]() {
					return !this->workQueue.empty() || this->stopProcessing;
				});
				
				if (this->stopProcessing) {
					break;
				}
				
				inputBuffer = this->workQueue.front();
				this->workQueue.pop();
			}
			
			// Resize buffer if needed (only happens on first frame or config change)
			size_t inputDataSize = this->config.dataParams.samplesPerBuffer() *
								(this->config.dataParams.getBitDepth() / 8);
			if (this->processingBuffer.size() != inputDataSize) {
				this->processingBuffer.resize(inputDataSize);
			}
			
			// Copy input data (reusing pre-allocated buffer)
			std::memcpy(this->processingBuffer.data(), inputBuffer->getDataPointer(), inputDataSize);

			// Store buffer ID before returning input buffer
			uint64_t bufferId = inputBuffer->getBufferId();

			// Return input buffer right after copying to processingBuffer so user can reuse it
			{
				std::lock_guard<std::mutex> lock(this->freeQueueMutex);
				this->freeBuffersQueue.push(inputBuffer);
			}
			this->freeQueueCV.notify_one();

			// Wait for an output buffer to be available (prevents buffer reuse before consumer release)
			{
				std::unique_lock<std::mutex> lock(this->outputSemaphoreMutex);
				this->outputSemaphoreCV.wait(lock, [this]() {
					return this->availableOutputBuffers > 0;
				});
				this->availableOutputBuffers--;
			}

			this->currentOutputBuffer = (this->currentOutputBuffer + 1) % 2;
			IOBuffer& output = (this->currentOutputBuffer == 0)
				? this->outputBuffer1
				: this->outputBuffer2;

			// Propagate buffer ID to output
			output.setBufferId(bufferId);

			// Hold the state lock for the whole buffer so setters cannot replace
			// background/config state mid-buffer; released before the callback
			{
				std::lock_guard<std::mutex> stateLock(this->processingStateMutex);
				this->processData(this->processingBuffer.data(), output);
			}

			// Invoke callback
			if (this->callback) {
				this->callback(output);
			}
		}
	}
	
	void processData(const void* inputData, IOBuffer& output) {
		const ProcessorConfiguration& config = this->config;
		const int signalLength = config.dataParams.signalLength;
		const int ascansPerBscan = config.dataParams.ascansPerBscan;
		const int bscansPerBuffer = config.dataParams.bscansPerBuffer;
		const int totalAscans = ascansPerBscan * bscansPerBuffer;
		const int outputSamplesPerAscan = signalLength / 2;

		float* outputPtr = static_cast<float*>(output.getDataPointer());

		// Line-field OCT: buffer-level preparation (spectral averages, recording, EMA, smoothing).
		// Must run before the per-A-scan loop so the FINAL background of this buffer is
		// subtracted from ALL its A-scans (see prepareBackgroundFrame)
		bool backgroundFrameActive = config.processingParams.backgroundFrame.enabled;
		bool frameCorrectionActive = config.processingParams.frameCorrection.enabled;
		if (backgroundFrameActive || frameCorrectionActive || this->backgroundFrameRecordingInProgress) {
			this->prepareBackgroundFrame(inputData);
		}
		const std::vector<float>& activeBackgroundFrame =
			(config.processingParams.backgroundFrame.smoothSpectra && !this->smoothedBackgroundFrame.empty())
			? this->smoothedBackgroundFrame
			: this->backgroundFrame;
		bool applyBackgroundFrame = backgroundFrameActive && !activeBackgroundFrame.empty();
		float frameNormalizationScale = std::sqrt(std::pow(2.0f, static_cast<float>(config.dataParams.getBitDepth())));

		// Check if we need FPN processing (requires allIfftOutputs storage)
		bool needsFPN = this->fixedPatternNoiseDeterminationRequested ||
		                config.processingParams.fixedPatternNoise.enabled ||
		                config.processingParams.fixedPatternNoise.continuous;

		// Prepare storage for per-A-scan IFFT outputs (only if FPN needed)
		if (needsFPN) {
			this->allIfftOutputs.clear();
			this->allIfftOutputs.resize(totalAscans);
		}

		// Process each A-scan
		for (int ascanIdx = 0; ascanIdx < totalAscans; ++ascanIdx) {
			const void* ascanStart = static_cast<const uint8_t*>(inputData) +
			                         (ascanIdx * signalLength * (config.dataParams.getBitDepth() / 8));
			
			// 1. Convert input data
			cpu_kernels::convertInputData<float>(
				ascanStart,
				signalLength,
				config.dataParams.getBitDepth(),
				this->spectrum
			);
			
			// 1.5 Background frame subtraction (line-field OCT, before DC removal to match OCTproZ order)
			if (applyBackgroundFrame) {
				const float* backgroundRow = activeBackgroundFrame.data() +
				                             (ascanIdx % ascansPerBscan) * signalLength;
				cpu_kernels::backgroundFrameSubtraction<float>(
					this->spectrum,
					backgroundRow,
					config.processingParams.backgroundFrame.normalize,
					frameNormalizationScale
				);
			}

			// 2. Background removal (if enabled)
			if (config.processingParams.dcRemoval.enabled) {
				cpu_kernels::rollingAverageDCRemoval<float>(
					this->spectrum,
					config.processingParams.dcRemoval.windowSize
				);
			}

			// 3. K-linearization (if enabled)
			if (config.processingParams.resampling.enabled) {
				switch (config.processingParams.resampling.method) {
					case InterpolationMethod::LINEAR:
						cpu_kernels::kLinearizationLinear<float>(this->spectrum, this->resampleCurve, this->linearizedSpectrum);
						break;
					case InterpolationMethod::CUBIC:
						cpu_kernels::kLinearizationCubic<float>(this->spectrum, this->resampleCurve, this->linearizedSpectrum);
						break;
					case InterpolationMethod::LANCZOS:
						cpu_kernels::kLinearizationLanczos<float>(this->spectrum, this->resampleCurve, this->linearizedSpectrum);
						break;
				}
			} else {
				this->linearizedSpectrum = this->spectrum;
			}
			
			// 4. Windowing (if enabled)
			if (config.processingParams.windowing.enabled) {
				cpu_kernels::applyWindow<float>(this->linearizedSpectrum, this->windowCurve);
			}

			// 5. Dispersion compensation (if enabled)
			if (config.processingParams.dispersion.enabled) {
				cpu_kernels::dispersionCompensation<float>(this->linearizedSpectrum, this->dispersionPhaseComplex);
			}
			
			// 6. IFFT
			this->computeIFFT();

			// 6.5 Post-FFT frame correction: divide by sqrt of the pre-subtraction spectral average
			if (frameCorrectionActive) {
				cpu_kernels::normalizeBySqrtSpectralAverage<float>(
					this->ifftOutput,
					this->liveSpectralAverages[ascanIdx],
					frameNormalizationScale
				);
			}

			if (needsFPN) {
				// Store IFFT output for later post-processing (fixed-pattern noise removal requires whole-buffer context)
				this->allIfftOutputs[ascanIdx] = this->ifftOutput;
			} else {
				// Process immediately to output (no storage, better performance)
				// 8. Magnitude calculation, grayscale conversion, truncation
				if (config.processingParams.intensity.logScale) {
					cpu_kernels::logScaleAndTruncate<float>(
						this->ifftOutput,
						this->processedAscan,
						config.processingParams.intensity.preScale,
						config.processingParams.intensity.rangeMin,
						config.processingParams.intensity.rangeMax,
						config.processingParams.intensity.postOffset,
						(config.processingParams.intensity.rangeMin == config.processingParams.intensity.rangeMax)
					);
				} else {
					cpu_kernels::linearScaleAndTruncate<float>(
						this->ifftOutput,
						this->processedAscan,
						config.processingParams.intensity.preScale,
						config.processingParams.intensity.rangeMin,
						config.processingParams.intensity.rangeMax,
						config.processingParams.intensity.postOffset
					);
				}
				// Copy to output
				int outputStartIdx = ascanIdx * outputSamplesPerAscan;
				std::copy(this->processedAscan.begin(), this->processedAscan.begin() + outputSamplesPerAscan, outputPtr + outputStartIdx);
			}
		}

		if (needsFPN) {
			// 7. Fixed-pattern-noise determination
			this->computeFixedPatternNoiseIfRequested(config, signalLength, totalAscans);

			for (int ascanIdx = 0; ascanIdx < totalAscans; ++ascanIdx) {
				std::vector<std::complex<float>>& ifftOutputRef = this->allIfftOutputs[ascanIdx];
				if (config.processingParams.fixedPatternNoise.enabled && !this->recordedFixedPatternNoise.empty()) {
					const std::vector<float>& meanVec = this->recordedFixedPatternNoise;
					if (!meanVec.empty()) {
						cpu_kernels::meanALineSubtraction<float>(ifftOutputRef, meanVec);
					}
				}
				// 8. Magnitude calculation, grayscale conversion, truncation
				if (config.processingParams.intensity.logScale) {
					cpu_kernels::logScaleAndTruncate<float>(
						ifftOutputRef,
						this->processedAscan,
						config.processingParams.intensity.preScale,
						config.processingParams.intensity.rangeMin,
						config.processingParams.intensity.rangeMax,
						config.processingParams.intensity.postOffset,
						(config.processingParams.intensity.rangeMin == config.processingParams.intensity.rangeMax)
					);
				} else {
					cpu_kernels::linearScaleAndTruncate<float>(
						ifftOutputRef,
						this->processedAscan,
						config.processingParams.intensity.preScale,
						config.processingParams.intensity.rangeMin,
						config.processingParams.intensity.rangeMax,
						config.processingParams.intensity.postOffset
					);
				}
				// Copy to output
				int outputStartIdx = ascanIdx * outputSamplesPerAscan;
				std::copy(this->processedAscan.begin(), this->processedAscan.begin() + outputSamplesPerAscan, outputPtr + outputStartIdx);
			}
		}

		// 9. Post-process background profile subtraction
		if (config.processingParams.background.enabled) {
			// Record background if requested (must happen BEFORE removal)
			if (this->postProcessBackgroundRecordingRequested) {
				this->recordPostProcessBackground(outputPtr, outputSamplesPerAscan, totalAscans);
			}
			
			// Apply background removal if we have a background curve
			if (!this->postProcessBackgroundProfile.empty()) {
				cpu_kernels::applyPostProcessBackgroundSubtraction<float>(
					outputPtr,
					this->postProcessBackgroundProfile.data(),
					config.processingParams.background.weight,
					config.processingParams.background.offset,
					outputSamplesPerAscan,
					totalAscans
				);
			}
		}
	}
};

void CpuBackend::Impl::computeFixedPatternNoiseIfRequested(const ProcessorConfiguration& config, int signalLength, int totalAscans) {
	if (!(this->fixedPatternNoiseDeterminationRequested || config.processingParams.fixedPatternNoise.continuous)) {
		return;  
	}

	const int FPN_SEGMENTS = 8; // match CUDA //todo: make configurable?

	for (const auto& ifftOutput : this->allIfftOutputs) {
		this->accumulatedIfftOutputs.push_back(ifftOutput);
		this->accumulatedAscanCount++;
	}

	int requiredAscanCount = config.processingParams.fixedPatternNoise.bscanAverageCount * config.dataParams.ascansPerBscan;

	if (this->accumulatedAscanCount < requiredAscanCount) {
		return;
	}

	std::vector<std::vector<std::complex<float>>> inputs;
	if (this->accumulatedAscanCount == requiredAscanCount) {
		inputs = this->accumulatedIfftOutputs;
	} else {
		inputs.assign(this->accumulatedIfftOutputs.begin(),
					  this->accumulatedIfftOutputs.begin() + requiredAscanCount);
	}

	std::vector<float> meanInterleaved = cpu_kernels::getMinimumVarianceMean<float>(inputs, signalLength, FPN_SEGMENTS);

	int positivePairs = signalLength / 2;
	this->recordedFixedPatternNoise.clear();
	this->recordedFixedPatternNoise.reserve(static_cast<size_t>(positivePairs) * 2);
	for (int i = 0; i < positivePairs; ++i) {
		this->recordedFixedPatternNoise.push_back(meanInterleaved[i * 2]);
		this->recordedFixedPatternNoise.push_back(meanInterleaved[i * 2 + 1]);
	}

	//sync recorded profile to configuration
	if (!this->recordedFixedPatternNoise.empty()) {
		this->config.setFixedPatternNoiseProfile(this->recordedFixedPatternNoise);
	}

	// If continuous mode, keep the last requiredAscanCount A-scans as a sliding window.
	if (config.processingParams.fixedPatternNoise.continuous) {
		if (this->accumulatedAscanCount > requiredAscanCount) {
			// erase newer entries, keep the earliest ones
			this->accumulatedIfftOutputs.erase(this->accumulatedIfftOutputs.begin() + requiredAscanCount,
											   this->accumulatedIfftOutputs.end());
			this->accumulatedAscanCount = requiredAscanCount;
		}
	} else {
		// one-shot determination: clear accumulation after computing
		this->accumulatedIfftOutputs.clear();
		this->accumulatedAscanCount = 0;
	}

	this->fixedPatternNoiseDeterminationRequested = false;
}

void CpuBackend::Impl::prepareBackgroundFrame(const void* inputData) {
	const ProcessorConfiguration& config = this->config;
	const int signalLength = config.dataParams.signalLength;
	const int ascansPerBscan = config.dataParams.ascansPerBscan;
	const int bscansPerBuffer = config.dataParams.bscansPerBuffer;
	const int totalAscans = ascansPerBscan * bscansPerBuffer;
	const int samplesPerBscan = signalLength * ascansPerBscan;
	const int bytesPerSample = config.dataParams.getBitDepth() / 8;

	bool frameCorrection = config.processingParams.frameCorrection.enabled;
	bool recording = this->backgroundFrameRecordingInProgress;
	// Continuous EMA only runs together with subtraction (matches OCTproZ); it bootstraps
	// from a zeroed background and converges over ~bscansToAverage B-scans.
	// Recording takes precedence: EMA is suppressed while a recording is in progress and
	// resumes on the next buffer, seeded by the freshly recorded frame
	bool continuous = config.processingParams.backgroundFrame.continuousUpdate &&
	                  config.processingParams.backgroundFrame.enabled &&
	                  !recording;

	if (frameCorrection || recording || continuous) {
		if (frameCorrection) {
			this->liveSpectralAverages.resize(totalAscans);
		}
		if (continuous && this->backgroundFrame.empty()) {
			this->backgroundFrame.assign(samplesPerBscan, 0.0f);
		}

		int bscansToRecord = 0;
		if (recording) {
			// The target is latched at request time: changing the averaging setting
			// mid-recording must not corrupt the count or the normalization
			int bscansRemaining = this->backgroundFrameBscansTarget - this->backgroundFrameBscansRecorded;
			bscansToRecord = std::min(bscansPerBuffer, bscansRemaining);
		}
		float alpha = 1.0f / static_cast<float>(config.processingParams.backgroundFrame.bscansToAverage);

		// One conversion pass over the buffer; each consumer reads the raw (pre-subtraction) spectra.
		// Sequential A-scan order folds the B-scans into the EMA background in buffer order,
		// matching the CUDA updateBackgroundFrameEMA kernel semantics.
		for (int ascanIdx = 0; ascanIdx < totalAscans; ++ascanIdx) {
			const void* ascanStart = static_cast<const uint8_t*>(inputData) +
			                         (ascanIdx * signalLength * bytesPerSample);
			cpu_kernels::convertInputData<float>(
				ascanStart,
				signalLength,
				config.dataParams.getBitDepth(),
				this->spectrum
			);

			if (frameCorrection) {
				this->liveSpectralAverages[ascanIdx] = cpu_kernels::spectralAverage<float>(this->spectrum);
			}

			int bscanIdx = ascanIdx / ascansPerBscan;
			int rowOffset = (ascanIdx % ascansPerBscan) * signalLength;

			if (recording && bscanIdx < bscansToRecord) {
				for (int s = 0; s < signalLength; ++s) {
					this->backgroundFrameAccumulator[rowOffset + s] += this->spectrum[s].real();
				}
			}
			if (continuous) {
				for (int s = 0; s < signalLength; ++s) {
					float& bg = this->backgroundFrame[rowOffset + s];
					bg = alpha * this->spectrum[s].real() + (1.0f - alpha) * bg;
				}
			}
		}

		if (recording) {
			this->backgroundFrameBscansRecorded += bscansToRecord;
			if (this->backgroundFrameBscansRecorded >= this->backgroundFrameBscansTarget) {
				// Finalize: the recorded frame applies starting with the current buffer
				float normFactor = 1.0f / static_cast<float>(this->backgroundFrameBscansRecorded);
				this->backgroundFrame.resize(samplesPerBscan);
				for (int i = 0; i < samplesPerBscan; ++i) {
					this->backgroundFrame[i] = this->backgroundFrameAccumulator[i] * normFactor;
				}
				this->backgroundFrameRecordingInProgress = false;
				this->smoothedFrameDirty = true;

				//sync recorded profile to configuration
				this->config.setBackgroundFrameProfile(this->backgroundFrame, signalLength, ascansPerBscan);
			}
		}
		if (continuous) {
			this->smoothedFrameDirty = true;
		}
	}

	// Rebuild the cached smoothed frame only when the background or the smoothing settings changed
	if (config.processingParams.backgroundFrame.enabled &&
		config.processingParams.backgroundFrame.smoothSpectra &&
		!this->backgroundFrame.empty() && this->smoothedFrameDirty) {
		cpu_kernels::smoothBackgroundFrame<float>(
			this->backgroundFrame,
			this->smoothedBackgroundFrame,
			config.processingParams.backgroundFrame.smoothingWindowRadius,
			signalLength
		);
		this->smoothedFrameDirty = false;
	}
}


// ============================================
// CpuBackend Implementation
// ============================================

CpuBackend::CpuBackend() : impl(std::make_unique<Impl>()) {
}

CpuBackend::~CpuBackend() {
	this->cleanup();
}

void CpuBackend::initialize(const ProcessorConfiguration& config) {
	this->impl->config = config;
	
	int signalLength = config.dataParams.signalLength;
	
	this->impl->fftShape = {static_cast<size_t>(signalLength)};
	
	// Allocate output buffers
	size_t outputSize = (config.dataParams.samplesPerBuffer() / 2) * sizeof(float);

	if (!this->impl->outputBuffer1.allocateMemory(outputSize) ||
		!this->impl->outputBuffer2.allocateMemory(outputSize)) {
		throw std::runtime_error("Failed to allocate output buffers");
	}

	this->impl->outputBuffer1.setDataType(IOBuffer::DataType::FLOAT32);
	this->impl->outputBuffer2.setDataType(IOBuffer::DataType::FLOAT32);

	// Allocate input buffers
	size_t inputSize = config.dataParams.samplesPerBuffer() * (config.dataParams.getBytesPerSample());
	//size_t inputSize = config.dataParams.samplesPerBuffer * sizeof(float);
	this->impl->hostInputBuffers.resize(this->impl->numInputBuffers);
	
	for (int i = 0; i < this->impl->numInputBuffers; ++i) {
		if (!this->impl->hostInputBuffers[i].allocateMemory(inputSize)) {
			throw std::runtime_error("Failed to allocate input buffer " + std::to_string(i));
		}
		this->impl->hostInputBuffers[i].setDataType(config.dataParams.inputDataType);
		//this->impl->hostInputBuffers[i].setDataType(IOBuffer::DataType::FLOAT32);
		this->impl->freeBuffersQueue.push(&this->impl->hostInputBuffers[i]);
	}
	
	// Initialize curves
	this->impl->resampleCurve.resize(signalLength);
	this->impl->dispersionPhaseComplex.resize(signalLength);
	this->impl->windowCurve.resize(signalLength);

	// Pre-allocate temporary processing buffers
	this->impl->spectrum.resize(signalLength);
	this->impl->linearizedSpectrum.resize(signalLength);
	this->impl->ifftOutput.resize(signalLength);
	this->impl->processedAscan.resize(signalLength / 2);

	// Warm the plan cache before starting the worker. The inverse is unnormalized.
	std::fill(this->impl->linearizedSpectrum.begin(), this->impl->linearizedSpectrum.end(),
		std::complex<float>(0.0f, 0.0f));
	this->impl->computeIFFT();

	//load recorded profiles from configuration
	if (config.hasCustomPostProcessBackgroundProfile()) {
		this->impl->postProcessBackgroundProfile = config.getBackgroundProfile();
	}
	if (config.hasCustomFixedPatternNoiseProfile()) {
		this->impl->recordedFixedPatternNoise = config.getFixedPatternNoiseProfile();
	}
	if (config.hasCustomBackgroundFrameProfile()) {
		this->impl->backgroundFrame = config.getBackgroundFrameProfile();
		this->impl->smoothedFrameDirty = true;
	}

	// Start processing thread
	this->impl->stopProcessing = false;
	this->impl->processingThread = std::thread([this]() {
		this->impl->processingThreadFunc();
	});
}

void CpuBackend::cleanup() {
	// Stop processing thread
	this->impl->stopProcessing = true;
	{
		std::lock_guard<std::mutex> lock(this->impl->workQueueMutex);
	}
	this->impl->workQueueCV.notify_one();
	
	if (this->impl->processingThread.joinable()) {
		this->impl->processingThread.join();
	}
	
	// Release output buffers
	this->impl->outputBuffer1.releaseMemory();
	this->impl->outputBuffer2.releaseMemory();
	
	// Release input buffers
	for (auto& buffer : this->impl->hostInputBuffers) {
		buffer.releaseMemory();
	}
	this->impl->hostInputBuffers.clear();
	
	// Clear queues
	while (!this->impl->freeBuffersQueue.empty()) {
		this->impl->freeBuffersQueue.pop();
	}
	while (!this->impl->workQueue.empty()) {
		this->impl->workQueue.pop();
	}

	// Release vector memory
	std::vector<std::vector<std::complex<float>>>().swap(this->impl->accumulatedIfftOutputs);
	this->impl->accumulatedAscanCount = 0;
	std::vector<std::vector<std::complex<float>>>().swap(this->impl->allIfftOutputs);
	std::vector<uint8_t>().swap(this->impl->processingBuffer);
	std::vector<float>().swap(this->impl->resampleCurve);
	std::vector<std::complex<float>>().swap(this->impl->dispersionPhaseComplex);
	std::vector<float>().swap(this->impl->windowCurve);
	std::vector<float>().swap(this->impl->recordedFixedPatternNoise);
	std::vector<float>().swap(this->impl->postProcessBackgroundProfile);
	std::vector<float>().swap(this->impl->backgroundFrame);
	std::vector<float>().swap(this->impl->smoothedBackgroundFrame);
	std::vector<float>().swap(this->impl->backgroundFrameAccumulator);
	std::vector<float>().swap(this->impl->liveSpectralAverages);
	this->impl->backgroundFrameBscansRecorded = 0;
	this->impl->backgroundFrameRecordingInProgress = false;
	this->impl->smoothedFrameDirty = true;
	std::vector<std::complex<float>>().swap(this->impl->spectrum);
	std::vector<std::complex<float>>().swap(this->impl->linearizedSpectrum);
	std::vector<std::complex<float>>().swap(this->impl->ifftOutput);
	std::vector<float>().swap(this->impl->processedAscan);
}

void CpuBackend::setOutputCallback(std::function<void(const IOBuffer&)> callback) {
	this->impl->callback = callback;
}

// ============================================
// UPDATED: process() - NOW ASYNC!
// ============================================

void CpuBackend::process(IOBuffer& input) {
	{
		std::lock_guard<std::mutex> lock(this->impl->workQueueMutex);
		this->impl->workQueue.push(&input);
	}
	this->impl->workQueueCV.notify_one();
}

void CpuBackend::updateConfig(const ProcessorConfiguration& config) {
	std::lock_guard<std::mutex> lock(this->impl->processingStateMutex);
	// A changed smoothing radius must invalidate the cached smoothed background frame
	if (config.processingParams.backgroundFrame.smoothingWindowRadius !=
		this->impl->config.processingParams.backgroundFrame.smoothingWindowRadius) {
		this->impl->smoothedFrameDirty = true;
	}
	this->impl->config = config;
}

void CpuBackend::updateResamplingCurve(const float* curve, size_t length) {
	if (!curve && length > 0) {
		throw std::runtime_error("Invalid resampling curve pointer");
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
// Buffer management
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

int CpuBackend::getOutputBufferCount() const {
	return 2;  // Fixed ping-pong buffers. todo: make configurable
}

void CpuBackend::releaseOutputBuffer(IOBuffer* buffer) {
	(void)buffer;
	{
		std::lock_guard<std::mutex> lock(this->impl->outputSemaphoreMutex);
		this->impl->availableOutputBuffers++;
	}
	this->impl->outputSemaphoreCV.notify_one();
}

float CpuBackend::cubicHermiteInterpolation(float y0, float y1, float y2, float y3, float t) {
	return cpu_kernels::cubicHermiteInterpolation<float>(y0, y1, y2, y3, t);
}

float CpuBackend::clamp(float value, float low, float high) {
	return cpu_kernels::clamp<float>(value, low, high);
}

void CpuBackend::requestPostProcessBackgroundRecording() {
	this->impl->postProcessBackgroundRecordingRequested = true;
}

void CpuBackend::setPostProcessBackgroundProfile(const float* background, size_t length) {
	if (!background) {
		throw std::invalid_argument("Background curve data is null");
	}
	
	int expectedSize = this->impl->config.dataParams.signalLength / 2;
	if (static_cast<int>(length) != expectedSize) {
		throw std::invalid_argument("Invalid background buffer size. Expected " + 
		                            std::to_string(expectedSize) + " but got " + 
		                            std::to_string(length));
	}
	
	this->impl->postProcessBackgroundProfile.assign(background, background + length);
}

const std::vector<float>& CpuBackend::getPostProcessBackgroundProfile() const {
	return this->impl->postProcessBackgroundProfile;
}

// Fixed-pattern noise management
void CpuBackend::requestFixedPatternNoiseDetermination() {
	this->impl->fixedPatternNoiseDeterminationRequested = true;
}

void CpuBackend::setFixedPatternNoiseProfile(const float* profileInterleaved, size_t complexPairs) {
	if (!profileInterleaved && complexPairs > 0) {
		throw std::invalid_argument("Invalid fixed-pattern noise profile pointer");
	}
	this->impl->recordedFixedPatternNoise.assign(profileInterleaved, profileInterleaved + complexPairs * 2);
}

const std::vector<float>& CpuBackend::getFixedPatternNoiseProfile() const {
	return this->impl->recordedFixedPatternNoise;
}

// Background frame management (line-field OCT)
void CpuBackend::requestBackgroundFrameRecording() {
	std::lock_guard<std::mutex> lock(this->impl->processingStateMutex);
	int samplesPerBscan = this->impl->config.dataParams.signalLength *
	                      this->impl->config.dataParams.ascansPerBscan;
	this->impl->backgroundFrameAccumulator.assign(samplesPerBscan, 0.0f);
	this->impl->backgroundFrameBscansRecorded = 0;
	this->impl->backgroundFrameBscansTarget = this->impl->config.processingParams.backgroundFrame.bscansToAverage;
	this->impl->backgroundFrameRecordingInProgress = true;
}

void CpuBackend::setBackgroundFrameProfile(const float* frame, size_t samplesPerLine, size_t ascansPerBscan) {
	if (!frame || samplesPerLine == 0 || ascansPerBscan == 0) {
		throw std::invalid_argument("Invalid background frame profile pointer");
	}
	if (samplesPerLine != static_cast<size_t>(this->impl->config.dataParams.signalLength) ||
		ascansPerBscan != static_cast<size_t>(this->impl->config.dataParams.ascansPerBscan)) {
		throw std::invalid_argument("Background frame profile dimensions do not match current configuration");
	}

	std::lock_guard<std::mutex> lock(this->impl->processingStateMutex);
	this->impl->backgroundFrame.assign(frame, frame + samplesPerLine * ascansPerBscan);
	this->impl->smoothedFrameDirty = true;
}

std::vector<float> CpuBackend::getBackgroundFrameProfile() const {
	std::lock_guard<std::mutex> lock(this->impl->processingStateMutex);
	return this->impl->backgroundFrame;
}

bool CpuBackend::hasBackgroundFrameProfile() const {
	std::lock_guard<std::mutex> lock(this->impl->processingStateMutex);
	return !this->impl->backgroundFrame.empty();
}

void CpuBackend::resetBackgroundFrame() {
	std::lock_guard<std::mutex> lock(this->impl->processingStateMutex);
	this->impl->backgroundFrame.clear();
	this->impl->smoothedBackgroundFrame.clear();
	this->impl->backgroundFrameAccumulator.clear();
	this->impl->backgroundFrameBscansRecorded = 0;
	this->impl->backgroundFrameBscansTarget = 0;
	this->impl->backgroundFrameRecordingInProgress = false;
	this->impl->smoothedFrameDirty = true;
}


} // namespace ope