#include "../../include/processor.h"
#include "../backends/backend_interface.h"
#include "../backends/cuda/cuda_backend.h"
#include "../backends/cpu/cpu_backend.h"
#include "callback_manager.h" 
#include <stdexcept>
#include <fstream>
#include <cstring>
#include "processor.h"

namespace ope {

// ============================================
// PIMPL Implementation - All logic here!
// ============================================

class Processor::Impl {
public:
	ProcessorConfiguration config;
	std::unique_ptr<ProcessingBackend> backend;
	Backend backendType;
	
	bool initialized = false;
	ProcessorConfiguration::DataParameters lastInitializedDataParams = {};
	
	CallbackManager callbackManager;

	Impl(Backend type) : backendType(type) {
		this->createBackend(type);
	}
	
	~Impl() {
		if (this->backend && this->initialized) {
			this->backend->cleanup();
		}
	}
	
	void createBackend(Backend type) {
		switch (type) {
			case Backend::CUDA:
				this->backend = std::make_unique<CudaBackend>();
				break;
			case Backend::CPU:
				this->backend = std::make_unique<CpuBackend>();
				break;
			default:
				throw std::runtime_error("Unknown backend type");
		}
		this->backendType = type;
	}

	// Helper methods: Get appropriate curve based on flags
	std::vector<float> getResamplingCurve() const {
		if (this->config.resamplingParams.useCustomCurve && 
		    this->config.hasCustomResamplingCurve()) {
			const float* data = this->config.getCustomResamplingCurve();
			size_t size = this->config.getCustomResamplingCurveSize();
			return std::vector<float>(data, data + size);
		}
		const float* data = this->config.getGeneratedResamplingCurve();
		size_t size = this->config.getGeneratedResamplingCurveSize();
		return std::vector<float>(data, data + size);
	}
	
	std::vector<float> getWindowCurve() const {
		if (this->config.windowingParams.useCustomCurve && 
		    this->config.hasCustomWindowCurve()) {
			const float* data = this->config.getCustomWindowCurve();
			size_t size = this->config.getCustomWindowCurveSize();
			return std::vector<float>(data, data + size);
		}
		const float* data = this->config.getGeneratedWindowCurve();
		size_t size = this->config.getGeneratedWindowCurveSize();
		return std::vector<float>(data, data + size);
	}
	
	std::vector<float> getDispersionCurve() const {
	// Get phase values (custom or generated - both are phase after refactoring)
	const float* phaseData;
	size_t phaseSize;
	
	if (this->config.dispersionParams.useCustomCurve && 
	    this->config.hasCustomDispersionCurve()) {
		phaseData = this->config.getCustomDispersionCurve();
		phaseSize = this->config.getCustomDispersionCurveSize();
	} else {
		phaseData = this->config.getGeneratedDispersionCurve();
		phaseSize = this->config.getGeneratedDispersionCurveSize();
	}
	
	// Convert phase values to complex format (cos/sin) for backend
	std::vector<float> complexCurve(phaseSize * 2);
	for (size_t i = 0; i < phaseSize; ++i) {
		float phase = phaseData[i];
		complexCurve[i * 2] = std::cos(phase);
		complexCurve[i * 2 + 1] = std::sin(phase);
	}
	return complexCurve;
}

	// Backend update methods
	void updateBackendResamplingCurve() {
		if (!this->initialized) return;
		std::vector<float> curve = this->getResamplingCurve();
		this->backend->updateResamplingCurve(curve.data(), curve.size());
	}
	
	void updateBackendWindowCurve() {
		if (!this->initialized) return;
		std::vector<float> curve = this->getWindowCurve();
		this->backend->updateWindowCurve(curve.data(), curve.size());
	}
	
	void updateBackendDispersionCurve() {
		if (!this->initialized) return;
		std::vector<float> curve = this->getDispersionCurve();
		this->backend->updateDispersionCurve(curve.data(), curve.size());
	}
	
	void updateAllBackendCurves() {
		if (!this->initialized) return;
		this->updateBackendResamplingCurve();
		this->updateBackendWindowCurve();
		this->updateBackendDispersionCurve();
	}

	void ensureInitialized() {
		if (!this->initialized) {
			this->initialize();
		} else if (this->needsReinit()) {
			this->reinitialize();
		}
	}
	
	void initialize() {
		if (!this->config.validate()) {
			throw std::runtime_error("Invalid processor configuration");
		}

		// (re-)initialize backend
		if (this->initialized) {
			this->backend->cleanup();
		}
		this->backend->initialize(this->config);

		// Setup internal callback that distributes to all consumers
		this->backend->setOutputCallback([this](const IOBuffer& output) {
			this->internalCallback(output);
		});

		this->initialized = true;

		this->updateAllBackendCurves();

		this->lastInitializedDataParams = this->config.dataParams;
	}
	
	void reinitialize() {
		if (!this->config.validate()) {
			throw std::runtime_error("Invalid processor configuration");
		}
		
		this->backend->cleanup();
		this->backend->initialize(this->config);

		// Send curves to backend
		this->updateAllBackendCurves();

		this->lastInitializedDataParams = this->config.dataParams;
	}
	
	void cleanup() {
		if (this->initialized) {
			this->backend->cleanup();
			this->initialized = false;
		}
	}
	
	bool needsReinit() const {
		const auto& current = this->config.dataParams;
		const auto& last = this->lastInitializedDataParams;
		
		return current.signalLength != last.signalLength ||
		       current.samplesPerBuffer != last.samplesPerBuffer ||
		       current.ascansPerBscan != last.ascansPerBscan ||
		       current.bscansPerBuffer != last.bscansPerBuffer;
	}

	void internalCallback(const IOBuffer& output) {
		this->callbackManager.invokeAll(output);
	}
};

// ============================================
// PUBLIC API - Thin wrappers
// ============================================

Processor::Processor(Backend backend)
	: impl(std::make_unique<Impl>(backend))
{
}

Processor::~Processor() = default;

// ============================================
// LIFECYCLE
// ============================================

void Processor::initialize() {
	this->impl->ensureInitialized();
}

void Processor::cleanup() {
	this->impl->cleanup();
}

bool Processor::isInitialized() const {
	return this->impl->initialized;
}

// ============================================
// CONFIGURATION - FILE-BASED
// ============================================

void Processor::loadConfigurationFromFile(const std::string& filepath) {
	if (!this->impl->config.loadFromFile(filepath)) {
		throw std::runtime_error("Failed to load configuration from: " + filepath);
	}
}

void Processor::saveConfigurationToFile(const std::string& filepath) const {
	if (!this->impl->config.saveToFile(filepath)) {
		throw std::runtime_error("Failed to save configuration to: " + filepath);
	}
}

// ============================================
// CONFIGURATION - READ ACCESS
// ============================================

const ProcessorConfiguration& Processor::getConfig() const {
	return this->impl->config;
}

void Processor::setConfig(const ProcessorConfiguration& config) {
	// Check if buffer dimensions changed
	bool dimensionsChanged = 
		this->impl->config.dataParams.signalLength != config.dataParams.signalLength ||
		this->impl->config.dataParams.samplesPerBuffer != config.dataParams.samplesPerBuffer ||
		this->impl->config.dataParams.ascansPerBscan != config.dataParams.ascansPerBscan ||
		this->impl->config.dataParams.bscansPerBuffer != config.dataParams.bscansPerBuffer;
	
	// Copy the entire configuration (including custom curves)
	this->impl->config = config;
	
	// If initialized, handle backend updates
	if (this->impl->initialized) {
		if (dimensionsChanged) {
			// Dimensions changed - must reinitialize backend
			this->impl->reinitialize();
		} else {
			// Dimensions same - just update curves and parameters
			this->impl->updateAllBackendCurves();
		}
	}
	// If not initialized, config is just stored and will be used during initialize()
}

// ============================================
// CONFIGURATION - CONTROLLED WRITE ACCESS
// ============================================

void Processor::setInputParameters(
	int samplesPerRawAscan,
	int ascansPerBscan,
	int bscansPerBuffer,
	DataType type)
{
	int oldSignalLength = this->impl->config.dataParams.signalLength;
	
	this->impl->config.dataParams.signalLength = samplesPerRawAscan;
	this->impl->config.dataParams.ascansPerBscan = ascansPerBscan;
	this->impl->config.dataParams.bscansPerBuffer = bscansPerBuffer;
	this->impl->config.dataParams.samplesPerBuffer = samplesPerRawAscan * ascansPerBscan * bscansPerBuffer;
	this->impl->config.dataParams.inputDataType = type;
	
	// If signalLength changed, re-adjust all custom curves
	if (samplesPerRawAscan != oldSignalLength) {
		this->impl->config.adjustAllCustomCurves();
		
		// Update backend with new curves if initialized
		if (this->impl->initialized) {
			this->impl->updateAllBackendCurves();
		}
	}
}

// ============================================
// BACKEND MANAGEMENT
// ============================================

void Processor::setBackend(Backend backend) {
	if (this->impl->backendType == backend) {
		return;
	}
	
	// Clean up old backend if initialized
	if (this->impl->initialized) {
		this->impl->backend->cleanup();
		this->impl->initialized = false;
	}
	
	// Create new backend
	this->impl->createBackend(backend);
}

Backend Processor::getBackend() const {
	return this->impl->backendType;
}

// ============================================
// PROCESSING
// ============================================
Processor::CallbackId Processor::addOutputCallback(OutputCallback callback) {
	return this->impl->callbackManager.addCallback(callback);
}

bool Processor::removeOutputCallback(CallbackId id) {
	return this->impl->callbackManager.removeCallback(id);
}

void Processor::clearOutputCallbacks() {
	this->impl->callbackManager.clear();
}

size_t Processor::getCallbackCount() const {
	return this->impl->callbackManager.getCallbackCount();
}

void Processor::setOutputCallback(OutputCallback callback) {
	// Legacy method: clear all and add one
	// Provided for backwards compatibility
	// todo: remove this method und update processor.h and all examples, tests, etc
	this->impl->callbackManager.clear();
	this->impl->callbackManager.addCallback(callback);
}

void Processor::process(IOBuffer& input) {
	this->impl->ensureInitialized();
	this->impl->backend->process(input);
}

// ============================================
// BUFFER MANAGEMENT
// ============================================

IOBuffer& Processor::getInputBuffer(int index) {
	//this->impl->ensureInitialized();
	return this->impl->backend->getInputBuffer(index);
}

IOBuffer& Processor::getNextAvailableInputBuffer() {
	//this->impl->ensureInitialized();
	return this->impl->backend->getNextAvailableInputBuffer();
}

int Processor::getNumInputBuffers() const {
	if (!this->impl->initialized) {
		return 0;
	}
	return this->impl->backend->getNumInputBuffers();
}

// ============================================
// HOT-SWAP METHODS
// ============================================

// Resampling - Curve generation (needs backend call)

void Processor::setResamplingCoefficients(const float coefficients[4]) {
	std::copy(coefficients, coefficients + 4, this->impl->config.resamplingParams.coefficients);
	this->impl->config.resamplingParams.useCoefficients = true;
	this->impl->config.resamplingParams.useCustomCurve = false;
	
	this->impl->updateBackendResamplingCurve();
}

void Processor::setCustomResamplingCurve(const float* curve, size_t length) {
	if (!curve || length == 0) {
		throw std::invalid_argument("Invalid custom resampling curve");
	}
	
	// Store in config
	this->impl->config.setCustomResamplingCurve(curve, length);
	
	// Update config flags
	this->impl->config.resamplingParams.useCustomCurve = true;
	this->impl->config.resamplingParams.useCoefficients = false;
	
	// Update backend
	this->impl->updateBackendResamplingCurve();
}

void Processor::useCustomResamplingCurve(bool useCustom) {
	if (useCustom) {
		if (!this->impl->config.hasCustomResamplingCurve()) {
			throw std::runtime_error("No custom resampling curve set. Call setCustomResamplingCurve() first.");
		}
		this->impl->config.resamplingParams.useCustomCurve = true;
		this->impl->config.resamplingParams.useCoefficients = false;
	} else {
		this->impl->config.resamplingParams.useCoefficients = true;
		this->impl->config.resamplingParams.useCustomCurve = false;
	}
	
	// Update backend
	this->impl->updateBackendResamplingCurve();
}

void Processor::enableResampling(bool enable) {
	this->impl->config.resamplingParams.enabled = enable;
}

void Processor::setInterpolationMethod(InterpolationMethod method) {
	this->impl->config.resamplingParams.interpolationMethod = method;
}

// ============================================
// HOT-SWAP METHODS - DISPERSION
// ============================================

void Processor::setDispersionCoefficients(const float coefficients[4], float factor) {
	std::copy(coefficients, coefficients + 4, this->impl->config.dispersionParams.coefficients);
	this->impl->config.dispersionParams.factor = factor;
	this->impl->config.dispersionParams.useCoefficients = true;
	this->impl->config.dispersionParams.useCustomCurve = false;
	
	this->impl->updateBackendDispersionCurve();
}

void Processor::setCustomDispersionCurve(const float* curve, size_t length) {
	if (!curve || length == 0) {
		throw std::invalid_argument("Invalid custom dispersion curve");
	}
	
	// Store in config (phase values)
	this->impl->config.setCustomDispersionCurve(curve, length);
	
	// Update config flags
	this->impl->config.dispersionParams.useCustomCurve = true;
	this->impl->config.dispersionParams.useCoefficients = false;
	
	// Update backend
	this->impl->updateBackendDispersionCurve();
}

void Processor::useCustomDispersionCurve(bool useCustom) {
	if (useCustom) {
		if (!this->impl->config.hasCustomDispersionCurve()) {
			throw std::runtime_error("No custom dispersion curve set. Call setCustomDispersionCurve() first.");
		}
		this->impl->config.dispersionParams.useCustomCurve = true;
		this->impl->config.dispersionParams.useCoefficients = false;
	} else {
		this->impl->config.dispersionParams.useCoefficients = true;
		this->impl->config.dispersionParams.useCustomCurve = false;
	}
	
	// Update backend
	this->impl->updateBackendDispersionCurve();
}

void Processor::enableDispersionCompensation(bool enable) {
	this->impl->config.dispersionParams.enabled = enable;
}

// ============================================
// HOT-SWAP METHODS - WINDOWING
// ============================================

void Processor::setWindowParameters(WindowType type, float centerPosition, float fillFactor) {
	// Update config
	this->impl->config.windowingParams.windowType = type;
	this->impl->config.windowingParams.windowCenterPosition = centerPosition;
	this->impl->config.windowingParams.windowFillFactor = fillFactor;
	this->impl->config.windowingParams.useCustomCurve = false;
	// Don't clear custom curve - keep it for later toggling!
	
	// Update backend
	this->impl->updateBackendWindowCurve();
}

void Processor::setCustomWindowCurve(const float* curve, size_t length) {
	if (!curve || length == 0) {
		throw std::invalid_argument("Invalid custom window curve");
	}
	
	// Store in config
	this->impl->config.setCustomWindowCurve(curve, length);
	
	// Update config flags
	this->impl->config.windowingParams.useCustomCurve = true;
	
	// Update backend
	this->impl->updateBackendWindowCurve();
}

void Processor::useCustomWindowCurve(bool useCustom) {
	if (useCustom) {
		if (!this->impl->config.hasCustomWindowCurve()) {
			throw std::runtime_error("No custom window curve set. Call setCustomWindowCurve() first.");
		}
		this->impl->config.windowingParams.useCustomCurve = true;
	} else {
		this->impl->config.windowingParams.useCustomCurve = false;
	}
	
	// Update backend
	this->impl->updateBackendWindowCurve();
}

void Processor::enableWindowing(bool enable) {
	this->impl->config.windowingParams.enabled = enable;
}

// Post-processing - Simple parameters (backend reads from config)

void Processor::setGrayscaleRange(float min, float max) {
	this->impl->config.postProcessingParams.grayscaleMin = min;
	this->impl->config.postProcessingParams.grayscaleMax = max;
}

void Processor::setSignalMultiplicatorAndAddend(float multiplicator, float addend) {
	this->impl->config.postProcessingParams.multiplicator = multiplicator;
	this->impl->config.postProcessingParams.addend = addend;
}

void Processor::enableLogScaling(bool enable) {
	this->impl->config.postProcessingParams.logScaling = enable;
}

// Background removal - Simple parameters (backend reads from config)

void Processor::enableBackgroundRemoval(bool enable) {
	this->impl->config.backgroundRemovalParams.enabled = enable;
}

void Processor::setBackgroundRemovalWindowSize(int windowSize) {
	this->impl->config.backgroundRemovalParams.rollingAverageWindowSize = windowSize;
}

// Other toggles - Simple flags (backend reads from config)

void Processor::enableBscanFlip(bool enable) {
	this->impl->config.postProcessingParams.bscanFlip = enable;
}

void Processor::enableSinusoidalScanCorrection(bool enable) {
	this->impl->config.postProcessingParams.sinusoidalScanCorrection = enable;
}

void Processor::enableFixedPatternNoiseRemoval(bool enable) {
	this->impl->config.postProcessingParams.fixedPatternNoiseRemoval = enable;
}

void Processor::enablePostProcessBackgroundRemoval(bool enable) {
	this->impl->config.postProcessingParams.backgroundRemoval = enable;
}

// ============================================
// LOW-LEVEL API - Forward to backend
// ============================================

std::vector<float> Processor::convertInput(const void* input, IOBuffer::DataType inputType, int bitDepth, int samples, bool applyBitshift) {
	this->impl->ensureInitialized();
	return this->impl->backend->convertInput(input, inputType, bitDepth, samples, applyBitshift);
}

std::vector<float> Processor::kLinearization(const float* input, const float* resampleCurve, InterpolationMethod method, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->kLinearization(input, resampleCurve, method, lineWidth, samples);
}

std::vector<float> Processor::windowing(const float* input, const float* windowCurve, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->windowing(input, windowCurve, lineWidth, samples);
}

std::vector<float> Processor::dispersionCompensation(const float* input, const float* phaseComplex, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->dispersionCompensation(input, phaseComplex, lineWidth, samples);
}

std::vector<float> Processor::fft(const float* input, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->fft(input, lineWidth, samples);
}

std::vector<float> Processor::ifft(const float* input, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->ifft(input, lineWidth, samples);
}

std::vector<float> Processor::rollingAverageBackgroundRemoval(const float* input, int windowSize, int lineWidth, int numLines) {
	this->impl->ensureInitialized();
	return this->impl->backend->rollingAverageBackgroundRemoval(input, windowSize, lineWidth, numLines);
}

std::vector<float> Processor::kLinearizationAndWindowing(const float* input, const float* resampleCurve, const float* windowCurve, InterpolationMethod method, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->kLinearizationAndWindowing(input, resampleCurve, windowCurve, method, lineWidth, samples);
}

std::vector<float> Processor::kLinearizationAndWindowingAndDispersion(const float* input, const float* resampleCurve, const float* windowCurve, const float* phaseComplex, InterpolationMethod method, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->kLinearizationAndWindowingAndDispersion(input, resampleCurve, windowCurve, phaseComplex, method, lineWidth, samples);
}

std::vector<float> Processor::dispersionCompensationAndWindowing(const float* input, const float* phaseComplex, const float* windowCurve, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->dispersionCompensationAndWindowing(input, phaseComplex, windowCurve, lineWidth, samples);
}

std::vector<float> Processor::getMinimumVarianceMean(const float* input, int width, int height, int segments) {
	this->impl->ensureInitialized();
	return this->impl->backend->getMinimumVarianceMean(input, width, height, segments);
}

std::vector<float> Processor::fixedPatternNoiseRemoval(const float* input, const float* meanALine, int lineWidth, int numLines) {
	this->impl->ensureInitialized();
	return this->impl->backend->fixedPatternNoiseRemoval(input, meanALine, lineWidth, numLines);
}

std::vector<float> Processor::postProcessTruncate(const float* input, bool logScaling, float grayscaleMax, float grayscaleMin, float addend, float multiplicator, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->postProcessTruncate(input, logScaling, grayscaleMax, grayscaleMin, addend, multiplicator, lineWidth, samples);
}

std::vector<float> Processor::bscanFlip(const float* input, int lineWidth, int linesPerBscan, int numBscans) {
	this->impl->ensureInitialized();
	return this->impl->backend->bscanFlip(input, lineWidth, linesPerBscan, numBscans);
}

std::vector<float> Processor::sinusoidalScanCorrection(const float* input, const float* resampleCurve, int lineWidth, int linesPerBscan, int numBscans) {
	this->impl->ensureInitialized();
	return this->impl->backend->sinusoidalScanCorrection(input, resampleCurve, lineWidth, linesPerBscan, numBscans);
}

std::vector<float> Processor::postProcessBackgroundRemoval(const float* input, const float* backgroundLine, float weight, float offset, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->postProcessBackgroundRemoval(input, backgroundLine, weight, offset, lineWidth, samples);
}

} // namespace ope