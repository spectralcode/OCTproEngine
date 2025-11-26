#include "../../include/processor.h"
#include "../backends/backend_interface.h"
#ifdef OPE_CUDA_AVAILABLE
#include "../backends/cuda/cuda_backend.h"
#endif
#ifdef OPE_CPU_AVAILABLE
#include "../backends/cpu/cpu_backend.h"
#endif
#ifdef OPE_OPENCL_AVAILABLE
#include "../backends/opencl/opencl_backend.h"
#endif
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

	CallbackManager outputCallbackManager;
	CallbackManager inputCallbackManager;
	uint64_t nextBufferId = 0;  // Simple counter, not atomic

	// Backend-specific settings (NOT in config)
	int numBuffers = 2;           // currently default for all backends. todo: make configurable per backend?
	std::unique_ptr<BackendConfig> backendConfig;  // Unified backend configuration

	Impl(Backend type) : backendType(type) {
		// Create default configuration for the backend
		this->backendConfig = BackendUtils::createDefaultConfig(type);
		this->createBackend(type);
	}
	
	~Impl() {
		if (this->backend && this->initialized) {
			this->backend->cleanup();
		}
	}
	
	void createBackend(Backend type) {
		// Ensure we have a valid backend configuration
		if (!this->backendConfig || this->backendConfig->getBackendType() != type) {
			this->backendConfig = BackendUtils::createDefaultConfig(type);
		}

		switch (type) {
			case Backend::CUDA:
#ifdef OPE_CUDA_AVAILABLE
				{
					auto cudaBackend = std::make_unique<CudaBackend>();

					// Apply CUDA settings from config
					const auto* cudaConfig = static_cast<const CudaConfig*>(this->backendConfig.get());
					cudaBackend->setDeviceId(cudaConfig->deviceId);
					cudaBackend->setNumInputBuffers(this->numBuffers); //todo: numBuffers should be member of cudaConfig
					//cudaBackend->setEnableZeroCopy(cudaConfig->enableZeroCopy); //todo: add setter

					this->backend = std::move(cudaBackend);
				}
#else
				throw std::runtime_error(
					"CUDA backend not available. "
					"OCTproEngine was compiled without CUDA support. "
					"Use Backend::CPU instead."
				);
#endif
				break;
			case Backend::CPU:
#ifdef OPE_CPU_AVAILABLE
				{
					auto cpuBackend = std::make_unique<CpuBackend>();

					// Apply CPU settings from config
					const auto* cpuConfig = static_cast<const CpuConfig*>(this->backendConfig.get());
					// todo: think if it makes sense to add these settings to backend interface
					// could be used to configure FFTW multi-threading, SIMD optimizations, etc.
					// cpuBackend->setNumThreads(cpuConfig->numThreads);
					// cpuBackend->setEnableSimd(cpuConfig->enableSimd);

					this->backend = std::move(cpuBackend);
				}
#else
				throw std::runtime_error(
					"CPU backend not available. "
					"OCTproEngine was compiled without CPU backend support. "
					"Use Backend::CUDA instead."
				);
#endif
				break;
			case Backend::OPENCL:
#ifdef OPE_OPENCL_AVAILABLE
				{
					auto openclBackend = std::make_unique<OpenClBackend>();

					// Apply OpenCL settings from config
					const auto* openclConfig = static_cast<const OpenCLConfig*>(this->backendConfig.get());
					openclBackend->setPlatformId(openclConfig->platformId);
					openclBackend->setPreferGpu(openclConfig->preferGpu);
					openclBackend->setDeviceId(openclConfig->deviceId);
					openclBackend->setNumInputBuffers(this->numBuffers); //todo: numBuffers should be member of openclConfig

					this->backend = std::move(openclBackend);
				}
#else
				throw std::runtime_error(
					"OpenCL backend not available. "
					"OCTproEngine was compiled without OpenCL support. "
					"Use Backend::CUDA or Backend::CPU instead."
				);
#endif
				break;
			default:
				throw std::runtime_error("Unknown backend type");
		}
		this->backendType = type;
	}

	// Helper methods: Get appropriate curve based on flags
	std::vector<float> getResamplingCurve() const {
		if (this->config.processingParams.resampling.useCustomLut &&
		    this->config.hasCustomResamplingCurve()) {
			return this->config.getResamplingLut();
		}
		return this->config.generateResamplingLut();
	}

	std::vector<float> getWindowCurve() const {
		if (this->config.processingParams.windowing.useCustomFunction &&
		    this->config.hasCustomWindowCurve()) {
			return this->config.getWindowFunction();
		}
		return this->config.generateWindowFunction();
	}

	std::vector<float> getDispersionCurve() const {
		if (this->config.processingParams.dispersion.useCustomPhase &&
		    this->config.hasCustomDispersionCurve()) {
			// Custom curve returns adjusted phase values, need to convert to complex
			std::vector<float> phaseValues = this->config.getDispersionPhase();
			size_t phaseSize = phaseValues.size();
			std::vector<float> complexCurve(phaseSize * 2);
			for (size_t i = 0; i < phaseSize; ++i) {
				float phase = phaseValues[i];
				complexCurve[i * 2] = std::cos(phase);
				complexCurve[i * 2 + 1] = std::sin(phase);
			}
			return complexCurve;
		}
		// Generated curve is already in complex format
		return this->config.generateDispersionPhase();
	}

	// Backend update methods
	void updateBackendResamplingCurve() {
		if (!this->initialized) return;
		std::vector<float> curve = this->getResamplingCurve();
		if (!curve.empty()) {
			this->backend->updateResamplingCurve(curve.data(), curve.size());
		}
	}

	void updateBackendWindowCurve() {
		if (!this->initialized) return;
		std::vector<float> curve = this->getWindowCurve();
		if (!curve.empty()) {
			this->backend->updateWindowCurve(curve.data(), curve.size());
		}
	}
	
	void updateBackendDispersionCurve() {
		if (!this->initialized) return;
		std::vector<float> curve = this->getDispersionCurve();
		if (!curve.empty()) {
			this->backend->updateDispersionCurve(curve.data(), curve.size());
		}
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

		this->nextBufferId = 0;

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
		       current.samplesPerBuffer() != last.samplesPerBuffer() ||
		       current.ascansPerBscan != last.ascansPerBscan ||
		       current.bscansPerBuffer != last.bscansPerBuffer;
	}

	void internalCallback(const IOBuffer& output) {
		this->outputCallbackManager.invokeAll(output);
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
	ProcessorConfiguration loadedConfig;
	if (!loadedConfig.loadFromFile(filepath)) {
		throw std::runtime_error("Failed to load configuration from: " + filepath);
	}
	this->setConfig(loadedConfig);
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
		this->impl->config.dataParams.samplesPerBuffer() != config.dataParams.samplesPerBuffer() ||
		this->impl->config.dataParams.ascansPerBscan != config.dataParams.ascansPerBscan ||
		this->impl->config.dataParams.bscansPerBuffer != config.dataParams.bscansPerBuffer;

	// Copy the entire configuration (including custom curves)
	this->impl->config = config;

	// Automatically adjust all custom curves to match the new dimensions
	// This ensures curves are always the correct size without user intervention
	this->impl->config.adjustAllCustomCurves();

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
	this->impl->config.dataParams.inputDataType = type;
	// samplesPerBuffer and outputSignalLength are computed properties now

	// If signalLength changed, re-adjust all custom curves
	if (samplesPerRawAscan != oldSignalLength) {
		this->impl->config.adjustAllCustomCurves();

		// Update backend with new curves if initialized
		if (this->impl->initialized) {
			this->impl->updateAllBackendCurves();
		}
	}
}

void Processor::setBuffersPerVolume(int buffersPerVolume) {
	this->impl->config.dataParams.buffersPerVolume = buffersPerVolume;
}

int Processor::getBuffersPerVolume() const {
	return this->impl->config.dataParams.buffersPerVolume;
}

// ============================================
// BACKEND MANAGEMENT
// ============================================

void Processor::setBackend(Backend backend) {
	if (this->impl->backendType == backend) {
		return;
	}

	// Remember if old backend was initialized
	bool wasInitialized = this->impl->initialized;

	// Sync backend's recorded profiles to processor's config before cleanup
	if (this->impl->initialized) {
		// Get profiles from backend and update processor's config
		const std::vector<float>& bgProfile = this->impl->backend->getPostProcessBackgroundProfile();
		if (!bgProfile.empty()) {
			this->impl->config.setBackgroundProfile(bgProfile);
		}

		const std::vector<float>& fpnProfile = this->impl->backend->getFixedPatternNoiseProfile();
		if (!fpnProfile.empty()) {
			this->impl->config.setFixedPatternNoiseProfile(fpnProfile);
		}

		// Clean up old backend
		this->impl->backend->cleanup();
		this->impl->initialized = false;
	}

	// Create new backend
	this->impl->createBackend(backend);

	// If old backend was initialized, initialize new backend with config
	// The new backend will load any recorded profiles from config during initialization
	if (wasInitialized) {
		this->impl->backend->initialize(this->impl->config);

		// Setup internal callback that distributes to all consumers
		this->impl->backend->setOutputCallback([this](const IOBuffer& output) {
			this->impl->internalCallback(output);
		});

		this->impl->initialized = true;

		// Send curves to backend
		this->impl->updateAllBackendCurves();
	}
}

Backend Processor::getBackend() const {
	return this->impl->backendType;
}

// ============================================
// PROCESSING
// ============================================
Processor::CallbackId Processor::addOutputCallback(OutputCallback callback) {
	return this->impl->outputCallbackManager.addCallback(callback);
}

bool Processor::removeOutputCallback(CallbackId id) {
	return this->impl->outputCallbackManager.removeCallback(id);
}

void Processor::clearOutputCallbacks() {
	this->impl->outputCallbackManager.clear();
}

size_t Processor::getOutputCallbackCount() const {
	return this->impl->outputCallbackManager.getCallbackCount();
}

// Input callbacks
Processor::CallbackId Processor::addInputCallback(InputCallback callback) {
	return this->impl->inputCallbackManager.addCallback(callback);
}

bool Processor::removeInputCallback(CallbackId id) {
	return this->impl->inputCallbackManager.removeCallback(id);
}

void Processor::clearInputCallbacks() {
	this->impl->inputCallbackManager.clear();
}

size_t Processor::getInputCallbackCount() const {
	return this->impl->inputCallbackManager.getCallbackCount();
}

void Processor::setOutputCallback(OutputCallback callback) {
	// Legacy method: clear all and add one
	// Provided for backwards compatibility
	// todo: remove this method und update processor.h and all examples, tests, etc
	this->impl->outputCallbackManager.clear();
	this->impl->outputCallbackManager.addCallback(callback);
}

void Processor::process(IOBuffer& input) {
	this->impl->ensureInitialized();

	uint64_t bufferId = this->impl->nextBufferId++;
	input.setBufferId(bufferId);

	this->impl->inputCallbackManager.invokeAll(input);

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
	std::copy(coefficients, coefficients + 4, this->impl->config.processingParams.resampling.coefficients);
	this->impl->config.processingParams.resampling.useCustomLut = false;
	
	this->impl->updateBackendResamplingCurve();
}

void Processor::setCustomResamplingCurve(const float* curve, size_t length) {
	if (!curve || length == 0) {
		throw std::invalid_argument("Invalid custom resampling curve");
	}

	// Store in config using new vector API
	this->impl->config.setResamplingLut(std::vector<float>(curve, curve + length));

	// Update config flags
	this->impl->config.processingParams.resampling.useCustomLut = true;

	// Update backend
	this->impl->updateBackendResamplingCurve();
}

void Processor::useCustomResamplingCurve(bool useCustom) {
	if (useCustom) {
		if (!this->impl->config.hasCustomResamplingCurve()) {
			throw std::runtime_error("No custom resampling curve set. Call setCustomResamplingCurve() first.");
		}
		this->impl->config.processingParams.resampling.useCustomLut = true;
	} else {
		this->impl->config.processingParams.resampling.useCustomLut = false;
	}
	
	// Update backend
	this->impl->updateBackendResamplingCurve();
}

void Processor::enableResampling(bool enable) {
	this->impl->config.processingParams.resampling.enabled = enable;
}

void Processor::setInterpolationMethod(InterpolationMethod method) {
	this->impl->config.processingParams.resampling.method = method;
}

// ============================================
// HOT-SWAP METHODS - DISPERSION
// ============================================

void Processor::setDispersionCoefficients(const float coefficients[4], float factor) {
	std::copy(coefficients, coefficients + 4, this->impl->config.processingParams.dispersion.coefficients);
	this->impl->config.processingParams.dispersion.factor = factor;
	this->impl->config.processingParams.dispersion.useCustomPhase = false;
	
	this->impl->updateBackendDispersionCurve();
}

void Processor::setCustomDispersionCurve(const float* curve, size_t length) {
	if (!curve || length == 0) {
		throw std::invalid_argument("Invalid custom dispersion curve");
	}

	// Store in config (phase values) using new vector API
	this->impl->config.setDispersionPhase(std::vector<float>(curve, curve + length));

	// Update config flags
	this->impl->config.processingParams.dispersion.useCustomPhase = true;

	// Update backend
	this->impl->updateBackendDispersionCurve();
}

void Processor::useCustomDispersionCurve(bool useCustom) {
	if (useCustom) {
		if (!this->impl->config.hasCustomDispersionCurve()) {
			throw std::runtime_error("No custom dispersion curve set. Call setCustomDispersionCurve() first.");
		}
		this->impl->config.processingParams.dispersion.useCustomPhase = true;
	} else {
		this->impl->config.processingParams.dispersion.useCustomPhase = false;
	}
	
	// Update backend
	this->impl->updateBackendDispersionCurve();
}

void Processor::enableDispersionCompensation(bool enable) {
	this->impl->config.processingParams.dispersion.enabled = enable;
}

// ============================================
// HOT-SWAP METHODS - WINDOWING
// ============================================

void Processor::setWindowParameters(WindowType type, float centerPosition, float fillFactor) {
	// Update config
	this->impl->config.processingParams.windowing.type = type;
	this->impl->config.processingParams.windowing.centerPosition = centerPosition;
	this->impl->config.processingParams.windowing.fillFactor = fillFactor;
	this->impl->config.processingParams.windowing.useCustomFunction = false;
	// Don't clear custom curve - keep it for later toggling!
	
	// Update backend
	this->impl->updateBackendWindowCurve();
}

void Processor::setCustomWindowCurve(const float* curve, size_t length) {
	if (!curve || length == 0) {
		throw std::invalid_argument("Invalid custom window curve");
	}

	// Store in config using new vector API
	this->impl->config.setWindowFunction(std::vector<float>(curve, curve + length));

	// Update config flags
	this->impl->config.processingParams.windowing.useCustomFunction = true;

	// Update backend
	this->impl->updateBackendWindowCurve();
}

void Processor::useCustomWindowCurve(bool useCustom) {
	if (useCustom) {
		if (!this->impl->config.hasCustomWindowCurve()) {
			throw std::runtime_error("No custom window curve set. Call setCustomWindowCurve() first.");
		}
		this->impl->config.processingParams.windowing.useCustomFunction = true;
	} else {
		this->impl->config.processingParams.windowing.useCustomFunction = false;
	}
	
	// Update backend
	this->impl->updateBackendWindowCurve();
}

void Processor::enableWindowing(bool enable) {
	this->impl->config.processingParams.windowing.enabled = enable;
}

// Post-processing - Simple parameters (backend reads from config)

void Processor::setGrayscaleRange(float min, float max) {
	this->impl->config.processingParams.intensity.rangeMin = min;
	this->impl->config.processingParams.intensity.rangeMax = max;
}

void Processor::setSignalMultiplicatorAndAddend(float multiplicator, float addend) {
	this->impl->config.processingParams.intensity.preScale = multiplicator;
	this->impl->config.processingParams.intensity.postOffset = addend;
}

void Processor::enableLogScaling(bool enable) {
	this->impl->config.processingParams.intensity.logScale = enable;
}

// Background removal - Simple parameters (backend reads from config)

void Processor::enableBackgroundRemoval(bool enable) {
	this->impl->config.processingParams.dcRemoval.enabled = enable;
}

void Processor::setBackgroundRemovalWindowSize(int windowSize) {
	this->impl->config.processingParams.dcRemoval.windowSize = windowSize;
}

// Other toggles - Simple flags (backend reads from config)

void Processor::enableBscanFlip(bool enable) {
	this->impl->config.processingParams.geometry.alternatingBscanFlip = enable;
}

void Processor::enableSinusoidalScanCorrection(bool enable) {
	this->impl->config.processingParams.geometry.sinusoidalCorrection = enable;
}

void Processor::enableFixedPatternNoiseRemoval(bool enable) {
	this->impl->config.processingParams.fixedPatternNoise.enabled = enable;

	// Update backend config if initialized
	if (this->impl->initialized) {
		this->impl->backend->updateConfig(this->impl->config);
	}
}

void Processor::enablePostProcessBackgroundSubtraction(bool enable) {
	this->impl->config.processingParams.background.enabled = enable;

	// Update backend config if initialized
	if (this->impl->initialized) {
		this->impl->backend->updateConfig(this->impl->config);
	}
}


// Post-processing background profile management
void Processor::requestPostProcessBackgroundRecording() {
	this->impl->backend->requestPostProcessBackgroundRecording();
}

void Processor::requestFixedPatternNoiseDetermination() {
	this->impl->backend->requestFixedPatternNoiseDetermination();
}

void Processor::setFixedPatternNoiseBscanCount(int numberOfBscans) {
	if (numberOfBscans < 1) throw std::invalid_argument("numberOfBscans must be >= 1");
	this->impl->config.processingParams.fixedPatternNoise.bscanAverageCount = numberOfBscans;
	if (this->impl->initialized) {
		this->impl->backend->updateConfig(this->impl->config);
	}
}

void Processor::enableContinuousFixedPatternNoiseDetermination(bool enable) {
	this->impl->config.processingParams.fixedPatternNoise.continuous = enable;
	if (this->impl->initialized) {
		this->impl->backend->updateConfig(this->impl->config);
	}
}

void Processor::setFixedPatternNoiseProfile(const float* data, size_t complexPairs) {
	if (!data || complexPairs == 0) throw std::invalid_argument("Invalid fixed pattern noise profile");
	
	// Store in configuration (for metadata persistence) using new vector API
	// Convert from raw pointer to vector (data contains interleaved real/imag, so size is complexPairs * 2)
	this->impl->config.setFixedPatternNoiseProfile(std::vector<float>(data, data + complexPairs * 2));

	// Update backend if initialized
	if (this->impl->initialized) {
		this->impl->backend->setFixedPatternNoiseProfile(data, complexPairs);
	}
}

const float* Processor::getFixedPatternNoiseProfile() const {
	// Get from config (single source of truth for recorded profiles)
	const std::vector<float>& profile = this->impl->config.getFixedPatternNoiseProfile();
	return profile.empty() ? nullptr : profile.data();
}

size_t Processor::getFixedPatternNoiseProfileSize() const {
	// Get from config (single source of truth for recorded profiles)
	// Returns complex pairs (vector size / 2)
	return this->impl->config.getFixedPatternNoiseProfile().size() / 2;
}

bool Processor::hasFixedPatternNoiseProfile() const {
	// Get from config (single source of truth for recorded profiles)
	return this->impl->config.hasCustomFixedPatternNoiseProfile();
}

void Processor::saveFixedPatternNoiseProfileToFile(const std::string& filepath) const {
	// Get from config (single source of truth)
	const std::vector<float>& profileVec = this->impl->config.getFixedPatternNoiseProfile();

	if (profileVec.empty()) {
		throw std::runtime_error("No fixed pattern noise profile to save");
	}

	const float* profile = profileVec.data();
	size_t complexPairs = profileVec.size() / 2;
	std::ofstream file(filepath);
	if (!file.is_open()) throw std::runtime_error("Failed to open file for writing: " + filepath);
	file << "Sample Number;Real;Imag\n";
	for (size_t i = 0; i < complexPairs; ++i) {
		file << i << ";" << profile[i*2] << ";" << profile[i*2+1] << "\n";
	}
	file.close();

	if (!file.good()) {
		throw std::runtime_error("Error writing to file: " + filepath);
	}
}

void Processor::loadFixedPatternNoiseProfileFromFile(const std::string& filepath) {
	std::ifstream file(filepath);
	if (!file.is_open()) throw std::runtime_error("Failed to open file for reading: " + filepath);
	std::string line;
	if (!std::getline(file, line)) throw std::runtime_error("Empty file: " + filepath);
	std::vector<float> profile;
	int lineNumber = 1;
	while (std::getline(file, line)) {
		++lineNumber;
		if (line.empty()) continue;
		size_t p1 = line.find(';');
		if (p1 == std::string::npos) throw std::runtime_error("Invalid format at line " + std::to_string(lineNumber));
		size_t p2 = line.find(';', p1 + 1);
		if (p2 == std::string::npos) throw std::runtime_error("Invalid format at line " + std::to_string(lineNumber));
		std::string realStr = line.substr(p1 + 1, p2 - p1 - 1);
		std::string imagStr = line.substr(p2 + 1);
		try {
			float real = std::stof(realStr);
			float imag = std::stof(imagStr);
			profile.push_back(real);
			profile.push_back(imag);
		} catch (...) {
			throw std::runtime_error("Invalid number at line " + std::to_string(lineNumber));
		}
	}
	file.close();
	if (profile.empty()) throw std::runtime_error("No data found in file: " + filepath);

	// Forward to backend if initialized
	if (!this->impl->initialized) {
		throw std::runtime_error("Processor must be initialized before loading fixed pattern noise profile");
	}
	this->impl->backend->setFixedPatternNoiseProfile(profile.data(), profile.size()/2);
}

void Processor::setPostProcessBackgroundWeight(float weight) {
	this->impl->config.processingParams.background.weight = weight;

	// Update backend config (hot-swap)
	if (this->impl->initialized) {
		this->impl->backend->updateConfig(this->impl->config);
	}
}

void Processor::setPostProcessBackgroundOffset(float offset) {
	this->impl->config.processingParams.background.offset = offset;

	// Update backend config (hot-swap)
	if (this->impl->initialized) {
		this->impl->backend->updateConfig(this->impl->config);
	}
}

const float* Processor::getPostProcessBackgroundProfile() const {
	// Check backend first if initialized (it has the most recent data)
	if (this->impl->initialized) {
		const std::vector<float>& profile = this->impl->backend->getPostProcessBackgroundProfile();
		if (!profile.empty()) {
			return profile.data();
		}
	}
	// Fall back to config
	const std::vector<float>& profile = this->impl->config.getBackgroundProfile();
	return profile.empty() ? nullptr : profile.data();
}

size_t Processor::getPostProcessBackgroundProfileSize() const {
	// Check backend first if initialized (it has the most recent data)
	if (this->impl->initialized) {
		const std::vector<float>& profile = this->impl->backend->getPostProcessBackgroundProfile();
		if (!profile.empty()) {
			return profile.size();
		}
	}
	// Fall back to config
	return this->impl->config.getBackgroundProfile().size();
}

bool Processor::hasPostProcessBackgroundProfile() const {
	// Check backend first if initialized (it has the most recent data)
	if (this->impl->initialized) {
		const std::vector<float>& profile = this->impl->backend->getPostProcessBackgroundProfile();
		if (!profile.empty()) {
			return true;
		}
	}
	// Fall back to config
	return this->impl->config.hasCustomPostProcessBackgroundProfile();
}

void Processor::setPostProcessBackgroundProfile(const float* data, size_t size) {
	if (!data || size == 0) {
		throw std::invalid_argument("Invalid post-process background curve data");
	}

	// Store in configuration (for metadata persistence) using new vector API
	this->impl->config.setBackgroundProfile(std::vector<float>(data, data + size));

	// Update backend if initialized
	if (this->impl->initialized) {
		this->impl->backend->setPostProcessBackgroundProfile(data, size);
	}
}

//todo: use csvhelper here!
void Processor::savePostProcessBackgroundProfileToFile(const std::string& filepath) const {
	const std::vector<float>& curveVec = this->impl->config.getBackgroundProfile();

	if (curveVec.empty()) {
		throw std::runtime_error("No post-process background curve to save");
	}

	const float* curve = curveVec.data();
	size_t size = curveVec.size();
	std::ofstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file for writing: " + filepath);
	}
	file << "Sample Number;Sample Value\n";
	for (size_t i = 0; i < size; ++i) {
		file << i << ";" << curve[i] << "\n";
	}
	
	file.close();
	
	if (!file.good()) {
		throw std::runtime_error("Error writing to file: " + filepath);
	}
}

void Processor::loadPostProcessBackgroundProfileFromFile(const std::string& filepath) {
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file for reading: " + filepath);
	}
	
	std::vector<float> curve;
	std::string line;
	
	if (!std::getline(file, line)) {
		throw std::runtime_error("Empty file: " + filepath);
	}
	
	int lineNumber = 1;
	while (std::getline(file, line)) {
		lineNumber++;
		if (line.empty()) {
			continue;
		}
		
		size_t semicolonPos = line.find(';');
		if (semicolonPos == std::string::npos) {
			throw std::runtime_error("Invalid format at line " + std::to_string(lineNumber) + 
			                        ": missing semicolon");
		}
		
		std::string valueStr = line.substr(semicolonPos + 1);
		
		try {
			float value = std::stof(valueStr);
			curve.push_back(value);
		} catch (const std::exception& e) {
			throw std::runtime_error("Invalid number at line " + std::to_string(lineNumber) + 
			                        ": " + valueStr);
		}
	}
	
	file.close();
	
	if (curve.empty()) {
		throw std::runtime_error("No data found in file: " + filepath);
	}
	
	this->setPostProcessBackgroundProfile(curve.data(), curve.size());
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

std::vector<float> Processor::postProcessBackgroundSubtraction(const float* input, const float* backgroundLine, float weight, float offset, int lineWidth, int samples) {
	this->impl->ensureInitialized();
	return this->impl->backend->postProcessBackgroundSubtraction(input, backgroundLine, weight, offset, lineWidth, samples);
}

// ============================================
// BACKEND-SPECIFIC SETTINGS
// ============================================

void Processor::setNumBuffers(int numBuffers) {
	if (this->impl->initialized) {
		throw std::runtime_error(
			"Cannot change number of buffers after initialization. "
			"Call cleanup() first."
		);
	}
	if (numBuffers < 1) {
		throw std::invalid_argument("Number of buffers must be at least 1");
	}

	this->impl->numBuffers = numBuffers;

	// Apply to CUDA backend if it exists
#ifdef OPE_CUDA_AVAILABLE
	if (this->impl->backendType == Backend::CUDA && this->impl->backend) {
		auto* cudaBackend = static_cast<CudaBackend*>(this->impl->backend.get());
		cudaBackend->setNumInputBuffers(numBuffers);
	}
#endif
}

int Processor::getNumBuffers() const {
	return this->impl->numBuffers;
}

// ============================================
// Unified Backend Configuration API
// ============================================

void Processor::setBackendConfig(const BackendConfig& config) {
	// Check if we need to switch backends
	Backend newBackend = config.getBackendType();
	if (this->impl->backendType != newBackend) {
		// Store new configuration
		this->impl->backendConfig = config.clone();

		// Switch backend (this will preserve all processing configuration)
		this->setBackend(newBackend);
	} else {
		// Same backend, just update configuration
		if (this->impl->initialized) {
			throw std::runtime_error(
				"Cannot change backend configuration after initialization. "
				"Call cleanup() first or use setBackend() to switch backends."
			);
		}

		// Update configuration
		this->impl->backendConfig = config.clone();

		// Recreate backend with new configuration
		this->impl->createBackend(newBackend);
	}
}

std::unique_ptr<BackendConfig> Processor::getBackendConfig() const {
	if (!this->impl->backendConfig) {
		return nullptr;
	}
	return this->impl->backendConfig->clone();
}

//todo: use inihelper to save and load backend config
void Processor::saveBackendConfigToFile(const std::string& filepath) const { 
	std::ofstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file for writing: " + filepath);
	}

	file << "# Backend Configuration File\n";
	file << "# Auto-generated by OCTproEngine\n\n";

	// Save buffer settings
	file << "[Buffer]\n";
	file << "numBuffers=" << this->impl->numBuffers << "\n\n";

	// Save backend configuration
	if (this->impl->backendConfig) {
		std::string configStr = BackendUtils::serializeConfig(*this->impl->backendConfig);
		file << "[Backend]\n";
		file << "config=" << configStr << "\n";

		// Also save backend type for clarity
		switch (this->impl->backendConfig->getBackendType()) {
		case Backend::CUDA:
			file << "type=CUDA\n";
			break;
		case Backend::OPENCL:
			file << "type=OpenCL\n";
			break;
		case Backend::CPU:
			file << "type=CPU\n";
			break;
		}
	}

	file.close();
	if (!file.good()) {
		throw std::runtime_error("Error writing to file: " + filepath);
	}
}

void Processor::loadBackendConfigFromFile(const std::string& filepath) {
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file for reading: " + filepath);
	}

	std::string line;
	std::string currentSection;
	std::string backendConfigStr;

	while (std::getline(file, line)) {
		// Trim whitespace
		line.erase(0, line.find_first_not_of(" \t\r\n"));
		line.erase(line.find_last_not_of(" \t\r\n") + 1);

		// Skip empty lines and comments
		if (line.empty() || line[0] == '#') {
			continue;
		}

		// Section headers
		if (line[0] == '[' && line[line.length() - 1] == ']') {
			currentSection = line.substr(1, line.length() - 2);
			continue;
		}

		// Parse key=value
		size_t pos = line.find('=');
		if (pos == std::string::npos) {
			continue;
		}

		std::string key = line.substr(0, pos);
		std::string value = line.substr(pos + 1);

		// Trim key and value
		key.erase(0, key.find_first_not_of(" \t"));
		key.erase(key.find_last_not_of(" \t") + 1);
		value.erase(0, value.find_first_not_of(" \t"));
		value.erase(value.find_last_not_of(" \t") + 1);

		// Process based on section
		if (currentSection == "Buffer") {
			if (key == "numBuffers") {
				this->setNumBuffers(std::stoi(value));
			}
		} else if (currentSection == "Backend") {
			if (key == "config") {
				backendConfigStr = value;
			}
		}
	}

	file.close();

	// Apply backend configuration if found
	if (!backendConfigStr.empty()) {
		auto config = BackendUtils::parseConfig(backendConfigStr);
		if (config) {
			this->setBackendConfig(*config);
		} else {
			throw std::runtime_error("Failed to parse backend configuration from file");
		}
	}
}

} // namespace ope