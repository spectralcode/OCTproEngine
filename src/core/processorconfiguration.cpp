#include "../../include/processorconfiguration.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ope {

// ============================================
// PIMPL Implementation
// ============================================

struct ProcessorConfiguration::Impl {
	// Original custom curves (as provided by user)
	std::vector<float> customResamplingCurveOriginal;
	std::vector<float> customWindowCurveOriginal;
	std::vector<float> customDispersionCurveOriginal;
	
	// Adjusted custom curves (truncated/zero-padded to match signalLength)
	std::vector<float> customResamplingCurve;
	std::vector<float> customWindowCurve;
	std::vector<float> customDispersionCurve;
	
	// Generated curves (regenerated on demand, no caching)
	mutable std::vector<float> generatedResamplingCurve;
	mutable std::vector<float> generatedWindowCurve;
	mutable std::vector<float> generatedDispersionCurve;
	
	Impl() = default;
	
	Impl(const Impl& other)
		: customResamplingCurveOriginal(other.customResamplingCurveOriginal)
		, customWindowCurveOriginal(other.customWindowCurveOriginal)
		, customDispersionCurveOriginal(other.customDispersionCurveOriginal)
		, customResamplingCurve(other.customResamplingCurve)
		, customWindowCurve(other.customWindowCurve)
		, customDispersionCurve(other.customDispersionCurve)
	{}
};

// ============================================
// Constructor and Rule of 5
// ============================================

ProcessorConfiguration::ProcessorConfiguration()
	: impl(new Impl())
{
}

ProcessorConfiguration::~ProcessorConfiguration() {
	delete this->impl;
}

ProcessorConfiguration::ProcessorConfiguration(const ProcessorConfiguration& other)
	: dataParams(other.dataParams)
	, processingParams(other.processingParams)
	, resamplingParams(other.resamplingParams)
	, windowingParams(other.windowingParams)
	, dispersionParams(other.dispersionParams)
	, backgroundRemovalParams(other.backgroundRemovalParams)
	, postProcessingParams(other.postProcessingParams)
	, impl(new Impl(*other.impl))
{
}

ProcessorConfiguration::ProcessorConfiguration(ProcessorConfiguration&& other) noexcept
	: dataParams(other.dataParams)
	, processingParams(other.processingParams)
	, resamplingParams(other.resamplingParams)
	, windowingParams(other.windowingParams)
	, dispersionParams(other.dispersionParams)
	, backgroundRemovalParams(other.backgroundRemovalParams)
	, postProcessingParams(other.postProcessingParams)
	, impl(other.impl)
{
	other.impl = nullptr;
}

ProcessorConfiguration& ProcessorConfiguration::operator=(const ProcessorConfiguration& other) {
	if (this != &other) {
		this->dataParams = other.dataParams;
		this->processingParams = other.processingParams;
		this->resamplingParams = other.resamplingParams;
		this->windowingParams = other.windowingParams;
		this->dispersionParams = other.dispersionParams;
		this->backgroundRemovalParams = other.backgroundRemovalParams;
		this->postProcessingParams = other.postProcessingParams;
		
		delete this->impl;
		this->impl = new Impl(*other.impl);
	}
	return *this;
}

ProcessorConfiguration& ProcessorConfiguration::operator=(ProcessorConfiguration&& other) noexcept {
	if (this != &other) {
		this->dataParams = other.dataParams;
		this->processingParams = other.processingParams;
		this->resamplingParams = other.resamplingParams;
		this->windowingParams = other.windowingParams;
		this->dispersionParams = other.dispersionParams;
		this->backgroundRemovalParams = other.backgroundRemovalParams;
		this->postProcessingParams = other.postProcessingParams;
		
		delete this->impl;
		this->impl = other.impl;
		other.impl = nullptr;
	}
	return *this;
}

// ============================================
// Custom Curve Setters
// ============================================

void ProcessorConfiguration::setCustomResamplingCurve(const float* data, size_t size) {
	if (!data) {
		throw std::invalid_argument("Custom resampling curve data is null");
	}
	if (size == 0) {
		throw std::invalid_argument("Custom resampling curve size is zero");
	}
	
	// Store original curve
	this->impl->customResamplingCurveOriginal.assign(data, data + size);
	
	// Adjust to current signalLength
	this->adjustCustomResamplingCurve();
}

void ProcessorConfiguration::setCustomWindowCurve(const float* data, size_t size) {
	if (!data) {
		throw std::invalid_argument("Custom window curve data is null");
	}
	if (size == 0) {
		throw std::invalid_argument("Custom window curve size is zero");
	}
	
	// Store original curve
	this->impl->customWindowCurveOriginal.assign(data, data + size);
	
	// Adjust to current signalLength
	this->adjustCustomWindowCurve();
}

void ProcessorConfiguration::setCustomDispersionCurve(const float* data, size_t size) {
	if (!data) {
		throw std::invalid_argument("Custom dispersion curve data is null");
	}
	if (size == 0) {
		throw std::invalid_argument("Custom dispersion curve size is zero");
	}
	
	// Store original curve (phase values)
	this->impl->customDispersionCurveOriginal.assign(data, data + size);
	
	// Adjust to current signalLength
	this->adjustCustomDispersionCurve();
}

// ============================================
// Custom Curve Getters
// ============================================

const float* ProcessorConfiguration::getCustomResamplingCurve() const {
	if (this->impl->customResamplingCurve.empty()) {
		return nullptr;
	}
	return this->impl->customResamplingCurve.data();
}

const float* ProcessorConfiguration::getCustomWindowCurve() const {
	if (this->impl->customWindowCurve.empty()) {
		return nullptr;
	}
	return this->impl->customWindowCurve.data();
}

const float* ProcessorConfiguration::getCustomDispersionCurve() const {
	if (this->impl->customDispersionCurve.empty()) {
		return nullptr;
	}
	return this->impl->customDispersionCurve.data();
}

size_t ProcessorConfiguration::getCustomResamplingCurveSize() const {
	return this->impl->customResamplingCurve.size();
}

size_t ProcessorConfiguration::getCustomWindowCurveSize() const {
	return this->impl->customWindowCurve.size();
}

size_t ProcessorConfiguration::getCustomDispersionCurveSize() const {
	// Note: Returns phase curve size (not complex curve size which is 2x)
	return this->impl->customDispersionCurve.size();
}

bool ProcessorConfiguration::hasCustomResamplingCurve() const {
	return !this->impl->customResamplingCurveOriginal.empty();
}

bool ProcessorConfiguration::hasCustomWindowCurve() const {
	return !this->impl->customWindowCurveOriginal.empty();
}

bool ProcessorConfiguration::hasCustomDispersionCurve() const {
	return !this->impl->customDispersionCurveOriginal.empty();
}

// ============================================
// Generated Curve Getters (Cached)
// ============================================

const float* ProcessorConfiguration::getGeneratedResamplingCurve() const {
	// Always regenerate - no caching for simplicity
	this->generateResamplingCurve();
	return this->impl->generatedResamplingCurve.data();
}

const float* ProcessorConfiguration::getGeneratedWindowCurve() const {
	// Always regenerate - no caching for simplicity
	this->generateWindowCurve();
	return this->impl->generatedWindowCurve.data();
}

const float* ProcessorConfiguration::getGeneratedDispersionCurve() const {
	// Always regenerate - no caching for simplicity
	this->generateDispersionCurve();
	return this->impl->generatedDispersionCurve.data();
}

size_t ProcessorConfiguration::getGeneratedResamplingCurveSize() const {
	return this->dataParams.signalLength;
}

size_t ProcessorConfiguration::getGeneratedWindowCurveSize() const {
	return this->dataParams.signalLength;
}

size_t ProcessorConfiguration::getGeneratedDispersionCurveSize() const {
	// Note: Returns phase curve size (same as signalLength)
	return this->dataParams.signalLength;
}

// ============================================
// Curve Adjustment Methods
// ============================================

void ProcessorConfiguration::adjustCustomResamplingCurve() {
	if (this->impl->customResamplingCurveOriginal.empty()) {
		this->impl->customResamplingCurve.clear();
		return;
	}
	
	size_t targetSize = static_cast<size_t>(this->dataParams.signalLength);
	size_t originalSize = this->impl->customResamplingCurveOriginal.size();
	
	this->impl->customResamplingCurve = this->impl->customResamplingCurveOriginal;
	
	if (originalSize < targetSize) {
		// Zero-pad
		this->impl->customResamplingCurve.resize(targetSize, 0.0f);
	} else if (originalSize > targetSize) {
		// Truncate
		this->impl->customResamplingCurve.resize(targetSize);
	}
}

void ProcessorConfiguration::adjustCustomWindowCurve() {
	if (this->impl->customWindowCurveOriginal.empty()) {
		this->impl->customWindowCurve.clear();
		return;
	}
	
	size_t targetSize = static_cast<size_t>(this->dataParams.signalLength);
	size_t originalSize = this->impl->customWindowCurveOriginal.size();
	
	this->impl->customWindowCurve = this->impl->customWindowCurveOriginal;
	
	if (originalSize < targetSize) {
		// Zero-pad
		this->impl->customWindowCurve.resize(targetSize, 0.0f);
	} else if (originalSize > targetSize) {
		// Truncate
		this->impl->customWindowCurve.resize(targetSize);
	}
}

void ProcessorConfiguration::adjustCustomDispersionCurve() {
	if (this->impl->customDispersionCurveOriginal.empty()) {
		this->impl->customDispersionCurve.clear();
		return;
	}
	
	size_t targetSize = static_cast<size_t>(this->dataParams.signalLength);
	size_t originalSize = this->impl->customDispersionCurveOriginal.size();
	
	this->impl->customDispersionCurve = this->impl->customDispersionCurveOriginal;
	
	if (originalSize < targetSize) {
		// Zero-pad
		this->impl->customDispersionCurve.resize(targetSize, 0.0f);
	} else if (originalSize > targetSize) {
		// Truncate
		this->impl->customDispersionCurve.resize(targetSize);
	}
}

void ProcessorConfiguration::adjustAllCustomCurves() {
	this->adjustCustomResamplingCurve();
	this->adjustCustomWindowCurve();
	this->adjustCustomDispersionCurve();
	// No cache invalidation needed - curves regenerate on demand
}

// ============================================
// Curve Generation Methods (Private)
// ============================================

void ProcessorConfiguration::generateResamplingCurve() const {
	size_t size = static_cast<size_t>(this->dataParams.signalLength);
	this->impl->generatedResamplingCurve.resize(size);
	
	// Normalize coefficients
	float coeff0 = this->resamplingParams.coefficients[0];
	float coeff1 = this->resamplingParams.coefficients[1] / (size - 1.0f);
	float coeff2 = this->resamplingParams.coefficients[2] / ((size - 1.0f) * (size - 1.0f));
	float coeff3 = this->resamplingParams.coefficients[3] / ((size - 1.0f) * (size - 1.0f) * (size - 1.0f));
	
	// Clamp to safe range for interpolation methods. this avoids out-of-bounds access during resampling and slightly improves cuda kernel performance (no extra boundry check needed). However with this the first and last few samples will not be resampled (usually this has no real world impact).
	// Lanczos resampling reads 16 samples: from index [n-7] to [n+8]
	// Therefore: n must be >= 7 (so n-7 >= 0) and n <= size-9 (so n+8 < size)
	float minIndex = 7.0f;
	float maxIndex = static_cast<float>(size - 9);
	
	for (size_t i = 0; i < size; ++i) {
		float x = static_cast<float>(i);
		float val = coeff0 + x * (coeff1 + x * (coeff2 + x * coeff3));
		this->impl->generatedResamplingCurve[i] = this->clamp(val, minIndex, maxIndex);
	}
}

void ProcessorConfiguration::generateDispersionCurve() const {
	size_t size = static_cast<size_t>(this->dataParams.signalLength);
	this->impl->generatedDispersionCurve.resize(size);
	
	// Normalize coefficients
	float denom = static_cast<float>(size - 1);
	float normCoeffs[4];
	normCoeffs[0] = this->dispersionParams.coefficients[0];
	normCoeffs[1] = this->dispersionParams.coefficients[1] / denom;
	normCoeffs[2] = this->dispersionParams.coefficients[2] / (denom * denom);
	normCoeffs[3] = this->dispersionParams.coefficients[3] / (denom * denom * denom);
	
	// Compute and store phase values only (user-facing representation)
	for (size_t i = 0; i < size; ++i) {
		float phase = normCoeffs[0] 
			+ i * (normCoeffs[1] 
			+ i * (normCoeffs[2] 
			+ i * normCoeffs[3]));
		
		this->impl->generatedDispersionCurve[i] = phase;
	}
}

void ProcessorConfiguration::generateWindowCurve() const {
	unsigned int size = static_cast<unsigned int>(this->dataParams.signalLength);
	this->impl->generatedWindowCurve.resize(size);
	
	float centerPosition = this->windowingParams.windowCenterPosition;
	float fillFactor = this->windowingParams.windowFillFactor;
	WindowType type = this->windowingParams.windowType;
	
	// Clamp centerPosition to [0, 1]
	if (centerPosition > 1.0f) centerPosition = 1.0f;
	if (centerPosition < 0.0f) centerPosition = 0.0f;
	
	unsigned int width = static_cast<unsigned int>(fillFactor * size);
	unsigned int center = static_cast<unsigned int>(centerPosition * size);
	int minPos = static_cast<int>(center - width / 2);
	
	switch (type) {
		case WindowType::HANN: {
			for (unsigned int i = 0; i < size; i++) {
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					this->impl->generatedWindowCurve[i] = 0.0f;
				} else {
					this->impl->generatedWindowCurve[i] = static_cast<float>(0.5 * (1.0 - cos(2.0 * M_PI * static_cast<double>(xiNorm))));
				}
			}
			break;
		}
		
		case WindowType::GAUSS: {
			for (unsigned int i = 0; i < size; i++) {
				int xi = static_cast<int>(i) - static_cast<int>(center);
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(size) - 1.0f)) / fillFactor;
				this->impl->generatedWindowCurve[i] = expf(-10.0f * (xiNorm * xiNorm));
			}
			break;
		}
		
		case WindowType::SINE: {
			for (unsigned int i = 0; i < size; i++) {
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					this->impl->generatedWindowCurve[i] = 0.0f;
				} else {
					this->impl->generatedWindowCurve[i] = static_cast<float>(sin(M_PI * static_cast<double>(xiNorm)));
				}
			}
			break;
		}
		
		case WindowType::LANCZOS: {
			for (unsigned int i = 0; i < size; i++) {
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					this->impl->generatedWindowCurve[i] = 0.0f;
				} else {
					float argument = 2.0f * xiNorm - 1.0f;
					if (argument == 0.0f) {
						this->impl->generatedWindowCurve[i] = 1.0f;
					} else {
						this->impl->generatedWindowCurve[i] = static_cast<float>(sin(M_PI * static_cast<double>(argument)) / (M_PI * static_cast<double>(argument)));
					}
				}
			}
			break;
		}
		
		case WindowType::RECTANGULAR: {
			for (unsigned int i = 0; i < size; i++) {
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					this->impl->generatedWindowCurve[i] = 0.0f;
				} else {
					this->impl->generatedWindowCurve[i] = 1.0f;
				}
			}
			break;
		}
		
		case WindowType::FLAT_TOP: {
			float a0 = 0.215578948f;
			float a1 = 0.416631580f;
			float a2 = 0.277263158f;
			float a3 = 0.083578947f;
			float a4 = 0.006947368f;
			for (unsigned int i = 0; i < size; i++) {
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					this->impl->generatedWindowCurve[i] = 0.0f;
				} else {
					this->impl->generatedWindowCurve[i] = a0 - a1 * static_cast<float>(cos(2.0 * M_PI * static_cast<double>(xiNorm))) + 
							a2 * static_cast<float>(cos(4.0 * M_PI * static_cast<double>(xiNorm))) - 
							a3 * static_cast<float>(cos(6.0 * M_PI * static_cast<double>(xiNorm))) +
							a4 * static_cast<float>(cos(8.0 * M_PI * static_cast<double>(xiNorm)));
				}
			}
			break;
		}
	}
	
}

// ============================================
// File I/O (TODO: Implement)
// ============================================

bool ProcessorConfiguration::saveToFile(const std::string& filepath) const {
	// TODO: Implement INI-style file format with custom curves
	return false;
}

bool ProcessorConfiguration::loadFromFile(const std::string& filepath) {
	// TODO: Implement INI-style file format with custom curves
	return false;
}

// ============================================
// Validation
// ============================================

bool ProcessorConfiguration::validate() const {
	if (this->dataParams.signalLength <= 0 || 
		this->dataParams.samplesPerBuffer <= 0 ||
		this->dataParams.ascansPerBscan <= 0 ||
		this->dataParams.bscansPerBuffer <= 0) {
		return false;
	}
	return true;
}

} // namespace ope