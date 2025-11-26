#include "../../include/processorconfiguration.h"
#include "../utils/inihelper.h"
#include "../utils/csvhelper.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ope {

// ============================================
// PIMPL Implementation
// ============================================

struct ProcessorConfiguration::Impl {
	// Original custom curves (as provided by user)
	std::vector<float> resamplingLutOriginal;
	std::vector<float> windowFunctionOriginal;
	std::vector<float> dispersionPhaseOriginal;
	std::vector<float> backgroundProfileOriginal;
	std::vector<float> fixedPatternNoiseProfileOriginal;

	// Adjusted custom curves (truncated/zero-padded to match signalLength)
	std::vector<float> resamplingLut;
	std::vector<float> windowFunction;
	std::vector<float> dispersionPhase;
	std::vector<float> backgroundProfile;
	std::vector<float> fixedPatternNoiseProfile;

	Impl() = default;

	Impl(const Impl& other)
		: resamplingLutOriginal(other.resamplingLutOriginal)
		, windowFunctionOriginal(other.windowFunctionOriginal)
		, dispersionPhaseOriginal(other.dispersionPhaseOriginal)
		, backgroundProfileOriginal(other.backgroundProfileOriginal)
		, fixedPatternNoiseProfileOriginal(other.fixedPatternNoiseProfileOriginal)
		, resamplingLut(other.resamplingLut)
		, windowFunction(other.windowFunction)
		, dispersionPhase(other.dispersionPhase)
		, backgroundProfile(other.backgroundProfile)
		, fixedPatternNoiseProfile(other.fixedPatternNoiseProfile)
	{}

	void adjustAllCurves(int signalLength) {
		adjustCurve(resamplingLutOriginal, resamplingLut, signalLength);
		adjustCurve(windowFunctionOriginal, windowFunction, signalLength);
		adjustCurve(dispersionPhaseOriginal, dispersionPhase, signalLength);
		adjustCurve(backgroundProfileOriginal, backgroundProfile, signalLength / 2);
		// Fixed pattern noise is complex pairs (interleaved real/imag)
		adjustCurve(fixedPatternNoiseProfileOriginal, fixedPatternNoiseProfile, signalLength);
	}

private:
	void adjustCurve(const std::vector<float>& original, std::vector<float>& adjusted, size_t targetSize) {
		if (original.empty()) {
			adjusted.clear();
			return;
		}

		adjusted = original;
		if (adjusted.size() < targetSize) {
			adjusted.resize(targetSize, 0.0f);  // Zero-pad
		} else if (adjusted.size() > targetSize) {
			adjusted.resize(targetSize);  // Truncate
		}
	}
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
	, impl(new Impl(*other.impl))
{
}

ProcessorConfiguration::ProcessorConfiguration(ProcessorConfiguration&& other) noexcept
	: dataParams(std::move(other.dataParams))
	, processingParams(std::move(other.processingParams))
	, impl(other.impl)
{
	other.impl = nullptr;
}

ProcessorConfiguration& ProcessorConfiguration::operator=(const ProcessorConfiguration& other) {
	if (this != &other) {
		this->dataParams = other.dataParams;
		this->processingParams = other.processingParams;

		delete this->impl;
		this->impl = new Impl(*other.impl);
	}
	return *this;
}

ProcessorConfiguration& ProcessorConfiguration::operator=(ProcessorConfiguration&& other) noexcept {
	if (this != &other) {
		this->dataParams = std::move(other.dataParams);
		this->processingParams = std::move(other.processingParams);

		delete this->impl;
		this->impl = other.impl;
		other.impl = nullptr;
	}
	return *this;
}

// ============================================
// New Custom Data Management (std::vector API)
// ============================================

void ProcessorConfiguration::setResamplingLut(const std::vector<float>& data) {
	this->impl->resamplingLutOriginal = data;
	this->impl->adjustAllCurves(this->dataParams.signalLength);
	this->processingParams.resampling.useCustomLut = !data.empty();
}

void ProcessorConfiguration::setWindowFunction(const std::vector<float>& data) {
	this->impl->windowFunctionOriginal = data;
	this->impl->adjustAllCurves(this->dataParams.signalLength);
	this->processingParams.windowing.useCustomFunction = !data.empty();
}

void ProcessorConfiguration::setDispersionPhase(const std::vector<float>& data) {
	this->impl->dispersionPhaseOriginal = data;
	this->impl->adjustAllCurves(this->dataParams.signalLength);
	this->processingParams.dispersion.useCustomPhase = !data.empty();
}

void ProcessorConfiguration::setBackgroundProfile(const std::vector<float>& data) {
	this->impl->backgroundProfileOriginal = data;
	this->impl->adjustAllCurves(this->dataParams.signalLength);
	this->processingParams.background.useCustomProfile = !data.empty();
}

void ProcessorConfiguration::setFixedPatternNoiseProfile(const std::vector<float>& complexPairs) {
	this->impl->fixedPatternNoiseProfileOriginal = complexPairs;
	this->impl->adjustAllCurves(this->dataParams.signalLength);
	this->processingParams.fixedPatternNoise.useCustomProfile = !complexPairs.empty();
}

std::vector<float> ProcessorConfiguration::getResamplingLut() const {
	return this->impl->resamplingLut;
}

std::vector<float> ProcessorConfiguration::getWindowFunction() const {
	return this->impl->windowFunction;
}

std::vector<float> ProcessorConfiguration::getDispersionPhase() const {
	return this->impl->dispersionPhase;
}

std::vector<float> ProcessorConfiguration::getBackgroundProfile() const {
	return this->impl->backgroundProfile;
}

std::vector<float> ProcessorConfiguration::getFixedPatternNoiseProfile() const {
	return this->impl->fixedPatternNoiseProfile;
}

// ============================================
// Generate Curves
// ============================================

std::vector<float> ProcessorConfiguration::generateResamplingLut() const {
	size_t size = static_cast<size_t>(this->dataParams.signalLength);
	std::vector<float> curve(size);

	// Normalize coefficients
	float coeff0 = this->processingParams.resampling.coefficients[0];
	float coeff1 = this->processingParams.resampling.coefficients[1] / (size - 1.0f);
	float coeff2 = this->processingParams.resampling.coefficients[2] / ((size - 1.0f) * (size - 1.0f));
	float coeff3 = this->processingParams.resampling.coefficients[3] / ((size - 1.0f) * (size - 1.0f) * (size - 1.0f));

	// Clamp to safe range for interpolation methods. this avoids out-of-bounds access during resampling and slightly improves cuda kernel performance (no extra boundry check needed). However with this the first and last few samples will not be resampled (usually this has no real world impact).
	// Lanczos resampling reads 16 samples: from index [n-7] to [n+8]
	// Therefore: n must be >= 7 (so n-7 >= 0) and n <= size-9 (so n+8 < size)
	float minIndex = 7.0f;
	float maxIndex = static_cast<float>(size - 9);

	for (size_t i = 0; i < size; ++i) {
		float t = static_cast<float>(i);
		float tt = t * t;
		float ttt = tt * t;
		float index = coeff0 + coeff1 * t + coeff2 * tt + coeff3 * ttt;

		// Clamp to safe range
		if (index < minIndex) index = minIndex;
		if (index > maxIndex) index = maxIndex;

		curve[i] = index;
	}

	return curve;
}

std::vector<float> ProcessorConfiguration::generateWindowFunction() const {
	size_t size = static_cast<size_t>(this->dataParams.signalLength);
	std::vector<float> curve(size, 0.0f);

	float fillFactor = this->processingParams.windowing.fillFactor;
	float centerPosition = this->processingParams.windowing.centerPosition;
	unsigned int width = static_cast<unsigned int>(fillFactor * size);
	unsigned int center = static_cast<unsigned int>(centerPosition * size);
	int minPos = static_cast<int>(center) - static_cast<int>(width / 2);
	int maxPos = minPos + static_cast<int>(width);

	switch (this->processingParams.windowing.type) {
		case WindowType::HANN: {
			for (unsigned int i = 0; i < size; i++) {
				if (static_cast<int>(i) < minPos || static_cast<int>(i) > maxPos) {
					curve[i] = 0.0f;
					continue;
				}
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					curve[i] = 0.0f;
				} else {
					curve[i] = static_cast<float>(0.5 * (1.0 - cos(2.0 * M_PI * static_cast<double>(xiNorm))));
				}
			}
			break;
		}

		case WindowType::GAUSS: {
			for (unsigned int i = 0; i < size; i++) {
				int xi = static_cast<int>(i) - static_cast<int>(center);
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(size) - 1.0f)) / fillFactor;
				curve[i] = expf(-10.0f * (xiNorm * xiNorm));
			}
			break;
		}

		case WindowType::SINE: {
			for (unsigned int i = 0; i < size; i++) {
				if (static_cast<int>(i) < minPos || static_cast<int>(i) > maxPos) {
					curve[i] = 0.0f;
					continue;
				}
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					curve[i] = 0.0f;
				} else {
					curve[i] = static_cast<float>(sin(M_PI * static_cast<double>(xiNorm)));
				}
			}
			break;
		}

		case WindowType::LANCZOS: {
			for (unsigned int i = 0; i < size; i++) {
				if (static_cast<int>(i) < minPos || static_cast<int>(i) > maxPos) {
					curve[i] = 0.0f;
					continue;
				}
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					curve[i] = 0.0f;
				} else {
					float argument = 2.0f * xiNorm - 1.0f;
					if (argument == 0.0f) {
						curve[i] = 1.0f;
					} else {
						curve[i] = static_cast<float>(sin(M_PI * static_cast<double>(argument)) / (M_PI * static_cast<double>(argument)));
					}
				}
			}
			break;
		}

		case WindowType::RECTANGULAR: {
			for (unsigned int i = 0; i < size; i++) {
				if (static_cast<int>(i) < minPos || static_cast<int>(i) > maxPos) {
					curve[i] = 0.0f;
				} else {
					curve[i] = 1.0f;
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
				if (static_cast<int>(i) < minPos || static_cast<int>(i) > maxPos) {
					curve[i] = 0.0f;
					continue;
				}
				int xi = static_cast<int>(i) - minPos;
				float xiNorm = (static_cast<float>(xi) / (static_cast<float>(width) - 1.0f));
				if (xiNorm > 0.999f || xiNorm < 0.0001f) {
					curve[i] = 0.0f;
				} else {
					curve[i] = a0 - a1 * static_cast<float>(cos(2.0 * M_PI * static_cast<double>(xiNorm))) +
							a2 * static_cast<float>(cos(4.0 * M_PI * static_cast<double>(xiNorm))) -
							a3 * static_cast<float>(cos(6.0 * M_PI * static_cast<double>(xiNorm))) +
							a4 * static_cast<float>(cos(8.0 * M_PI * static_cast<double>(xiNorm)));
				}
			}
			break;
		}
	}

	return curve;
}

std::vector<float> ProcessorConfiguration::generateDispersionPhase() const {
	size_t size = static_cast<size_t>(this->dataParams.signalLength);
	std::vector<float> phaseValues(size);

	//float factor = this->processingParams.dispersion.factor; //todo: remove from processingParams
	float normalizationFactor = static_cast<float>(size - 1);
	float d0 = this->processingParams.dispersion.coefficients[0];
	float d1 = this->processingParams.dispersion.coefficients[1] / normalizationFactor;
	float d2 = this->processingParams.dispersion.coefficients[2] / (normalizationFactor * normalizationFactor);
	float d3 = this->processingParams.dispersion.coefficients[3] / (normalizationFactor * normalizationFactor * normalizationFactor);
	for (size_t i = 0; i < size; ++i) {
		float k = static_cast<float>(i);
		float kk = k * k;
		float kkk = kk * k;
		phaseValues[i] = d0 + d1 * k + d2 * kk + d3 * kkk;
	}

	// Convert phase to complex (for backward compatibility with backends)
	// Backends expect interleaved real/imag values
	std::vector<float> complexValues(size * 2);
	for (size_t i = 0; i < size; ++i) {
		float phase = phaseValues[i];
		complexValues[i * 2] = cosf(phase);      // Real
		complexValues[i * 2 + 1] = sinf(phase);  // Imaginary
	}

	return complexValues;
}

// ============================================
// Clear methods
// ============================================

void ProcessorConfiguration::clearResamplingLut() {
	this->impl->resamplingLutOriginal.clear();
	this->impl->resamplingLut.clear();
	this->processingParams.resampling.useCustomLut = false;
}

void ProcessorConfiguration::clearWindowFunction() {
	this->impl->windowFunctionOriginal.clear();
	this->impl->windowFunction.clear();
	this->processingParams.windowing.useCustomFunction = false;
}

void ProcessorConfiguration::clearDispersionPhase() {
	this->impl->dispersionPhaseOriginal.clear();
	this->impl->dispersionPhase.clear();
	this->processingParams.dispersion.useCustomPhase = false;
}

void ProcessorConfiguration::clearBackgroundProfile() {
	this->impl->backgroundProfileOriginal.clear();
	this->impl->backgroundProfile.clear();
	this->processingParams.background.useCustomProfile = false;
}

void ProcessorConfiguration::clearFixedPatternNoiseProfile() {
	this->impl->fixedPatternNoiseProfileOriginal.clear();
	this->impl->fixedPatternNoiseProfile.clear();
	this->processingParams.fixedPatternNoise.useCustomProfile = false;
}

// ============================================
// Utility Methods
// ============================================

bool ProcessorConfiguration::hasCustomResamplingCurve() const {
	return !this->impl->resamplingLutOriginal.empty();
}

bool ProcessorConfiguration::hasCustomWindowCurve() const {
	return !this->impl->windowFunctionOriginal.empty();
}

bool ProcessorConfiguration::hasCustomDispersionCurve() const {
	return !this->impl->dispersionPhaseOriginal.empty();
}

bool ProcessorConfiguration::hasCustomPostProcessBackgroundProfile() const {
	return !this->impl->backgroundProfileOriginal.empty();
}

bool ProcessorConfiguration::hasCustomFixedPatternNoiseProfile() const {
	return !this->impl->fixedPatternNoiseProfileOriginal.empty();
}

void ProcessorConfiguration::adjustAllCustomCurves() {
	this->impl->adjustAllCurves(this->dataParams.signalLength);
}

// ============================================
// File I/O - ioFields bidirectional helper
// ============================================

void ProcessorConfiguration::ioFields(std::map<std::string, std::string>& m, bool saving, SaveMode saveMode, LoadMode loadMode) {
	// Data parameters
	IniHelper::field(m, "data.signal_length", this->dataParams.signalLength, saving);
	IniHelper::field(m, "data.ascans_per_bscan", this->dataParams.ascansPerBscan, saving);
	IniHelper::field(m, "data.bscans_per_buffer", this->dataParams.bscansPerBuffer, saving);
	IniHelper::field(m, "data.buffers_per_volume", this->dataParams.buffersPerVolume, saving);
	IniHelper::fieldEnum(m, "data.input_type", this->dataParams.inputDataType, saving);
	IniHelper::fieldEnum(m, "data.output_type", this->dataParams.outputDataType, saving);

	// Processing parameters - Input
	IniHelper::field(m, "input.bitshift", this->processingParams.input.bitshift, saving);

	// DC Removal
	IniHelper::field(m, "dc_removal.enabled", this->processingParams.dcRemoval.enabled, saving);
	IniHelper::field(m, "dc_removal.window_size", this->processingParams.dcRemoval.windowSize, saving);

	// Resampling
	IniHelper::field(m, "resampling.enabled", this->processingParams.resampling.enabled, saving);
	IniHelper::fieldEnum(m, "resampling.method", this->processingParams.resampling.method, saving);
	IniHelper::field(m, "resampling.c0", this->processingParams.resampling.coefficients[0], saving);
	IniHelper::field(m, "resampling.c1", this->processingParams.resampling.coefficients[1], saving);
	IniHelper::field(m, "resampling.c2", this->processingParams.resampling.coefficients[2], saving);
	IniHelper::field(m, "resampling.c3", this->processingParams.resampling.coefficients[3], saving);
	IniHelper::field(m, "resampling.use_custom_lut", this->processingParams.resampling.useCustomLut, saving);

	// Windowing
	IniHelper::field(m, "windowing.enabled", this->processingParams.windowing.enabled, saving);
	IniHelper::fieldEnum(m, "windowing.type", this->processingParams.windowing.type, saving);
	IniHelper::field(m, "windowing.center_position", this->processingParams.windowing.centerPosition, saving);
	IniHelper::field(m, "windowing.fill_factor", this->processingParams.windowing.fillFactor, saving);
	IniHelper::field(m, "windowing.use_custom_function", this->processingParams.windowing.useCustomFunction, saving);

	// Dispersion
	IniHelper::field(m, "dispersion.enabled", this->processingParams.dispersion.enabled, saving);
	IniHelper::field(m, "dispersion.c0", this->processingParams.dispersion.coefficients[0], saving);
	IniHelper::field(m, "dispersion.c1", this->processingParams.dispersion.coefficients[1], saving);
	IniHelper::field(m, "dispersion.c2", this->processingParams.dispersion.coefficients[2], saving);
	IniHelper::field(m, "dispersion.c3", this->processingParams.dispersion.coefficients[3], saving);
	IniHelper::field(m, "dispersion.use_custom_phase", this->processingParams.dispersion.useCustomPhase, saving);
	// Fixed Pattern Noise
	IniHelper::field(m, "fixed_pattern_noise.enabled", this->processingParams.fixedPatternNoise.enabled, saving);
	IniHelper::field(m, "fixed_pattern_noise.bscan_average_count", this->processingParams.fixedPatternNoise.bscanAverageCount, saving);
	IniHelper::field(m, "fixed_pattern_noise.continuous", this->processingParams.fixedPatternNoise.continuous, saving);
	IniHelper::field(m, "fixed_pattern_noise.use_custom_profile", this->processingParams.fixedPatternNoise.useCustomProfile, saving);

	// Background
	IniHelper::field(m, "background.enabled", this->processingParams.background.enabled, saving);
	IniHelper::field(m, "background.weight", this->processingParams.background.weight, saving);
	IniHelper::field(m, "background.offset", this->processingParams.background.offset, saving);
	IniHelper::field(m, "background.use_custom_profile", this->processingParams.background.useCustomProfile, saving);

	// Intensity
	IniHelper::field(m, "intensity.log_scale", this->processingParams.intensity.logScale, saving);
	IniHelper::field(m, "intensity.range_min", this->processingParams.intensity.rangeMin, saving);
	IniHelper::field(m, "intensity.range_max", this->processingParams.intensity.rangeMax, saving);
	IniHelper::field(m, "intensity.pre_scale", this->processingParams.intensity.preScale, saving);
	IniHelper::field(m, "intensity.post_offset", this->processingParams.intensity.postOffset, saving);

	// Geometry
	IniHelper::field(m, "geometry.alternating_bscan_flip", this->processingParams.geometry.alternatingBscanFlip, saving);
	IniHelper::field(m, "geometry.sinusoidal_correction", this->processingParams.geometry.sinusoidalCorrection, saving);
	// Custom data vectors
	if (saving) {
		// When saving: only if mode is COMPLETE
		if (saveMode == SaveMode::COMPLETE) {
			if (!this->impl->resamplingLutOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.resampling_lut", this->impl->resamplingLutOriginal, true);
			}
			if (!this->impl->windowFunctionOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.window_function", this->impl->windowFunctionOriginal, true);
			}
			if (!this->impl->dispersionPhaseOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.dispersion_phase", this->impl->dispersionPhaseOriginal, true);
			}
			if (!this->impl->backgroundProfileOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.background_profile", this->impl->backgroundProfileOriginal, true);
			}
			if (!this->impl->fixedPatternNoiseProfileOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.fixed_pattern_noise", this->impl->fixedPatternNoiseProfileOriginal, true);
			}
		}
	} else {
		// When loading: depends on mode
		if (loadMode == LoadMode::OVERWRITE_ALL) {
			IniHelper::fieldVector(m, "~custom_data.resampling_lut", this->impl->resamplingLutOriginal, false);
			IniHelper::fieldVector(m, "~custom_data.window_function", this->impl->windowFunctionOriginal, false);
			IniHelper::fieldVector(m, "~custom_data.dispersion_phase", this->impl->dispersionPhaseOriginal, false);
			IniHelper::fieldVector(m, "~custom_data.background_profile", this->impl->backgroundProfileOriginal, false);
			IniHelper::fieldVector(m, "~custom_data.fixed_pattern_noise", this->impl->fixedPatternNoiseProfileOriginal, false);
		} else if (loadMode == LoadMode::MERGE_IF_MISSING) {
			if (this->impl->resamplingLutOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.resampling_lut", this->impl->resamplingLutOriginal, false);
			}
			if (this->impl->windowFunctionOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.window_function", this->impl->windowFunctionOriginal, false);
			}
			if (this->impl->dispersionPhaseOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.dispersion_phase", this->impl->dispersionPhaseOriginal, false);
			}
			if (this->impl->backgroundProfileOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.background_profile", this->impl->backgroundProfileOriginal, false);
			}
			if (this->impl->fixedPatternNoiseProfileOriginal.empty()) {
				IniHelper::fieldVector(m, "~custom_data.fixed_pattern_noise", this->impl->fixedPatternNoiseProfileOriginal, false);
			}
		}
	}
}

bool ProcessorConfiguration::saveToFile(const std::string& filepath, SaveMode mode) const {
	IniHelper::IniMap m;
	const_cast<ProcessorConfiguration*>(this)->ioFields(m, true, mode, LoadMode::OVERWRITE_ALL);
	return IniHelper::saveToFile(filepath, m);
}

bool ProcessorConfiguration::loadFromFile(const std::string& filepath, LoadMode mode) {
	IniHelper::IniMap m;
	if (!IniHelper::loadFromFile(filepath, m)) {
		return false;
	}
	this->ioFields(m, false, SaveMode::COMPLETE, mode);
	this->adjustAllCustomCurves();
	return true;
}

// ============================================
// CSV export/import
// ============================================

bool ProcessorConfiguration::saveResamplingLutToFile(const std::string& filepath) const {
	return CSVHelper::save(filepath, this->impl->resamplingLutOriginal, "Resampling LUT");
}

bool ProcessorConfiguration::loadResamplingLutFromFile(const std::string& filepath) {
	auto data = CSVHelper::load(filepath);
	if (!data.empty()) {
		this->setResamplingLut(data);
		return true;
	}
	return false;
}

bool ProcessorConfiguration::saveWindowFunctionToFile(const std::string& filepath) const {
	return CSVHelper::save(filepath, this->impl->windowFunctionOriginal, "Window Function");
}

bool ProcessorConfiguration::loadWindowFunctionFromFile(const std::string& filepath) {
	auto data = CSVHelper::load(filepath);
	if (!data.empty()) {
		this->setWindowFunction(data);
		return true;
	}
	return false;
}

bool ProcessorConfiguration::saveDispersionPhaseToFile(const std::string& filepath) const {
	return CSVHelper::save(filepath, this->impl->dispersionPhaseOriginal, "Dispersion Phase");
}

bool ProcessorConfiguration::loadDispersionPhaseFromFile(const std::string& filepath) {
	auto data = CSVHelper::load(filepath);
	if (!data.empty()) {
		this->setDispersionPhase(data);
		return true;
	}
	return false;
}

bool ProcessorConfiguration::saveBackgroundProfileToFile(const std::string& filepath) const {
	return CSVHelper::save(filepath, this->impl->backgroundProfileOriginal, "Background Profile");
}

bool ProcessorConfiguration::loadBackgroundProfileFromFile(const std::string& filepath) {
	auto data = CSVHelper::load(filepath);
	if (!data.empty()) {
		this->setBackgroundProfile(data);
		return true;
	}
	return false;
}

bool ProcessorConfiguration::saveFixedPatternNoiseProfileToFile(const std::string& filepath) const {
	return CSVHelper::saveComplex(filepath, this->impl->fixedPatternNoiseProfileOriginal,
								  "Fixed Pattern Noise Profile");
}

bool ProcessorConfiguration::loadFixedPatternNoiseProfileFromFile(const std::string& filepath) {
	auto data = CSVHelper::load(filepath);
	if (!data.empty()) {
		this->setFixedPatternNoiseProfile(data);
		return true;
	}
	return false;
}

// ============================================
// Validation
// ============================================

bool ProcessorConfiguration::validate() const {
	if (this->dataParams.signalLength <= 0 ||
		this->dataParams.ascansPerBscan <= 0 ||
		this->dataParams.bscansPerBuffer <= 0) {
		return false;
	}
	return true;
}

} // namespace ope