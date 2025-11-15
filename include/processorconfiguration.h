#ifndef OPE_PROCESSORCONFIGURATION_H
#define OPE_PROCESSORCONFIGURATION_H

#include <string>
#include "types.h"
#include "export.h"


namespace ope {

enum class Backend {
	CUDA,
	CPU
};

enum class InterpolationMethod {
	LINEAR = 0,
	CUBIC = 1,
	LANCZOS = 2
};

enum class WindowType {
	HANN = 0,
	GAUSS = 1,
	SINE = 2,
	LANCZOS = 3,
	RECTANGULAR = 4,
	FLAT_TOP = 5
};

class OPE_API ProcessorConfiguration {
public:
	struct OPE_API DataParameters {
		int signalLength;
		int samplesPerBuffer;
		int ascansPerBscan;
		int bscansPerBuffer;
		DataType inputDataType;
		bool bitshift;
		
		int getBitDepth() const {
			return getDataTypeBitDepth(this->inputDataType);
		}
		
		int getBytesPerSample() const {
			return getDataTypeByteSize(this->inputDataType);
		}

		DataParameters()
			: signalLength(1024)
			, samplesPerBuffer(1024 * 512)
			, ascansPerBscan(512)
			, bscansPerBuffer(1)
			, inputDataType(DataType::UINT16)
			, bitshift(false)
		{}
	};
	
	struct OPE_API ProcessingParameters { //todo: rename to CudaParameters and rethink if this struct is really needed here
		int nStreams;
		int nBuffers;
		int gridSize;
		int blockSize;
		
		ProcessingParameters()
			: nStreams(8)
			, nBuffers(4)
			, gridSize(4096) //gridSize = samplesPerBuffer / blockSize;
			, blockSize(128)
		{}
	};
	
	struct OPE_API ResamplingParameters {
		bool enabled;
		InterpolationMethod interpolationMethod;
		bool useCoefficients;
		float coefficients[4];  // c0, c1, c2, c3
		bool useCustomCurve;
		
		ResamplingParameters()
			: enabled(false)
			, interpolationMethod(InterpolationMethod::LINEAR)
			, useCoefficients(true)
			, useCustomCurve(false)
		{
			coefficients[0] = 0.0f;
			coefficients[1] = 1.0f;
			coefficients[2] = 0.0f;
			coefficients[3] = 0.0f;
		}
	};
	
	struct OPE_API WindowingParameters {
		bool enabled;
		WindowType windowType;
		float windowCenterPosition;
		float windowFillFactor;
		bool useCustomCurve;
	
		WindowingParameters()
			: enabled(false)
			, windowType(WindowType::HANN)
			, windowCenterPosition(0.5f)
			, windowFillFactor(0.95f)
			, useCustomCurve(false)
		{}
	};
	
	struct OPE_API DispersionCompensationParameters {
		bool enabled;
		bool useCoefficients;
		float coefficients[4];  // d0, d1, d2, d3
		float factor;
		bool useCustomCurve;
		
		DispersionCompensationParameters()
			: enabled(false)
			, useCoefficients(true)
			, factor(1.0f)
			, useCustomCurve(false)
		{
			coefficients[0] = 0.0f;
			coefficients[1] = 0.0f;
			coefficients[2] = 0.0f;
			coefficients[3] = 0.0f;
		}
	};
	
	struct OPE_API BackgroundRemovalParameters {
		bool enabled;
		int rollingAverageWindowSize;
		
		BackgroundRemovalParameters()
			: enabled(false)
			, rollingAverageWindowSize(64)
		{}
	};
	
	struct OPE_API PostProcessingParameters {
		bool backgroundRemoval;
		float backgroundWeight;
		float backgroundOffset;
		
		bool logScaling;
		float grayscaleMax;
		float grayscaleMin;
		float addend;
		float multiplicator;
		
		bool bscanFlip;
		bool sinusoidalScanCorrection;
		
		bool fixedPatternNoiseRemoval;
		int fixedPatternNoiseBscanCount;
		bool continuousFixedPatternNoiseDetermination;
		
		PostProcessingParameters()
			: backgroundRemoval(false)
			, backgroundWeight(1.0f)
			, backgroundOffset(0.0f)
			, logScaling(true)
			, grayscaleMax(100.0f)
			, grayscaleMin(30.0f)
			, addend(0.0f)
			, multiplicator(1.0f)
			, bscanFlip(false)
			, sinusoidalScanCorrection(false)
			, fixedPatternNoiseRemoval(false)
			, fixedPatternNoiseBscanCount(1)
			, continuousFixedPatternNoiseDetermination(false)
		{}
	};
	
	DataParameters dataParams;
	ProcessingParameters processingParams;
	ResamplingParameters resamplingParams;
	WindowingParameters windowingParams;
	DispersionCompensationParameters dispersionParams;
	BackgroundRemovalParameters backgroundRemovalParams;
	PostProcessingParameters postProcessingParams;
	
	// Constructor and Rule of 5
	ProcessorConfiguration();
	~ProcessorConfiguration();
	ProcessorConfiguration(const ProcessorConfiguration& other);
	ProcessorConfiguration(ProcessorConfiguration&& other) noexcept;
	ProcessorConfiguration& operator=(const ProcessorConfiguration& other);
	ProcessorConfiguration& operator=(ProcessorConfiguration&& other) noexcept;
	
	void setCustomResamplingCurve(const float* data, size_t size);
	void setCustomWindowCurve(const float* data, size_t size);
	void setCustomDispersionCurve(const float* data, size_t size);
	void setCustomPostProcessBackgroundProfile(const float* data, size_t size); //todo: think about removing "custom" from background subtraction methods.  
	void setCustomFixedPatternNoiseProfile(const float* data, size_t complexPairs);

	const float* getCustomResamplingCurve() const;
	const float* getCustomWindowCurve() const;
	const float* getCustomDispersionCurve() const;
	const float* getCustomPostProcessBackgroundProfile() const;
	const float* getCustomFixedPatternNoiseProfile() const;
	
	size_t getCustomResamplingCurveSize() const;
	size_t getCustomWindowCurveSize() const;
	size_t getCustomDispersionCurveSize() const;
	size_t getCustomPostProcessBackgroundProfileSize() const;
	size_t getCustomFixedPatternNoiseProfileSize() const;

	bool hasCustomResamplingCurve() const;
	bool hasCustomWindowCurve() const;
	bool hasCustomDispersionCurve() const;
	bool hasCustomPostProcessBackgroundProfile() const;
	bool hasCustomFixedPatternNoiseProfile() const;

	const float* getGeneratedResamplingCurve() const;
	const float* getGeneratedWindowCurve() const;
	const float* getGeneratedDispersionCurve() const;
	
	size_t getGeneratedResamplingCurveSize() const;
	size_t getGeneratedWindowCurveSize() const;
	size_t getGeneratedDispersionCurveSize() const;
	
	
	void adjustAllCustomCurves(); // call after changing signalLength
	
	bool saveToFile(const std::string& filepath) const;
	bool loadFromFile(const std::string& filepath);
	bool validate() const;

private:
	struct Impl;
	Impl* impl;
	
	void generateResamplingCurve() const;
	void generateWindowCurve() const;
	void generateDispersionCurve() const;
	
	void adjustCustomResamplingCurve();
	void adjustCustomWindowCurve();
	void adjustCustomDispersionCurve();
	void adjustCustomPostProcessBackgroundProfile();
	void adjustCustomFixedPatternNoiseProfile();
	
	template<typename T>
	T clamp(T value, T min, T max) const {
		return (value < min) ? min : ((value > max) ? max : value);
	}
};

} // namespace ope

#endif // OPE_PROCESSORCONFIGURATION_H