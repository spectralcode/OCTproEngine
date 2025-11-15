// CPU Kernels for optical coherence tomography signal processing
// This file is part of OCTproEngine
// Copyright (c) 2025 Miroslav Zabic

#ifndef OPE_CPU_KERNELS_TPP
#define OPE_CPU_KERNELS_TPP

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <limits>
#include "fftw3.h"

namespace ope {
namespace cpu_kernels {

template <typename T>
void convertInputData(
	const void* inputData,
	size_t totalSamples,
	int inputBitDepth,
	std::vector<std::complex<T>>& outputData)
{
	outputData.resize(totalSamples);

	T scaleFactor = static_cast<T>(1.0);

	if (inputBitDepth <= 8) {
		const uint8_t* in = static_cast<const uint8_t*>(inputData);
		scaleFactor = static_cast<T>(1.0);// / static_cast<T>(255);
		for (size_t i = 0; i < totalSamples; ++i) {
			outputData[i] = std::complex<T>(static_cast<T>(in[i]) * scaleFactor, static_cast<T>(0));
		}
	} else if (inputBitDepth <= 16) {
		const uint16_t* in = static_cast<const uint16_t*>(inputData);
		scaleFactor = static_cast<T>(1.0);// / static_cast<T>(65535);
		for (size_t i = 0; i < totalSamples; ++i) {
			outputData[i] = std::complex<T>(static_cast<T>(in[i]) * scaleFactor, static_cast<T>(0));
		}
	} else {
		const uint32_t* in = static_cast<const uint32_t*>(inputData);
		scaleFactor = static_cast<T>(1.0);// / static_cast<T>(4294967295);
		for (size_t i = 0; i < totalSamples; ++i) {
			outputData[i] = std::complex<T>(static_cast<T>(in[i]) * scaleFactor, static_cast<T>(0));
		}
	}
}

template <typename T>
void rollingAverageDCRemoval(
	std::vector<std::complex<T>>& spectrum,
	size_t rollingAverageWindowSize)
{
	size_t numSamples = spectrum.size();
	size_t rollingWindowSize = rollingAverageWindowSize;

	// Compute cumulative sum for efficient rolling average computation
	std::vector<T> cumulativeSum(numSamples + 1, static_cast<T>(0));
	for (size_t i = 0; i < numSamples; ++i) {
		cumulativeSum[i + 1] = cumulativeSum[i] + spectrum[i].real();
	}

	// Perform rolling average subtraction
	for (size_t i = 0; i < numSamples; ++i) {
		size_t startIdx = (i >= rollingWindowSize - 1) ? i - (rollingWindowSize - 1) : 0;
		size_t endIdx = std::min(i + rollingWindowSize, numSamples - 1);

		T sum = cumulativeSum[endIdx + 1] - cumulativeSum[startIdx];
		size_t windowSize = endIdx - startIdx + 1;
		T rollingAverage = sum / static_cast<T>(windowSize);

		// Subtract rolling average
		spectrum[i] -= std::complex<T>(rollingAverage, static_cast<T>(0));
	}
}

template <typename T>
void kLinearizationLinear(
	const std::vector<std::complex<T>>& inputSpectrum,
	const std::vector<T>& resampleCurve,
	std::vector<std::complex<T>>& outputSpectrum)
{
	size_t width = resampleCurve.size();
	outputSpectrum.resize(width);

	size_t inputSize = inputSpectrum.size();

	for (size_t j = 0; j < width; ++j) {
		T nx = resampleCurve[j];
		int n1 = static_cast<int>(nx);
		int n2 = n1 + 1;

		// Handle boundary conditions
		n1 = std::min(n1, static_cast<int>(inputSize) - 1);
		n2 = std::min(n2, static_cast<int>(inputSize) - 1);

		T y1 = inputSpectrum[n1].real();
		T y2 = inputSpectrum[n2].real();

		T interpolatedValue = y1 + (y2 - y1) * (nx - n1);

		outputSpectrum[j] = std::complex<T>(interpolatedValue, static_cast<T>(0));
	}
}

template <typename T>
void kLinearizationCubic(
	const std::vector<std::complex<T>>& inputSpectrum,
	const std::vector<T>& resampleCurve,
	std::vector<std::complex<T>>& outputSpectrum)
{
	size_t width = resampleCurve.size();
	outputSpectrum.resize(width);

	size_t inputSize = inputSpectrum.size();

	for (size_t j = 0; j < width; ++j) {
		T nx = resampleCurve[j];
		int n1 = static_cast<int>(nx);
		int n0 = std::abs(n1 - 1);
		int n2 = n1 + 1;
		int n3 = n2 + 1;

		// Handle boundary conditions
		n0 = std::min(n0, static_cast<int>(inputSize) - 1);
		n1 = std::min(n1, static_cast<int>(inputSize) - 1);
		n2 = std::min(n2, static_cast<int>(inputSize) - 1);
		n3 = std::min(n3, static_cast<int>(inputSize) - 1);

		T y0 = inputSpectrum[n0].real();
		T y1 = inputSpectrum[n1].real();
		T y2 = inputSpectrum[n2].real();
		T y3 = inputSpectrum[n3].real();

		T interpolatedValue = cubicHermiteInterpolation(y0, y1, y2, y3, nx - n1);

		outputSpectrum[j] = std::complex<T>(interpolatedValue, static_cast<T>(0));
	}
}

template <typename T>
inline T lanczosKernel(T x, int a = 8) {
	if (std::abs(x) < static_cast<T>(1e-5)) {
		return static_cast<T>(1.0);
	}
	if (std::abs(x) >= static_cast<T>(a)) {
		return static_cast<T>(0.0);
	}
	
	T pi = static_cast<T>(M_PI);
	T piX = pi * x;
	T piXOverA = piX / static_cast<T>(a);
	
	return (std::sin(piX) / piX) * (std::sin(piXOverA) / piXOverA);
}

template <typename T>
void kLinearizationLanczos(
	const std::vector<std::complex<T>>& inputSpectrum,
	const std::vector<T>& resampleCurve,
	std::vector<std::complex<T>>& outputSpectrum)
{
	size_t width = resampleCurve.size();
	outputSpectrum.resize(width);

	size_t inputSize = inputSpectrum.size();
	const int a = 8; // Lanczos kernel size

	for (size_t j = 0; j < width; ++j) {
		T nx = resampleCurve[j];
		int n0 = static_cast<int>(nx);
		
		T sum = static_cast<T>(0.0);
		
		// Sum over kernel window [-a+1, a]
		for (int i = -a + 1; i <= a; ++i) {
			int idx = n0 + i;
			
			// Clamp to valid range
			if (idx < 0) idx = 0;
			if (idx >= static_cast<int>(inputSize)) idx = inputSize - 1;
			
			T y = inputSpectrum[idx].real();
			sum += y * lanczosKernel<T>(nx - idx, a);
		}
		
		outputSpectrum[j] = std::complex<T>(sum, static_cast<T>(0));
	}
}

template <typename T>
void dispersionCompensation(
	std::vector<std::complex<T>>& data,
	const std::vector<std::complex<T>>& phaseComplex)
{
	size_t dataSize = data.size();

	if (phaseComplex.empty() || phaseComplex.size() != dataSize) {
		// Cannot apply dispersion compensation without valid phase data
		return;
	}

	//full complex multiplication
//	for (size_t i = 0; i < dataSize; ++i) {
//		data[i] *= phaseComplex[i];
//	}

	// Perform only real part of complex multiplication
	// this works because imaginary part of input data is always 0
	for (size_t i = 0; i < dataSize; ++i) {
		T inReal = data[i].real();

		T outReal = inReal * phaseComplex[i].real();
		T outImag = inReal * phaseComplex[i].imag();

		data[i] = std::complex<T>(outReal, outImag);
	}
}

template <typename T>
void applyWindow(
	std::vector<std::complex<T>>& data,
	const std::vector<T>& windowFunction)
{
	for (size_t i = 0; i < data.size(); ++i) {
		data[i] *= windowFunction[i];
	}
}

// Specialization for float using fftwf
template <>
inline void computeIFFT<float>(
	const std::vector<std::complex<float>>& input,
	std::vector<std::complex<float>>& output,
	void* fftPlan,
	void* fftIn,
	void* fftOut)
{
	fftwf_plan plan = static_cast<fftwf_plan>(fftPlan);
	fftwf_complex* in = static_cast<fftwf_complex*>(fftIn);
	fftwf_complex* out = static_cast<fftwf_complex*>(fftOut);
	
	size_t samplesPerSpectrum = input.size();
	
	// Copy input data to FFTW input array
	for (size_t i = 0; i < samplesPerSpectrum; ++i) {
		in[i][0] = input[i].real();
		in[i][1] = input[i].imag();
	}

	// Execute the IFFT
	fftwf_execute(plan);

	// Normalize and copy output data
	output.resize(samplesPerSpectrum);
	float normFactor = static_cast<float>(1) / static_cast<float>(samplesPerSpectrum);
	for (size_t i = 0; i < samplesPerSpectrum; ++i) {
		output[i] = std::complex<float>(out[i][0], out[i][1]);// * normFactor;
	}
}

// Specialization for double using fftw
template <>
inline void computeIFFT<double>(
	const std::vector<std::complex<double>>& input,
	std::vector<std::complex<double>>& output,
	void* fftPlan,
	void* fftIn,
	void* fftOut)
{
	fftw_plan plan = static_cast<fftw_plan>(fftPlan);
	fftw_complex* in = static_cast<fftw_complex*>(fftIn);
	fftw_complex* out = static_cast<fftw_complex*>(fftOut);
	
	size_t samplesPerSpectrum = input.size();
	
	// Copy input data to FFTW input array
	for (size_t i = 0; i < samplesPerSpectrum; ++i) {
		in[i][0] = input[i].real();
		in[i][1] = input[i].imag();
	}

	// Execute the IFFT
	fftw_execute(plan);

	// Normalize and copy output data
	output.resize(samplesPerSpectrum);
	double normFactor = static_cast<double>(1) / static_cast<double>(samplesPerSpectrum);
	for (size_t i = 0; i < samplesPerSpectrum; ++i) {
		output[i] = std::complex<double>(out[i][0], out[i][1]);// * normFactor;
	}
}

template <typename T>
void logScaleAndTruncate(
	const std::vector<std::complex<T>>& input,
	std::vector<T>& output,
	T coeff,
	T minVal,
	T maxVal,
	T addend,
	bool autoComputeMinMax)
{
	size_t size = input.size()/2; // the output size is half of input size because we only use first half of each a-scan to remove mirror artifact
	output.resize(size);

	T outputAscanLength = static_cast<T>(size);

	// Compute magnitude squared and initial value array
	std::vector<T> valueArray(size);
	for (size_t i = 0; i < size; ++i) {
		T realComponent = input[i].real();
		T imaginaryComponent = input[i].imag();
		T magnitudeSquared = realComponent * realComponent + imaginaryComponent * imaginaryComponent;
		T value = static_cast<T>(10.0) * std::log10(magnitudeSquared / outputAscanLength);
		valueArray[i] = value;
	}

	// Auto-compute min and max if enabled
	if (autoComputeMinMax) {
		minVal = *std::min_element(valueArray.begin(), valueArray.end());
		maxVal = *std::max_element(valueArray.begin(), valueArray.end());
	}

	T range = maxVal - minVal;// Avoid division by zero
	if (range < static_cast<T>(1e-6)) {
		range = static_cast<T>(1.0); // Avoid division by zero
	}

	for (size_t i = 0; i < size; ++i) {
		T normalizedValue = (valueArray[i] - minVal) / range;
		output[i] = coeff * (normalizedValue + addend);
	}
}

template <typename T>
void linearScaleAndTruncate(
	const std::vector<std::complex<T>>& input,
	std::vector<T>& output,
	T coeff,
	T minVal,
	T maxVal,
	T addend)
{
	size_t size = input.size()/2; // the output size is half of input size because we only use first half of each a-scan to remove mirror artifact
	output.resize(size);

	T outputAscanLength = static_cast<T>(size);
	T range = maxVal - minVal;
	
	// Avoid division by zero
	if (range < static_cast<T>(1e-6)) {
		range = static_cast<T>(1.0);
	}

	// Compute linear magnitude with normalization and grayscale mapping (note: input is unnormalized IFFT output)
	for (size_t i = 0; i < size; ++i) {
		T realComponent = input[i].real();
		T imaginaryComponent = input[i].imag();
		
		// Compute normalized magnitude
		T magnitude = std::sqrt(realComponent * realComponent + 
		                       imaginaryComponent * imaginaryComponent) / outputAscanLength;
		
		// Apply grayscale mapping
		T normalized = (magnitude - minVal) / range;
		
		// Apply multiplicator and addend
		output[i] = coeff * (normalized + addend);
	}
}

template<typename T>
void applyPostProcessBackgroundSubtraction(
	T* data,
	const T* background,
	T weight,
	T offset,
	int samplesPerAscan,
	int totalAscans
) {
	int totalSamples = samplesPerAscan * totalAscans;
	
	for (int i = 0; i < totalSamples; ++i) {
		int bgIndex = i % samplesPerAscan;
		T bgValue = weight * background[bgIndex] + offset;
		T result = data[i] - bgValue;
		
		// Saturate to [0, 1] range
		//data[i] = std::max(static_cast<T>(0), std::min(static_cast<T>(1), result));
		data[i] = result;
	}
}

template <typename T>
std::vector<T> getMinimumVarianceMean(
	const std::vector<std::vector<std::complex<T>>>& allIfftOutputs,
	int width,
	int segments
)
{
	int height = static_cast<int>(allIfftOutputs.size());
	if (height == 0 || width <= 0 || segments <= 0) {
		return std::vector<T>();
	}

	int segWidth = height / segments;
	T factor = static_cast<T>(1.0) / static_cast<T>(segWidth);

	std::vector<T> out;
	out.resize(static_cast<size_t>(width) * 2);

	for (int col = 0; col < width; ++col) {
		T minVariance = static_cast<T>(std::numeric_limits<T>::infinity());
		T meanAtMinX = static_cast<T>(0);
		T meanAtMinY = static_cast<T>(0);

		for (int seg = 0; seg < segments; ++seg) {
			int offsetRow = seg * segWidth;

			double sumX = 0.0;
			double sumY = 0.0;
			double sumXX = 0.0;

			for (int r = 0; r < segWidth; ++r) {
				int row = offsetRow + r;
				const std::complex<T>& c = allIfftOutputs[row][col];
				double dx = static_cast<double>(c.real());
				double dy = static_cast<double>(c.imag());
				sumX += dx;
				sumY += dy;
				sumXX += dx * dx + dy * dy;
			}

			T meanX = static_cast<T>(sumX * static_cast<double>(factor));
			T meanY = static_cast<T>(sumY * static_cast<double>(factor));
			T variance = static_cast<T>((sumXX * static_cast<double>(factor)) - (static_cast<double>(meanX) * static_cast<double>(meanX) + static_cast<double>(meanY) * static_cast<double>(meanY)));

			if (variance < minVariance) {
				minVariance = variance;
				meanAtMinX = meanX;
				meanAtMinY = meanY;
			}
		}

		out[col * 2] = meanAtMinX;
		out[col * 2 + 1] = meanAtMinY;
	}

	return out;
}

template <typename T>
void meanALineSubtraction(
	std::vector<std::complex<T>>& spectrum,
	const std::vector<T>& meanInterleaved
)
{
	if (meanInterleaved.empty()) return;

	int half = static_cast<int>(spectrum.size()) / 2;
	int pairs = static_cast<int>(meanInterleaved.size()) / 2;
	int use = std::min(half, pairs);

	for (int i = 0; i < use; ++i) {
		int mi = i * 2;
		T mx = meanInterleaved[mi];
		T my = meanInterleaved[mi + 1];
		spectrum[i].real(spectrum[i].real() - mx);
		spectrum[i].imag(spectrum[i].imag() - my);
	}
}


// ============================================
// Helper functions
// ============================================

template <typename T>
T cubicHermiteInterpolation(
	const T y0,
	const T y1,
	const T y2,
	const T y3,
	const T positionBetweenY1andY2)
{
	const T a = -y0 + 3.0f * (y1 - y2) + y3;
	const T b = 2.0f * y0 - 5.0f * y1 + 4.0f * y2 - y3;
	const T c = -y0 + y2;

	const T pos = positionBetweenY1andY2;
	const T pos2 = pos * pos;

	return static_cast<T>(0.5) * pos * (a * pos2 + b * pos + c) + y1;
}

template <typename T>
const T& clamp(const T& value, const T& low, const T& high)
{
	return (value < low) ? low : (high < value) ? high : value;
}

} // namespace cpu_kernels
} // namespace ope

#endif // OPE_CPU_KERNELS_TPP