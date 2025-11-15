#ifndef OPE_CPU_KERNELS_H
#define OPE_CPU_KERNELS_H

#include <vector>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace ope {
namespace cpu_kernels {


template <typename T>
void convertInputData(
	const void* inputData,
	size_t totalSamples,
	int inputBitDepth,
	std::vector<std::complex<T>>& outputData
);

template <typename T>
void rollingAverageDCRemoval(
	std::vector<std::complex<T>>& spectrum,
	size_t rollingAverageWindowSize
);

template <typename T>
void kLinearizationLinear(
	const std::vector<std::complex<T>>& inputSpectrum,
	const std::vector<T>& resampleCurve,
	std::vector<std::complex<T>>& outputSpectrum
);

template <typename T>
void kLinearizationCubic(
	const std::vector<std::complex<T>>& inputSpectrum,
	const std::vector<T>& resampleCurve,
	std::vector<std::complex<T>>& outputSpectrum
);

template <typename T>
void kLinearizationLanczos(
	const std::vector<std::complex<T>>& inputSpectrum,
	const std::vector<T>& resampleCurve,
	std::vector<std::complex<T>>& outputSpectrum
);

template <typename T>
void dispersionCompensation(
	std::vector<std::complex<T>>& data,
	const std::vector<std::complex<T>>& phaseComplex
);

template <typename T>
void applyWindow(
	std::vector<std::complex<T>>& data,
	const std::vector<T>& windowFunction
);

template <typename T>
void computeIFFT(
	const std::vector<std::complex<T>>& input,
	std::vector<std::complex<T>>& output,
	void* fftPlan,
	void* fftIn,
	void* fftOut
);

template <typename T>
void logScaleAndTruncate(
	const std::vector<std::complex<T>>& input,
	std::vector<T>& output,
	T coeff,
	T minVal,
	T maxVal,
	T addend,
	bool autoComputeMinMax
);

template <typename T>
void linearScaleAndTruncate(
	const std::vector<std::complex<T>>& input,
	std::vector<T>& output,
	T coeff,
	T minVal,
	T maxVal,
	T addend
);

// Fixed-pattern-noise helpers
template <typename T>
std::vector<T> getMinimumVarianceMean(
	const std::vector<std::vector<std::complex<T>>>& allIfftOutputs,
	int width,
	int segments
);

template <typename T>
void meanALineSubtraction(
	std::vector<std::complex<T>>& spectrum,
	const std::vector<T>& meanInterleaved
);

template <typename T>
T cubicHermiteInterpolation(
	const T y0,
	const T y1,
	const T y2,
	const T y3,
	const T positionBetweenY1andY2
);

template <typename T>
const T& clamp(const T& value, const T& low, const T& high);

} // namespace cpu_kernels
} // namespace ope

// Include template implementations
#include "cpu_kernels.tpp"

#endif // OPE_CPU_KERNELS_H