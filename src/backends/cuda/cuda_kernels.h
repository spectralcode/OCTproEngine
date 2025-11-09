#ifndef OPE_CUDA_KERNELS_H
#define OPE_CUDA_KERNELS_H

#include <cufft.h>
#include <cuda_runtime.h>

namespace ope {
namespace cuda_kernels {

// Constants
#define EIGHT_OVER_PI_SQUARED 0.8105694691f
#define PI_OVER_8 0.3926990817f
#define PI 3.141592654f
#define FIXED_PATTERN_NOISE_REMOVAL_SEGMENTS 8

// ============================================
// Input Conversion Kernels
// ============================================

__global__ void inputToCufftComplex(
	cufftComplex* __restrict__ output,
	const void* __restrict__ input,
	int samplesPerLine,
	int linesPerFrame,
	int bitDepth,
	int samplesPerBuffer
);

__global__ void inputToCufftComplex_and_bitshift(
	cufftComplex* __restrict__ output,
	const void* __restrict__ input,
	int samplesPerLine,
	int linesPerFrame,
	int bitDepth,
	int samplesPerBuffer
);

// ============================================
// Background Removal Kernels
// ============================================

__global__ void rollingAverageBackgroundRemoval(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	int windowSize,
	int samplesPerLine,
	int linesPerFrame,
	int samplesPerFrame,
	int samplesPerBuffer
);

// ============================================
// K-Linearization Kernels
// ============================================

__global__ void klinearization(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void klinearizationQuadratic(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void klinearizationCubic(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void klinearizationLanczos(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

// ============================================
// Windowing Kernels
// ============================================

__global__ void windowing(
	cufftComplex* output,
	const cufftComplex* input,
	const float* windowCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

// ============================================
// Fused K-Linearization + Windowing Kernels
// ============================================

__global__ void klinearizationAndWindowing(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	const float* __restrict__ windowCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void klinearizationCubicAndWindowing(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	const float* __restrict__ windowCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void klinearizationLanczosAndWindowing(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	const float* __restrict__ windowCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

// ============================================
// Fused K-Linearization + Windowing + Dispersion Kernels
// ============================================

__global__ void klinearizationAndWindowingAndDispersionCompensation(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	const float* __restrict__ windowCurve,
	const cufftComplex* __restrict__ phaseComplex,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void klinearizationCubicAndWindowingAndDispersionCompensation(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	const float* __restrict__ windowCurve,
	const cufftComplex* __restrict__ phaseComplex,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void klinearizationLanczosAndWindowingAndDispersionCompensation(
	cufftComplex* __restrict__ out,
	const cufftComplex* __restrict__ in,
	const float* __restrict__ resampleCurve,
	const float* __restrict__ windowCurve,
	const cufftComplex* __restrict__ phaseComplex,
	int samplesPerLine,
	int samplesPerBuffer
);

// ============================================
// Dispersion Compensation Kernels
// ============================================

__device__ cufftComplex cuMultiply(const cufftComplex& a, const cufftComplex& b);

__global__ void dispersionCompensation(
	cufftComplex* out,
	const cufftComplex* in,
	const cufftComplex* phaseComplex,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void dispersionCompensationAndWindowing(
	cufftComplex* out,
	const cufftComplex* in,
	const cufftComplex* phaseComplex,
	const float* windowCurve,
	int samplesPerLine,
	int samplesPerBuffer
);

__global__ void fillDispersivePhase(
	cufftComplex* __restrict__ phaseComplex,
	const float* __restrict__ dispersionCurve,
	float factor,
	int size,
	int direction
);

// ============================================
// Post-Processing Kernels
// ============================================

__global__ void postProcessTruncateLog(
	float* __restrict__ output,
	const cufftComplex* __restrict__ input,
	int samplesPerLine,
	int samplesPerBuffer,
	int currentBufferNr,
	float max,
	float min,
	float addend,
	float coeff
);

__global__ void postProcessTruncateLin(
	float* __restrict__ output,
	const cufftComplex* __restrict__ input,
	int samplesPerLine,
	int samplesPerBuffer,
	float max,
	float min,
	float addend,
	float coeff
);

// ============================================
// Fixed Pattern Noise Removal Kernels
// ============================================

__global__ void getMinimumVarianceMean(
	cufftComplex* __restrict__ meanLine,
	const cufftComplex* __restrict__ input,
	int width,
	int height,
	int segments
);

__global__ void meanALineSubtraction(
	cufftComplex* __restrict__ in_out,
	const cufftComplex* __restrict__ meanLine,
	int width,
	int samples
);

// ============================================
// B-Scan Flip Kernels
// ============================================

__global__ void cuda_bscanFlip(
	float* output,
	const float* input,
	int samplesPerAscan,
	int ascansPerBscan,
	int samplesPerBscan,
	int samplesPerBuffer
);

// ============================================
// Sinusoidal Scan Correction Kernels
// ============================================

__global__ void sinusoidalScanCorrection(
	float* __restrict__ out,
	const float* __restrict__ in,
	const float* __restrict__ sinusoidalResampleCurve,
	int samplesPerLine,
	int linesPerBscan,
	int bscansPerBuffer,
	int samplesPerBuffer
);

__global__ void fillSinusoidalScanCorrectionCurve(
	float* sinusoidalResampleCurve,
	const int length
);

// ============================================
// Post-Process Background Removal Kernels
// ============================================

__global__ void getPostProcessBackground(
	float* __restrict__ output,
	const float* __restrict__ input,
	int samplesPerLine,
	int ascansPerBscan
);

__global__ void postProcessBackgroundRemoval(
	float* data,
	const float* background,
	float weight,
	float offset,
	int samplesPerLine,
	int samplesPerBuffer
);

} // namespace cuda_kernels
} // namespace ope

#endif // OPE_CUDA_KERNELS_H