/**
**  This file is part of OCTproEngine.
**  OpenCL kernels for optical coherence tomography signal processing
**  Based on the original CUDA kernels from OCTproEngine
**  Translation from CUDA to OpenCL was done with extensive LLM assistance
**
**  IMPORTANT: 
**	opencl_kernels.cl is used to auto-generate opencl_kernels_source.cpp
**  All changes made in the opencl_kernels_source.cpp file will be automatically
**  overwritten when cmake is re-run. 
**
**  Copyright (C) 2025 Miroslav Zabic
**
**/

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================
// Input Conversion Kernels
// ============================================

__kernel void inputToCufftComplex(
	__global float2* output,
	__global const uchar* input,
	const int width_out,
	const int width_in,
	const int inputBitdepth,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	float real_val;
	if (inputBitdepth <= 8) {
		real_val = convert_float_rtn(input[index]);
	} else if (inputBitdepth <= 16) {
		__global const ushort* in_ushort = (__global const ushort*)input;
		real_val = convert_float_rtn(in_ushort[index]);
	} else {
		__global const uint* in_uint = (__global const uint*)input;
		real_val = convert_float_rtn(in_uint[index]);
	}
	output[index] = (float2)(real_val, 0.0f);
}

__kernel void inputToCufftComplex_and_bitshift(
	__global float2* output,
	__global const uchar* input,
	const int width_out,
	const int width_in,
	const int inputBitdepth,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	float real_val;
	if (inputBitdepth <= 8) {
		real_val = convert_float_rtn(input[index] >> 4);
	} else if (inputBitdepth <= 16) {
		__global const ushort* in_ushort = (__global const ushort*)input;
		real_val = convert_float_rtn(in_ushort[index] >> 4);
	} else {
		__global const uint* in_uint = (__global const uint*)input;
		real_val = in_uint[index] / 4294967296.0f;
	}
	output[index] = (float2)(real_val, 0.0f);
}

// ============================================
// Background Removal Kernel
// ============================================

__kernel void rollingAverageBackgroundRemoval(
	__global float2* out,
	__global const float2* in,
	const int rollingAverageWindowSize,
	const int width,
	const int height,
	const int samplesPerFrame,
	const int samples,
	__local float* s_data)
{
	int index = get_global_id(0);
	int local_id = get_local_id(0);
	int group_id = get_group_id(0);
	int local_size = get_local_size(0);

	if (index < samples) {
		int currentBscan = index / samplesPerFrame;
		int currentLine = (index / width) % height;
		int firstIndexOfCurrentLine = currentLine * width + (samplesPerFrame * currentBscan);
		int lastIndexOfCurrentLine = firstIndexOfCurrentLine + width - 1;

		int startIdx = max(firstIndexOfCurrentLine, index - rollingAverageWindowSize + 1);
		int endIdx = min(lastIndexOfCurrentLine, index + rollingAverageWindowSize);
		int windowSize = endIdx - startIdx + 1;

		int blockFirstIdx = group_id * local_size;
		int blockStartIdx = max(0, blockFirstIdx - rollingAverageWindowSize + 1);
		int blockEndIdx = min(samples - 1, (blockFirstIdx + local_size - 1) + rollingAverageWindowSize);

		for (int i = blockStartIdx + local_id; i <= blockEndIdx; i += local_size) {
			s_data[i - blockStartIdx] = in[i].x;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		float rollingSum = 0.0f;
		for (int i = startIdx; i <= endIdx; i++) {
			rollingSum += s_data[i - blockStartIdx];
		}

		float rollingAverage = rollingSum / windowSize;
		out[index] = (float2)(in[index].x - rollingAverage, 0.0f);
	}
}

// ============================================
// K-Linearization Kernels
// ============================================

__kernel void klinearization(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;

	out[index] = (float2)(f_x0 + (f_x1 - f_x0) * (x - x0), 0.0f);
}

__kernel void klinearizationQuadratic(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;
	int x2 = x0 + 2;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;
	float f_x2 = in[offset + x2].x;
	float b0 = f_x0;
	float b1 = f_x1 - f_x0;
	float b2 = ((f_x2 - f_x1) - b1) / (x2 - x0);

	out[index] = (float2)(b0 + b1 * (x - x0) + b2 * (x - x0) * (x - x1), 0.0f);
}

inline float cubicHermiteInterpolation(
	const float y0,
	const float y1,
	const float y2,
	const float y3,
	const float positionBetweenY1andY2)
{
	const float a = -y0 + 3.0f * (y1 - y2) + y3;
	const float b = 2.0f * y0 - 5.0f * y1 + 4.0f * y2 - y3;
	const float c = -y0 + y2;

	const float pos = positionBetweenY1andY2;
	const float pos2 = pos * pos;

	return 0.5f * pos * (a * pos2 + b * pos + c) + y1;
}

__kernel void klinearizationCubic(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float nx = resampleCurve[j];
	const int n1 = (int)nx;
	int n0 = abs(n1 - 1);
	int n2 = n1 + 1;
	int n3 = n1 + 2;

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;

	out[index] = (float2)(cubicHermiteInterpolation(y0, y1, y2, y3, nx - n1), 0.0f);
}

inline float sinc(float x) {
	if (fabs(x) < 1e-7f) {
		return 1.0f;
	}
	float pix = M_PI * x;
	return sin(pix) / pix;
}

inline float lanczosKernel(float x, int a) {
	if (fabs(x) >= a) {
		return 0.0f;
	}
	return sinc(x) * sinc(x / a);
}

__kernel void klinearizationLanczos(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float x = resampleCurve[j];
	int center = (int)floor(x);
	const int a = 3;

	float sum = 0.0f;
	for (int k = center - a + 1; k <= center + a; k++) {
		int idx = clamp(k, 0, width - 1);
		sum += in[offset + idx].x * lanczosKernel(x - k, a);
	}

	out[index] = (float2)(sum, 0.0f);
}

// ============================================
// Windowing Kernel
// ============================================

__kernel void windowing(
	__global float2* out,
	__global const float2* in,
	__global const float* windowCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	float windowValue = windowCurve[j];

	out[index] = (float2)(in[index].x * windowValue, in[index].y * windowValue);
}

// ============================================
// Fused K-Linearization + Windowing Kernels
// ============================================

__kernel void klinearizationAndWindowing(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	__global const float* windowCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;

	float interpolated = f_x0 + (f_x1 - f_x0) * (x - x0);
	float windowValue = windowCurve[j];

	out[index] = (float2)(interpolated * windowValue, 0.0f);
}

__kernel void klinearizationCubicAndWindowing(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	__global const float* windowCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float nx = resampleCurve[j];
	const int n1 = (int)nx;
	int n0 = abs(n1 - 1);
	int n2 = n1 + 1;
	int n3 = n1 + 2;

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;

	float interpolated = cubicHermiteInterpolation(y0, y1, y2, y3, nx - n1);
	float windowValue = windowCurve[j];

	out[index] = (float2)(interpolated * windowValue, 0.0f);
}

__kernel void klinearizationLanczosAndWindowing(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	__global const float* windowCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float x = resampleCurve[j];
	int center = (int)floor(x);
	const int a = 3;

	float sum = 0.0f;
	for (int k = center - a + 1; k <= center + a; k++) {
		int idx = clamp(k, 0, width - 1);
		sum += in[offset + idx].x * lanczosKernel(x - k, a);
	}

	float windowValue = windowCurve[j];

	out[index] = (float2)(sum * windowValue, 0.0f);
}

// ============================================
// Dispersion Compensation Kernels
// ============================================

inline float2 complexMultiply(float2 a, float2 b) {
	return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void fillDispersivePhase(
	__global float2* phaseCartesian,
	__global const float* dispersionCurve,
	const int width)
{
	int j = get_global_id(0);
	if (j >= width) return;

	float phase = dispersionCurve[j];
	phaseCartesian[j] = (float2)(cos(phase), sin(phase));
}

__kernel void dispersionCompensation(
	__global float2* out,
	__global const float2* in,
	__global const float2* phaseCartesian,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	out[index] = complexMultiply(in[index], phaseCartesian[j]);
}

__kernel void dispersionCompensationAndWindowing(
	__global float2* out,
	__global const float2* in,
	__global const float2* phaseCartesian,
	__global const float* windowCurve,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	float2 compensated = complexMultiply(in[index], phaseCartesian[j]);
	float windowValue = windowCurve[j];

	out[index] = (float2)(compensated.x * windowValue, compensated.y * windowValue);
}

// ============================================
// Fused K-Linearization + Windowing + Dispersion Kernels
// ============================================

__kernel void klinearizationAndWindowingAndDispersionCompensation(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	__global const float* windowCurve,
	__global const float2* phaseCartesian,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;

	float interpolated = f_x0 + (f_x1 - f_x0) * (x - x0);
	float windowValue = windowCurve[j];
	float2 windowed = (float2)(interpolated * windowValue, 0.0f);

	out[index] = complexMultiply(windowed, phaseCartesian[j]);
}

__kernel void klinearizationCubicAndWindowingAndDispersionCompensation(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	__global const float* windowCurve,
	__global const float2* phaseCartesian,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float nx = resampleCurve[j];
	const int n1 = (int)nx;
	int n0 = abs(n1 - 1);
	int n2 = n1 + 1;
	int n3 = n1 + 2;

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;

	float interpolated = cubicHermiteInterpolation(y0, y1, y2, y3, nx - n1);
	float windowValue = windowCurve[j];
	float2 windowed = (float2)(interpolated * windowValue, 0.0f);

	out[index] = complexMultiply(windowed, phaseCartesian[j]);
}

__kernel void klinearizationLanczosAndWindowingAndDispersionCompensation(
	__global float2* out,
	__global const float2* in,
	__global const float* resampleCurve,
	__global const float* windowCurve,
	__global const float2* phaseCartesian,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int j = index % width;
	int offset = index - j;

	float x = resampleCurve[j];
	int center = (int)floor(x);
	const int a = 3;

	float sum = 0.0f;
	for (int k = center - a + 1; k <= center + a; k++) {
		int idx = clamp(k, 0, width - 1);
		sum += in[offset + idx].x * lanczosKernel(x - k, a);
	}

	float windowValue = windowCurve[j];
	float2 windowed = (float2)(sum * windowValue, 0.0f);

	out[index] = complexMultiply(windowed, phaseCartesian[j]);
}

// ============================================
// Post-Processing Kernels
// ============================================

__kernel void postProcessTruncateLog(
	__global float* output,
	__global const float2* input,
	const float max,
	const float min,
	const float addend,
	const float multiplicator,
	const int width_in,
	const int width_out,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	//	Calculate input array index (accounts for stride differences)
	int lineIndex = index / width_out;
	int posInAscan = index % width_out;
	int inputArrayIndex = lineIndex * width_in + posInAscan;

	//	Note: log scaling: log(sqrt(x*x+y*y)) == 0.5*log(x*x+y*y)
	//	The calculation below is 20*log(magnitude) and not 10*log...
	//	Note fft normalization: (1/(2*width_out)) is the FFT normalization factor.
	//	In addition a multiplication by 2 is performed since the acquired OCT raw signal is a real valued signal,
	//	so (1/(2*width_out)) becomes 1/width_out.
	float realComponent = input[inputArrayIndex].x;
	float imaginaryComponent = input[inputArrayIndex].y;
	float magSquared = realComponent * realComponent + imaginaryComponent * imaginaryComponent;

	//	10 * log10(mag^2 / width_out) with FFT normalization
	output[index] = multiplicator * ((((10.0f * log10(magSquared / (float)width_out)) - min) / (max - min)) + addend);
}

__kernel void postProcessTruncateLin(
	__global float* output,
	__global const float2* input,
	const float max,
	const float min,
	const float addend,
	const float multiplicator,
	const int width_in,
	const int width_out,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	//	Calculate input array index (accounts for stride differences)
	int lineIndex = index / width_out;
	int posInAscan = index % width_out;
	int inputArrayIndex = lineIndex * width_in + posInAscan;

	//	Calculate amplitude with FFT normalization
	float realComponent = input[inputArrayIndex].x;
	float imaginaryComponent = input[inputArrayIndex].y;
	float magnitude = sqrt(realComponent * realComponent + imaginaryComponent * imaginaryComponent) / width_out;

	//	Range normalization: (magnitude - min) / (max - min), then scale
	output[index] = multiplicator * ((((magnitude) - min) / (max - min)) + addend);
}

// ============================================
// Fixed-Pattern Noise Removal Kernels
// ============================================

__kernel void getMinimumVarianceMean(
	__global float2* meanALine,
	__global const float2* input,
	const int width,
	const int height,
	const int segments)
{
	int index = get_global_id(0);
	if (index >= width) return;

	int segWidth = height / segments;
	int stride = width;
	float factor = 1.0f / segWidth;

	float minVariance = FLT_MAX;
	float2 meanAtMinVariance = (float2)(0.0f, 0.0f);

	for (int i = 0; i < segments; i++) {
		int offset = i * segWidth * stride + index;

		float sumX = 0.0f, sumY = 0.0f;
		float sumXX = 0.0f;

		for (int j = 0; j < segWidth; j++) {
			float2 val = input[offset + j * stride];
			float dx = val.x;
			float dy = val.y;
			sumX += dx;
			sumY += dy;
			sumXX += dx * dx + dy * dy;
		}

		float meanX = sumX * factor;
		float meanY = sumY * factor;
		float variance = (sumXX * factor) - (meanX * meanX + meanY * meanY);

		if (variance < minVariance) {
			minVariance = variance;
			meanAtMinVariance.x = meanX;
			meanAtMinVariance.y = meanY;
		}
	}

	meanALine[index] = meanAtMinVariance;
}

__kernel void meanALineSubtraction(
	__global float2* in_out,
	__global const float2* meanLine,
	const int width,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples) return;

	int meanLineIndex = index % width;
	int lineIndex = index / width;
	int volumeArrayIndex = lineIndex * width + index;

	//	Subtract mean line from positive-depth part of each A-scan
	float2 meanValue = meanLine[meanLineIndex];
	in_out[volumeArrayIndex] -= meanValue;
}

// ============================================
// B-scan Operations
// ============================================

__kernel void bscanFlip(
	__global float* data,
	__global const float* unused,
	const int samplesPerAscan,
	const int ascansPerBscan,
	const int samplesPerBscan,
	const int halfSamplesInVolume)
{
	int index = get_global_id(0);
	if (index >= halfSamplesInVolume) return;

	//	Multiplication by 2 gets us just even bscanIndex-values (0, 2, 4, 6, ...)
	//	This is necessary because we just want to flip every second Bscan
	int bscanIndex = (index / samplesPerBscan) * 2;

	//	Recalculation of index is necessary here to skip every second Bscan
	index = bscanIndex * samplesPerBscan + index % samplesPerBscan;

	int sampleIndex = index % samplesPerBscan;
	int ascanIndex = sampleIndex / samplesPerAscan;
	int mirrorIndex = bscanIndex * samplesPerBscan +
	                  ((ascansPerBscan - 1) - ascanIndex) * samplesPerAscan +
	                  (sampleIndex % samplesPerAscan);

	//	Only process half the A-scans to avoid double-swapping
	if (ascanIndex >= ascansPerBscan / 2) {
		float tmp = data[mirrorIndex];
		data[mirrorIndex] = data[index];
		data[index] = tmp;
	}
}

// ============================================
// Sinusoidal Scan Correction Kernels
// ============================================

__kernel void fillSinusoidalScanCorrectionCurve(
	__global float* sinusoidalResampleCurve,
	const int length)
{
	int index = get_global_id(0);
	if (index < length) {
		sinusoidalResampleCurve[index] = ((float)length/M_PI)*acos((float)(1.0-((2.0*(float)index)/(float)length)));
	}
}

__kernel void sinusoidalScanCorrection(
	__global float* out,
	__global const float* in,
	__global const float* resampleCurve,
	const int width,
	const int linesPerBscan,
	const int numBscans,
	const int samples)
{
	int index = get_global_id(0);
	if (index >= samples - width) return;

	//	Match CUDA implementation exactly
	int j = index % width;                      //	pos within ascan
	int k = (index / width) % linesPerBscan;    //	pos within bscan
	int l = index / (width * linesPerBscan);    //	pos within buffer

	float n_sinusoidal = resampleCurve[k];
	float x = n_sinusoidal;
	int x0 = (int)x * width + j + l * width * linesPerBscan;
	int x1 = x0 + width;

	float f_x0 = in[x0];
	float f_x1 = in[x1];

	out[index] = f_x0 + (f_x1 - f_x0) * (x - (int)x);
}

// ============================================
// Post-Process Background Subtraction Kernels
// ============================================

__kernel void getPostProcessBackground(
	__global float* backgroundLine,
	__global const float* input,
	const int width,
	const int height)
{
	int j = get_global_id(0);
	if (j >= width) return;

	float sum = 0.0f;
	for (int line = 0; line < height; line++) {
		sum += input[line * width + j];
	}

	backgroundLine[j] = sum / (float)height;
}

__kernel void postProcessBackgroundSubtraction(
	__global float* data,
	__global const float* background,
	const float backgroundWeight,
	const float backgroundOffset,
	const int samplesPerAscan,
	const int samplesPerBuffer)
{
	int index = get_global_id(0);
	if (index >= samplesPerBuffer) return;

	//	In-place modification: subtract background from data
	data[index] = data[index] - (backgroundWeight * background[index % samplesPerAscan] + backgroundOffset);
}
