/**
**  This file is part of OCTproEngine.
**  CUDA kernels for optical coherence tomography signal processing
**  Originally created for OCTproZ
**  Adapted for OCTproEngine library architecture
**
**  Copyright (C) 2019-2025 Miroslav Zabic
**  
**/

#include "cuda_kernels.h"
#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace ope {
namespace cuda_kernels {

__global__ void inputToCufftComplex(cufftComplex* __restrict__ output,
                                   const void* __restrict__ input,
                                   int width_out,
                                   int width_in,
                                   int inputBitdepth,
                                   int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(inputBitdepth <= 8){
		unsigned char* in = (unsigned char*)input;
		output[index].x = __uint2float_rd(in[index]);
	}else if(inputBitdepth > 8 && inputBitdepth <= 16){
		unsigned short* in = (unsigned short*)input;
		output[index].x = __uint2float_rd(in[index]);
	}else{
		unsigned int* in = (unsigned int*)input;
		output[index].x = __uint2float_rd(in[index]);
	}
	output[index].y = 0;
}

__global__ void inputToCufftComplex_and_bitshift(cufftComplex* __restrict__ output,
                                                const void* __restrict__ input,
                                                int width_out,
                                                int width_in,
                                                int inputBitdepth,
                                                int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(inputBitdepth <= 8){
		unsigned char* in = (unsigned char*)input;
		output[index].x = __uint2float_rd(in[index] >> 4);
	}else if(inputBitdepth > 8 && inputBitdepth <= 16){
		unsigned short* in = (unsigned short*)input;
		output[index].x = __uint2float_rd(in[index] >> 4);
	}else{
		unsigned int* in = (unsigned int*)input;
		output[index].x = (in[index])/4294967296.0;
	}
	output[index].y = 0;
}

//device functions for endian byte swap //todo: check if big endian to little endian conversion may be needed and extend inputToCufftComplex kernel if necessary
inline __device__ uint32_t endianSwapUint32(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | ((val >> 16));
}
inline __device__ int32_t endianSwapInt32(int32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF );
	return (val << 16) | ((val >> 16) & 0xFFFF);
}
inline __device__ uint16_t endianSwapUint16(uint16_t val) {
	return (val << 8) | (val >> 8 );
}
inline __device__ int16_t endianSwapInt16(int16_t val) {
	return (val << 8) | ((val >> 8) & 0xFF);
}

__global__ void rollingAverageBackgroundRemoval(cufftComplex* __restrict__ out,
                                              const cufftComplex* __restrict__ in,
                                              const int rollingAverageWindowSize,
                                              const int width, //samplesPerAscan
                                              const int height, //ascansPerBscan
                                              const int samplesPerFrame,
                                              const int samples) //total number of samples in buffer
                                              {
	extern __shared__ float s_data[];

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < samples) {
		int currentBscan = index / samplesPerFrame;
		int currentLine = (index / width) % height;
		int firstIndexOfCurrentLine = currentLine * width + (samplesPerFrame * currentBscan);
		int lastIndexOfCurrentLine = firstIndexOfCurrentLine + width - 1;

		int startIdx = max(firstIndexOfCurrentLine, index - rollingAverageWindowSize + 1);
		int endIdx = min(lastIndexOfCurrentLine, index + rollingAverageWindowSize);
		int windowSize = endIdx - startIdx + 1;

		//load data into shared memory for this line segment
		//first determine the range of data this block will process
		int blockFirstIdx = blockIdx.x * blockDim.x;
		int blockStartIdx = max(0, blockFirstIdx - rollingAverageWindowSize + 1);
		int blockEndIdx = min(samples-1, (blockFirstIdx + blockDim.x - 1) + rollingAverageWindowSize);

		//load data collaboratively (each thread loads one or more elements)
		for (int i = blockStartIdx + threadIdx.x; i <= blockEndIdx ; i += blockDim.x) {
				s_data[i - blockStartIdx] = in[i].x;
		}

		//ensure all data is loaded before proceeding
		__syncthreads();

		//calculate rolling average using shared memory
		float rollingSum = 0.0f;
		for (int i = startIdx; i <= endIdx; i++) {
			rollingSum += s_data[i - blockStartIdx];
		}

		float rollingAverage = rollingSum / windowSize;
		out[index].x = in[index].x - rollingAverage;
		out[index].y = 0;
	}
}

__global__ void klinearization(cufftComplex* __restrict__ out,
                              const cufftComplex* __restrict__ in,
                              const float* __restrict__ resampleCurve,
                              const int width,
                              const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;

	out[index].x = f_x0 + (f_x1 - f_x0) * (x - x0);
	out[index].y = 0;
}

__global__ void klinearizationQuadratic(cufftComplex* __restrict__ out,
                                       const cufftComplex* __restrict__ in,
                                       const float* __restrict__ resampleCurve,
                                       const int width,
                                       const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float x = resampleCurve[j];
	int x0 = (int)x;
	int x1 = x0 + 1;
	int x2 = x0 + 2;

	float f_x0 = in[offset + x0].x;
	float f_x1 = in[offset + x1].x;
	float f_x2 = in[offset + x2].x;
	float b0 = f_x0;
	float b1 = f_x1-f_x0;
	float b2 = ((f_x2-f_x1)-b1)/(x2-x0);

	out[index].x = b0 + b1 * (x - x0) + b2*(x-x0)*(x-x1);
	out[index].y = 0;
}

__forceinline__ __device__ float cubicHermiteInterpolation(const float y0,
                                                          const float y1,
                                                          const float y2,
                                                          const float y3,
                                                          const float positionBetweenY1andY2) {
	const float a = -y0 + 3.0f*(y1-y2) + y3;
	const float b = 2.0f*y0 - 5.0f*y1 + 4.0f*y2 - y3;
	const float c = -y0 + y2;

	const float pos = positionBetweenY1andY2;
	const float pos2 = pos*pos;

	return 0.5f*pos*(a * pos2 + b * pos + c) + y1;
}

__global__ void klinearizationCubic(cufftComplex* __restrict__ out,
                                   const cufftComplex* __restrict__ in,
                                   const float* __restrict__ resampleCurve,
                                   const int width,
                                   const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float nx = resampleCurve[j];
	const int n1 = (int)nx;
	int n0 = abs(n1 - 1); //using abs() to avoid negative n0 because offset can be 0 and out of bounds memory access may occur
	int n2 = n1 + 1;
	int n3 = n2 + 1; //we do not need to worry here about out of bounds memory access as the resampleCurve is restricted to values that avoid out of bound memory acces in resample kernels

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;

	out[index].x = cubicHermiteInterpolation(y0,y1,y2,y3,nx-n1);
	out[index].y = 0;
}

inline __device__ float lanczosKernel8(const float x) {
	const float absX = fabsf(x);
	const float sincX = sinf(PI*absX)/(PI*absX);
	const float sincXOver8 = sinf(PI_OVER_8*absX)/(PI_OVER_8*absX);
	return (absX < 0.00001f) ? 1.0f :(sincX * sincXOver8);
}

__global__ void klinearizationLanczos(cufftComplex* __restrict__ out,
                                          const cufftComplex* __restrict__ in,
                                          const float* __restrict__ resampleCurve,
                                          const int width,
                                          const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	int j = index%width;
	int offset = index-j;
	//offset = min(samples-9, max(offset, 8)); //no need for offset calculation to aviod out of bounds here because resampleCurve is already restricted to valid range

	const float nx = resampleCurve[j];
	const int n0 = (int)nx;
	float sum = 0.0f;

	#pragma unroll
	for (int i = -7; i <= 8; i++) {
		float y = in[offset + (n0 + i)].x;
		sum += y * lanczosKernel8(nx - (n0 + i));
	}

	out[index].x = sum;
	out[index].y = 0;
}

__global__ void windowing(cufftComplex* output,
                         const cufftComplex* input,
                         const float* __restrict__ window,
                         const int lineWidth,
                         const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int line_index = index % lineWidth;
		output[index].x = input[index].x * window[line_index];
		output[index].y = 0;
	}
}

__global__ void klinearizationAndWindowing(cufftComplex* __restrict__ out,
                                          const cufftComplex* __restrict__ in,
                                          const float* __restrict__ resampleCurve,
                                          const float* __restrict__ window,
                                          const int width,
                                          const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float n_m = resampleCurve[j];
	int n1 = (int)n_m;
	int n2 = n1 + 1;

	float inN1 = in[offset + n1].x;
	float inN2 = in[offset + n2].x;

	out[index].x = (inN1 + (inN2 - inN1) * (n_m - n1)) * window[j];
	out[index].y = 0;
}

__global__ void klinearizationCubicAndWindowing(cufftComplex* __restrict__ out,
                                               const cufftComplex* __restrict__ in,
                                               const float* __restrict__ resampleCurve,
                                               const float* __restrict__ window,
                                               const int width,
                                               const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float nx = resampleCurve[j];
	const int n1 = (int)nx;
	int n0 = abs(n1 - 1);
	int n2 = n1 + 1;
	int n3 = n2 + 1;

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;
	float pos = nx-n1;

	out[index].x = cubicHermiteInterpolation(y0,y1,y2,y3,pos) * window[j];
	out[index].y = 0;
}

__global__ void klinearizationLanczosAndWindowing(cufftComplex* __restrict__ out,
                                                const cufftComplex* __restrict__ in,
                                                const float* __restrict__ resampleCurve,
                                                const float* __restrict__ window,
                                                const int width,
                                                const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;
	//offset = min(samples-9, max(offset, 8)); //no need for offset calculation to aviod out of bounds here because resampleCurve is already restricted to valid range

	const float nx = resampleCurve[j];
	const int n0 = (int)nx;
	float sum = 0.0f;

	#pragma unroll
	for (int i = -7; i <= 8; i++) {
		float y = in[offset + (n0 + i)].x;
		sum += y * lanczosKernel8(nx - (n0 + i));
	}

	out[index].x = sum * window[j];
	out[index].y = 0;
}

__global__ void klinearizationAndWindowingAndDispersionCompensation(cufftComplex* __restrict__ out,
                                                                   const cufftComplex* __restrict__ in,
                                                                   const float* __restrict__ resampleCurve,
                                                                   const float* __restrict__ window,
                                                                   const cufftComplex* __restrict__ phaseComplex,
                                                                   const int width,
                                                                   const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float n_m = resampleCurve[j];
	int n1 = (int)n_m;
	int n2 = n1 + 1;

	float inN1 = in[offset + n1].x;
	float inN2 = in[offset + n2].x;

	float linearizedAndWindowedInX = (inN1 + (inN2 - inN1) * (n_m - n1)) * window[j];
	out[index].x = linearizedAndWindowedInX * phaseComplex[j].x;
	out[index].y = linearizedAndWindowedInX * phaseComplex[j].y;
}

__global__ void klinearizationCubicAndWindowingAndDispersionCompensation(cufftComplex* __restrict__ out,
                                                                       const cufftComplex* __restrict__ in,
                                                                       const float* __restrict__ resampleCurve,
                                                                       const float* __restrict__ window,
                                                                       const cufftComplex* __restrict__ phaseComplex,
                                                                       const int width,
                                                                       const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;

	float nx = resampleCurve[j];
	int n1 = (int)nx;
	int n0 = abs(n1 - 1);
	int n2 = n1 + 1;
	int n3 = n2 + 1;

	float y0 = in[offset + n0].x;
	float y1 = in[offset + n1].x;
	float y2 = in[offset + n2].x;
	float y3 = in[offset + n3].x;
	float pos = nx-n1;

	float linearizedAndWindowedInX = cubicHermiteInterpolation(y0,y1,y2,y3,pos) * window[j];
	out[index].x = linearizedAndWindowedInX * phaseComplex[j].x;
	out[index].y = linearizedAndWindowedInX * phaseComplex[j].y;
}

__global__ void klinearizationLanczosAndWindowingAndDispersionCompensation(cufftComplex* __restrict__ out,
                                                                        const cufftComplex* __restrict__ in,
                                                                        const float* __restrict__ resampleCurve,
                                                                        const float* __restrict__ window,
                                                                        const cufftComplex* __restrict__ phaseComplex,
                                                                        const int width,
                                                                        const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j = index%width;
	int offset = index-j;
	//offset = min(samples-9, max(offset, 8)); //no need for offset calculation to aviod out of bounds here because resampleCurve is already restricted to valid range

	const float nx = resampleCurve[j];
	const int n0 = (int)nx;
	float sum = 0.0f;

	#pragma unroll
	for (int i = -7; i <= 8; i++) {
		float y = in[offset + (n0 + i)].x;
		sum += y * lanczosKernel8(nx - (n0 + i));
	}

	float linearizedAndWindowedInX = sum * window[j];
	out[index].x = linearizedAndWindowedInX * phaseComplex[j].x;
	out[index].y = linearizedAndWindowedInX * phaseComplex[j].y;
}

__global__ void sinusoidalScanCorrection(float* __restrict__ out,
                                        const float* __restrict__ in,
                                        const float* __restrict__ sinusoidalResampleCurve,
                                        const int width, //samplesPerAscan
                                        const int height,  //ascansPerBscan
                                        const int depth, //bscansPerBuffer
                                        const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < samples-width){
		int j = index%(width); //pos within ascan
		int k = (index/width)%height; //pos within bscan
		int l = index/(width*height); //pos within buffer

		float n_sinusoidal = sinusoidalResampleCurve[k];
		float x = n_sinusoidal;
		int x0 = (int)x*width+j+l*width*height;
		int x1 = x0 + width;

		float f_x0 = in[x0];
		float f_x1 = in[x1];

		out[index] = f_x0 + (f_x1 - f_x0) * (x - (int)(x));
	}
}

__global__ void fillSinusoidalScanCorrectionCurve(float* sinusoidalResampleCurve,  const int length) {
	int index = blockIdx.x;
	if (index < length) {
		sinusoidalResampleCurve[index] = ((float)length/M_PI)*acos((float)(1.0-((2.0*(float)index)/(float)length)));
	}
}

__global__ void getMinimumVarianceMean(cufftComplex* __restrict__ meanLine,
                                      const cufftComplex* __restrict__ in,
                                      const int width,
                                      const int height,
                                      const int segs) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= width) return;

	int segWidth = height / segs;
	int stride = width;
	float factor = 1.0f / segWidth;

	float minVariance = FLT_MAX;
	cufftComplex meanAtMinVariance = {0.0f, 0.0f};

	for (int i = 0; i < segs; i++) {
		int offset = i * segWidth * stride + index;

		float sumX = 0.0f, sumY = 0.0f;
		float sumXX = 0.0f;

		for (int j = 0; j < segWidth; j++) {
			cufftComplex val = in[offset + j * stride];
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

	meanLine[index] = meanAtMinVariance;
}

__global__ void meanALineSubtraction(cufftComplex* __restrict__ in_out,
                                    const cufftComplex* __restrict__ meanLine,
                                    const int width,
                                    const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int meanLineIndex = index % width;
		int lineIndex = index / width;
		int volumeArrayIndex = lineIndex * width + index;
		//in_out contains data after IFFT with lines that have both positive and negative depths.
		//the volumeArrayIndex points to the positive-depth (first half) of each line
		//since width is the number of positive samples in one line and index only spans (total samples in buffer)/2,
		//this subtracts the mean line only from the positive-depth part of each line in the buffer.
		cufftComplex meanValue = meanLine[meanLineIndex];
		in_out[volumeArrayIndex].x -= meanValue.x;
		in_out[volumeArrayIndex].y -= meanValue.y;
	}
}

__device__ cufftComplex cuMultiply(const cufftComplex& a, const cufftComplex& b) {
	cufftComplex result;
	result.x = a.x*b.x - a.y*b.y;
	result.y = a.x*b.y + a.y*b.x;
	return result;
}

__global__ void dispersionCompensation(cufftComplex* out,
                                      const cufftComplex* in,
                                      const cufftComplex* __restrict__ phaseComplex,
                                      const int width,
                                      const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int phaseIndex = index%width;
		//because in[].y is always zero we can omit full complex multiplication and just multiply in[].x
		//for full multiplication the device kernel "cuMultiply" can be used
		float inX = in[index].x;
		out[index].x = inX * phaseComplex[phaseIndex].x;
		out[index].y = inX * phaseComplex[phaseIndex].y;
	}
}

__global__ void dispersionCompensationAndWindowing(cufftComplex* out,
                                                  const cufftComplex* in,
                                                  const cufftComplex* __restrict__ phaseComplex,
                                                  const float* __restrict__ window,
                                                  const int width,
                                                  const int samples) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples) {
		int lineIndex = index%width;
		float inX = in[index].x * window[lineIndex];
		out[index].x = inX * phaseComplex[lineIndex].x;
		out[index].y = inX * phaseComplex[lineIndex].y;
	}
}

__global__ void fillDispersivePhase(cufftComplex* __restrict__ phaseComplex,
                                   const float* __restrict__ dispersionCurve,
                                   float factor,
                                   int size,
                                   int direction) {
	int index = blockIdx.x;
	if (index < size) {
		phaseComplex[index].x = cosf(factor*dispersionCurve[index]);
		phaseComplex[index].y = sinf(factor*dispersionCurve[index]) * direction;
	}
}

__global__ void cuda_bscanFlip(float* output,
                             const float* input,
                             int samplesPerAscan,
                             int ascansPerBscan,
                             int samplesPerBscan,
                             int halfSamplesInVolume) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < halfSamplesInVolume) {
		int bscanIndex = (index / samplesPerBscan)*2; //multiplication by 2 gets us just even bscanIndex-values (0, 2, 4, 6, ...) This is necessary because we just want to flip every second Bscan.
		index = bscanIndex*samplesPerBscan + index%samplesPerBscan; //recalculation of index is necessary here to skip every second Bscan
		int sampleIndex = index % samplesPerBscan;
		int ascanIndex = sampleIndex / samplesPerAscan;
		int mirrorIndex = bscanIndex*samplesPerBscan + ((ascansPerBscan - 1) - ascanIndex)*samplesPerAscan + (sampleIndex%samplesPerAscan);

		if (ascanIndex >= ascansPerBscan / 2) {
			float tmp = input[mirrorIndex];
			output[mirrorIndex] = input[index];
			output[index] = tmp;
		}
	}
}

//Removes half of each processed A-scan (the mirror artefacts), logarithmizes each value of magnitude of remaining A-scan and copies it into an output array. This output array can be used to display the processed OCT data.
__global__ void postProcessTruncateLog(float* __restrict__ output,
                                       const cufftComplex* __restrict__ input,
                                       const int outputAscanLength,
                                       const int samples,
                                       const int bufferNumberInVolume,
                                       const float max,
                                       const float min,
                                       const float addend,
                                       const float coeff) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples / 2) {
		int lineIndex = index / outputAscanLength;
		int inputArrayIndex = lineIndex *outputAscanLength + index;

		//Note log scaling: log(sqrt(x*x+y*y)) == 0.5*log(x*x+y*y) --> the calculation in the code below is 20*log(magnitude) and not 10*log...
		//Note fft normalization://(1/(2*outputAscanLength)) is the FFT normalization factor. In addition a multiplication by 2 is performed since the acquired OCT raw signal is a real valued signal, so (1/(2*outputAscanLength)) becomes 1/outputAscanLength. (Why multiply by 2: FFT of a real-valued signal is a complex-valued signal with a symmetric spectrum, where the positive and negative frequency components are identical in magnitude. And since the signal is truncated (negative or positive frequency components are removed), doubling of the remaining components is performed here to preserve the total energy of the signal after truncation)
		//amplitude:
		float realComponent = input[inputArrayIndex].x;
		float imaginaryComponent = input[inputArrayIndex].y;
		output[index] = coeff*((((10.0f*log10f(((realComponent*realComponent) + (imaginaryComponent*imaginaryComponent))/(outputAscanLength))) - min) / (max - min)) + addend);
	}
}

//Removes half of each processed A-scan (the mirror artefacts), calculates magnitude of remaining A-scan and copies it into an output array. This output array can be used to display the processed OCT data.
__global__ void postProcessTruncateLin(float* __restrict__ output,
                                      const cufftComplex* __restrict__ input,
                                      const int outputAscanLength,
                                      const int samples,
                                      const float max,
                                      const float min,
                                      const float addend,
                                      const float coeff) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samples / 2) {
		int lineIndex = index / outputAscanLength;
		int inputArrayIndex = lineIndex * outputAscanLength + index;

		//amplitude:
		float realComponent = input[inputArrayIndex].x;
		float imaginaryComponent = input[inputArrayIndex].y;
		output[index] = coeff * ((((sqrt((realComponent*realComponent) + (imaginaryComponent*imaginaryComponent))/(outputAscanLength)) - min) / (max - min)) + addend);
	}
}

__global__ void getPostProcessBackground(float* __restrict__ output,
                                        const float* __restrict__ input,
                                        const int samplesPerAscan,
                                        const int ascansPerBuffer) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samplesPerAscan) {
		float sum = 0;
		for (int i = 0; i < ascansPerBuffer; i++){
			sum += input[index+i*samplesPerAscan];
		}
		output[index] = sum/ascansPerBuffer;
	}
}

__global__ void postProcessBackgroundSubtraction(float* data,
                                           const float* __restrict__ background,
                                           const float backgroundWeight,
                                           const float backgroundOffset,
                                           const int samplesPerAscan,
                                           const int samplesPerBuffer) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < samplesPerBuffer) {
		//data[index] = __saturatef(data[index] - (backgroundWeight * background[index%samplesPerAscan] + backgroundOffset)); //saturatef only needed if we want later to convert to original integer datatype (uchar, ushort, etc)
		data[index] = data[index] - (backgroundWeight * background[index%samplesPerAscan] + backgroundOffset);
	}
}


} // namespace cuda_kernels
} // namespace ope
