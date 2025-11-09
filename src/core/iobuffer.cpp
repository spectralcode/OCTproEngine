#include "../../include/iobuffer.h"

#if defined(_WIN32) //windows replacement for posix_memalign
	#include <conio.h>
	#include <Windows.h>
	#include <errno.h>
	static inline int _posix_memalign_wrapper(void** p, size_t a, size_t s) {
		*p = _aligned_malloc(s, a);
		return (*p) ? 0 : ENOMEM;
	}
	#define posix_memalign(p, a, s) _posix_memalign_wrapper((p), (a), (s))
	#define posix_memalign_free _aligned_free
#elif defined(__aarch64__) //jetson nano 
	#include <cuda_runtime.h>
	#include <stdlib.h>
	#include <errno.h>
	static inline int _posix_memalign_wrapper(void** p, size_t a, size_t s) {
		(void)a; //alignment parameter ignored by cudaHostAlloc
		cudaError_t err;
		#ifdef ENABLE_CUDA_ZERO_COPY
			err = cudaHostAlloc(p, s, cudaHostAllocMapped);
		#else
			err = cudaHostAlloc(p, s, cudaHostAllocPortable);
		#endif
		return (err == cudaSuccess) ? 0 : ENOMEM;
	}
	#define posix_memalign(p, a, s) _posix_memalign_wrapper((p), (a), (s))
	#define posix_memalign_free(p) cudaFreeHost((p))
#else //default posix_memalign for linux
	#include <stdlib.h>
	#define posix_memalign_free free
#endif

namespace ope {
IOBuffer::IOBuffer()
	: dataPtr(nullptr), 
	sizeInBytes(0), 
	dataType(DataType::UINT8)
{
}

IOBuffer::~IOBuffer() {
	this->releaseMemory();
}

bool IOBuffer::allocateMemory(size_t sizeInBytes) {
	this->releaseMemory();

	if (sizeInBytes == 0) {
		return true;
	}

	const size_t alignment = 64;
	if (posix_memalign(&this->dataPtr, alignment, sizeInBytes) != 0) {
		return false;
	}

	memset(this->dataPtr, 0, sizeInBytes);

	this->sizeInBytes = sizeInBytes;
	return true;
}

void IOBuffer::releaseMemory() {
	if (this->dataPtr) {
		posix_memalign_free(this->dataPtr);
		this->dataPtr = nullptr;
		this->sizeInBytes = 0;
	}
}


void IOBuffer::setDataType(DataType type) {
	this->dataType = type;
}

IOBuffer::DataType IOBuffer::getDataType() const {
	return this->dataType;
}	

int IOBuffer::getBitDepth() const {
	switch (this->dataType) {
	case DataType::UINT8:
	case DataType::INT8:
		return 8;
	case DataType::UINT16:
	case DataType::INT16:
		return 16;
	case DataType::UINT32:
	case DataType::INT32:
	case DataType::FLOAT32:
		return 32;
	case DataType::UINT64:
	case DataType::INT64:
	case DataType::FLOAT64:
		return 64;
	case DataType::COMPLEX_FLOAT32:
		return 64;
	case DataType::COMPLEX_FLOAT64:
		return 128;
	default:
		return 0;
	}
}

void* IOBuffer::getDataPointer() {
	return this->dataPtr;
}

const void* IOBuffer::getDataPointer() const {
	return dataPtr;
}

size_t IOBuffer::getSizeInBytes() const {
	return this->sizeInBytes;
}

} // namespace ope

