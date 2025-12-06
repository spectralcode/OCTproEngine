#include "../../include/iobuffer.h"
#include <cstring>

#if defined(_WIN32)
	#include <Windows.h>
#elif defined(__aarch64__)
	#include <cuda_runtime.h>
#else
	#include <stdlib.h>
#endif


static inline bool allocate_aligned(void** dataPointer, size_t sizeInBytes, ope::IOBuffer::AllocationHint hint) {
#if defined(_WIN32)
	(void)hint;
	*dataPointer = _aligned_malloc(sizeInBytes, 64);
	return (*dataPointer != nullptr);
#elif defined(__aarch64__)
	unsigned int cudaFlags = (hint == ope::IOBuffer::AllocationHint::DEVICE_MAPPED)
		? cudaHostAllocMapped
		: cudaHostAllocPortable;
	return cudaHostAlloc(dataPointer, sizeInBytes, cudaFlags) == cudaSuccess;
#else
	(void)hint;
	return posix_memalign(dataPointer, 64, sizeInBytes) == 0;
#endif
}

static inline void free_aligned(void* dataPointer) {
#if defined(_WIN32)
	_aligned_free(dataPointer);
#elif defined(__aarch64__)
	cudaFreeHost(dataPointer);
#else
	free(dataPointer);
#endif
}

namespace ope {
IOBuffer::IOBuffer()
	: dataPtr(nullptr),
	sizeInBytes(0),
	dataType(DataType::UINT8),
	bufferId(0),
	allocationHint(AllocationHint::DEFAULT)
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

	if (!allocate_aligned(&this->dataPtr, sizeInBytes, this->allocationHint)) {
		return false;
	}

	memset(this->dataPtr, 0, sizeInBytes);
	this->sizeInBytes = sizeInBytes;
	return true;
}

void IOBuffer::releaseMemory() {
	if (this->dataPtr) {
		if (!this->externalMemory) {
			free_aligned(this->dataPtr);
		}
		this->dataPtr = nullptr;
		this->sizeInBytes = 0;
		this->externalMemory = false;
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

void IOBuffer::setBufferId(uint64_t id) {
	this->bufferId = id;
}

uint64_t IOBuffer::getBufferId() const {
	return this->bufferId;
}

void IOBuffer::setBackendIndex(int index) {
	this->backendIndex = index;
}

int IOBuffer::getBackendIndex() const {
	return this->backendIndex;
}

void IOBuffer::setAllocationHint(AllocationHint hint) {
	this->allocationHint = hint;
}

IOBuffer::AllocationHint IOBuffer::getAllocationHint() const {
	return this->allocationHint;
}

//workaround to allow vulkan to use its own allocated memory
// todo: think about a better way to handle this
void IOBuffer::setExternalMemory(void* ptr, size_t size) { 
	this->releaseMemory();
	this->dataPtr = ptr;
	this->sizeInBytes = size;
	this->externalMemory = true;
}

bool IOBuffer::isUsingExternalMemory() const {
	return this->externalMemory;
}

} // namespace ope