#ifndef OPE_IOBUFFER_H
#define OPE_IOBUFFER_H

#include <cstddef>
#include <cstdint>
#include "types.h"
#include "export.h"

namespace ope {

class OPE_API IOBuffer {
public:
	using DataType = ope::DataType;

	// Allocation hints for platform-specific memory optimizations
	enum class AllocationHint {
		DEFAULT,         // Platform default allocation
		DEVICE_MAPPED,   // Device-mapped memory (zero-copy on integrated GPUs like Jetson)
		PORTABLE         // Portable pinned memory (optimized for PCIe transfers)
	};

	IOBuffer();
	~IOBuffer();

	bool allocateMemory(size_t sizeInBytes);
	void releaseMemory();

	void setDataType(DataType type);
	DataType getDataType() const;
	int getBitDepth() const;

	void* getDataPointer();
	const void* getDataPointer() const;

	size_t getSizeInBytes() const;

	// Buffer ID for correlating raw and processed data
	void setBufferId(uint64_t id);
	uint64_t getBufferId() const;

	// Backend buffer index (for internal backend use to map IOBuffer to staging buffers/command buffers)
	void setBackendIndex(int index);
	int getBackendIndex() const;

	// Allocation hint (must be set before allocateMemory)
	void setAllocationHint(AllocationHint hint);
	AllocationHint getAllocationHint() const;

	// Zero-copy external memory support (for direct GPU staging buffer access)
	// External memory is NOT owned by IOBuffer and will NOT be freed on destruction
	// todo: think about a better way to handle this. 
	// find a simple and clean way to handle memory for all backends in a unified way and not to have special cases like this here for vulkan
	void setExternalMemory(void* ptr, size_t size);
	bool isUsingExternalMemory() const;

private:
	void* dataPtr;
	size_t sizeInBytes;
	DataType dataType;
	uint64_t bufferId;
	int backendIndex = -1;  // Internal backend index (-1 = unset)
	AllocationHint allocationHint;
	bool externalMemory = false;  // If true, dataPtr points to external memory (don't free)
};

} // namespace ope

#endif // OPE_IOBUFFER_H