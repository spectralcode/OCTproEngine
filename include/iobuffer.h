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

	// Allocation hint (must be set before allocateMemory)
	void setAllocationHint(AllocationHint hint);
	AllocationHint getAllocationHint() const;

private:
	void* dataPtr;
	size_t sizeInBytes;
	DataType dataType;
	uint64_t bufferId;
	AllocationHint allocationHint;
};

} // namespace ope

#endif // OPE_IOBUFFER_H