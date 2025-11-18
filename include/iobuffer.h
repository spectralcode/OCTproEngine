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

private:
	void* dataPtr;
	size_t sizeInBytes;
	DataType dataType;
	uint64_t bufferId;
};

} // namespace ope

#endif // OPE_IOBUFFER_H