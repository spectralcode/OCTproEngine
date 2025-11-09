#ifndef OPE_TYPES_H
#define OPE_TYPES_H

namespace ope {

enum class DataType {
	UINT8 = 1,
	UINT16 = 2,
	UINT32 = 3,
	UINT64 = 4,
	INT8 = 5,
	INT16 = 6,
	INT32 = 7,
	INT64 = 8,
	FLOAT32 = 9,
	FLOAT64 = 10,
	COMPLEX_FLOAT32 = 11,
	COMPLEX_FLOAT64 = 12
};

int getDataTypeBitDepth(DataType type);

int getDataTypeByteSize(DataType type);

} // namespace ope

#endif // OPE_TYPES_H