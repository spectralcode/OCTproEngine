#include "../../include/types.h"

namespace ope {

int getDataTypeBitDepth(DataType type) {
	switch (type) {
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

int getDataTypeByteSize(DataType type) {
	switch (type) {
	case DataType::UINT8:
	case DataType::INT8:
		return 1;
	case DataType::UINT16:
	case DataType::INT16:
		return 2;
	case DataType::UINT32:
	case DataType::INT32:
	case DataType::FLOAT32:
		return 4;
	case DataType::UINT64:
	case DataType::INT64:
	case DataType::FLOAT64:
		return 8;
	case DataType::COMPLEX_FLOAT32:
		return 8;
	case DataType::COMPLEX_FLOAT64:
		return 16;
	default:
		return 0;
	}
}

} // namespace ope