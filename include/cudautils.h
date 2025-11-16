#ifndef OPE_CUDA_UTILS_H
#define OPE_CUDA_UTILS_H

#include <vector>
#include <string>
#include "export.h"

namespace ope {

struct OPE_API CudaDeviceInfo {
	int deviceId;
	std::string name;
	size_t totalMemory;
	size_t freeMemory;
	int computeCapabilityMajor;
	int computeCapabilityMinor;
	int maxThreadsPerBlock;
	int multiProcessorCount;
	bool isAvailable;

	std::string getComputeCapability() const;
};

class OPE_API CudaUtils {
public:
	static std::vector<CudaDeviceInfo> getAvailableDevices();
	static CudaDeviceInfo getDeviceInfo(int deviceId);
	static bool isDeviceAvailable(int deviceId);
	static int getDeviceCount();
	static bool isAvailable();
	static int getCurrentDevice();

private:
	// No instances allowed, static utility class only
	CudaUtils() = delete;
	~CudaUtils() = delete;
	CudaUtils(const CudaUtils&) = delete;
	CudaUtils& operator=(const CudaUtils&) = delete;
};

} // namespace ope

#endif // OPE_CUDA_UTILS_H
