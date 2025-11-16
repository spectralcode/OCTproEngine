#include "../../include/cudautils.h"
#include <stdexcept>

#ifdef OPE_CUDA_AVAILABLE
#include "../backends/cuda/cuda_backend.h"
#endif

namespace ope {

std::string CudaDeviceInfo::getComputeCapability() const {
	return std::to_string(computeCapabilityMajor) + "." +
	       std::to_string(computeCapabilityMinor);
}

std::vector<CudaDeviceInfo> CudaUtils::getAvailableDevices() {
#ifdef OPE_CUDA_AVAILABLE
	auto gpuDevices = CudaBackend::getAvailableDevices();
	std::vector<CudaDeviceInfo> devices;
	for (const auto& gpu : gpuDevices) {
		CudaDeviceInfo info;
		info.deviceId = gpu.deviceId;
		info.name = gpu.name;
		info.totalMemory = gpu.totalMemory;
		info.freeMemory = gpu.freeMemory;
		info.computeCapabilityMajor = gpu.computeCapabilityMajor;
		info.computeCapabilityMinor = gpu.computeCapabilityMinor;
		info.maxThreadsPerBlock = gpu.maxThreadsPerBlock;
		info.multiProcessorCount = gpu.multiProcessorCount;
		info.isAvailable = gpu.isAvailable;
		devices.push_back(info);
	}
	return devices;
#else
	return std::vector<CudaDeviceInfo>();
#endif
}

CudaDeviceInfo CudaUtils::getDeviceInfo(int deviceId) {
#ifdef OPE_CUDA_AVAILABLE
	auto gpu = CudaBackend::getDeviceInfo(deviceId);
	CudaDeviceInfo info;
	info.deviceId = gpu.deviceId;
	info.name = gpu.name;
	info.totalMemory = gpu.totalMemory;
	info.freeMemory = gpu.freeMemory;
	info.computeCapabilityMajor = gpu.computeCapabilityMajor;
	info.computeCapabilityMinor = gpu.computeCapabilityMinor;
	info.maxThreadsPerBlock = gpu.maxThreadsPerBlock;
	info.multiProcessorCount = gpu.multiProcessorCount;
	info.isAvailable = gpu.isAvailable;
	return info;
#else
	throw std::runtime_error("CUDA not available - OCTproEngine was compiled without CUDA support");
#endif
}

bool CudaUtils::isDeviceAvailable(int deviceId) {
#ifdef OPE_CUDA_AVAILABLE
	return CudaBackend::isDeviceAvailable(deviceId);
#else
	return false;
#endif
}

int CudaUtils::getDeviceCount() {
#ifdef OPE_CUDA_AVAILABLE
	return static_cast<int>(CudaBackend::getAvailableDevices().size());
#else
	return 0;
#endif
}

bool CudaUtils::isAvailable() {
#ifdef OPE_CUDA_AVAILABLE
	return getDeviceCount() > 0;
#else
	return false;
#endif
}

int CudaUtils::getCurrentDevice() {
#ifdef OPE_CUDA_AVAILABLE
	return CudaBackend::getCurrentDevice();
#else
	return -1;
#endif
}

} // namespace ope
