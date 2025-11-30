#include "../../include/backendconfig.h"
#include <sstream>
#include <thread>

#ifdef OPE_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

#ifdef OPE_OPENCL_AVAILABLE
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifdef OPE_VULKAN_AVAILABLE
#include "../backends/vulkan/vulkan_backend.h"
#endif

namespace ope {

//	========================================
//	CudaConfig Implementation
//	========================================

std::string CudaConfig::toString() const {
	std::ostringstream oss;
	oss << "CudaConfig(device=" << deviceId;
	if (enableZeroCopy) {
		oss << ", zero_copy=true";
	}
	oss << ")";
	return oss.str();
}

//	========================================
//	OpenCLConfig Implementation
//	========================================

std::string OpenCLConfig::toString() const {
	std::ostringstream oss;
	oss << "OpenCLConfig(platform=" << platformId
		<< ", device=" << deviceId
		<< ", prefer_gpu=" << (preferGpu ? "true" : "false")
		<< ")";
	return oss.str();
}

//	========================================
//	VulkanConfig Implementation
//	========================================

std::string VulkanConfig::toString() const {
	std::ostringstream oss;
	oss << "VulkanConfig(device=" << deviceId << ")";
	return oss.str();
}

//	========================================
//	CpuConfig Implementation
//	========================================

std::string CpuConfig::toString() const {
	std::ostringstream oss;
	oss << "CpuConfig(threads=";
	if (numThreads == 0) {
		oss << "auto";
	} else {
		oss << numThreads;
	}
	oss << ", simd=" << (enableSimd ? "true" : "false") << ")";
	return oss.str();
}

//	========================================
//	BackendUtils Implementation
//	========================================

std::vector<DeviceInfo> BackendUtils::getCudaDevices() {
	std::vector<DeviceInfo> devices;

#ifdef OPE_CUDA_AVAILABLE
	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	if (error != cudaSuccess || deviceCount == 0) {
		return devices;
	}

	for (int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			DeviceInfo info;
			info.id = i;
			info.name = prop.name;
			info.totalMemory = prop.totalGlobalMem;
			info.computeCapabilityMajor = prop.major;
			info.computeCapabilityMinor = prop.minor;

			//	Try to get available memory
			size_t free = 0, total = 0;
			cudaSetDevice(i);
			if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
				info.availableMemory = free;
			} else {
				info.availableMemory = 0;
			}

			devices.push_back(info);
		}
	}
#endif

	return devices;
}

std::vector<DeviceInfo> BackendUtils::getOpenCLDevices() {
	std::vector<DeviceInfo> devices;

#ifdef OPE_OPENCL_AVAILABLE
	cl_uint num_platforms = 0;
	cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);

	if (err != CL_SUCCESS || num_platforms == 0) {
		return devices;
	}

	std::vector<cl_platform_id> platforms(num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

	if (err != CL_SUCCESS) {
		return devices;
	}

	int globalDeviceIndex = 0;
	for (cl_uint p = 0; p < num_platforms; ++p) {
		cl_uint num_devices = 0;
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);

		if (err != CL_SUCCESS || num_devices == 0) {
			continue;
		}

		std::vector<cl_device_id> platform_devices(num_devices);
		err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, platform_devices.data(), nullptr);

		if (err != CL_SUCCESS) {
			continue;
		}

		for (cl_uint d = 0; d < num_devices; ++d) {
			DeviceInfo info;
			info.id = globalDeviceIndex++;

			//	Get device name
			size_t size;
			clGetDeviceInfo(platform_devices[d], CL_DEVICE_NAME, 0, nullptr, &size);
			std::vector<char> name(size);
			clGetDeviceInfo(platform_devices[d], CL_DEVICE_NAME, size, name.data(), nullptr);
			info.name = std::string(name.data());

			//	Get vendor name
			clGetDeviceInfo(platform_devices[d], CL_DEVICE_VENDOR, 0, nullptr, &size);
			std::vector<char> vendor(size);
			clGetDeviceInfo(platform_devices[d], CL_DEVICE_VENDOR, size, vendor.data(), nullptr);
			info.vendorName = std::string(vendor.data());

			//	Get device version
			clGetDeviceInfo(platform_devices[d], CL_DEVICE_VERSION, 0, nullptr, &size);
			std::vector<char> version(size);
			clGetDeviceInfo(platform_devices[d], CL_DEVICE_VERSION, size, version.data(), nullptr);
			info.deviceVersion = std::string(version.data());

			//	Get memory info
			cl_ulong mem_size;
			clGetDeviceInfo(platform_devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);
			info.totalMemory = static_cast<size_t>(mem_size);

			clGetDeviceInfo(platform_devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(mem_size), &mem_size, nullptr);
			info.availableMemory = static_cast<size_t>(mem_size);

			devices.push_back(info);
		}
	}
#endif

	return devices;
}

std::vector<DeviceInfo> BackendUtils::getVulkanDevices() {
	std::vector<DeviceInfo> devices;

#ifdef OPE_VULKAN_AVAILABLE
	std::vector<VulkanDeviceInfo> vulkanDevices = VulkanBackend::getAvailableDevices();

	for (const auto& vkDev : vulkanDevices) {
		DeviceInfo info;
		info.id = vkDev.deviceId;
		info.name = vkDev.name;
		info.totalMemory = vkDev.totalMemory;
		info.availableMemory = vkDev.freeMemory;
		info.computeCapabilityMajor = static_cast<int>(vkDev.apiVersionMajor);
		info.computeCapabilityMinor = static_cast<int>(vkDev.apiVersionMinor);
		info.vendorName = "Vulkan";  // Generic vendor name
		info.deviceVersion = vkDev.getApiVersion();

		devices.push_back(info);
	}
#endif

	return devices;
}

DeviceInfo BackendUtils::getCpuInfo() {
	DeviceInfo info;
	info.id = 0;
	info.name = "CPU";
	info.vendorName = "Host System";

	//	Get number of threads
	unsigned int numThreads = std::thread::hardware_concurrency();
	if (numThreads == 0) {
		numThreads = 1;
	}

	std::ostringstream oss;
	oss << numThreads << " threads";
	info.deviceVersion = oss.str();

	//	CPU doesn't have dedicated memory like GPUs
	info.totalMemory = 0;
	info.availableMemory = 0;

	return info;
}

bool BackendUtils::isCudaAvailable() {
#ifdef OPE_CUDA_AVAILABLE
	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	return (error == cudaSuccess && deviceCount > 0);
#else
	return false;
#endif
}

bool BackendUtils::isOpenCLAvailable() {
#ifdef OPE_OPENCL_AVAILABLE
	cl_uint num_platforms = 0;
	cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
	return (err == CL_SUCCESS && num_platforms > 0);
#else
	return false;
#endif
}

bool BackendUtils::isVulkanAvailable() {
#ifdef OPE_VULKAN_AVAILABLE
	//	todo: ceck if Vulkan runtime is available
	return true;
#else
	return false;
#endif
}

bool BackendUtils::isCpuAvailable() {
#ifdef OPE_CPU_AVAILABLE
	return true;
#else
	return false;
#endif
}

std::unique_ptr<BackendConfig> BackendUtils::createDefaultConfig(Backend backend) {
	switch (backend) {
	case Backend::CUDA:
		return std::make_unique<CudaConfig>();
	case Backend::OPENCL:
		return std::make_unique<OpenCLConfig>();
	case Backend::VULKAN:
		return std::make_unique<VulkanConfig>();
	case Backend::CPU:
		return std::make_unique<CpuConfig>();
	default:
		return nullptr;
	}
}

std::unique_ptr<BackendConfig> BackendUtils::parseConfig(const std::string& configString) {
	//	Simple parser for config strings like "cuda:device=1" or "opencl:platform=0,device=1"
	std::istringstream iss(configString);
	std::string backendName;
	std::getline(iss, backendName, ':');

	if (backendName == "cuda") {
		auto config = std::make_unique<CudaConfig>();
		//	Parse additional parameters if present
		std::string params;
		if (std::getline(iss, params)) {
			//	Parse key=value pairs
			std::istringstream paramStream(params);
			std::string param;
			while (std::getline(paramStream, param, ',')) {
				size_t pos = param.find('=');
				if (pos != std::string::npos) {
					std::string key = param.substr(0, pos);
					std::string value = param.substr(pos + 1);
					if (key == "device") {
						config->deviceId = std::stoi(value);
					} else if (key == "zero_copy") {
						config->enableZeroCopy = (value == "true");
					}
				}
			}
		}
		return config;
	} else if (backendName == "opencl") {
		auto config = std::make_unique<OpenCLConfig>();
		std::string params;
		if (std::getline(iss, params)) {
			std::istringstream paramStream(params);
			std::string param;
			while (std::getline(paramStream, param, ',')) {
				size_t pos = param.find('=');
				if (pos != std::string::npos) {
					std::string key = param.substr(0, pos);
					std::string value = param.substr(pos + 1);
					if (key == "platform") {
						config->platformId = std::stoi(value);
					} else if (key == "device") {
						config->deviceId = std::stoi(value);
					} else if (key == "prefer_gpu") {
						config->preferGpu = (value == "true");
					}
				}
			}
		}
		return config;
	} else if (backendName == "vulkan") {
		auto config = std::make_unique<VulkanConfig>();
		std::string params;
		if (std::getline(iss, params)) {
			std::istringstream paramStream(params);
			std::string param;
			while (std::getline(paramStream, param, ',')) {
				size_t pos = param.find('=');
				if (pos != std::string::npos) {
					std::string key = param.substr(0, pos);
					std::string value = param.substr(pos + 1);
					if (key == "device") {
						config->deviceId = std::stoi(value);
					}
				}
			}
		}
		return config;
	} else if (backendName == "cpu") {
		auto config = std::make_unique<CpuConfig>();
		std::string params;
		if (std::getline(iss, params)) {
			std::istringstream paramStream(params);
			std::string param;
			while (std::getline(paramStream, param, ',')) {
				size_t pos = param.find('=');
				if (pos != std::string::npos) {
					std::string key = param.substr(0, pos);
					std::string value = param.substr(pos + 1);
					if (key == "threads") {
						config->numThreads = std::stoi(value);
					} else if (key == "simd") {
						config->enableSimd = (value == "true");
					}
				}
			}
		}
		return config;
	}

	return nullptr;
}

std::string BackendUtils::serializeConfig(const BackendConfig& config) {
	std::ostringstream oss;

	switch (config.getBackendType()) {
	case Backend::CUDA: {
		const auto& cudaConfig = static_cast<const CudaConfig&>(config);
		oss << "cuda:device=" << cudaConfig.deviceId;
		if (cudaConfig.enableZeroCopy) {
			oss << ",zero_copy=true";
		}
		break;
	}
	case Backend::OPENCL: {
		const auto& openclConfig = static_cast<const OpenCLConfig&>(config);
		oss << "opencl:platform=" << openclConfig.platformId
			<< ",device=" << openclConfig.deviceId
			<< ",prefer_gpu=" << (openclConfig.preferGpu ? "true" : "false");
		break;
	}
	case Backend::VULKAN: {
		const auto& vulkanConfig = static_cast<const VulkanConfig&>(config);
		oss << "vulkan:device=" << vulkanConfig.deviceId;
		break;
	}
	case Backend::CPU: {
		const auto& cpuConfig = static_cast<const CpuConfig&>(config);
		oss << "cpu:threads=" << cpuConfig.numThreads
			<< ",simd=" << (cpuConfig.enableSimd ? "true" : "false");
		break;
	}
	}

	return oss.str();
}

} // namespace ope