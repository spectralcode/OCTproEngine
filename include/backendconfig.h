#ifndef OPE_BACKENDCONFIG_H
#define OPE_BACKENDCONFIG_H

#include <string>
#include <vector>
#include <memory>
#include "export.h"
#include "processorconfiguration.h"

namespace ope {

// Forward declarations
struct DeviceInfo {
	int id;
	std::string name;
	size_t totalMemory;
	size_t availableMemory;
	int computeCapabilityMajor;  // For CUDA
	int computeCapabilityMinor;  // For CUDA
	std::string vendorName;       // For OpenCL
	std::string deviceVersion;    // For OpenCL
};

// Base class for backend configuration
class OPE_API BackendConfig {
public:
	BackendConfig() = default;
	virtual ~BackendConfig() = default;

	// Get the backend type
	virtual Backend getBackendType() const = 0;

	// Clone the configuration
	virtual std::unique_ptr<BackendConfig> clone() const = 0;

	// Validate the configuration
	virtual bool isValid() const = 0;

	// Get human-readable description
	virtual std::string toString() const = 0;
};

// CUDA backend configuration
class OPE_API CudaConfig : public BackendConfig {
public:
	int deviceId;
	bool enableZeroCopy;
	int numOutputBuffers;  // 0 = auto (default: numStreams * 2)

	CudaConfig() : deviceId(0), enableZeroCopy(false), numOutputBuffers(0) {}

	Backend getBackendType() const override { return Backend::CUDA; }

	std::unique_ptr<BackendConfig> clone() const override {
		return std::make_unique<CudaConfig>(*this);
	}

	bool isValid() const override {
		return deviceId >= 0;
	}

	std::string toString() const override;
};

// OpenCL backend configuration
class OPE_API OpenCLConfig : public BackendConfig {
public:
	int platformId;
	int deviceId;  // -1 = auto-select based on preferGpu
	bool preferGpu;
	int numOutputBuffers;  // 0 = auto (default: numCommandQueues * 2)

	OpenCLConfig() : platformId(0), deviceId(-1), preferGpu(true), numOutputBuffers(0) {}

	Backend getBackendType() const override { return Backend::OPENCL; }

	std::unique_ptr<BackendConfig> clone() const override {
		return std::make_unique<OpenCLConfig>(*this);
	}

	bool isValid() const override {
		return platformId >= 0 && deviceId >= -1;  // -1 = auto-select
	}

	std::string toString() const override;
};

// Vulkan backend configuration
class OPE_API VulkanConfig : public BackendConfig {
public:
	int deviceId;  // Physical device index
	int numOutputBuffers;  // 0 = auto (default: numCommandBuffers * 2)
	int numStagingInputBuffers;  // 0 = auto (default: numCommandBuffers * 2)

	VulkanConfig() : deviceId(0), numOutputBuffers(0), numStagingInputBuffers(0) {}

	Backend getBackendType() const override { return Backend::VULKAN; }

	std::unique_ptr<BackendConfig> clone() const override {
		return std::make_unique<VulkanConfig>(*this);
	}

	bool isValid() const override {
		return deviceId >= 0;
	}

	std::string toString() const override;
};

// CPU backend configuration
class OPE_API CpuConfig : public BackendConfig {
public:
	int numThreads; //todo: currently not used, but could be fun to play around with FFTW threading, SIMD optimizations, etc.
	bool enableSimd; //todo: only as a reminder here. use it or remove it!

	CpuConfig() : numThreads(0), enableSimd(true) {}  // 0 = auto-detect

	Backend getBackendType() const override { return Backend::CPU; }

	std::unique_ptr<BackendConfig> clone() const override {
		return std::make_unique<CpuConfig>(*this);
	}

	bool isValid() const override {
		return numThreads >= 0;
	}

	std::string toString() const override;
};

// Utility class for backend management
class OPE_API BackendUtils {
public:
	// Device enumeration
	static std::vector<DeviceInfo> getCudaDevices();
	static std::vector<DeviceInfo> getOpenCLDevices();
	static std::vector<DeviceInfo> getVulkanDevices();
	static DeviceInfo getCpuInfo();

	// Backend availability
	static bool isCudaAvailable();
	static bool isOpenCLAvailable();
	static bool isVulkanAvailable();
	static bool isCpuAvailable();

	// Create default configuration for a backend
	static std::unique_ptr<BackendConfig> createDefaultConfig(Backend backend);

	// Parse configuration from string (for config files)
	static std::unique_ptr<BackendConfig> parseConfig(const std::string& configString);

	// Serialize configuration to string
	static std::string serializeConfig(const BackendConfig& config);
};

} // namespace ope

#endif // OPE_BACKENDCONFIG_H