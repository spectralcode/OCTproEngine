#include "bindings_common.h"

void register_backend_config(py::module& m) {
	// DeviceInfo structure
	py::class_<ope::DeviceInfo>(m, "DeviceInfo")
		.def(py::init<>())
		.def_readonly("id", &ope::DeviceInfo::id)
		.def_readonly("name", &ope::DeviceInfo::name)
		.def_readonly("total_memory", &ope::DeviceInfo::totalMemory)
		.def_readonly("available_memory", &ope::DeviceInfo::availableMemory)
		.def_readonly("compute_capability_major", &ope::DeviceInfo::computeCapabilityMajor)
		.def_readonly("compute_capability_minor", &ope::DeviceInfo::computeCapabilityMinor)
		.def_readonly("vendor_name", &ope::DeviceInfo::vendorName)
		.def_readonly("device_version", &ope::DeviceInfo::deviceVersion)
		.def("__repr__", [](const ope::DeviceInfo& info) {
			return "<DeviceInfo(id=" + std::to_string(info.id) +
			       ", name='" + info.name + "')>";
		});

	// Base BackendConfig class
	py::class_<ope::BackendConfig>(m, "BackendConfig")
		.def("get_backend_type", &ope::BackendConfig::getBackendType,
			"Get the backend type for this configuration")
		.def("is_valid", &ope::BackendConfig::isValid,
			"Check if the configuration is valid")
		.def("to_string", &ope::BackendConfig::toString,
			"Get human-readable string representation")
		.def("__repr__", [](const ope::BackendConfig& config) {
			return "<BackendConfig(" + config.toString() + ")>";
		});

	// CudaConfig derived class
	py::class_<ope::CudaConfig, ope::BackendConfig>(m, "CudaConfig")
		.def(py::init<>())
		.def_readwrite("device_id", &ope::CudaConfig::deviceId,
			"CUDA device ID (default: 0)")
		.def_readwrite("enable_zero_copy", &ope::CudaConfig::enableZeroCopy,
			"Enable zero-copy mode for Jetson devices (default: False)")
		.def("__repr__", [](const ope::CudaConfig& config) {
			return "<CudaConfig(device_id=" + std::to_string(config.deviceId) +
			       ", enable_zero_copy=" + (config.enableZeroCopy ? "True" : "False") + ")>";
		});

	// OpenCLConfig derived class
	py::class_<ope::OpenCLConfig, ope::BackendConfig>(m, "OpenCLConfig")
		.def(py::init<>())
		.def_readwrite("platform_id", &ope::OpenCLConfig::platformId,
			"OpenCL platform ID (default: 0)")
		.def_readwrite("device_id", &ope::OpenCLConfig::deviceId,
			"OpenCL device ID (default: 0)")
		.def_readwrite("prefer_gpu", &ope::OpenCLConfig::preferGpu,
			"Prefer GPU devices (default: True)")
		.def("__repr__", [](const ope::OpenCLConfig& config) {
			return "<OpenCLConfig(platform_id=" + std::to_string(config.platformId) +
			       ", device_id=" + std::to_string(config.deviceId) +
			       ", prefer_gpu=" + (config.preferGpu ? "True" : "False") + ")>";
		});

	// VulkanConfig derived class
	py::class_<ope::VulkanConfig, ope::BackendConfig>(m, "VulkanConfig")
		.def(py::init<>())
		.def_readwrite("device_id", &ope::VulkanConfig::deviceId,
			"Vulkan physical device ID (default: 0)")
		.def("__repr__", [](const ope::VulkanConfig& config) {
			return "<VulkanConfig(device_id=" + std::to_string(config.deviceId) + ")>";
		});

	// CpuConfig derived class
	py::class_<ope::CpuConfig, ope::BackendConfig>(m, "CpuConfig")
		.def(py::init<>())
		.def_readwrite("num_threads", &ope::CpuConfig::numThreads,
			"Number of threads (0 = auto-detect, default: 0)")
		.def_readwrite("enable_simd", &ope::CpuConfig::enableSimd,
			"Enable SIMD optimizations (default: True)")
		.def("__repr__", [](const ope::CpuConfig& config) {
			return "<CpuConfig(num_threads=" + std::to_string(config.numThreads) +
			       ", enable_simd=" + (config.enableSimd ? "True" : "False") + ")>";
		});

	// BackendUtils class
	py::class_<ope::BackendUtils>(m, "BackendUtils")
		.def_static("get_cuda_devices", &ope::BackendUtils::getCudaDevices,
			"Get list of available CUDA devices")
		.def_static("get_opencl_devices", &ope::BackendUtils::getOpenCLDevices,
			"Get list of available OpenCL devices")
		.def_static("get_vulkan_devices", &ope::BackendUtils::getVulkanDevices,
			"Get list of available Vulkan devices")
		.def_static("get_cpu_info", &ope::BackendUtils::getCpuInfo,
			"Get CPU information")
		.def_static("is_cuda_available", &ope::BackendUtils::isCudaAvailable,
			"Check if CUDA is available")
		.def_static("is_opencl_available", &ope::BackendUtils::isOpenCLAvailable,
			"Check if OpenCL is available")
		.def_static("is_vulkan_available", &ope::BackendUtils::isVulkanAvailable,
			"Check if Vulkan is available")
		.def_static("is_cpu_available", &ope::BackendUtils::isCpuAvailable,
			"Check if CPU backend is available")
		.def_static("create_default_config", &ope::BackendUtils::createDefaultConfig,
			py::arg("backend"),
			"Create default configuration for a backend")
		.def_static("parse_config", &ope::BackendUtils::parseConfig,
			py::arg("config_string"),
			"Parse configuration from string")
		.def_static("serialize_config", &ope::BackendUtils::serializeConfig,
			py::arg("config"),
			"Serialize configuration to string");

	// CudaDeviceInfo structure
	py::class_<ope::CudaDeviceInfo>(m, "CudaDeviceInfo")
		.def(py::init<>())
		.def_readonly("device_id", &ope::CudaDeviceInfo::deviceId)
		.def_readonly("name", &ope::CudaDeviceInfo::name)
		.def_readonly("total_memory", &ope::CudaDeviceInfo::totalMemory)
		.def_readonly("free_memory", &ope::CudaDeviceInfo::freeMemory)
		.def_readonly("compute_capability_major", &ope::CudaDeviceInfo::computeCapabilityMajor)
		.def_readonly("compute_capability_minor", &ope::CudaDeviceInfo::computeCapabilityMinor)
		.def_readonly("max_threads_per_block", &ope::CudaDeviceInfo::maxThreadsPerBlock)
		.def_readonly("multiprocessor_count", &ope::CudaDeviceInfo::multiProcessorCount)
		.def_readonly("is_available", &ope::CudaDeviceInfo::isAvailable)
		.def("get_compute_capability", &ope::CudaDeviceInfo::getComputeCapability,
			"Get compute capability as string (e.g., '8.6')")
		.def("__repr__", [](const ope::CudaDeviceInfo& info) {
			return "<CudaDeviceInfo(id=" + std::to_string(info.deviceId) +
			       ", name='" + info.name + "', compute=" + info.getComputeCapability() + ")>";
		});

	// CudaUtils - pure static utility class (no instances)
	// Use module-level bindings instead of py::class_ to avoid instantiation issues
	auto cuda_utils = m.def_submodule("CudaUtils", "CUDA utility functions");

	cuda_utils.def("get_available_devices", &ope::CudaUtils::getAvailableDevices,
		"Get list of available CUDA devices\n\n"
		"Returns:\n"
		"    List[CudaDeviceInfo]: List of available GPUs (empty if no CUDA support)");

	cuda_utils.def("get_device_info", &ope::CudaUtils::getDeviceInfo,
		py::arg("device_id"),
		"Get detailed information about specific GPU\n\n"
		"Args:\n"
		"    device_id: GPU device ID (0-based)\n\n"
		"Returns:\n"
		"    CudaDeviceInfo: Device information\n\n"
		"Raises:\n"
		"    RuntimeError: If CUDA not available or device doesn't exist");

	cuda_utils.def("is_device_available", &ope::CudaUtils::isDeviceAvailable,
		py::arg("device_id"),
		"Check if specific GPU device is available\n\n"
		"Args:\n"
		"    device_id: GPU device ID (0-based)\n\n"
		"Returns:\n"
		"    bool: True if device exists and is available");

	cuda_utils.def("get_device_count", &ope::CudaUtils::getDeviceCount,
		"Get number of available GPU devices\n\n"
		"Returns:\n"
		"    int: Number of GPUs (0 if no CUDA support)");

	cuda_utils.def("is_available", &ope::CudaUtils::isAvailable,
		"Check if CUDA is available in this build\n\n"
		"Returns:\n"
		"    bool: True if CUDA compiled and devices available");

	cuda_utils.def("get_current_device", &ope::CudaUtils::getCurrentDevice,
		"Get current CUDA device ID\n\n"
		"Returns:\n"
		"    int: Current device ID, or -1 if no CUDA");
}
