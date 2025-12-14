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
}
