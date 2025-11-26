#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "processor.h"
#include "processorconfiguration.h"
#include "backendconfig.h"
#include "iobuffer.h"
#include "types.h"
#include "version.h"
#include "cudautils.h"
#include "tools/recorder.h"

namespace py = pybind11;

// ============================================
// EXCEPTION DEFINITIONS
// ============================================

class InitializationError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class ConfigurationError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class BufferError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class ProcessingError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class BackendError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

// ============================================
// NUMPY DTYPE HELPERS
// ============================================

ope::DataType numpy_dtype_to_ope(py::dtype dtype) {
	if (dtype.is(py::dtype::of<uint8_t>())) {
		return ope::DataType::UINT8;
	} else if (dtype.is(py::dtype::of<uint16_t>())) {
		return ope::DataType::UINT16;
	} else if (dtype.is(py::dtype::of<uint32_t>())) {
		return ope::DataType::UINT32;
	} else if (dtype.is(py::dtype::of<float>())) {
		return ope::DataType::FLOAT32;
	} else if (dtype.is(py::dtype::of<double>())) {
		return ope::DataType::FLOAT64;
	} else {
		throw BufferError("Unsupported NumPy dtype. Supported types: uint8, uint16, uint32, float32, float64");
	}
}

std::string ope_dtype_to_string(ope::DataType dtype) {
	switch (dtype) {
		case ope::DataType::UINT8: return "uint8";
		case ope::DataType::UINT16: return "uint16";
		case ope::DataType::UINT32: return "uint32";
		case ope::DataType::FLOAT32: return "float32";
		case ope::DataType::FLOAT64: return "float64";
		default: return "unknown";
	}
}

// ============================================
// BUFFER <-> NUMPY CONVERSION
// ============================================

// Return NumPy array view of IOBuffer (zero-copy)
py::array buffer_to_numpy(ope::IOBuffer& buffer) {
	void* ptr = buffer.getDataPointer();
	size_t size_bytes = buffer.getSizeInBytes();
	ope::DataType dtype = buffer.getDataType();
	
	// Determine NumPy dtype and element count
	py::dtype np_dtype;
	size_t num_elements;
	
	switch (dtype) {
		case ope::DataType::UINT8:
			np_dtype = py::dtype::of<uint8_t>();
			num_elements = size_bytes;
			break;
		case ope::DataType::UINT16:
			np_dtype = py::dtype::of<uint16_t>();
			num_elements = size_bytes / 2;
			break;
		case ope::DataType::UINT32:
			np_dtype = py::dtype::of<uint32_t>();
			num_elements = size_bytes / 4;
			break;
		case ope::DataType::FLOAT32:
			np_dtype = py::dtype::of<float>();
			num_elements = size_bytes / 4;
			break;
		case ope::DataType::FLOAT64:
			np_dtype = py::dtype::of<double>();
			num_elements = size_bytes / 8;
			break;
		default:
			throw BufferError("Unsupported IOBuffer data type");
	}
	
	// Create NumPy array view (no copy, doesn't own data)
	return py::array(np_dtype, {num_elements}, {np_dtype.itemsize()}, ptr, py::cast(&buffer));
}

// ============================================
// PROCESSOR WRAPPER WITH CALLBACK SUPPORT
// ============================================

class ProcessorWrapper {
public:
	ope::Processor processor;
	py::function callback;
	py::function error_callback;

	std::map<ope::Processor::CallbackId, py::function> pyCallbacks;
	std::mutex callbacksMutex;
	
	ProcessorWrapper(ope::Backend backend) : processor(backend) {}
	
	void set_callback(py::function cb, py::object error_cb = py::none()) {
		callback = cb;
		if (!error_cb.is_none()) {
			error_callback = error_cb.cast<py::function>();
		}
		
		// Set C++ callback that will call Python callback
		processor.setOutputCallback([this](const ope::IOBuffer& output) {
			// Re-acquire GIL to call Python code
			py::gil_scoped_acquire acquire;
			
			try {
				// Create NumPy view of output buffer (cast away const for view)
				ope::IOBuffer& output_ref = const_cast<ope::IOBuffer&>(output);
				py::array output_array = buffer_to_numpy(output_ref);
				
				// Call Python callback with buffer ID
				callback(output_array, output.getBufferId());
			} catch (const std::exception& e) {
				// Handle errors in callback
				if (!error_callback.is_none()) {
					try {
						error_callback(py::str(e.what()));
					} catch (...) {
						py::print("Error in error_callback:", py::str(e.what()));
					}
				} else {
					py::print("Error in callback:", py::str(e.what()));
				}
			}
		});
	}
	
ope::Processor::CallbackId add_output_callback(py::function cb) {
		// Create C++ wrapper callback that handles GIL
		auto wrappedCallback = [this, cb](const ope::IOBuffer& buffer) {
			// CRITICAL: Re-acquire GIL before calling Python code
			// We're in a C++ callback thread, need GIL to call Python
			py::gil_scoped_acquire acquire;
			
			try {
				// Create NumPy view of buffer (zero-copy)
				ope::IOBuffer& buffer_ref = const_cast<ope::IOBuffer&>(buffer);
				py::array output_array = buffer_to_numpy(buffer_ref);
				
				// Call Python callback with buffer ID
				cb(output_array, buffer.getBufferId());
				
			} catch (const py::error_already_set& e) {
				// Python exception
				py::print("Python error in callback:", e.what());
			} catch (const std::exception& e) {
				// C++ exception
				py::print("C++ error in callback:", e.what());
			}
		};
		
		// Register with C++ processor
		ope::Processor::CallbackId id;
		{
			// Release GIL for C++ operation
			py::gil_scoped_release release;
			id = processor.addOutputCallback(wrappedCallback);
		}
		
		// Store Python function to prevent garbage collection
		{
			std::lock_guard<std::mutex> lock(callbacksMutex);
			pyCallbacks[id] = cb;
		}
		
		return id;
	}
	
	bool remove_output_callback(ope::Processor::CallbackId id) {
		bool removed;
		{
			py::gil_scoped_release release;
			removed = processor.removeOutputCallback(id);
		}
		
		if (removed) {
			std::lock_guard<std::mutex> lock(callbacksMutex);
			pyCallbacks.erase(id);
		}
		
		return removed;
	}
	
	void clear_output_callbacks() {
		{
			py::gil_scoped_release release;
			processor.clearOutputCallbacks();
		}
		
		std::lock_guard<std::mutex> lock(callbacksMutex);
		pyCallbacks.clear();
	}
	
	size_t get_output_callback_count() const {
		return processor.getOutputCallbackCount();
	}

	// Input callback methods (for raw data before processing)
	ope::Processor::CallbackId add_input_callback(py::function cb) {
		// Create C++ wrapper callback that handles GIL
		auto wrappedCallback = [this, cb](const ope::IOBuffer& buffer) {
			// CRITICAL: Re-acquire GIL before calling Python code
			py::gil_scoped_acquire acquire;

			try {
				// Create NumPy view of buffer (zero-copy)
				ope::IOBuffer& buffer_ref = const_cast<ope::IOBuffer&>(buffer);
				py::array input_array = buffer_to_numpy(buffer_ref);

				// Call Python callback with buffer ID
				cb(input_array, buffer.getBufferId());

			} catch (const py::error_already_set& e) {
				// Python exception
				py::print("Python error in input callback:", e.what());
			} catch (const std::exception& e) {
				// C++ exception
				py::print("C++ error in input callback:", e.what());
			}
		};

		// Register with C++ processor
		ope::Processor::CallbackId id;
		{
			// Release GIL for C++ operation
			py::gil_scoped_release release;
			id = processor.addInputCallback(wrappedCallback);
		}

		// Store Python function to prevent garbage collection
		{
			std::lock_guard<std::mutex> lock(callbacksMutex);
			pyCallbacks[id] = cb;
		}

		return id;
	}

	bool remove_input_callback(ope::Processor::CallbackId id) {
		bool removed;
		{
			py::gil_scoped_release release;
			removed = processor.removeInputCallback(id);
		}

		if (removed) {
			std::lock_guard<std::mutex> lock(callbacksMutex);
			pyCallbacks.erase(id);
		}

		return removed;
	}

	void clear_input_callbacks() {
		{
			py::gil_scoped_release release;
			processor.clearInputCallbacks();
		}

		// Note: We don't clear pyCallbacks here as they might contain output callbacks too
		// They'll be cleaned up when individual callbacks are removed
	}

	size_t get_input_callback_count() const {
		return processor.getInputCallbackCount();
	}

	void process(py::array buffer_array) {
		// Get buffer from the array's base object (if it's a view of IOBuffer)
		py::object base = buffer_array.attr("base");
		if (base.is_none()) {
			throw BufferError("Buffer array is not a view of an IOBuffer. Use get_next_available_buffer() first.");
		}

		// Extract IOBuffer reference
		ope::IOBuffer* buffer_ptr = base.cast<ope::IOBuffer*>();

		// Release GIL during processing
		{
			py::gil_scoped_release release;
			processor.process(*buffer_ptr);
		}
	}
	
	py::array get_next_available_buffer() {
		ope::IOBuffer* buffer;
		
		// Release GIL while waiting for buffer
		{
			py::gil_scoped_release release;
			buffer = &processor.getNextAvailableInputBuffer();
		}
		
		return buffer_to_numpy(*buffer);
	}
	
	// Wrapper methods that release GIL
	void initialize() {
		try {
			py::gil_scoped_release release;
			processor.initialize();
		} catch (const std::exception& e) {
			throw InitializationError(std::string("Initialization failed: ") + e.what());
		}
	}
	
	void stop() { // todo: rename to cleanup?
		py::gil_scoped_release release;
		processor.cleanup();
	}
	
	void load_config(const std::string& filepath) {
		try {
			processor.loadConfigurationFromFile(filepath);
		} catch (const std::exception& e) {
			throw ConfigurationError(std::string("Failed to load config: ") + e.what());
		}
	}
	
	void save_config(const std::string& filepath) const {
		try {
			processor.saveConfigurationToFile(filepath);
		} catch (const std::exception& e) {
			throw ConfigurationError(std::string("Failed to save config: ") + e.what());
		}
	}
	
	// Context manager support
	ProcessorWrapper& enter() {
		return *this;
	}
	
	void exit(py::object exc_type, py::object exc_value, py::object traceback) {
		stop();
	}
};

// ============================================
// PYBIND11 MODULE DEFINITION
// ============================================

PYBIND11_MODULE(octproengine, m) {
	m.doc() = "OCTproEngine - High-performance OCT processing library";
	
	// ============================================
	// EXCEPTIONS
	// ============================================
	
	py::register_exception<InitializationError>(m, "InitializationError");
	py::register_exception<ConfigurationError>(m, "ConfigurationError");
	py::register_exception<BufferError>(m, "BufferError");
	py::register_exception<ProcessingError>(m, "ProcessingError");
	py::register_exception<BackendError>(m, "BackendError");
	py::register_exception<ope::tools::RecordingException>(m, "RecordingException");

	// ============================================
	// IOBUFFER
	// ============================================

	py::class_<ope::IOBuffer>(m, "IOBuffer", py::module_local())
		.def("get_size", &ope::IOBuffer::getSizeInBytes, "Get buffer size in bytes")
		.def("get_data_type", &ope::IOBuffer::getDataType, "Get buffer data type");
	
	// ============================================
	// ENUMS
	// ============================================
	
	py::enum_<ope::Backend>(m, "Backend")
		.value("CUDA", ope::Backend::CUDA, "NVIDIA CUDA GPU backend")
		.value("CPU", ope::Backend::CPU, "CPU backend")
		.value("OPENCL", ope::Backend::OPENCL, "OpenCL GPU backend")
		.export_values();
	
	py::enum_<ope::DataType>(m, "DataType")
		.value("UINT8", ope::DataType::UINT8)
		.value("UINT16", ope::DataType::UINT16)
		.value("UINT32", ope::DataType::UINT32)
		.value("UINT64", ope::DataType::UINT64)
		.value("INT8", ope::DataType::INT8)
		.value("INT16", ope::DataType::INT16)
		.value("INT32", ope::DataType::INT32)
		.value("INT64", ope::DataType::INT64)
		.value("FLOAT32", ope::DataType::FLOAT32)
		.value("FLOAT64", ope::DataType::FLOAT64)
		.value("COMPLEX_FLOAT32", ope::DataType::COMPLEX_FLOAT32)
		.value("COMPLEX_FLOAT64", ope::DataType::COMPLEX_FLOAT64)
		.export_values();
	
	py::enum_<ope::InterpolationMethod>(m, "InterpolationMethod")
		.value("LINEAR", ope::InterpolationMethod::LINEAR, "Linear interpolation")
		.value("CUBIC", ope::InterpolationMethod::CUBIC, "Cubic interpolation")
		.value("LANCZOS", ope::InterpolationMethod::LANCZOS, "Lanczos interpolation")
		.export_values();
	
	py::enum_<ope::WindowType>(m, "WindowType")
		.value("HANN", ope::WindowType::HANN, "Hann window")
		.value("GAUSS", ope::WindowType::GAUSS, "Gaussian window")
		.value("SINE", ope::WindowType::SINE, "Sine window")
		.value("LANCZOS", ope::WindowType::LANCZOS, "Lanczos window")
		.value("RECTANGULAR", ope::WindowType::RECTANGULAR, "Rectangular window")
		.value("FLAT_TOP", ope::WindowType::FLAT_TOP, "Flat-top window")
		.export_values();

	py::enum_<ope::tools::Recorder::Mode>(m, "RecorderMode")
		.value("RAW_ONLY", ope::tools::Recorder::Mode::RAW_ONLY, "Record only raw (input) data")
		.value("PROCESSED_ONLY", ope::tools::Recorder::Mode::PROCESSED_ONLY, "Record only processed (output) data")
		.value("BOTH", ope::tools::Recorder::Mode::BOTH, "Record both raw and processed data")
		.export_values();

	py::enum_<ope::tools::Recorder::Format>(m, "RecorderFormat")
		.value("RAW_BINARY", ope::tools::Recorder::Format::RAW_BINARY, "Raw binary format")
		.export_values();

	py::enum_<ope::tools::Recorder::Status>(m, "RecorderStatus")
		.value("IDLE", ope::tools::Recorder::Status::IDLE, "Newly created or last recording aborted")
		.value("RECORDING", ope::tools::Recorder::Status::RECORDING, "Currently collecting buffers")
		.value("WRITING", ope::tools::Recorder::Status::WRITING, "Recording complete, writing to disk")
		.value("COMPLETE", ope::tools::Recorder::Status::COMPLETE, "Last recording succeeded")
		.value("ERROR_STATUS", ope::tools::Recorder::Status::ERROR, "Last recording failed")
		.export_values();

	// ============================================
	// BACKEND CONFIGURATION
	// ============================================

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
		.def_static("get_cpu_info", &ope::BackendUtils::getCpuInfo,
			"Get CPU information")
		.def_static("is_cuda_available", &ope::BackendUtils::isCudaAvailable,
			"Check if CUDA is available")
		.def_static("is_opencl_available", &ope::BackendUtils::isOpenCLAvailable,
			"Check if OpenCL is available")
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

	// ============================================
	// CUDA UTILITIES
	// ============================================

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

	// ============================================
	// CONFIGURATION STRUCTS AND ENUMS
	// ============================================

	// Enums for ProcessorConfiguration
	py::enum_<ope::ProcessorConfiguration::LoadMode>(m, "LoadMode")
		.value("OVERWRITE_ALL", ope::ProcessorConfiguration::LoadMode::OVERWRITE_ALL)
		.value("PARAMETERS_ONLY", ope::ProcessorConfiguration::LoadMode::PARAMETERS_ONLY)
		.value("MERGE_IF_MISSING", ope::ProcessorConfiguration::LoadMode::MERGE_IF_MISSING);

	py::enum_<ope::ProcessorConfiguration::SaveMode>(m, "SaveMode")
		.value("PARAMETERS_ONLY", ope::ProcessorConfiguration::SaveMode::PARAMETERS_ONLY)
		.value("COMPLETE", ope::ProcessorConfiguration::SaveMode::COMPLETE);

	// DataParameters
	py::class_<ope::ProcessorConfiguration::DataParameters>(m, "DataParameters")
		.def(py::init<>())
		.def_readwrite("signalLength", &ope::ProcessorConfiguration::DataParameters::signalLength)
		.def_readwrite("ascansPerBscan", &ope::ProcessorConfiguration::DataParameters::ascansPerBscan)
		.def_readwrite("bscansPerBuffer", &ope::ProcessorConfiguration::DataParameters::bscansPerBuffer)
		.def_readwrite("buffersPerVolume", &ope::ProcessorConfiguration::DataParameters::buffersPerVolume)
		.def_readwrite("inputDataType", &ope::ProcessorConfiguration::DataParameters::inputDataType)
		.def_readwrite("outputDataType", &ope::ProcessorConfiguration::DataParameters::outputDataType)
		.def("samplesPerBuffer", &ope::ProcessorConfiguration::DataParameters::samplesPerBuffer)
		.def("outputSignalLength", &ope::ProcessorConfiguration::DataParameters::outputSignalLength)
		.def("getBitDepth", &ope::ProcessorConfiguration::DataParameters::getBitDepth)
		.def("getBytesPerSample", &ope::ProcessorConfiguration::DataParameters::getBytesPerSample)
		.def("getOutputBytesPerSample", &ope::ProcessorConfiguration::DataParameters::getOutputBytesPerSample);

	// ProcessingParameters sub-structs
	py::class_<ope::ProcessorConfiguration::ProcessingParameters::Input>(m, "ProcessingInput")
		.def(py::init<>())
		.def_readwrite("bitshift", &ope::ProcessorConfiguration::ProcessingParameters::Input::bitshift);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::DCRemoval>(m, "ProcessingDCRemoval")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::ProcessingParameters::DCRemoval::enabled)
		.def_readwrite("windowSize", &ope::ProcessorConfiguration::ProcessingParameters::DCRemoval::windowSize);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::Resampling>(m, "ProcessingResampling")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::ProcessingParameters::Resampling::enabled)
		.def_readwrite("method", &ope::ProcessorConfiguration::ProcessingParameters::Resampling::method)
		.def_property("coefficients",
			[](const ope::ProcessorConfiguration::ProcessingParameters::Resampling& self) {
				return std::vector<float>(self.coefficients, self.coefficients + 4);
			},
			[](ope::ProcessorConfiguration::ProcessingParameters::Resampling& self, const std::vector<float>& coeffs) {
				if (coeffs.size() != 4) throw std::runtime_error("Coefficients must have exactly 4 elements");
				std::copy(coeffs.begin(), coeffs.end(), self.coefficients);
			})
		.def_readwrite("useCustomLut", &ope::ProcessorConfiguration::ProcessingParameters::Resampling::useCustomLut);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::Windowing>(m, "ProcessingWindowing")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::ProcessingParameters::Windowing::enabled)
		.def_readwrite("type", &ope::ProcessorConfiguration::ProcessingParameters::Windowing::type)
		.def_readwrite("centerPosition", &ope::ProcessorConfiguration::ProcessingParameters::Windowing::centerPosition)
		.def_readwrite("fillFactor", &ope::ProcessorConfiguration::ProcessingParameters::Windowing::fillFactor)
		.def_readwrite("useCustomFunction", &ope::ProcessorConfiguration::ProcessingParameters::Windowing::useCustomFunction);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::Dispersion>(m, "ProcessingDispersion")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::ProcessingParameters::Dispersion::enabled)
		.def_property("coefficients",
			[](const ope::ProcessorConfiguration::ProcessingParameters::Dispersion& self) {
				return std::vector<float>(self.coefficients, self.coefficients + 4);
			},
			[](ope::ProcessorConfiguration::ProcessingParameters::Dispersion& self, const std::vector<float>& coeffs) {
				if (coeffs.size() != 4) throw std::runtime_error("Coefficients must have exactly 4 elements");
				std::copy(coeffs.begin(), coeffs.end(), self.coefficients);
			})
		.def_readwrite("factor", &ope::ProcessorConfiguration::ProcessingParameters::Dispersion::factor)
		.def_readwrite("useCustomPhase", &ope::ProcessorConfiguration::ProcessingParameters::Dispersion::useCustomPhase);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::FixedPatternNoise>(m, "ProcessingFixedPatternNoise")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::ProcessingParameters::FixedPatternNoise::enabled)
		.def_readwrite("bscanAverageCount", &ope::ProcessorConfiguration::ProcessingParameters::FixedPatternNoise::bscanAverageCount)
		.def_readwrite("continuous", &ope::ProcessorConfiguration::ProcessingParameters::FixedPatternNoise::continuous)
		.def_readwrite("useCustomProfile", &ope::ProcessorConfiguration::ProcessingParameters::FixedPatternNoise::useCustomProfile);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::Background>(m, "ProcessingBackground")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::ProcessingParameters::Background::enabled)
		.def_readwrite("weight", &ope::ProcessorConfiguration::ProcessingParameters::Background::weight)
		.def_readwrite("offset", &ope::ProcessorConfiguration::ProcessingParameters::Background::offset)
		.def_readwrite("useCustomProfile", &ope::ProcessorConfiguration::ProcessingParameters::Background::useCustomProfile);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::Intensity>(m, "ProcessingIntensity")
		.def(py::init<>())
		.def_readwrite("logScale", &ope::ProcessorConfiguration::ProcessingParameters::Intensity::logScale)
		.def_readwrite("rangeMin", &ope::ProcessorConfiguration::ProcessingParameters::Intensity::rangeMin)
		.def_readwrite("rangeMax", &ope::ProcessorConfiguration::ProcessingParameters::Intensity::rangeMax)
		.def_readwrite("preScale", &ope::ProcessorConfiguration::ProcessingParameters::Intensity::preScale)
		.def_readwrite("postOffset", &ope::ProcessorConfiguration::ProcessingParameters::Intensity::postOffset);

	py::class_<ope::ProcessorConfiguration::ProcessingParameters::Geometry>(m, "ProcessingGeometry")
		.def(py::init<>())
		.def_readwrite("alternatingBscanFlip", &ope::ProcessorConfiguration::ProcessingParameters::Geometry::alternatingBscanFlip)
		.def_readwrite("sinusoidalCorrection", &ope::ProcessorConfiguration::ProcessingParameters::Geometry::sinusoidalCorrection);

	// ProcessingParameters main struct
	py::class_<ope::ProcessorConfiguration::ProcessingParameters>(m, "ProcessingParameters")
		.def(py::init<>())
		.def_readwrite("input", &ope::ProcessorConfiguration::ProcessingParameters::input)
		.def_readwrite("dcRemoval", &ope::ProcessorConfiguration::ProcessingParameters::dcRemoval)
		.def_readwrite("resampling", &ope::ProcessorConfiguration::ProcessingParameters::resampling)
		.def_readwrite("windowing", &ope::ProcessorConfiguration::ProcessingParameters::windowing)
		.def_readwrite("dispersion", &ope::ProcessorConfiguration::ProcessingParameters::dispersion)
		.def_readwrite("fixedPatternNoise", &ope::ProcessorConfiguration::ProcessingParameters::fixedPatternNoise)
		.def_readwrite("background", &ope::ProcessorConfiguration::ProcessingParameters::background)
		.def_readwrite("intensity", &ope::ProcessorConfiguration::ProcessingParameters::intensity)
		.def_readwrite("geometry", &ope::ProcessorConfiguration::ProcessingParameters::geometry);

	// ============================================
	// CONFIGURATION CLASS
	// ============================================
	
	py::class_<ope::ProcessorConfiguration>(m, "ProcessorConfiguration")
		.def(py::init<>())
		// Nested structure API
		.def_readwrite("dataParams", &ope::ProcessorConfiguration::dataParams)
		.def_readwrite("processingParams", &ope::ProcessorConfiguration::processingParams)

		// Vector-based curve methods
		.def("setResamplingLut", [](ope::ProcessorConfiguration& self, const std::vector<float>& data) {
			self.setResamplingLut(data);
		})
		.def("setWindowFunction", [](ope::ProcessorConfiguration& self, const std::vector<float>& data) {
			self.setWindowFunction(data);
		})
		.def("setDispersionPhase", [](ope::ProcessorConfiguration& self, const std::vector<float>& data) {
			self.setDispersionPhase(data);
		})
		.def("setBackgroundProfile", [](ope::ProcessorConfiguration& self, const std::vector<float>& data) {
			self.setBackgroundProfile(data);
		})
		.def("setFixedPatternNoiseProfile", [](ope::ProcessorConfiguration& self, const std::vector<float>& data) {
			self.setFixedPatternNoiseProfile(data);
		})

		.def("getResamplingLut", &ope::ProcessorConfiguration::getResamplingLut)
		.def("getWindowFunction", &ope::ProcessorConfiguration::getWindowFunction)
		.def("getDispersionPhase", &ope::ProcessorConfiguration::getDispersionPhase)
		.def("getBackgroundProfile", &ope::ProcessorConfiguration::getBackgroundProfile)
		.def("getFixedPatternNoiseProfile", &ope::ProcessorConfiguration::getFixedPatternNoiseProfile)

		// Generate curves
		.def("generateResamplingLut", &ope::ProcessorConfiguration::generateResamplingLut)
		.def("generateWindowFunction", &ope::ProcessorConfiguration::generateWindowFunction)
		.def("generateDispersionPhase", &ope::ProcessorConfiguration::generateDispersionPhase)

		// Clear curves
		.def("clearResamplingLut", &ope::ProcessorConfiguration::clearResamplingLut)
		.def("clearWindowFunction", &ope::ProcessorConfiguration::clearWindowFunction)
		.def("clearDispersionPhase", &ope::ProcessorConfiguration::clearDispersionPhase)
		.def("clearBackgroundProfile", &ope::ProcessorConfiguration::clearBackgroundProfile)
		.def("clearFixedPatternNoiseProfile", &ope::ProcessorConfiguration::clearFixedPatternNoiseProfile)

		// Adjust curves when dimensions change
		.def("adjustAllCustomCurves", &ope::ProcessorConfiguration::adjustAllCustomCurves)

		// Check if custom curves are set
		.def("hasCustomResamplingCurve", &ope::ProcessorConfiguration::hasCustomResamplingCurve)
		.def("hasCustomWindowCurve", &ope::ProcessorConfiguration::hasCustomWindowCurve)
		.def("hasCustomDispersionCurve", &ope::ProcessorConfiguration::hasCustomDispersionCurve)
		.def("hasCustomPostProcessBackgroundProfile", &ope::ProcessorConfiguration::hasCustomPostProcessBackgroundProfile)
		.def("hasCustomFixedPatternNoiseProfile", &ope::ProcessorConfiguration::hasCustomFixedPatternNoiseProfile)

		// File I/O with modes
		.def("saveToFile", &ope::ProcessorConfiguration::saveToFile,
			py::arg("filepath"),
			py::arg("mode") = ope::ProcessorConfiguration::SaveMode::COMPLETE)
		.def("loadFromFile", &ope::ProcessorConfiguration::loadFromFile,
			py::arg("filepath"),
			py::arg("mode") = ope::ProcessorConfiguration::LoadMode::OVERWRITE_ALL)

		// CSV export/import for curves
		.def("saveResamplingLutToFile", &ope::ProcessorConfiguration::saveResamplingLutToFile)
		.def("loadResamplingLutFromFile", &ope::ProcessorConfiguration::loadResamplingLutFromFile)
		.def("saveWindowFunctionToFile", &ope::ProcessorConfiguration::saveWindowFunctionToFile)
		.def("loadWindowFunctionFromFile", &ope::ProcessorConfiguration::loadWindowFunctionFromFile)
		.def("saveDispersionPhaseToFile", &ope::ProcessorConfiguration::saveDispersionPhaseToFile)
		.def("loadDispersionPhaseFromFile", &ope::ProcessorConfiguration::loadDispersionPhaseFromFile)
		.def("saveBackgroundProfileToFile", &ope::ProcessorConfiguration::saveBackgroundProfileToFile)
		.def("loadBackgroundProfileFromFile", &ope::ProcessorConfiguration::loadBackgroundProfileFromFile)
		.def("saveFixedPatternNoiseProfileToFile", &ope::ProcessorConfiguration::saveFixedPatternNoiseProfileToFile)
		.def("loadFixedPatternNoiseProfileFromFile", &ope::ProcessorConfiguration::loadFixedPatternNoiseProfileFromFile)

		// Validation
		.def("validate", &ope::ProcessorConfiguration::validate);
	
	// ============================================
	// PROCESSOR CLASS
	// ============================================
	
	py::class_<ProcessorWrapper>(m, "Processor")
		.def(py::init<ope::Backend>(), py::arg("backend") = ope::Backend::CPU,
			"Create a new Processor instance\n\n"
			"Args:\n"
			"    backend: Backend to use (Backend.CUDA or Backend.CPU)")
		
		// Lifecycle
		.def("initialize", &ProcessorWrapper::initialize,
			"Initialize the processor and allocate buffers.\n"
			"Must be called before processing data.")
		.def("stop", &ProcessorWrapper::stop,
			"Stop the processor and free all resources.")
		
		// Configuration
		.def("load_config", &ProcessorWrapper::load_config, py::arg("filepath"),
			"Load configuration from INI file\n\n"
			"Args:\n"
			"    filepath: Path to configuration file")
		.def("save_config", &ProcessorWrapper::save_config, py::arg("filepath"),
			"Save current configuration to INI file\n\n"
			"Args:\n"
			"    filepath: Path to save configuration")
		.def_property_readonly("config", 
			[](ProcessorWrapper& self) -> ope::ProcessorConfiguration& {
				return const_cast<ope::ProcessorConfiguration&>(self.processor.getConfig());
			},
			py::return_value_policy::reference_internal,
			"Access to configuration object (read/write)")
		.def("set_config", [](ProcessorWrapper& self, const ope::ProcessorConfiguration& config) {
			self.processor.setConfig(config);
		}, py::arg("config"), "Set entire configuration at once")
		
		// Input parameters
		.def("set_input_parameters", 
			[](ProcessorWrapper& self, int signal_length, int ascans_per_bscan, int bscans_per_buffer, ope::DataType dtype) {
				py::gil_scoped_release release;
				self.processor.setInputParameters(signal_length, ascans_per_bscan, bscans_per_buffer, dtype);
			},
			py::arg("signal_length"),
			py::arg("ascans_per_bscan"),
			py::arg("bscans_per_buffer"),
			py::arg("data_type"),
			"Set input buffer parameters (requires reinitialization)")
		
		// Backend management
		.def("set_backend", [](ProcessorWrapper& self, ope::Backend backend) {
			py::gil_scoped_release release;
			self.processor.setBackend(backend);
		}, py::arg("backend"), "Switch backend (CUDA <-> CPU <-> OpenCL)")
		.def("get_backend", [](const ProcessorWrapper& self) {
			return self.processor.getBackend();
		}, "Get current backend")

		// Unified backend configuration API
		.def("set_backend_config", [](ProcessorWrapper& self, const ope::BackendConfig& config) {
			py::gil_scoped_release release;
			self.processor.setBackendConfig(config);
		}, py::arg("config"),
			"Set backend configuration\n\n"
			"Automatically switches backend if type differs from current.\n"
			"Preserves all processing configuration.\n\n"
			"Args:\n"
			"    config: Backend-specific configuration (CudaConfig, OpenCLConfig, or CpuConfig)\n\n"
			"Example:\n"
			"    >>> cuda_config = CudaConfig()\n"
			"    >>> cuda_config.device_id = 1\n"
			"    >>> processor.set_backend_config(cuda_config)")

		.def("get_backend_config", [](const ProcessorWrapper& self) -> std::unique_ptr<ope::BackendConfig> {
			return self.processor.getBackendConfig();
		}, "Get current backend configuration")

		.def("save_backend_config_to_file", [](const ProcessorWrapper& self, const std::string& filepath) {
			self.processor.saveBackendConfigToFile(filepath);
		}, py::arg("filepath"),
			"Save backend configuration to file\n\n"
			"Args:\n"
			"    filepath: Path to save configuration")

		.def("load_backend_config_from_file", [](ProcessorWrapper& self, const std::string& filepath) {
			py::gil_scoped_release release;
			self.processor.loadBackendConfigFromFile(filepath);
		}, py::arg("filepath"),
			"Load backend configuration from file\n\n"
			"Automatically switches backend if type differs from current.\n\n"
			"Args:\n"
			"    filepath: Path to load configuration from")
		
		// Processing
		.def("set_callback", &ProcessorWrapper::set_callback,
			py::arg("callback"),
			py::arg("error_callback") = py::none(),
			"Set callback function to receive processed output (legacy method)\n\n"
			"Note: Consider using add_output_callback() for new code.\n\n"
			"Args:\n"
			"    callback: Function that takes (NumPy array, buffer_id) as arguments\n"
			"    error_callback: Optional function to handle callback errors")

		.def("add_output_callback", &ProcessorWrapper::add_output_callback,
			py::arg("callback"),
			"Add output callback for multi-consumer support\n\n"
			"Allows multiple callbacks to receive processed data in parallel.\n"
			"Each callback runs in its own thread.\n\n"
			"Args:\n"
			"    callback: Function that takes (NumPy array, buffer_id) as arguments\n\n"
			"Returns:\n"
			"    int: Callback ID for later removal\n\n"
			"Example:\n"
			"    >>> def display(data, buffer_id):\n"
			"    ...     print(f'Displaying buffer {buffer_id}')\n"
			"    ...     cv2.imshow('OCT', data.copy())\n"
			"    >>> callback_id = processor.add_output_callback(display)\n"
			"    >>> processor.remove_output_callback(callback_id)")
		
		.def("remove_output_callback", &ProcessorWrapper::remove_output_callback,
			py::arg("callback_id"),
			"Remove a specific callback by ID\n\n"
			"Args:\n"
			"    callback_id: ID returned from add_output_callback()\n\n"
			"Returns:\n"
			"    bool: True if callback was found and removed")
		
		.def("clear_output_callbacks", &ProcessorWrapper::clear_output_callbacks,
			"Remove all output callbacks\n\n"
			"Stops all consumer threads and clears all registered callbacks.")
		
		.def("get_output_callback_count", &ProcessorWrapper::get_output_callback_count,
			"Get number of registered output callbacks\n\n"
			"Returns:\n"
			"    int: Number of active output callbacks")

		// Input callback methods (raw data before processing)
		.def("add_input_callback", &ProcessorWrapper::add_input_callback,
			py::arg("callback"),
			"Add input callback for raw data before processing\n\n"
			"Allows multiple callbacks to receive input data before processing.\n"
			"Each callback runs in its own thread.\n"
			"WARNING: Buffer is still in use by backend, copy data if needed beyond callback.\n\n"
			"Args:\n"
			"    callback: Function that takes (NumPy array, buffer_id) as arguments\n\n"
			"Returns:\n"
			"    int: Callback ID for later removal\n\n"
			"Example:\n"
			"    >>> def record_raw(data, buffer_id):\n"
			"    ...     print(f'Recording raw buffer {buffer_id}')\n"
			"    ...     np.save(f'raw_{buffer_id}.npy', data.copy())\n"
			"    >>> callback_id = processor.add_input_callback(record_raw)")

		.def("remove_input_callback", &ProcessorWrapper::remove_input_callback,
			py::arg("callback_id"),
			"Remove a specific input callback by ID\n\n"
			"Args:\n"
			"    callback_id: ID returned from add_input_callback()\n\n"
			"Returns:\n"
			"    bool: True if callback was found and removed")

		.def("clear_input_callbacks", &ProcessorWrapper::clear_input_callbacks,
			"Remove all input callbacks\n\n"
			"Stops all input callback threads and clears all registered input callbacks.")

		.def("get_input_callback_count", &ProcessorWrapper::get_input_callback_count,
			"Get number of registered input callbacks\n\n"
			"Returns:\n"
			"    int: Number of active input callbacks")

		.def("process", &ProcessorWrapper::process, py::arg("buffer"),
			"Process the input buffer asynchronously\n\n"
			"Args:\n"
			"    buffer: NumPy array obtained from get_next_available_buffer()\n\n"
			"Note: Callback must be set before calling process()")
		
		// Buffer management
		.def("get_next_available_buffer", &ProcessorWrapper::get_next_available_buffer,
			"Get next available input buffer (blocks if none available)\n\n"
			"Returns:\n"
			"    NumPy array view of the buffer (zero-copy)")
		
		// Hot-swap methods
		.def("set_resampling_coefficients", [](ProcessorWrapper& self, const std::array<float, 4>& coeffs) {
			self.processor.setResamplingCoefficients(coeffs.data());
		}, py::arg("coefficients"), "Set resampling coefficients [c0, c1, c2, c3]")
		.def("set_custom_resampling_curve", [](ProcessorWrapper& self, py::array_t<float> curve) {
			py::buffer_info buf = curve.request();
			self.processor.setCustomResamplingCurve(static_cast<float*>(buf.ptr), buf.size);
		}, py::arg("curve"), "Set custom resampling curve")
		.def("use_custom_resampling_curve", [](ProcessorWrapper& self, bool use_custom) {
			self.processor.useCustomResamplingCurve(use_custom);
		}, py::arg("use_custom"), "Enable/disable custom resampling curve")
		.def("enable_resampling", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableResampling(enable);
		}, py::arg("enable"), "Enable/disable resampling")
		.def("set_interpolation_method", [](ProcessorWrapper& self, ope::InterpolationMethod method) {
			self.processor.setInterpolationMethod(method);
		}, py::arg("method"), "Set interpolation method")
		
		.def("set_dispersion_coefficients", [](ProcessorWrapper& self, const std::array<float, 4>& coeffs, float factor) {
			self.processor.setDispersionCoefficients(coeffs.data(), factor);
		}, py::arg("coefficients"), py::arg("factor") = 1.0f, "Set dispersion coefficients [d0, d1, d2, d3] and factor")
		.def("set_custom_dispersion_curve", [](ProcessorWrapper& self, py::array_t<float> curve) {
			py::buffer_info buf = curve.request();
			self.processor.setCustomDispersionCurve(static_cast<float*>(buf.ptr), buf.size);
		}, py::arg("curve"), "Set custom dispersion curve")
		.def("use_custom_dispersion_curve", [](ProcessorWrapper& self, bool use_custom) {
			self.processor.useCustomDispersionCurve(use_custom);
		}, py::arg("use_custom"), "Enable/disable custom dispersion curve")
		.def("enable_dispersion_compensation", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableDispersionCompensation(enable);
		}, py::arg("enable"), "Enable/disable dispersion compensation")
		
		.def("set_window_parameters", [](ProcessorWrapper& self, ope::WindowType type, float center, float fill) {
			self.processor.setWindowParameters(type, center, fill);
		}, py::arg("window_type"), py::arg("center_position"), py::arg("fill_factor"),
			"Set windowing parameters")
		.def("set_custom_window_curve", [](ProcessorWrapper& self, py::array_t<float> curve) {
			py::buffer_info buf = curve.request();
			self.processor.setCustomWindowCurve(static_cast<float*>(buf.ptr), buf.size);
		}, py::arg("curve"), "Set custom window curve")
		.def("use_custom_window_curve", [](ProcessorWrapper& self, bool use_custom) {
			self.processor.useCustomWindowCurve(use_custom);
		}, py::arg("use_custom"), "Enable/disable custom window curve")
		.def("enable_windowing", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableWindowing(enable);
		}, py::arg("enable"), "Enable/disable windowing")
		
		.def("set_grayscale_range", [](ProcessorWrapper& self, float min, float max) {
			self.processor.setGrayscaleRange(min, max);
		}, py::arg("min"), py::arg("max"), "Set grayscale output range")
		.def("set_signal_multiplicator_and_addend", [](ProcessorWrapper& self, float mult, float add) {
			self.processor.setSignalMultiplicatorAndAddend(mult, add);
		}, py::arg("multiplicator"), py::arg("addend"), "Set signal multiplicator and addend")
		.def("enable_log_scaling", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableLogScaling(enable);
		}, py::arg("enable"), "Enable/disable logarithmic scaling")
		
		.def("enable_background_removal", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableBackgroundRemoval(enable);
		}, py::arg("enable"), "Enable/disable background removal")
		.def("set_background_removal_window_size", [](ProcessorWrapper& self, int window_size) {
			self.processor.setBackgroundRemovalWindowSize(window_size);
		}, py::arg("window_size"), "Set background removal window size")
		
		.def("enable_bscan_flip", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableBscanFlip(enable);
		}, py::arg("enable"), "Enable/disable B-scan flip")
		.def("enable_sinusoidal_scan_correction", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableSinusoidalScanCorrection(enable);
		}, py::arg("enable"), "Enable/disable sinusoidal scan correction")
		.def("enable_fixed_pattern_noise_removal", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableFixedPatternNoiseRemoval(enable);
		}, py::arg("enable"), "Enable/disable fixed pattern noise removal")
		
		.def("enable_post_process_background_subtraction", [](ProcessorWrapper& self, bool enable) {
			self.processor.enablePostProcessBackgroundSubtraction(enable);
		}, py::arg("enable"), "Enable/disable post-process background subtraction")
		
		.def("request_post_process_background_recording", [](ProcessorWrapper& self) {
			py::gil_scoped_release release;
			self.processor.requestPostProcessBackgroundRecording();
		}, "Request recording of next processed frame as background profile")
		
		.def("set_post_process_background_weight", [](ProcessorWrapper& self, float weight) {
			self.processor.setPostProcessBackgroundWeight(weight);
		}, py::arg("weight"), "Set post-process background subtraction weight")
		
		.def("set_post_process_background_offset", [](ProcessorWrapper& self, float offset) {
			self.processor.setPostProcessBackgroundOffset(offset);
		}, py::arg("offset"), "Set post-process background subtraction offset")
		
		.def("has_post_process_background_profile", [](const ProcessorWrapper& self) {
			return self.processor.hasPostProcessBackgroundProfile();
		}, "Check if a background profile is currently loaded")
		
		.def("get_post_process_background_profile_size", [](const ProcessorWrapper& self) {
			return self.processor.getPostProcessBackgroundProfileSize();
		}, "Get size of the background profile")
		
		.def("get_post_process_background_profile", [](const ProcessorWrapper& self) -> py::array_t<float> {
			if (!self.processor.hasPostProcessBackgroundProfile()) {
				throw BufferError("No background profile available");
			}
			size_t size = self.processor.getPostProcessBackgroundProfileSize();
			const float* data = self.processor.getPostProcessBackgroundProfile();
			py::array_t<float> result(size);
			py::buffer_info buf = result.request();
			float* ptr = static_cast<float*>(buf.ptr);
			std::memcpy(ptr, data, size * sizeof(float));
			return result;
		}, "Get the current background profile as a NumPy array")
		
		.def("set_post_process_background_profile", [](ProcessorWrapper& self, py::array_t<float> profile) {
			py::buffer_info buf = profile.request();
			
			if (buf.ndim != 1) {
				throw BufferError("Background profile must be a 1D array");
			}
			
			self.processor.setPostProcessBackgroundProfile(
				static_cast<float*>(buf.ptr), 
				buf.size
			);
		}, py::arg("profile"), "Set the background profile from a NumPy array")
		
		.def("save_post_process_background_profile_to_file", [](const ProcessorWrapper& self, const std::string& filepath) {
			try {
				self.processor.savePostProcessBackgroundProfileToFile(filepath);
			} catch (const std::exception& e) {
				throw ConfigurationError(std::string("Failed to save background profile: ") + e.what());
			}
		}, py::arg("filepath"), "Save the background profile to a file")
		
		.def("load_post_process_background_profile_from_file", [](ProcessorWrapper& self, const std::string& filepath) {
			try {
				self.processor.loadPostProcessBackgroundProfileFromFile(filepath);
			} catch (const std::exception& e) {
				throw ConfigurationError(std::string("Failed to load background profile: ") + e.what());
			}
		}, py::arg("filepath"), "Load a background profile from a file")
		
		.def("request_fixed_pattern_noise_determination", [](ProcessorWrapper& self) {
			self.processor.requestFixedPatternNoiseDetermination();
		}, "Request fixed-pattern noise determination on the next frame (one-shot mode)")
		
		.def("enable_continuous_fixed_pattern_noise_determination", [](ProcessorWrapper& self, bool enable) {
			self.processor.enableContinuousFixedPatternNoiseDetermination(enable);
		}, py::arg("enable"), "Enable or disable continuous fixed-pattern noise determination")
		
		.def("set_fixed_pattern_noise_bscan_count", [](ProcessorWrapper& self, int length) {
			self.processor.setFixedPatternNoiseBscanCount(length);
		}, py::arg("length"), "Set number of B-scans to use for fixed-pattern noise determination")
		
		.def("has_fixed_pattern_noise_profile", [](const ProcessorWrapper& self) {
			return self.processor.hasFixedPatternNoiseProfile();
		}, "Check if an fixed-pattern noise profile is currently loaded")
		
		.def("get_fixed_pattern_noise_profile_size", [](const ProcessorWrapper& self) {
			return self.processor.getFixedPatternNoiseProfileSize();
		}, "Get size of the fixed-pattern noise profile (in complex pairs)")
		
		.def("get_fixed_pattern_noise_profile", [](const ProcessorWrapper& self) -> py::array_t<float> {
			if (!self.processor.hasFixedPatternNoiseProfile()) {
				throw BufferError("No fixed-pattern noise profile available");
			}
			size_t pairs = self.processor.getFixedPatternNoiseProfileSize();
			const float* data = self.processor.getFixedPatternNoiseProfile();
			py::array_t<float> result(pairs * 2);  // interleaved real/imag
			py::buffer_info buf = result.request();
			float* ptr = static_cast<float*>(buf.ptr);
			std::memcpy(ptr, data, pairs * 2 * sizeof(float));
			return result;
		}, "Get the current fixed-pattern noise profile as a NumPy array (interleaved real/imag)")
		
		.def("set_fixed_pattern_noise_profile", [](ProcessorWrapper& self, py::array_t<float> profile) {
			py::buffer_info buf = profile.request();
			
			if (buf.ndim != 1) {
				throw BufferError("fixed-pattern noise profile must be a 1D array");
			}
			
			if (buf.size % 2 != 0) {
				throw BufferError("fixed-pattern noise profile must have even length (interleaved real/imag pairs)");
			}
			
			size_t complexPairs = buf.size / 2;
			self.processor.setFixedPatternNoiseProfile(
				static_cast<float*>(buf.ptr), 
				complexPairs
			);
		}, py::arg("profile"), "Set the fixed-pattern noise profile from a NumPy array (interleaved real/imag pairs)")
		
		.def("save_fixed_pattern_noise_profile_to_file", [](const ProcessorWrapper& self, const std::string& filepath) {
			try {
				self.processor.saveFixedPatternNoiseProfileToFile(filepath);
			} catch (const std::exception& e) {
				throw ConfigurationError(std::string("Failed to save fixed-pattern noise profile: ") + e.what());
			}
		}, py::arg("filepath"), "Save the fixed-pattern noise profile to a CSV file")
		
		.def("load_fixed_pattern_noise_profile_from_file", [](ProcessorWrapper& self, const std::string& filepath) {
			try {
				self.processor.loadFixedPatternNoiseProfileFromFile(filepath);
			} catch (const std::exception& e) {
				throw ConfigurationError(std::string("Failed to load fixed-pattern noise profile: ") + e.what());
			}
		}, py::arg("filepath"), "Load an fixed-pattern noise profile from a CSV file")

		// ============================================
		// BACKEND-SPECIFIC SETTINGS
		// ============================================

		.def("set_num_buffers", [](ProcessorWrapper& self, int num_buffers) {
			self.processor.setNumBuffers(num_buffers);
		}, py::arg("num_buffers"),
			"Set number of input buffers for pipelining\n\n"
			"Must be called before initialize() or after cleanup()\n\n"
			"Args:\n"
			"    num_buffers: Number of buffers (default: 2)")

		.def("get_num_buffers", [](const ProcessorWrapper& self) {
			return self.processor.getNumBuffers();
		}, "Get number of input buffers")

		// Context manager support
		.def("__enter__", &ProcessorWrapper::enter, py::return_value_policy::reference)
		.def("__exit__", &ProcessorWrapper::exit)
		
		// String representation
		.def("__repr__", [](const ProcessorWrapper& self) {
			std::string backend_str = (self.processor.getBackend() == ope::Backend::CUDA) ? "CUDA" :
			                          (self.processor.getBackend() == ope::Backend::OPENCL) ? "OpenCL" : "CPU";
			return "<Processor(backend=" + backend_str + ")>";
		});

	// ============================================
	// RECORDER
	// ============================================

	py::class_<ope::tools::Recorder::RecordingSummary>(m, "RecordingSummary")
		.def(py::init<>())
		.def_readonly("expected_buffers", &ope::tools::Recorder::RecordingSummary::expectedBuffers,
			"Number of buffers requested to record")
		.def_readonly("raw_recorded", &ope::tools::Recorder::RecordingSummary::rawRecorded,
			"Number of raw buffers actually recorded")
		.def_readonly("processed_recorded", &ope::tools::Recorder::RecordingSummary::processedRecorded,
			"Number of processed buffers actually recorded")
		.def_readonly("raw_buffer_ids", &ope::tools::Recorder::RecordingSummary::rawBufferIds,
			"List of buffer IDs for recorded raw buffers")
		.def_readonly("processed_buffer_ids", &ope::tools::Recorder::RecordingSummary::processedBufferIds,
			"List of buffer IDs for recorded processed buffers")
		.def_readonly("complete", &ope::tools::Recorder::RecordingSummary::complete,
			"True if all requested buffers were recorded")
		.def("__repr__", [](const ope::tools::Recorder::RecordingSummary& self) {
			return "<RecordingSummary(expected=" + std::to_string(self.expectedBuffers) +
			       ", raw=" + std::to_string(self.rawRecorded) +
			       ", processed=" + std::to_string(self.processedRecorded) +
			       ", complete=" + (self.complete ? "True" : "False") + ")>";
		});

	py::class_<ope::tools::Recorder>(m, "Recorder")
		.def(py::init<>(),
			"Create a new Recorder instance\n\n"
			"Must be attached to a Processor before use.")

		// Configuration
		.def("attach_to_processor", [](ope::tools::Recorder& self, ProcessorWrapper& wrapper) {
			py::gil_scoped_release release;
			self.attachToProcessor(&wrapper.processor);
		}, py::arg("processor"),
			"Attach recorder to a processor\n\n"
			"Args:\n"
			"    processor: Processor instance to attach to")

		.def("set_mode", &ope::tools::Recorder::setMode, py::arg("mode"),
			"Set recording mode\n\n"
			"Args:\n"
			"    mode: RecorderMode (RAW_ONLY, PROCESSED_ONLY, or BOTH)")

		.def("set_output_directory", &ope::tools::Recorder::setOutputDirectory, py::arg("directory"),
			"Set output directory for recorded files\n\n"
			"Args:\n"
			"    directory: Path to output directory")

		.def("set_output_base_name", &ope::tools::Recorder::setOutputBaseName, py::arg("base_name"),
			"Set base name for output files\n\n"
			"Args:\n"
			"    base_name: Base name (files will be named <base_name>_raw.bin, etc.)")

		.def("set_raw_format", &ope::tools::Recorder::setRawFormat, py::arg("format"),
			"Set format for raw data files\n\n"
			"Args:\n"
			"    format: RecorderFormat.RAW_BINARY")

		.def("set_processed_format", &ope::tools::Recorder::setProcessedFormat, py::arg("format"),
			"Set format for processed data files\n\n"
			"Args:\n"
			"    format: RecorderFormat.RAW_BINARY")

		.def("set_buffer_count", &ope::tools::Recorder::setBufferCount, py::arg("count"),
			"Set number of buffers to record\n\n"
			"Args:\n"
			"    count: Number of buffers to record")

		.def("get_buffer_count", &ope::tools::Recorder::getBufferCount,
			"Get number of buffers to record\n\n"
			"Returns:\n"
			"    int: Number of buffers")

		.def("set_write_metadata", &ope::tools::Recorder::setWriteMetadata, py::arg("enabled"),
			"Enable/disable writing metadata files\n\n"
			"Args:\n"
			"    enabled: True to write metadata")

		.def("set_use_timestamp", &ope::tools::Recorder::setUseTimestamp, py::arg("enabled"),
			"Enable/disable timestamps in filenames\n\n"
			"Args:\n"
			"    enabled: True to add timestamps")

		.def("set_manual_allocation", &ope::tools::Recorder::setManualAllocation, py::arg("manual"),
			"Enable manual buffer allocation mode\n\n"
			"If enabled, you must call allocate_buffers() before startRecording()\n\n"
			"Args:\n"
			"    manual: True for manual allocation")

		// Buffer management
		.def("allocate_buffers", [](ope::tools::Recorder& self) {
			py::gil_scoped_release release;
			self.allocateBuffers();
		},
			"Allocate recording buffers\n\n"
			"Only needed if set_manual_allocation(True) was called")

		.def("free_buffers", &ope::tools::Recorder::freeBuffers,
			"Free allocated recording buffers")

		.def("is_allocated", &ope::tools::Recorder::isAllocated,
			"Check if buffers are allocated\n\n"
			"Returns:\n"
			"    bool: True if buffers allocated")

		// Recording control
		.def("start_recording", [](ope::tools::Recorder& self, bool wait_for_volume_start) {
			py::gil_scoped_release release;
			self.startRecording(wait_for_volume_start);
		}, py::arg("wait_for_volume_start") = false,
			"Start recording\n\n"
			"Args:\n"
			"    wait_for_volume_start: If True, wait for volume boundary before recording")

		.def("stop_recording", [](ope::tools::Recorder& self) {
			py::gil_scoped_release release;
			self.stopRecording();
		},
			"Stop recording and write files to disk\n\n"
			"This will block until writing is complete")

		.def("abort_recording", &ope::tools::Recorder::abortRecording,
			"Abort recording without saving")

		.def("is_recording", &ope::tools::Recorder::isRecording,
			"Check if currently recording\n\n"
			"Returns:\n"
			"    bool: True if recording")

		.def("get_status", &ope::tools::Recorder::getStatus,
			"Get current recorder status\n\n"
			"Returns:\n"
			"    RecorderStatus: Current status")

		// Results
		.def("wait_for_completion", [](ope::tools::Recorder& self, int timeout_ms) {
			py::gil_scoped_release release;
			return self.waitForCompletion(timeout_ms);
		}, py::arg("timeout_ms"),
			"Wait for recording to complete\n\n"
			"Args:\n"
			"    timeout_ms: Timeout in milliseconds\n\n"
			"Returns:\n"
			"    bool: True if completed successfully, False if timed out or failed")

		.def("get_last_recording_summary", &ope::tools::Recorder::getLastRecordingSummary,
			"Get summary of last recording\n\n"
			"Returns:\n"
			"    RecordingSummary: Summary with buffer counts and IDs")

		.def("get_last_error", &ope::tools::Recorder::getLastError,
			"Get last error message\n\n"
			"Returns:\n"
			"    str: Error message or empty string if no error")

		.def("__repr__", [](const ope::tools::Recorder& self) {
			const char* status_str[] = {"IDLE", "READY", "RECORDING", "WRITING", "COMPLETED", "ERROR"};
			int status_idx = static_cast<int>(self.getStatus());
			return "<Recorder(status=" + std::string(status_str[status_idx]) + ")>";
		});

	// ============================================
	// MODULE VERSION
	// ============================================

	m.attr("__version__") = OPE_VERSION_STRING;
}
