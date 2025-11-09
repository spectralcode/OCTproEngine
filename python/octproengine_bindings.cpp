#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "processor.h"
#include "processorconfiguration.h"
#include "iobuffer.h"
#include "types.h"
#include "version.h"

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
				
				// Call Python callback
				callback(output_array);
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
	
	void process(py::array buffer_array) {
		// Validate that callback is set
		if (callback.is_none()) {
			throw ProcessingError("Callback must be set before calling process(). Use set_callback() first.");
		}
		
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
	
	void stop() {
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
	
	// ============================================
	// CONFIGURATION STRUCTS
	// ============================================
	
	py::class_<ope::ProcessorConfiguration::DataParameters>(m, "DataParameters")
		.def(py::init<>())
		.def_readwrite("signal_length", &ope::ProcessorConfiguration::DataParameters::signalLength)
		.def_readwrite("samples_per_buffer", &ope::ProcessorConfiguration::DataParameters::samplesPerBuffer)
		.def_readwrite("ascans_per_bscan", &ope::ProcessorConfiguration::DataParameters::ascansPerBscan)
		.def_readwrite("bscans_per_buffer", &ope::ProcessorConfiguration::DataParameters::bscansPerBuffer)
		.def_readwrite("input_data_type", &ope::ProcessorConfiguration::DataParameters::inputDataType)
		.def_readwrite("bitshift", &ope::ProcessorConfiguration::DataParameters::bitshift)
		.def("get_bit_depth", &ope::ProcessorConfiguration::DataParameters::getBitDepth)
		.def("get_bytes_per_sample", &ope::ProcessorConfiguration::DataParameters::getBytesPerSample);
	
	py::class_<ope::ProcessorConfiguration::ProcessingParameters>(m, "ProcessingParameters")
		.def(py::init<>())
		.def_readwrite("n_streams", &ope::ProcessorConfiguration::ProcessingParameters::nStreams)
		.def_readwrite("n_buffers", &ope::ProcessorConfiguration::ProcessingParameters::nBuffers)
		.def_readwrite("grid_size", &ope::ProcessorConfiguration::ProcessingParameters::gridSize)
		.def_readwrite("block_size", &ope::ProcessorConfiguration::ProcessingParameters::blockSize);
	
	py::class_<ope::ProcessorConfiguration::ResamplingParameters>(m, "ResamplingParameters")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::ResamplingParameters::enabled)
		.def_readwrite("interpolation_method", &ope::ProcessorConfiguration::ResamplingParameters::interpolationMethod)
		.def_readwrite("use_coefficients", &ope::ProcessorConfiguration::ResamplingParameters::useCoefficients)
		.def_property("coefficients",
			[](const ope::ProcessorConfiguration::ResamplingParameters& self) {
				return std::vector<float>(self.coefficients, self.coefficients + 4);
			},
			[](ope::ProcessorConfiguration::ResamplingParameters& self, const std::vector<float>& coeffs) {
				if (coeffs.size() != 4) throw std::runtime_error("Coefficients must have exactly 4 elements");
				std::copy(coeffs.begin(), coeffs.end(), self.coefficients);
			})
		.def_readwrite("use_custom_curve", &ope::ProcessorConfiguration::ResamplingParameters::useCustomCurve);
	
	py::class_<ope::ProcessorConfiguration::WindowingParameters>(m, "WindowingParameters")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::WindowingParameters::enabled)
		.def_readwrite("window_type", &ope::ProcessorConfiguration::WindowingParameters::windowType)
		.def_readwrite("window_center_position", &ope::ProcessorConfiguration::WindowingParameters::windowCenterPosition)
		.def_readwrite("window_fill_factor", &ope::ProcessorConfiguration::WindowingParameters::windowFillFactor)
		.def_readwrite("use_custom_curve", &ope::ProcessorConfiguration::WindowingParameters::useCustomCurve);
	
	py::class_<ope::ProcessorConfiguration::DispersionCompensationParameters>(m, "DispersionCompensationParameters")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::DispersionCompensationParameters::enabled)
		.def_readwrite("use_coefficients", &ope::ProcessorConfiguration::DispersionCompensationParameters::useCoefficients)
		.def_property("coefficients",
			[](const ope::ProcessorConfiguration::DispersionCompensationParameters& self) {
				return std::vector<float>(self.coefficients, self.coefficients + 4);
			},
			[](ope::ProcessorConfiguration::DispersionCompensationParameters& self, const std::vector<float>& coeffs) {
				if (coeffs.size() != 4) throw std::runtime_error("Coefficients must have exactly 4 elements");
				std::copy(coeffs.begin(), coeffs.end(), self.coefficients);
			})
		.def_readwrite("factor", &ope::ProcessorConfiguration::DispersionCompensationParameters::factor)
		.def_readwrite("use_custom_curve", &ope::ProcessorConfiguration::DispersionCompensationParameters::useCustomCurve);
	
	py::class_<ope::ProcessorConfiguration::BackgroundRemovalParameters>(m, "BackgroundRemovalParameters")
		.def(py::init<>())
		.def_readwrite("enabled", &ope::ProcessorConfiguration::BackgroundRemovalParameters::enabled)
		.def_readwrite("rolling_average_window_size", &ope::ProcessorConfiguration::BackgroundRemovalParameters::rollingAverageWindowSize);
	
	py::class_<ope::ProcessorConfiguration::PostProcessingParameters>(m, "PostProcessingParameters")
		.def(py::init<>())
		.def_readwrite("background_removal", &ope::ProcessorConfiguration::PostProcessingParameters::backgroundRemoval)
		.def_readwrite("background_weight", &ope::ProcessorConfiguration::PostProcessingParameters::backgroundWeight)
		.def_readwrite("background_offset", &ope::ProcessorConfiguration::PostProcessingParameters::backgroundOffset)
		.def_readwrite("log_scaling", &ope::ProcessorConfiguration::PostProcessingParameters::logScaling)
		.def_readwrite("grayscale_max", &ope::ProcessorConfiguration::PostProcessingParameters::grayscaleMax)
		.def_readwrite("grayscale_min", &ope::ProcessorConfiguration::PostProcessingParameters::grayscaleMin)
		.def_readwrite("addend", &ope::ProcessorConfiguration::PostProcessingParameters::addend)
		.def_readwrite("multiplicator", &ope::ProcessorConfiguration::PostProcessingParameters::multiplicator)
		.def_readwrite("bscan_flip", &ope::ProcessorConfiguration::PostProcessingParameters::bscanFlip)
		.def_readwrite("sinusoidal_scan_correction", &ope::ProcessorConfiguration::PostProcessingParameters::sinusoidalScanCorrection)
		.def_readwrite("fixed_pattern_noise_removal", &ope::ProcessorConfiguration::PostProcessingParameters::fixedPatternNoiseRemoval)
		.def_readwrite("bscans_for_noise_determination", &ope::ProcessorConfiguration::PostProcessingParameters::bscansForNoiseDetermination)
		.def_readwrite("continuous_fixed_pattern_noise_determination", &ope::ProcessorConfiguration::PostProcessingParameters::continuousFixedPatternNoiseDetermination);
	
	// ============================================
	// CONFIGURATION CLASS
	// ============================================
	
	py::class_<ope::ProcessorConfiguration>(m, "ProcessorConfiguration")
		.def(py::init<>())
		.def_readwrite("data", &ope::ProcessorConfiguration::dataParams)
		.def_readwrite("processing", &ope::ProcessorConfiguration::processingParams)
		.def_readwrite("resampling", &ope::ProcessorConfiguration::resamplingParams)
		.def_readwrite("windowing", &ope::ProcessorConfiguration::windowingParams)
		.def_readwrite("dispersion", &ope::ProcessorConfiguration::dispersionParams)
		.def_readwrite("background_removal", &ope::ProcessorConfiguration::backgroundRemovalParams)
		.def_readwrite("post_processing", &ope::ProcessorConfiguration::postProcessingParams)
		// Custom curve methods
		.def("set_custom_resampling_curve", [](ope::ProcessorConfiguration& self, py::array_t<float> curve) {
			py::buffer_info buf = curve.request();
			self.setCustomResamplingCurve(static_cast<float*>(buf.ptr), buf.size);
		})
		.def("set_custom_window_curve", [](ope::ProcessorConfiguration& self, py::array_t<float> curve) {
			py::buffer_info buf = curve.request();
			self.setCustomWindowCurve(static_cast<float*>(buf.ptr), buf.size);
		})
		.def("set_custom_dispersion_curve", [](ope::ProcessorConfiguration& self, py::array_t<float> curve) {
			py::buffer_info buf = curve.request();
			self.setCustomDispersionCurve(static_cast<float*>(buf.ptr), buf.size);
		})
		.def("get_custom_resampling_curve", [](const ope::ProcessorConfiguration& self) -> py::array_t<float> {
			if (!self.hasCustomResamplingCurve()) return py::array_t<float>(0);
			size_t size = self.getCustomResamplingCurveSize();
			const float* data = self.getCustomResamplingCurve();
			return py::array_t<float>(size, data);
		})
		.def("get_custom_window_curve", [](const ope::ProcessorConfiguration& self) -> py::array_t<float> {
			if (!self.hasCustomWindowCurve()) return py::array_t<float>(0);
			size_t size = self.getCustomWindowCurveSize();
			const float* data = self.getCustomWindowCurve();
			return py::array_t<float>(size, data);
		})
		.def("get_custom_dispersion_curve", [](const ope::ProcessorConfiguration& self) -> py::array_t<float> {
			if (!self.hasCustomDispersionCurve()) return py::array_t<float>(0);
			size_t size = self.getCustomDispersionCurveSize();
			const float* data = self.getCustomDispersionCurve();
			return py::array_t<float>(size, data);
		})
		.def("has_custom_resampling_curve", &ope::ProcessorConfiguration::hasCustomResamplingCurve)
		.def("has_custom_window_curve", &ope::ProcessorConfiguration::hasCustomWindowCurve)
		.def("has_custom_dispersion_curve", &ope::ProcessorConfiguration::hasCustomDispersionCurve)
		.def("save_to_file", &ope::ProcessorConfiguration::saveToFile)
		.def("load_from_file", &ope::ProcessorConfiguration::loadFromFile)
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
		}, py::arg("backend"), "Switch backend (CUDA <-> CPU)")
		.def("get_backend", [](const ProcessorWrapper& self) { 
			return self.processor.getBackend(); 
		}, "Get current backend")
		
		// Processing
		.def("set_callback", &ProcessorWrapper::set_callback,
			py::arg("callback"),
			py::arg("error_callback") = py::none(),
			"Set callback function to receive processed output\n\n"
			"Args:\n"
			"    callback: Function that takes a NumPy array as argument\n"
			"    error_callback: Optional function to handle callback errors")
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
		.def("enable_post_process_background_removal", [](ProcessorWrapper& self, bool enable) {
			self.processor.enablePostProcessBackgroundRemoval(enable);
		}, py::arg("enable"), "Enable/disable post-process background removal")
		
		// Context manager support
		.def("__enter__", &ProcessorWrapper::enter, py::return_value_policy::reference)
		.def("__exit__", &ProcessorWrapper::exit)
		
		// String representation
		.def("__repr__", [](const ProcessorWrapper& self) {
			std::string backend_str = (self.processor.getBackend() == ope::Backend::CUDA) ? "CUDA" : "CPU";
			return "<Processor(backend=" + backend_str + ")>";
		});
	
	// ============================================
	// MODULE VERSION
	// ============================================
	
	m.attr("__version__") = OPE_VERSION_STRING;
}
