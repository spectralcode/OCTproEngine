#include "bindings_common.h"

void register_processor(py::module& m) {
		py::class_<ProcessorWrapper>(m, "Processor")
			.def(py::init<ope::Backend>(), py::arg("backend") = ope::Backend::CPU,
				"Create a new Processor instance\n\n"
				"Args:\n"
				"    backend: Backend to use (Backend.VULKAN, Backend.CUDA, Backend.CPU, or Backend.OPENCL)")
		
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
			}, py::arg("backend"), "Switch backend (VULKAN <-> CUDA <-> CPU <-> OpenCL)")
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
				"The buffer is reference-protected while the callback runs; copy the\n"
				"data if you need it beyond the callback.\n\n"
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
				                          (self.processor.getBackend() == ope::Backend::OPENCL) ? "OpenCL" :
				                          (self.processor.getBackend() == ope::Backend::VULKAN) ? "Vulkan" : "CPU";
				return "<Processor(backend=" + backend_str + ")>";
			});
}
