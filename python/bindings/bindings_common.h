#ifndef OCTPROENGINE_BINDINGS_COMMON_H
#define OCTPROENGINE_BINDINGS_COMMON_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "processor.h"
#include "processorconfiguration.h"
#include "backendconfig.h"
#include "iobuffer.h"
#include "types.h"
#include "version.h"
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

inline ope::DataType numpy_dtype_to_ope(py::dtype dtype) {
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

inline std::string ope_dtype_to_string(ope::DataType dtype) {
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
// config: optional processor configuration for proper 3D shape
inline py::array buffer_to_numpy(ope::IOBuffer& buffer, const ope::ProcessorConfiguration* config = nullptr) {
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

	// Create properly shaped 3D array if configuration is available
	if (config) {
		int bscans = config->dataParams.bscansPerBuffer;
		int ascans = config->dataParams.ascansPerBscan;
		// Calculate signal length from actual buffer size
		int signal = num_elements / (bscans * ascans);

		// Shape: (bscans, ascans, signal_length)
		// Strides: in bytes for each dimension (row-major C order)
		size_t stride_signal = np_dtype.itemsize();
		size_t stride_ascan = stride_signal * signal;
		size_t stride_bscan = stride_ascan * ascans;

		return py::array(np_dtype,
			{bscans, ascans, signal},  // shape
			{stride_bscan, stride_ascan, stride_signal},  // strides
			ptr,
			py::cast(&buffer));
	} else {
		// Fallback to 1D array if no config available
		return py::array(np_dtype, {num_elements}, {np_dtype.itemsize()}, ptr, py::cast(&buffer));
	}
}

// ============================================
// PROCESSOR WRAPPER WITH CALLBACK SUPPORT
// ============================================

class ProcessorWrapper {
public:
	ope::Processor processor;
	py::function callback;
	py::function error_callback;

	// Keepalives for Python callback objects, split by callback direction so
	// clearing one direction cannot free the other direction's live callbacks
	std::map<ope::Processor::CallbackId, py::function> pyOutputCallbacks;
	std::map<ope::Processor::CallbackId, py::function> pyInputCallbacks;
	std::mutex callbacksMutex;

	ProcessorWrapper(ope::Backend backend) : processor(backend) {}

	~ProcessorWrapper() {
		// Tear down the callback workers with the GIL released: python destroys
		// this object while holding the GIL, but joining a callback worker
		// requires the worker to finish (callback execution and the callback
		// holder's GIL-taking deleter both need the GIL) - joining with the GIL
		// held deadlocks. After this body the members (including the keepalive
		// maps) destruct with the GIL held again, which is what they need.
		py::gil_scoped_release release;
		processor.clearOutputCallbacks();
		processor.clearInputCallbacks();
	}

	void set_callback(py::function cb, py::object error_cb = py::none()) {
		callback = cb;
		if (!error_cb.is_none()) {
			error_callback = error_cb.cast<py::function>();
		}

		// Set C++ callback that will call Python callback
		// Clear existing callbacks and add new one (legacy behavior)
		this->clear_output_callbacks();
		processor.addOutputCallback([this](const ope::IOBuffer& output) {
			// Capture buffer ID immediately before buffer can be recycled
			uint64_t bufferId = output.getBufferId();

			// Re-acquire GIL to call Python code
			py::gil_scoped_acquire acquire;

			try {
				// Create NumPy view of output buffer (cast away const for view)
				ope::IOBuffer& output_ref = const_cast<ope::IOBuffer&>(output);
				py::array output_array = buffer_to_numpy(output_ref, &processor.getConfig());

				// Call Python callback with captured buffer ID
				callback(output_array, bufferId);
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
		// Hold the Python callback through a shared_ptr whose deleter takes the
		// GIL: the closure below is copied into a callback worker thread, and the
		// last copy can be destroyed on that thread when it exits - destroying a
		// py::function without holding the GIL crashes the interpreter
		auto cbHolder = std::shared_ptr<py::function>(new py::function(cb), [](py::function* f) {
			py::gil_scoped_acquire acquire;
			delete f;
		});

		// Create C++ wrapper callback that handles GIL
		auto wrappedCallback = [this, cbHolder](const ope::IOBuffer& buffer) {
			// Capture buffer ID immediately before buffer can be recycled
			uint64_t bufferId = buffer.getBufferId();

			// Re-acquire GIL before calling Python code
			// We're in a C++ callback thread, need GIL to call Python
			py::gil_scoped_acquire acquire;

			try {
				// Create NumPy view of buffer (zero-copy)
				ope::IOBuffer& buffer_ref = const_cast<ope::IOBuffer&>(buffer);
				py::array output_array = buffer_to_numpy(buffer_ref, &processor.getConfig());

				// Call Python callback with captured buffer ID
				(*cbHolder)(output_array, bufferId);

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
			pyOutputCallbacks[id] = cb;
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
			pyOutputCallbacks.erase(id);
		}

		return removed;
	}

	void clear_output_callbacks() {
		{
			py::gil_scoped_release release;
			processor.clearOutputCallbacks();
		}

		std::lock_guard<std::mutex> lock(callbacksMutex);
		pyOutputCallbacks.clear();
	}

	size_t get_output_callback_count() const {
		return processor.getOutputCallbackCount();
	}

	// Input callback methods (for raw data before processing)
	ope::Processor::CallbackId add_input_callback(py::function cb) {
		// Hold the Python callback through a shared_ptr whose deleter takes the
		// GIL, see add_output_callback for the rationale
		auto cbHolder = std::shared_ptr<py::function>(new py::function(cb), [](py::function* f) {
			py::gil_scoped_acquire acquire;
			delete f;
		});

		// Create C++ wrapper callback that handles GIL
		auto wrappedCallback = [this, cbHolder](const ope::IOBuffer& buffer) {
			// Capture buffer ID immediately before buffer can be recycled
			uint64_t bufferId = buffer.getBufferId();

			// Re-acquire GIL before calling Python code
			py::gil_scoped_acquire acquire;

			try {
				// Create NumPy view of buffer (zero-copy)
				ope::IOBuffer& buffer_ref = const_cast<ope::IOBuffer&>(buffer);
				py::array input_array = buffer_to_numpy(buffer_ref, &processor.getConfig());

				// Call Python callback with captured buffer ID
				(*cbHolder)(input_array, bufferId);

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
			pyInputCallbacks[id] = cb;
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
			pyInputCallbacks.erase(id);
		}

		return removed;
	}

	void clear_input_callbacks() {
		{
			py::gil_scoped_release release;
			processor.clearInputCallbacks();
		}

		std::lock_guard<std::mutex> lock(callbacksMutex);
		pyInputCallbacks.clear();
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

		return buffer_to_numpy(*buffer, &processor.getConfig());
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
// REGISTRATION FUNCTION DECLARATIONS
// ============================================

void register_exceptions(py::module& m);
void register_enums(py::module& m);
void register_backend_config(py::module& m);
void register_configuration(py::module& m);
void register_processor(py::module& m);
void register_recorder(py::module& m);

#endif // OCTPROENGINE_BINDINGS_COMMON_H
