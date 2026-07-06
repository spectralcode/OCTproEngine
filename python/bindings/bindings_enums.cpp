#include "bindings_common.h"

void register_exceptions(py::module& m) {
	py::register_exception<InitializationError>(m, "InitializationError");
	py::register_exception<ConfigurationError>(m, "ConfigurationError");
	py::register_exception<BufferError>(m, "BufferError");
	py::register_exception<ProcessingError>(m, "ProcessingError");
	py::register_exception<BackendError>(m, "BackendError");
	py::register_exception<ope::tools::RecordingException>(m, "RecordingException");
}

void register_enums(py::module& m) {
	// IOBuffer class
	py::class_<ope::IOBuffer>(m, "IOBuffer", py::module_local())
		.def("get_size", &ope::IOBuffer::getSizeInBytes, "Get buffer size in bytes")
		.def("get_data_type", &ope::IOBuffer::getDataType, "Get buffer data type");

	// Backend enum
	py::enum_<ope::Backend>(m, "Backend")
		.value("CUDA", ope::Backend::CUDA, "NVIDIA CUDA GPU backend")
		.value("CPU", ope::Backend::CPU, "CPU backend")
		.value("OPENCL", ope::Backend::OPENCL, "OpenCL GPU backend")
		.value("VULKAN", ope::Backend::VULKAN, "Vulkan GPU backend")
		.export_values();

	// DataType enum
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

	// InterpolationMethod enum
	py::enum_<ope::InterpolationMethod>(m, "InterpolationMethod")
		.value("LINEAR", ope::InterpolationMethod::LINEAR, "Linear interpolation")
		.value("CUBIC", ope::InterpolationMethod::CUBIC, "Cubic interpolation")
		.value("LANCZOS", ope::InterpolationMethod::LANCZOS, "Lanczos interpolation")
		.export_values();

	// WindowType enum
	py::enum_<ope::WindowType>(m, "WindowType")
		.value("HANN", ope::WindowType::HANN, "Hann window")
		.value("GAUSS", ope::WindowType::GAUSS, "Gaussian window")
		.value("SINE", ope::WindowType::SINE, "Sine window")
		.value("LANCZOS", ope::WindowType::LANCZOS, "Lanczos window")
		.value("RECTANGULAR", ope::WindowType::RECTANGULAR, "Rectangular window")
		.value("FLAT_TOP", ope::WindowType::FLAT_TOP, "Flat-top window")
		.export_values();

	// Recorder enums
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

	// Configuration enums
	py::enum_<ope::ProcessorConfiguration::LoadMode>(m, "LoadMode")
		.value("OVERWRITE_ALL", ope::ProcessorConfiguration::LoadMode::OVERWRITE_ALL)
		.value("PARAMETERS_ONLY", ope::ProcessorConfiguration::LoadMode::PARAMETERS_ONLY)
		.value("MERGE_IF_MISSING", ope::ProcessorConfiguration::LoadMode::MERGE_IF_MISSING);

	py::enum_<ope::ProcessorConfiguration::SaveMode>(m, "SaveMode")
		.value("PARAMETERS_ONLY", ope::ProcessorConfiguration::SaveMode::PARAMETERS_ONLY)
		.value("COMPLETE", ope::ProcessorConfiguration::SaveMode::COMPLETE);
}
