#include "bindings_common.h"

void register_recorder(py::module& m) {
	// RecordingSummary structure
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

	// Recorder class
	py::class_<ope::tools::Recorder>(m, "Recorder")
		.def(py::init<>(),
			"Create a new Recorder instance\n\n"
			"Must be attached to a Processor before use.")

		// Configuration
		.def("attach_to_processor", [](ope::tools::Recorder& self, ProcessorWrapper& wrapper) {
			py::gil_scoped_release release;
			self.attachToProcessor(&wrapper.processor);
		}, py::arg("processor"), py::keep_alive<1, 2>(),
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
}
