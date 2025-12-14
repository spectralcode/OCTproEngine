#include "bindings_common.h"

void register_configuration(py::module& m) {
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

	// ProcessorConfiguration class
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
}
