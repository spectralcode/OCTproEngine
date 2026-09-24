#include "../src/utils/csvhelper.h"
#include "../include/processorconfiguration.h"
#include "../include/processor.h"
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace {

const char* fixturePath = "test_csv_helper_input.csv";
const char* outputPath = "test_csv_helper_output.csv";

void require(bool condition, const std::string& message) {
	if (!condition) throw std::runtime_error(message);
}

void writeFixture(const std::string& text) {
	std::ofstream file(fixturePath, std::ios::binary);
	file << text;
	file.close();
	require(file.good(), "Could not write CSV fixture");
}

void testImports() {
	const std::vector<std::string> realFiles = {
		"\n \t# Resampling LUT, with comments\n\nindex;value\n0;7.25\n1;8.5\n",
		"Sample Number;Sample Value\n0;7.25\n1;8.5\n",
		"Messpunkt;Wert\n0;7.25\n1;8.5\n",
		"\xEF\xBB\xBF  Sample Number ; Sample Value \r\n 0 ; 7.25 \r\n\r\n # comment\r\n1 ; 8.5\r\n",
		"index;value\n0;7.25e0\n1;+8.5E0\n"
	};
	for (size_t i = 0; i < realFiles.size(); ++i) {
		writeFixture(realFiles[i]);
		require(ope::CSVHelper::load(fixturePath) == std::vector<float>({7.25f, 8.5f}),
			"Real import " + std::to_string(i));
	}

	for (const auto& header : {"index;real;imaginary\n", "Sample Number;Real;Imag\n"}) {
		writeFixture("# FPN, complex\n" + std::string(header) + "0;1.25;-2.5\n # comment\n1;3;4\n");
		require(ope::CSVHelper::load(fixturePath) == std::vector<float>({1.25f, -2.5f, 3, 4}),
			"Complex import");
	}

	const std::vector<std::string> invalidFiles = {
		"", "# comment only\n", "index;value\n", "Value\n7.25\n8.5\n",
		"index,value\n0,7.25\n1,8.5\n", "index;value\n0;0,5\n",
		"index;value\n0;1.5garbage\n", "index;value\n0;nan\n",
		"index;value\n0;inf\n", "index;value\n0;1e100\n",
		"index;value\n0;1\n1;broken\n", "index;value\n0;1\nindex;value\n",
		"index;value\n0;\n", "index;value\n0;1;2\n", "index;value\n0\t7.25\n",
		"Sample Number;Raw;Processed\n0;1;2\n"
	};
	for (size_t i = 0; i < invalidFiles.size(); ++i) {
		writeFixture(invalidFiles[i]);
		require(ope::CSVHelper::load(fixturePath).empty(), "Invalid real import accepted: " + std::to_string(i));
	}
	for (const auto& text : {
		"index,real,imaginary\n0,1,2\n", "index;real;imaginary\n0;1,5;2\n",
		"index;real;imaginary\n0;1\n",
		"index;real;imaginary\n0;1;\n", "index;real;imaginary\n0;1;2;3\n",
		"index;real;imaginary\n0;1;2\n1;nan;3\n",
		"index;real;imaginary\n0;1;2\nindex;real;imaginary\n1;3;4\n",
		"Sample Number;Raw;Processed\n0;1;2\n"}) {
		writeFixture(text);
		require(ope::CSVHelper::load(fixturePath).empty(), "Invalid complex import accepted");
	}
	require(ope::CSVHelper::load("nonexistent_csv_helper_directory/file.csv").empty(), "Missing file accepted");
}

void testRoundTrips() {
	const std::vector<float> values = {1.23456789f, -8.5f, std::numeric_limits<float>::min(),
		std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min(), 0};
	require(ope::CSVHelper::save(outputPath, values, "Resampling LUT"), "Save real CSV");
	require(ope::CSVHelper::load(outputPath) == values, "Exact real round trip");
	require(ope::CSVHelper::saveComplex(outputPath, values, "Fixed Pattern Noise Profile"), "Save complex CSV");
	require(ope::CSVHelper::load(outputPath) == values, "Exact complex round trip");
	require(ope::CSVHelper::save(outputPath, values), "Default real export");
	std::ifstream file(outputPath);
	std::string line;
	std::getline(file, line);
	require(line == "index;value", "Default export must begin with a single semicolon header");
	// Mirror OCTproZ's reader: skip one line, then read the second semicolon field.
	std::vector<float> octprozValues;
	while (std::getline(file, line)) {
		const size_t separator = line.find(';');
		require(separator != std::string::npos && line.find(';', separator + 1) == std::string::npos,
			"OCTproZ rows must contain exactly two fields");
		std::istringstream valueStream(line.substr(separator + 1));
		valueStream.imbue(std::locale::classic());
		float value;
		require(static_cast<bool>(valueStream >> value), "Non-numeric OCTproZ export row");
		octprozValues.push_back(value);
	}
	file.close();
	require(octprozValues == values, "OCTproZ-compatible export must preserve all samples");

	require(!ope::CSVHelper::saveComplex(outputPath, {1}), "Odd complex value count accepted");
	require(ope::CSVHelper::load(outputPath) == values, "Rejected exports must not truncate existing files");
}

struct CommaDecimal : std::numpunct<char> {
	char do_decimal_point() const override { return ','; }
};

void testLocaleIndependentImport() {
	writeFixture("Sample Number;Sample Value\n0;1.25\n1;2.5\n");
	const std::locale previous = std::locale();
	std::locale::global(std::locale(previous, new CommaDecimal));
	const auto values = ope::CSVHelper::load(fixturePath);
	const bool saved = ope::CSVHelper::save(outputPath, values);
	const auto exported = ope::CSVHelper::load(outputPath);
	std::locale::global(previous);
	require(values == std::vector<float>({1.25f, 2.5f}), "Import must use dot decimals regardless of locale");
	require(saved && exported == values, "Export must use dot decimals regardless of locale");
}

void testProfileExchange() {
	ope::ProcessorConfiguration config;
	config.dataParams.signalLength = 32;
	config.dataParams.ascansPerBscan = 4;
	const std::vector<float> curve(32, 1.23456789f);
	const std::vector<float> background(16, 0.123456789f);
	std::vector<float> fpn(32);
	for (size_t i = 0; i < fpn.size(); ++i) fpn[i] = static_cast<float>(i) * -0.123456789f;
	config.setResamplingLut(curve);
	config.setWindowFunction(curve);
	config.setDispersionPhase(curve);
	config.setBackgroundProfile(background);
	config.setFixedPatternNoiseProfile(fpn);
	ope::Processor processor(ope::Backend::CPU);
	processor.setInputParameters(32, 4, 1, ope::DataType::UINT16);
	processor.initialize();
	require(config.saveResamplingLutToFile(outputPath) && ope::CSVHelper::load(outputPath) == curve,
		"Config resampling export");
	require(config.saveWindowFunctionToFile(outputPath) && ope::CSVHelper::load(outputPath) == curve,
		"Config window export");
	require(config.saveDispersionPhaseToFile(outputPath) && ope::CSVHelper::load(outputPath) == curve,
		"Config dispersion export");
	require(config.saveBackgroundProfileToFile(outputPath), "Config background export");
	processor.loadPostProcessBackgroundProfileFromFile(outputPath);
	const float* bg = processor.getPostProcessBackgroundProfile();
	require(processor.getPostProcessBackgroundProfileSize() == background.size() &&
		std::vector<float>(bg, bg + background.size()) == background, "Config to Processor background");
	processor.savePostProcessBackgroundProfileToFile(outputPath);
	require(config.loadBackgroundProfileFromFile(outputPath) && config.getBackgroundProfile() == background,
		"Processor to config background");

	require(config.saveFixedPatternNoiseProfileToFile(outputPath), "Config FPN export");
	processor.loadFixedPatternNoiseProfileFromFile(outputPath);
	const float* noise = processor.getFixedPatternNoiseProfile();
	require(processor.getFixedPatternNoiseProfileSize() * 2 == fpn.size() &&
		std::vector<float>(noise, noise + fpn.size()) == fpn, "Config to Processor FPN");
	processor.saveFixedPatternNoiseProfileToFile(outputPath);
	require(config.loadFixedPatternNoiseProfileFromFile(outputPath) && config.getFixedPatternNoiseProfile() == fpn,
		"Processor to config FPN");
	writeFixture("index;value\n0;1\n1;broken\n");
	bool rejected = false;
	try { processor.loadPostProcessBackgroundProfileFromFile(fixturePath); }
	catch (const std::runtime_error&) { rejected = true; }
	bg = processor.getPostProcessBackgroundProfile();
	require(rejected && std::vector<float>(bg, bg + background.size()) == background,
		"Invalid Processor import must preserve the existing profile");
}

void testConfigurationImports() {
	ope::ProcessorConfiguration config;
	config.dataParams.signalLength = 4;
	writeFixture("Sample Number;Sample Value\n0;1.25\n1;2.5\n2;3.75\n3;4\n");
	const std::vector<float> values = {1.25f, 2.5f, 3.75f, 4};
	require(config.loadResamplingLutFromFile(fixturePath) && config.getResamplingLut() == values &&
		config.processingParams.resampling.useCustomLut, "OCTproZ resampling config import");
	require(config.loadWindowFunctionFromFile(fixturePath) && config.getWindowFunction() == values,
		"OCTproZ window config import");
	require(config.loadDispersionPhaseFromFile(fixturePath) && config.getDispersionPhase() == values,
		"OCTproZ dispersion config import");
	require(config.loadBackgroundProfileFromFile(fixturePath) &&
		config.getBackgroundProfile() == std::vector<float>({1.25f, 2.5f}), "OCTproZ background config import");

	writeFixture("index;real;imaginary\n0;1;2\n1;3;4\n");
	require(config.loadFixedPatternNoiseProfileFromFile(fixturePath) &&
		config.getFixedPatternNoiseProfile() == std::vector<float>({1, 2, 3, 4}), "FPN config import");
	writeFixture("index;value\n0;1\n1;broken\n");
	require(!config.loadWindowFunctionFromFile(fixturePath) && config.getWindowFunction() == values,
		"Malformed import must preserve real curve");
}

} // namespace

int main() {
	int result = 0;
	try {
		testImports();
		testRoundTrips();
		testLocaleIndependentImport();
		testConfigurationImports();
		testProfileExchange();
		std::cout << "CSV helper tests passed\n";
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		result = 1;
	}
	std::remove(fixturePath);
	std::remove(outputPath);
	return result;
}
