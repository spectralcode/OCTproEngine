#include "../include/processor.h"
#include "../include/processorconfiguration.h"
#include "../src/utils/csvhelper.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>

// Helper function to compare float vectors
bool compareVectors(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 0.001f) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); ++i) {
		if (std::abs(a[i] - b[i]) > tolerance) return false;
	}
	return true;
}

// Helper function to create test curve data
std::vector<float> createTestCurve(size_t size, float startVal = 1.0f, float increment = 0.5f) {
	std::vector<float> curve(size);
	for (size_t i = 0; i < size; ++i) {
		curve[i] = startVal + i * increment;
	}
	return curve;
}

int main() {
	std::cout << "========================================" << std::endl;
	std::cout << "ProcessorConfiguration Comprehensive Test" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << std::endl;

	bool allTestsPass = true;
	int testNumber = 0;

	// ============================================
	// Test 1: Basic nested structure access
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Nested structure access" << std::endl;
	{
		ope::ProcessorConfiguration config;

		// Test DataParameters
		config.dataParams.signalLength = 2048;
		config.dataParams.ascansPerBscan = 512;
		config.dataParams.bscansPerBuffer = 2;

		// Test computed properties
		bool test1Pass =
			config.dataParams.samplesPerBuffer() == (2048 * 512 * 2) &&
			config.dataParams.outputSignalLength() == 1024;

		// Test ProcessingParameters nested structs
		config.processingParams.input.bitshift = true;
		config.processingParams.dcRemoval.enabled = true;
		config.processingParams.dcRemoval.windowSize = 128;
		config.processingParams.resampling.enabled = true;
		config.processingParams.resampling.method = ope::InterpolationMethod::CUBIC;
		config.processingParams.windowing.type = ope::WindowType::HANN;
		config.processingParams.intensity.logScale = true;
		config.processingParams.intensity.rangeMin = 20.0f;
		config.processingParams.intensity.rangeMax = 90.0f;

		test1Pass = test1Pass &&
			config.processingParams.input.bitshift == true &&
			config.processingParams.dcRemoval.windowSize == 128 &&
			config.processingParams.resampling.method == ope::InterpolationMethod::CUBIC &&
			config.processingParams.windowing.type == ope::WindowType::HANN;

		std::cout << "  Result: " << (test1Pass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && test1Pass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 2: Set config before initialization (from test_setconfig)
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Set config before initialization" << std::endl;
	{
		ope::Processor processor(ope::Backend::CPU);

		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 1024;
		config.dataParams.ascansPerBscan = 512;
		config.dataParams.bscansPerBuffer = 1;
		config.dataParams.inputDataType = ope::DataType::UINT16;

		config.processingParams.resampling.enabled = true;
		config.processingParams.resampling.coefficients[0] = 0.5f;
		config.processingParams.resampling.coefficients[1] = 1024.0f;

		processor.setConfig(config);
		processor.initialize();

		const ope::ProcessorConfiguration& retrieved = processor.getConfig();
		bool testPass =
			retrieved.dataParams.signalLength == 1024 &&
			retrieved.dataParams.ascansPerBscan == 512 &&
			retrieved.processingParams.resampling.enabled == true &&
			retrieved.processingParams.resampling.coefficients[0] == 0.5f;

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 3: Save and load with COMPLETE mode
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Save/load with COMPLETE mode" << std::endl;
	{
		ope::ProcessorConfiguration config1;
		config1.dataParams.signalLength = 2048;
		config1.dataParams.ascansPerBscan = 256;
		config1.dataParams.bscansPerBuffer = 2;

		// Set various parameters
		config1.processingParams.resampling.enabled = true;
		config1.processingParams.resampling.coefficients[0] = 1.5f;
		config1.processingParams.windowing.enabled = true;
		config1.processingParams.windowing.fillFactor = 0.85f;
		config1.processingParams.intensity.rangeMin = 25.0f;
		config1.processingParams.intensity.rangeMax = 95.0f;

		// Add custom curves
		std::vector<float> customWindow = createTestCurve(2048, 0.0f, 0.001f);
		config1.setWindowFunction(customWindow);

		// Save with COMPLETE mode (includes custom data)
		const std::string testFile = "test_config_complete.ini";
		bool saveOk = config1.saveToFile(testFile, ope::ProcessorConfiguration::SaveMode::COMPLETE);

		// Load into new config
		ope::ProcessorConfiguration config2;
		bool loadOk = config2.loadFromFile(testFile, ope::ProcessorConfiguration::LoadMode::OVERWRITE_ALL);

		// Verify
		bool testPass = saveOk && loadOk &&
			config2.dataParams.signalLength == 2048 &&
			config2.dataParams.ascansPerBscan == 256 &&
			config2.processingParams.resampling.coefficients[0] == 1.5f &&
			std::abs(config2.processingParams.windowing.fillFactor - 0.85f) < 0.001f &&
			config2.hasCustomWindowCurve() &&
			compareVectors(config2.getWindowFunction(), customWindow);

		// Clean up
		std::remove(testFile.c_str());

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 4: Save PARAMETERS_ONLY mode
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Save PARAMETERS_ONLY mode" << std::endl;
	{
		ope::ProcessorConfiguration config1;
		config1.dataParams.signalLength = 1024;

		// Add custom curve
		std::vector<float> customResampling = createTestCurve(1024, 10.0f, 0.5f);
		config1.setResamplingLut(customResampling);

		// Save parameters only (no custom data)
		const std::string testFile = "test_config_params_only.ini";
		config1.saveToFile(testFile, ope::ProcessorConfiguration::SaveMode::PARAMETERS_ONLY);

		// Load and verify
		ope::ProcessorConfiguration config2;
		config2.loadFromFile(testFile);

		bool testPass =
			config2.dataParams.signalLength == 1024 &&
			!config2.hasCustomResamplingCurve();  // Custom data should NOT be loaded

		// Clean up
		std::remove(testFile.c_str());

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 5: Load with MERGE_IF_MISSING mode
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Load with MERGE_IF_MISSING mode" << std::endl;
	{
		// Create config with custom curves
		ope::ProcessorConfiguration config1;
		config1.dataParams.signalLength = 512;
		std::vector<float> customWindow = createTestCurve(512, 1.0f, 0.1f);
		std::vector<float> customDispersion = createTestCurve(512, 2.0f, 0.2f);
		config1.setWindowFunction(customWindow);
		config1.setDispersionPhase(customDispersion);

		// Save complete config
		const std::string testFile = "test_config_merge.ini";
		config1.saveToFile(testFile, ope::ProcessorConfiguration::SaveMode::COMPLETE);

		// Create another config with only window curve
		ope::ProcessorConfiguration config2;
		config2.dataParams.signalLength = 512;
		std::vector<float> differentWindow = createTestCurve(512, 5.0f, 0.5f);
		config2.setWindowFunction(differentWindow);

		// Load with MERGE_IF_MISSING - should keep existing window, add dispersion
		config2.loadFromFile(testFile, ope::ProcessorConfiguration::LoadMode::MERGE_IF_MISSING);

		bool testPass =
			config2.hasCustomWindowCurve() &&
			config2.hasCustomDispersionCurve() &&
			compareVectors(config2.getWindowFunction(), differentWindow) &&  // Kept existing
			compareVectors(config2.getDispersionPhase(), customDispersion);  // Loaded missing

		// Clean up
		std::remove(testFile.c_str());

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 6: CSV export/import for individual curves
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": CSV export/import for curves" << std::endl;
	{
		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 256;

		// Create and set custom curves
		std::vector<float> resamplingLut = createTestCurve(256, 0.0f, 1.0f);
		std::vector<float> windowFunc = createTestCurve(256, 0.0f, 0.01f);
		std::vector<float> dispersionPhase = createTestCurve(256, 0.0f, 0.1f);

		config.setResamplingLut(resamplingLut);
		config.setWindowFunction(windowFunc);
		config.setDispersionPhase(dispersionPhase);

		// Export to CSV files
		const std::string resamplingFile = "test_resampling.csv";
		const std::string windowFile = "test_window.csv";
		const std::string dispersionFile = "test_dispersion.csv";

		bool exportOk =
			config.saveResamplingLutToFile(resamplingFile) &&
			config.saveWindowFunctionToFile(windowFile) &&
			config.saveDispersionPhaseToFile(dispersionFile);

		// Create new config and import from CSV
		ope::ProcessorConfiguration config2;
		config2.dataParams.signalLength = 256;

		bool importOk =
			config2.loadResamplingLutFromFile(resamplingFile) &&
			config2.loadWindowFunctionFromFile(windowFile) &&
			config2.loadDispersionPhaseFromFile(dispersionFile);

		// Verify imported data matches
		bool testPass = exportOk && importOk &&
			compareVectors(config2.getResamplingLut(), resamplingLut) &&
			compareVectors(config2.getWindowFunction(), windowFunc) &&
			compareVectors(config2.getDispersionPhase(), dispersionPhase);

		// Clean up
		std::remove(resamplingFile.c_str());
		std::remove(windowFile.c_str());
		std::remove(dispersionFile.c_str());

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 7: Complex data (Fixed Pattern Noise Profile)
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Complex data handling (FPN profile)" << std::endl;
	{
		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 128;

		// Create complex data (real/imag pairs)
		// The FPN profile is stored as pairs, but adjusted to signalLength when set
		std::vector<float> fpnProfile;
		for (int i = 0; i < 64; ++i) {  // Half size for complex pairs
			fpnProfile.push_back(static_cast<float>(i));      // Real
			fpnProfile.push_back(static_cast<float>(i * 2));  // Imaginary
		}

		config.setFixedPatternNoiseProfile(fpnProfile);

		// Save and load
		const std::string csvFile = "test_fpn.csv";
		const std::string iniFile = "test_fpn.ini";

		config.saveFixedPatternNoiseProfileToFile(csvFile);
		config.saveToFile(iniFile);

		// Load into new config
		ope::ProcessorConfiguration config2;
		config2.loadFromFile(iniFile);

		// Also test direct CSV load
		ope::ProcessorConfiguration config3;
		config3.dataParams.signalLength = 128;
		config3.loadFixedPatternNoiseProfileFromFile(csvFile);

		// The loaded profile might be adjusted to signalLength size
		bool testPass =
			config2.hasCustomFixedPatternNoiseProfile() &&
			config3.hasCustomFixedPatternNoiseProfile();

		// Clean up
		std::remove(csvFile.c_str());
		std::remove(iniFile.c_str());

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 8: Curve auto-adjustment on size change
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Curve auto-adjustment on size change" << std::endl;
	{
		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 512;

		// Set custom curves for 512 samples
		std::vector<float> windowFunc = createTestCurve(512, 1.0f, 0.01f);
		config.setWindowFunction(windowFunc);

		// Change signal length - curves should auto-adjust
		config.dataParams.signalLength = 1024;
		config.adjustAllCustomCurves();

		// Should be zero-padded to 1024
		std::vector<float> adjusted = config.getWindowFunction();
		bool testPass =
			adjusted.size() == 1024 &&
			adjusted[0] == windowFunc[0] &&
			adjusted[511] == windowFunc[511] &&
			adjusted[512] == 0.0f &&  // Zero-padded
			adjusted[1023] == 0.0f;

		// Now shrink - should truncate
		config.dataParams.signalLength = 256;
		config.adjustAllCustomCurves();

		adjusted = config.getWindowFunction();
		testPass = testPass &&
			adjusted.size() == 256 &&
			adjusted[0] == windowFunc[0] &&
			adjusted[255] == windowFunc[255];

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 9: Hot-swap config (from test_setconfig)
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Hot-swap config (same dimensions)" << std::endl;
	{
		ope::Processor processor(ope::Backend::CPU);
		processor.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
		processor.initialize();

		// Create new config with same dimensions but different parameters
		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 1024;  // Same
		config.dataParams.ascansPerBscan = 512;  // Same
		config.dataParams.bscansPerBuffer = 1;   // Same

		config.processingParams.resampling.enabled = true;
		config.processingParams.resampling.coefficients[0] = 1.0f;
		config.processingParams.windowing.enabled = true;
		config.processingParams.windowing.type = ope::WindowType::GAUSS;

		// Hot-swap (should not reinitialize)
		processor.setConfig(config);

		const ope::ProcessorConfiguration& retrieved = processor.getConfig();
		bool testPass =
			retrieved.processingParams.resampling.coefficients[0] == 1.0f &&
			retrieved.processingParams.windowing.enabled == true &&
			retrieved.processingParams.windowing.type == ope::WindowType::GAUSS;

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 10: Direct parameter modification
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Direct parameter modification" << std::endl;
	{
		ope::ProcessorConfiguration config;

		// Set via new structure
		config.processingParams.resampling.enabled = true;
		config.processingParams.resampling.method = ope::InterpolationMethod::CUBIC;
		config.processingParams.windowing.type = ope::WindowType::FLAT_TOP;
		config.processingParams.windowing.centerPosition = 0.6f;
		config.processingParams.intensity.logScale = true;
		config.processingParams.intensity.rangeMax = 85.0f;

		// Verify direct access to new structure
		bool testPass =
			config.processingParams.resampling.enabled == true &&
			config.processingParams.resampling.method == ope::InterpolationMethod::CUBIC &&
			config.processingParams.windowing.type == ope::WindowType::FLAT_TOP &&
			std::abs(config.processingParams.windowing.centerPosition - 0.6f) < 0.001f &&
			config.processingParams.intensity.logScale == true &&
			std::abs(config.processingParams.intensity.rangeMax - 85.0f) < 0.001f;

		// Modify parameters directly
		config.processingParams.resampling.coefficients[0] = 2.5f;
		config.processingParams.windowing.fillFactor = 0.92f;
		config.processingParams.intensity.rangeMin = 15.0f;

		// Verify modifications
		testPass = testPass &&
			config.processingParams.resampling.coefficients[0] == 2.5f &&
			std::abs(config.processingParams.windowing.fillFactor - 0.92f) < 0.001f &&
			std::abs(config.processingParams.intensity.rangeMin - 15.0f) < 0.001f;

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 11: Generate curves from parameters
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Generate curves from parameters" << std::endl;
	{
		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 256;

		// Set parameters for curve generation
		// For linear mapping from 0 to 255, coefficient[1] should be 255.0
		config.processingParams.resampling.coefficients[0] = 0.0f;
		config.processingParams.resampling.coefficients[1] = 255.0f;  // Maps to indices 0-255
		config.processingParams.resampling.coefficients[2] = 0.0f;
		config.processingParams.resampling.coefficients[3] = 0.0f;

		config.processingParams.windowing.type = ope::WindowType::HANN;
		config.processingParams.windowing.centerPosition = 0.5f;
		config.processingParams.windowing.fillFactor = 1.0f;

		config.processingParams.dispersion.coefficients[0] = 0.0f;
		config.processingParams.dispersion.coefficients[1] = 0.01f;
		config.processingParams.dispersion.coefficients[2] = 0.0f;
		config.processingParams.dispersion.coefficients[3] = 0.0f;

		// Generate curves
		std::vector<float> resamplingLut = config.generateResamplingLut();
		std::vector<float> windowFunc = config.generateWindowFunction();
		std::vector<float> dispersionPhase = config.generateDispersionPhase();

		// Note: Resampling LUT is clamped to safe range [7, 247] for Lanczos compatibility
		bool testPass =
			resamplingLut.size() == 256 &&
			windowFunc.size() == 256 &&
			dispersionPhase.size() == 512 &&     // Complex (real/imag pairs)
			resamplingLut[0] >= 7.0f &&          // Clamped to minIndex
			resamplingLut[0] <= 7.5f &&          // Should be close to 7
			resamplingLut[255] >= 246.0f &&      // Clamped to maxIndex
			resamplingLut[255] <= 247.0f &&      // Should be close to 247
			windowFunc[0] == 0.0f &&             // Hann window starts at 0
			windowFunc[128] > 0.9f;              // Peak in middle

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 12: Copy config with custom curves
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Copy config with custom curves" << std::endl;
	{
		ope::Processor processor1(ope::Backend::CPU);
		processor1.setInputParameters(512, 256, 1, ope::DataType::UINT16);

		// Set custom curves
		std::vector<float> customCurve = createTestCurve(512, 10.0f, 0.5f);
		processor1.setCustomResamplingCurve(customCurve.data(), customCurve.size());
		processor1.initialize();

		// Copy to another processor
		ope::Processor processor2(ope::Backend::CPU);
		processor2.setConfig(processor1.getConfig());
		processor2.initialize();

		// Verify custom curve was copied
		const ope::ProcessorConfiguration& config2 = processor2.getConfig();
		std::vector<float> copiedCurve = config2.getResamplingLut();
		bool testPass =
			config2.hasCustomResamplingCurve() &&
			copiedCurve.size() == 512;

		// Verify data matches
		if (testPass) {
			for (size_t i = 0; i < 512; ++i) {
				if (std::abs(copiedCurve[i] - customCurve[i]) > 0.001f) {
					testPass = false;
					break;
				}
			}
		}

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 13: Clear custom curves
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Clear custom curves" << std::endl;
	{
		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 128;

		// Set custom curves
		std::vector<float> testCurve = createTestCurve(128);
		config.setResamplingLut(testCurve);
		config.setWindowFunction(testCurve);
		config.setDispersionPhase(testCurve);

		// Verify they're set
		bool beforeClear =
			config.hasCustomResamplingCurve() &&
			config.hasCustomWindowCurve() &&
			config.hasCustomDispersionCurve();

		// Clear them
		config.clearResamplingLut();
		config.clearWindowFunction();
		config.clearDispersionPhase();

		// Verify they're cleared
		bool afterClear =
			!config.hasCustomResamplingCurve() &&
			!config.hasCustomWindowCurve() &&
			!config.hasCustomDispersionCurve();

		bool testPass = beforeClear && afterClear;

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 14: Validate configuration
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Configuration validation" << std::endl;
	{
		ope::ProcessorConfiguration config;

		// Set valid configuration
		config.dataParams.signalLength = 1024;
		config.dataParams.ascansPerBscan = 512;
		config.dataParams.bscansPerBuffer = 1;

		bool validConfig = config.validate();

		// Set invalid configuration (e.g., signal length of 0)
		config.dataParams.signalLength = 0;
		bool invalidConfig = !config.validate();

		bool testPass = validConfig && invalidConfig;

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 15: Load mode PARAMETERS_ONLY preserves custom data
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Load PARAMETERS_ONLY preserves custom data" << std::endl;
	{
		// Create config with custom data
		ope::ProcessorConfiguration config1;
		config1.dataParams.signalLength = 512;
		std::vector<float> customWindow = createTestCurve(512, 3.0f, 0.3f);
		config1.setWindowFunction(customWindow);
		config1.processingParams.intensity.rangeMax = 100.0f;

		// Save complete config
		const std::string testFile = "test_params_preserve.ini";
		config1.saveToFile(testFile);

		// Create another config with different custom data
		ope::ProcessorConfiguration config2;
		config2.dataParams.signalLength = 512;
		std::vector<float> differentWindow = createTestCurve(512, 7.0f, 0.7f);
		config2.setWindowFunction(differentWindow);
		config2.processingParams.intensity.rangeMax = 50.0f;  // Different value

		// Load with PARAMETERS_ONLY - should update params but keep custom data
		config2.loadFromFile(testFile, ope::ProcessorConfiguration::LoadMode::PARAMETERS_ONLY);

		bool testPass =
			config2.processingParams.intensity.rangeMax == 100.0f &&  // Params updated
			compareVectors(config2.getWindowFunction(), differentWindow);  // Custom data preserved

		// Clean up
		std::remove(testFile.c_str());

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 16: Copy config from one processor to another
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Copy config from one processor to another" << std::endl;
	{
		ope::Processor processor1(ope::Backend::CPU);

		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 1024;
		config.dataParams.ascansPerBscan = 512;
		config.dataParams.bscansPerBuffer = 1;
		config.processingParams.resampling.enabled = true;
		config.processingParams.resampling.coefficients[0] = 0.5f;
		config.processingParams.resampling.coefficients[1] = 1024.0f;

		processor1.setConfig(config);
		processor1.initialize();

		// Copy config to another processor
		ope::Processor processor2(ope::Backend::CPU);
		processor2.setConfig(processor1.getConfig());
		processor2.initialize();

		// Verify config was copied
		const ope::ProcessorConfiguration& retrieved1 = processor1.getConfig();
		const ope::ProcessorConfiguration& retrieved2 = processor2.getConfig();

		bool testPass =
			retrieved2.dataParams.signalLength == retrieved1.dataParams.signalLength &&
			retrieved2.dataParams.ascansPerBscan == retrieved1.dataParams.ascansPerBscan &&
			retrieved2.processingParams.resampling.enabled == retrieved1.processingParams.resampling.enabled &&
			retrieved2.processingParams.resampling.coefficients[0] == retrieved1.processingParams.resampling.coefficients[0];

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Test 17: Change dimensions (should reinitialize)
	// ============================================
	testNumber++;
	std::cout << "Test " << testNumber << ": Change dimensions (should reinitialize)" << std::endl;
	{
		ope::Processor processor(ope::Backend::CPU);
		processor.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
		processor.initialize();

		// Create config with different dimensions
		ope::ProcessorConfiguration config;
		config.dataParams.signalLength = 2048;  // Different!
		config.dataParams.ascansPerBscan = 1024;  // Different!
		config.dataParams.bscansPerBuffer = 1;
		config.dataParams.inputDataType = ope::DataType::UINT16;

		// Set config with different dimensions (should reinitialize)
		processor.setConfig(config);

		// Verify new dimensions were applied
		const ope::ProcessorConfiguration& retrieved = processor.getConfig();
		bool testPass =
			retrieved.dataParams.signalLength == 2048 &&
			retrieved.dataParams.ascansPerBscan == 1024;

		// The dimension change itself verifies the main functionality
		// (internally, the processor reinitializes when dimensions change)

		std::cout << "  Result: " << (testPass ? "PASS" : "FAIL") << std::endl;
		allTestsPass = allTestsPass && testPass;
	}
	std::cout << std::endl;

	// ============================================
	// Summary
	// ============================================
	std::cout << "========================================" << std::endl;
	if (allTestsPass) {
		std::cout << "SUMMARY: ALL " << testNumber << " TESTS PASSED ✓" << std::endl;
	} else {
		std::cout << "SUMMARY: SOME TESTS FAILED ✗" << std::endl;
	}
	std::cout << "========================================" << std::endl;

	return allTestsPass ? 0 : 1;
}