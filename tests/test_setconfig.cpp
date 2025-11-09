#include "../include/processor.h"
#include "../include/processorconfiguration.h"
#include <iostream>
#include <cstring>

int main() {
	std::cout << "Testing setConfig() functionality..." << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Test 1: Set config before initialization
	// ============================================
	std::cout << "Test 1: Set config before initialization" << std::endl;
	
	ope::Processor processor1(ope::Backend::CPU);
	
	// Create a config
	ope::ProcessorConfiguration config1;
	config1.dataParams.signalLength = 1024;
	config1.dataParams.ascansPerBscan = 512;
	config1.dataParams.bscansPerBuffer = 1;
	config1.dataParams.samplesPerBuffer = 1024 * 512 * 1;
	config1.dataParams.inputDataType = ope::DataType::UINT16;
	
	config1.resamplingParams.enabled = true;
	config1.resamplingParams.coefficients[0] = 0.5f;
	config1.resamplingParams.coefficients[1] = 1024.0f;
	config1.resamplingParams.coefficients[2] = -50.0f;
	config1.resamplingParams.coefficients[3] = 25.0f;
	
	// Set config before init
	processor1.setConfig(config1);
	
	// Initialize
	processor1.initialize();
	
	// Verify config was applied
	const ope::ProcessorConfiguration& retrieved1 = processor1.getConfig();
	bool test1Pass = 
		retrieved1.dataParams.signalLength == 1024 &&
		retrieved1.dataParams.ascansPerBscan == 512 &&
		retrieved1.resamplingParams.enabled == true &&
		retrieved1.resamplingParams.coefficients[0] == 0.5f;
	
	std::cout << "  Result: " << (test1Pass ? "PASS" : "FAIL") << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Test 2: Copy config from one processor to another
	// ============================================
	std::cout << "Test 2: Copy config from one processor to another" << std::endl;
	
	ope::Processor processor2(ope::Backend::CPU);
	
	// Copy config from processor1 to processor2
	processor2.setConfig(processor1.getConfig());
	processor2.initialize();
	
	// Verify config was copied
	const ope::ProcessorConfiguration& retrieved2 = processor2.getConfig();
	bool test2Pass = 
		retrieved2.dataParams.signalLength == retrieved1.dataParams.signalLength &&
		retrieved2.dataParams.ascansPerBscan == retrieved1.dataParams.ascansPerBscan &&
		retrieved2.resamplingParams.enabled == retrieved1.resamplingParams.enabled &&
		retrieved2.resamplingParams.coefficients[0] == retrieved1.resamplingParams.coefficients[0];
	
	std::cout << "  Result: " << (test2Pass ? "PASS" : "FAIL") << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Test 3: Hot-swap config (same dimensions)
	// ============================================
	std::cout << "Test 3: Hot-swap config (same dimensions)" << std::endl;
	
	ope::Processor processor3(ope::Backend::CPU);
	processor3.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
	processor3.enableResampling(true);
	processor3.setResamplingCoefficients(config1.resamplingParams.coefficients);
	processor3.initialize();
	
	// Create new config with same dimensions but different parameters
	ope::ProcessorConfiguration config3;
	config3.dataParams.signalLength = 1024;  // Same
	config3.dataParams.ascansPerBscan = 512;  // Same
	config3.dataParams.bscansPerBuffer = 1;  // Same
	config3.dataParams.samplesPerBuffer = 1024 * 512 * 1;
	config3.dataParams.inputDataType = ope::DataType::UINT16;
	
	config3.resamplingParams.enabled = true;
	config3.resamplingParams.coefficients[0] = 1.0f;  // Different
	config3.resamplingParams.coefficients[1] = 1024.0f;
	config3.resamplingParams.coefficients[2] = -25.0f;  // Different
	config3.resamplingParams.coefficients[3] = 12.0f;  // Different
	
	config3.windowingParams.enabled = true;  // Different
	config3.windowingParams.windowType = ope::WindowType::HANN;
	
	// Hot-swap config (should not reinitialize, just update curves)
	processor3.setConfig(config3);
	
	// Verify new config was applied
	const ope::ProcessorConfiguration& retrieved3 = processor3.getConfig();
	bool test3Pass = 
		retrieved3.resamplingParams.coefficients[0] == 1.0f &&
		retrieved3.resamplingParams.coefficients[2] == -25.0f &&
		retrieved3.windowingParams.enabled == true;
	
	std::cout << "  Result: " << (test3Pass ? "PASS" : "FAIL") << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Test 4: Change dimensions (should reinitialize)
	// ============================================
	std::cout << "Test 4: Change dimensions (should reinitialize)" << std::endl;
	
	ope::Processor processor4(ope::Backend::CPU);
	processor4.setInputParameters(1024, 512, 1, ope::DataType::UINT16);
	processor4.initialize();
	
	// Create config with different dimensions
	ope::ProcessorConfiguration config4;
	config4.dataParams.signalLength = 2048;  // Different!
	config4.dataParams.ascansPerBscan = 1024;  // Different!
	config4.dataParams.bscansPerBuffer = 1;
	config4.dataParams.samplesPerBuffer = 2048 * 1024 * 1;
	config4.dataParams.inputDataType = ope::DataType::UINT16;
	
	// Set config with different dimensions (should reinitialize)
	processor4.setConfig(config4);
	
	// Verify new dimensions were applied
	const ope::ProcessorConfiguration& retrieved4 = processor4.getConfig();
	bool test4Pass = 
		retrieved4.dataParams.signalLength == 2048 &&
		retrieved4.dataParams.ascansPerBscan == 1024;
	
	std::cout << "  Result: " << (test4Pass ? "PASS" : "FAIL") << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Test 5: Copy config with custom curves
	// ============================================
	std::cout << "Test 5: Copy config with custom curves" << std::endl;
	
	ope::Processor processor5a(ope::Backend::CPU);
	processor5a.setInputParameters(512, 256, 1, ope::DataType::UINT16);
	
	// Set custom curve
	std::vector<float> customCurve(512);
	for (int i = 0; i < 512; ++i) {
		customCurve[i] = 10.0f + i * 0.5f;
	}
	processor5a.setCustomResamplingCurve(customCurve.data(), customCurve.size());
	processor5a.initialize();
	
	// Copy to another processor
	ope::Processor processor5b(ope::Backend::CPU);
	processor5b.setConfig(processor5a.getConfig());
	processor5b.initialize();
	
	// Verify custom curve was copied
	const ope::ProcessorConfiguration& retrieved5 = processor5b.getConfig();
	bool test5Pass = 
		retrieved5.hasCustomResamplingCurve() &&
		retrieved5.getCustomResamplingCurveSize() == 512;
	
	// Verify data matches
	if (test5Pass) {
		const float* copiedCurve = retrieved5.getCustomResamplingCurve();
		for (int i = 0; i < 512; ++i) {
			if (copiedCurve[i] != customCurve[i]) {
				test5Pass = false;
				break;
			}
		}
	}
	
	std::cout << "  Result: " << (test5Pass ? "PASS" : "FAIL") << std::endl;
	std::cout << std::endl;
	
	// ============================================
	// Summary
	// ============================================
	bool allPass = test1Pass && test2Pass && test3Pass && test4Pass && test5Pass;
	
	std::cout << "========================================" << std::endl;
	std::cout << "SUMMARY: " << (allPass ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
	std::cout << "========================================" << std::endl;
	
	return allPass ? 0 : 1;
}
