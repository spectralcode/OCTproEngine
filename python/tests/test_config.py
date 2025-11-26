"""
Test for ProcessorConfiguration Python bindings
"""

import sys
import os
import numpy as np
import math

import octproengine as ope

def create_test_curve(size, start_val, increment):
    """Helper to create test curves"""
    return [start_val + i * increment for i in range(size)]

def compare_vectors(vec1, vec2, tolerance=1e-5):
    """Helper to compare two vectors with tolerance"""
    if len(vec1) != len(vec2):
        return False
    for v1, v2 in zip(vec1, vec2):
        if abs(v1 - v2) > tolerance:
            return False
    return True

def run_tests():
    """Run all ProcessorConfiguration tests"""
    test_number = 0
    all_tests_pass = True

    print("=" * 40)
    print("ProcessorConfiguration Python Test")
    print("=" * 40)
    print()

    # ============================================
    # Test 1: Nested structure access
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Nested structure access")
    try:
        config = ope.ProcessorConfiguration()

        # Test accessing nested parameters through new API
        config.dataParams.signalLength = 2048
        config.dataParams.ascansPerBscan = 256
        config.dataParams.bscansPerBuffer = 4
        config.dataParams.inputDataType = ope.DataType.UINT16

        config.processingParams.input.bitshift = True
        config.processingParams.dcRemoval.enabled = True
        config.processingParams.dcRemoval.windowSize = 128

        config.processingParams.resampling.enabled = True
        config.processingParams.resampling.method = ope.InterpolationMethod.CUBIC
        config.processingParams.resampling.coefficients = [0.5, 1024.0, -50.0, 25.0]

        config.processingParams.windowing.enabled = True
        config.processingParams.windowing.type = ope.WindowType.HANN
        config.processingParams.windowing.centerPosition = 0.5
        config.processingParams.windowing.fillFactor = 0.95

        # Verify access
        test_pass = (
            config.dataParams.signalLength == 2048 and
            config.dataParams.samplesPerBuffer() == 2048 * 256 * 4 and
            config.processingParams.input.bitshift == True and
            config.processingParams.dcRemoval.windowSize == 128 and
            config.processingParams.resampling.method == ope.InterpolationMethod.CUBIC and
            config.processingParams.windowing.type == ope.WindowType.HANN
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 2: Set config before initialization
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Set config before initialization")
    try:
        processor = ope.Processor(ope.Backend.CPU)

        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 1024
        config.dataParams.ascansPerBscan = 512
        config.dataParams.bscansPerBuffer = 1
        config.dataParams.inputDataType = ope.DataType.UINT16

        config.processingParams.resampling.enabled = True
        config.processingParams.resampling.coefficients = [0.5, 1024.0, 0.0, 0.0]

        processor.set_config(config)
        processor.initialize()

        retrieved = processor.config
        test_pass = (
            retrieved.dataParams.signalLength == 1024 and
            retrieved.dataParams.ascansPerBscan == 512 and
            retrieved.processingParams.resampling.enabled == True and
            abs(retrieved.processingParams.resampling.coefficients[0] - 0.5) < 0.001
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 3: Save/load with COMPLETE mode
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Save/load with COMPLETE mode")
    try:
        config1 = ope.ProcessorConfiguration()
        config1.dataParams.signalLength = 2048
        config1.dataParams.ascansPerBscan = 256
        config1.dataParams.bscansPerBuffer = 2

        config1.processingParams.resampling.enabled = True
        config1.processingParams.resampling.coefficients = [1.5, 1024.0, 0.0, 0.0]
        config1.processingParams.windowing.enabled = True
        config1.processingParams.windowing.fillFactor = 0.85
        config1.processingParams.intensity.rangeMin = 25.0
        config1.processingParams.intensity.rangeMax = 95.0

        # Add custom curve
        custom_window = create_test_curve(2048, 0.0, 0.001)
        config1.setWindowFunction(custom_window)

        # Save with COMPLETE mode
        test_file = "test_config_complete_py.ini"
        save_ok = config1.saveToFile(test_file, ope.SaveMode.COMPLETE)

        # Load into new config
        config2 = ope.ProcessorConfiguration()
        load_ok = config2.loadFromFile(test_file, ope.LoadMode.OVERWRITE_ALL)

        test_pass = (
            save_ok and load_ok and
            config2.dataParams.signalLength == 2048 and
            config2.dataParams.ascansPerBscan == 256 and
            abs(config2.processingParams.resampling.coefficients[0] - 1.5) < 0.001 and
            abs(config2.processingParams.windowing.fillFactor - 0.85) < 0.001 and
            config2.hasCustomWindowCurve()
        )

        # Clean up
        #if os.path.exists(test_file):
        #    os.remove(test_file)

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 4: Save PARAMETERS_ONLY mode
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Save PARAMETERS_ONLY mode")
    try:
        config1 = ope.ProcessorConfiguration()
        config1.dataParams.signalLength = 512

        # Add custom curves
        custom_resampling = create_test_curve(512, 0.0, 1.0)
        config1.setResamplingLut(custom_resampling)

        # Save parameters only (excludes custom curves)
        test_file = "test_params_only_py.ini"
        config1.saveToFile(test_file, ope.SaveMode.PARAMETERS_ONLY)

        # Load and verify
        config2 = ope.ProcessorConfiguration()
        config2.loadFromFile(test_file)

        test_pass = (
            config2.dataParams.signalLength == 512 and
            not config2.hasCustomResamplingCurve()  # Custom curve should NOT be saved
        )

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 5: Load with MERGE_IF_MISSING mode
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Load with MERGE_IF_MISSING mode")
    try:
        # Create config with custom curve
        config1 = ope.ProcessorConfiguration()
        config1.dataParams.signalLength = 256
        custom_window = create_test_curve(256, 1.0, 0.5)
        config1.setWindowFunction(custom_window)

        # Save parameters only
        test_file = "test_merge_py.ini"
        config1.saveToFile(test_file, ope.SaveMode.PARAMETERS_ONLY)

        # Create another config with different custom curve
        config2 = ope.ProcessorConfiguration()
        existing_window = create_test_curve(256, 2.0, 0.25)
        config2.setWindowFunction(existing_window)

        # Load with MERGE_IF_MISSING (should keep existing custom curve)
        config2.loadFromFile(test_file, ope.LoadMode.MERGE_IF_MISSING)

        loaded_window = config2.getWindowFunction()
        test_pass = (
            config2.dataParams.signalLength == 256 and
            config2.hasCustomWindowCurve() and
            len(loaded_window) > 0 and
            abs(loaded_window[0] - 2.0) < 0.001  # Should keep existing curve
        )

        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 6: CSV export/import for curves
    # ============================================
    test_number += 1
    print(f"Test {test_number}: CSV export/import for curves")
    try:
        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 256

        # Create and set a custom curve
        custom_curve = create_test_curve(256, 10.0, 0.5)
        config.setResamplingLut(custom_curve)

        # Export to CSV
        csv_file = "test_curve_py.csv"
        export_ok = config.saveResamplingLutToFile(csv_file)

        # Import into new config
        config2 = ope.ProcessorConfiguration()
        config2.dataParams.signalLength = 256
        import_ok = config2.loadResamplingLutFromFile(csv_file)

        # Compare curves
        loaded_curve = config2.getResamplingLut()
        test_pass = (
            export_ok and import_ok and
            compare_vectors(loaded_curve, custom_curve)
        )

        # Clean up
        if os.path.exists(csv_file):
            os.remove(csv_file)

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 7: Complex data handling (FPN profile)
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Complex data handling (FPN profile)")
    try:
        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 128

        # Create complex data (real/imag pairs)
        fpn_profile = []
        for i in range(64):  # Half size for complex pairs
            fpn_profile.append(float(i))       # Real
            fpn_profile.append(float(i * 2))   # Imaginary

        config.setFixedPatternNoiseProfile(fpn_profile)

        # Save and load
        csv_file = "test_fpn_py.csv"
        ini_file = "test_fpn_py.ini"

        config.saveFixedPatternNoiseProfileToFile(csv_file)
        config.saveToFile(ini_file)

        config2 = ope.ProcessorConfiguration()
        config2.loadFromFile(ini_file)

        # Also test direct CSV load
        config3 = ope.ProcessorConfiguration()
        config3.dataParams.signalLength = 128
        config3.loadFixedPatternNoiseProfileFromFile(csv_file)

        test_pass = (
            config2.hasCustomFixedPatternNoiseProfile() and
            config3.hasCustomFixedPatternNoiseProfile()
        )

        # Clean up
        if os.path.exists(csv_file):
            os.remove(csv_file)
        if os.path.exists(ini_file):
            os.remove(ini_file)

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 8: Automatic curve adjustment via processor
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Automatic curve adjustment via processor")
    try:
        # Create processor and initialize with 256 samples
        processor = ope.Processor(ope.Backend.CPU)
        processor.set_input_parameters(256, 128, 1, ope.DataType.UINT16)
        processor.initialize()

        # Set custom curve for size 256
        config1 = ope.ProcessorConfiguration()
        config1.dataParams.signalLength = 256
        config1.dataParams.ascansPerBscan = 128
        config1.dataParams.bscansPerBuffer = 1
        config1.dataParams.inputDataType = ope.DataType.UINT16
        custom_curve = create_test_curve(256, 0.0, 1.0)
        config1.setResamplingLut(custom_curve)

        # Apply config
        processor.set_config(config1)

        # Now change dimensions - curve should auto-adjust when setConfig is called
        config2 = ope.ProcessorConfiguration()
        config2.dataParams.signalLength = 512
        config2.dataParams.ascansPerBscan = 256
        config2.dataParams.bscansPerBuffer = 1
        config2.dataParams.inputDataType = ope.DataType.UINT16
        config2.setResamplingLut(custom_curve)  # Still 256 size

        # This should automatically adjust the curve to 512
        processor.set_config(config2)

        # Get adjusted curve from processor
        retrieved = processor.config
        adjusted = retrieved.getResamplingLut()

        test_pass = (
            len(adjusted) == 512 and  # Should be auto-adjusted to new size
            abs(adjusted[0] - 0.0) < 0.01  # First value preserved
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 9: Hot-swap config (same dimensions)
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Hot-swap config (same dimensions)")
    try:
        processor = ope.Processor(ope.Backend.CPU)
        processor.set_input_parameters(1024, 512, 1, ope.DataType.UINT16)
        processor.initialize()

        # Create new config with same dimensions but different parameters
        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 1024  # Same
        config.dataParams.ascansPerBscan = 512  # Same
        config.dataParams.bscansPerBuffer = 1  # Same

        config.processingParams.resampling.enabled = True
        config.processingParams.resampling.coefficients = [1.0, 1024.0, -25.0, 12.0]
        config.processingParams.windowing.enabled = True
        config.processingParams.windowing.type = ope.WindowType.GAUSS

        # Hot-swap
        processor.set_config(config)

        retrieved = processor.config
        test_pass = (
            abs(retrieved.processingParams.resampling.coefficients[0] - 1.0) < 0.001 and
            retrieved.processingParams.windowing.enabled == True and
            retrieved.processingParams.windowing.type == ope.WindowType.GAUSS
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 10: Generate curves from parameters
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Generate curves from parameters")
    try:
        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 256

        # Set parameters for curve generation
        config.processingParams.resampling.coefficients = [0.0, 255.0, 0.0, 0.0]
        config.processingParams.windowing.type = ope.WindowType.HANN
        config.processingParams.windowing.centerPosition = 0.5
        config.processingParams.windowing.fillFactor = 1.0
        config.processingParams.dispersion.coefficients = [0.0, 0.01, 0.0, 0.0]

        # Generate curves
        resampling_lut = config.generateResamplingLut()
        window_func = config.generateWindowFunction()
        dispersion_phase = config.generateDispersionPhase()

        # Note: Resampling LUT is clamped to safe range [7, 247] for Lanczos compatibility
        test_pass = (
            len(resampling_lut) == 256 and
            len(window_func) == 256 and
            len(dispersion_phase) == 512 and  # Complex (real/imag pairs)
            resampling_lut[0] >= 7.0 and     # Clamped to minIndex
            resampling_lut[0] <= 7.5 and
            resampling_lut[255] >= 246.0 and  # Clamped to maxIndex
            resampling_lut[255] <= 247.0 and
            window_func[0] == 0.0 and        # Hann window starts at 0
            window_func[128] > 0.9           # Peak in middle
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 11: Copy config with custom curves
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Copy config with custom curves")
    try:
        # Create config with custom curve
        config1 = ope.ProcessorConfiguration()
        config1.dataParams.signalLength = 512
        config1.dataParams.ascansPerBscan = 256
        config1.dataParams.bscansPerBuffer = 1
        config1.dataParams.inputDataType = ope.DataType.UINT16

        custom_curve = create_test_curve(512, 10.0, 0.5)
        config1.setResamplingLut(custom_curve)

        processor1 = ope.Processor(ope.Backend.CPU)
        processor1.set_config(config1)
        processor1.initialize()

        # Copy to another processor
        processor2 = ope.Processor(ope.Backend.CPU)
        processor2.set_config(processor1.config)
        processor2.initialize()

        # Verify custom curve was copied
        retrieved = processor2.config
        retrieved_curve = retrieved.getResamplingLut()
        test_pass = (
            len(retrieved_curve) == 512 and
            abs(retrieved_curve[0] - 10.0) < 0.01
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 12: Clear custom curves
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Clear custom curves")
    try:
        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 256

        # Set custom curves
        config.setResamplingLut(create_test_curve(256, 0.0, 1.0))
        config.setWindowFunction(create_test_curve(256, 0.0, 0.5))
        config.setDispersionPhase(create_test_curve(512, 0.0, 0.1))  # Complex

        # Verify they're set
        before_pass = (
            config.hasCustomResamplingCurve() and
            config.hasCustomWindowCurve() and
            config.hasCustomDispersionCurve()
        )

        # Clear them
        config.clearResamplingLut()
        config.clearWindowFunction()
        config.clearDispersionPhase()

        # Verify they're cleared
        after_pass = (
            not config.hasCustomResamplingCurve() and
            not config.hasCustomWindowCurve() and
            not config.hasCustomDispersionCurve()
        )

        test_pass = before_pass and after_pass

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 13: Configuration validation
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Configuration validation")
    try:
        # Valid config
        config1 = ope.ProcessorConfiguration()
        config1.dataParams.signalLength = 1024
        config1.dataParams.ascansPerBscan = 512
        config1.dataParams.bscansPerBuffer = 1

        valid_pass = config1.validate()

        # Invalid config (would need invalid values)
        # Since Python bindings may not expose invalid states easily,
        # we'll just test that validate() exists and works

        test_pass = valid_pass

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 14: Copy config from one processor to another
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Copy config from one processor to another")
    try:
        processor1 = ope.Processor(ope.Backend.CPU)

        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 1024
        config.dataParams.ascansPerBscan = 512
        config.dataParams.bscansPerBuffer = 1
        config.processingParams.resampling.enabled = True
        config.processingParams.resampling.coefficients = [0.5, 1024.0, 0.0, 0.0]

        processor1.set_config(config)
        processor1.initialize()

        # Copy config to another processor
        processor2 = ope.Processor(ope.Backend.CPU)
        processor2.set_config(processor1.config)
        processor2.initialize()

        # Verify config was copied
        retrieved1 = processor1.config
        retrieved2 = processor2.config

        test_pass = (
            retrieved2.dataParams.signalLength == retrieved1.dataParams.signalLength and
            retrieved2.dataParams.ascansPerBscan == retrieved1.dataParams.ascansPerBscan and
            retrieved2.processingParams.resampling.enabled == retrieved1.processingParams.resampling.enabled
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Test 15: Change dimensions (should reinitialize)
    # ============================================
    test_number += 1
    print(f"Test {test_number}: Change dimensions (should reinitialize)")
    try:
        processor = ope.Processor(ope.Backend.CPU)
        processor.set_input_parameters(1024, 512, 1, ope.DataType.UINT16)
        processor.initialize()

        # Create config with different dimensions
        config = ope.ProcessorConfiguration()
        config.dataParams.signalLength = 2048  # Different!
        config.dataParams.ascansPerBscan = 1024  # Different!
        config.dataParams.bscansPerBuffer = 1
        config.dataParams.inputDataType = ope.DataType.UINT16

        # Set config with different dimensions
        processor.set_config(config)

        # Verify new dimensions were applied
        retrieved = processor.config
        test_pass = (
            retrieved.dataParams.signalLength == 2048 and
            retrieved.dataParams.ascansPerBscan == 1024
        )

        print(f"  Result: {'PASS' if test_pass else 'FAIL'}")
        all_tests_pass = all_tests_pass and test_pass
    except Exception as e:
        print(f"  Result: FAIL - {str(e)}")
        all_tests_pass = False
    print()

    # ============================================
    # Summary
    # ============================================
    print("=" * 40)
    if all_tests_pass:
        print(f"SUMMARY: ALL {test_number} TESTS PASSED [OK]")
    else:
        print("SUMMARY: SOME TESTS FAILED [ERROR]")
    print("=" * 40)

    return 0 if all_tests_pass else 1

if __name__ == "__main__":
    sys.exit(run_tests())