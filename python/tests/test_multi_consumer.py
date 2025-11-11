"""
Python Test for Multi-Consumer Feature
Tests that multiple callbacks work correctly in Python
"""

import octproengine as ope
import numpy as np
import time

def test_basic_multi_consumer():
    """Test that multiple callbacks receive data"""
    print("TEST 1: Basic Multi-Consumer")
    print("  Testing that 3 callbacks all receive data...")
    
    processor = ope.Processor(ope.Backend.CPU)
    processor.set_input_parameters(1024, 256, 1, ope.DataType.UINT16)
    processor.initialize()
    
    # Counters for each callback
    counts = {'cb1': 0, 'cb2': 0, 'cb3': 0}
    
    # Add 3 callbacks
    id1 = processor.add_output_callback(lambda data: counts.update({'cb1': counts['cb1'] + 1}))
    id2 = processor.add_output_callback(lambda data: counts.update({'cb2': counts['cb2'] + 1}))
    id3 = processor.add_output_callback(lambda data: counts.update({'cb3': counts['cb3'] + 1}))
    
    # Verify count
    assert processor.get_callback_count() == 3, "Should have 3 callbacks"
    print("  [OK] Registered 3 callbacks")
    
    # Process one frame
    buffer = processor.get_next_available_buffer()
    buffer[:] = np.random.randint(0, 65535, buffer.shape, dtype=np.uint16)
    processor.process(buffer)
    
    # Wait for callbacks
    time.sleep(0.2)
    
    # Verify
    assert counts['cb1'] == 1, f"CB1 count={counts['cb1']}, expected 1"
    assert counts['cb2'] == 1, f"CB2 count={counts['cb2']}, expected 1"
    assert counts['cb3'] == 1, f"CB3 count={counts['cb3']}, expected 1"
    
    print("  [OK] All 3 callbacks received data")
    print("  PASSED\n")
    return True


def test_remove_callback():
    """Test callback removal"""
    print("TEST 2: Remove Callback")
    print("  Testing callback removal...")
    
    processor = ope.Processor(ope.Backend.CPU)
    processor.set_input_parameters(1024, 256, 1, ope.DataType.UINT16)
    processor.initialize()
    
    counts = {'cb1': 0, 'cb2': 0, 'cb3': 0}
    
    id1 = processor.add_output_callback(lambda data: counts.update({'cb1': counts['cb1'] + 1}))
    id2 = processor.add_output_callback(lambda data: counts.update({'cb2': counts['cb2'] + 1}))
    id3 = processor.add_output_callback(lambda data: counts.update({'cb3': counts['cb3'] + 1}))
    
    assert processor.get_callback_count() == 3
    print("  [OK] Added 3 callbacks")
    
    # Remove middle callback
    removed = processor.remove_output_callback(id2)
    assert removed, "remove_output_callback should return True"
    assert processor.get_callback_count() == 2
    print("  [OK] Removed callback 2")
    
    # Process frame
    buffer = processor.get_next_available_buffer()
    buffer[:] = np.random.randint(0, 65535, buffer.shape, dtype=np.uint16)
    processor.process(buffer)
    
    time.sleep(0.2)
    
    # Verify
    assert counts['cb1'] == 1, f"CB1 should be called, got {counts['cb1']}"
    assert counts['cb2'] == 0, f"CB2 should NOT be called, got {counts['cb2']}"
    assert counts['cb3'] == 1, f"CB3 should be called, got {counts['cb3']}"
    
    print("  [OK] Callbacks 1 and 3 called, callback 2 not called")
    print("  PASSED\n")
    return True


def test_clear_callbacks():
    """Test clearing all callbacks"""
    print("TEST 3: Clear All Callbacks")
    print("  Testing clear_output_callbacks()...")
    
    processor = ope.Processor(ope.Backend.CPU)
    processor.set_input_parameters(1024, 256, 1, ope.DataType.UINT16)
    processor.initialize()
    
    counts = {'cb1': 0, 'cb2': 0}
    
    processor.add_output_callback(lambda data: counts.update({'cb1': counts['cb1'] + 1}))
    processor.add_output_callback(lambda data: counts.update({'cb2': counts['cb2'] + 1}))
    
    assert processor.get_callback_count() == 2
    print("  [OK] Added 2 callbacks")
    
    # Clear all
    processor.clear_output_callbacks()
    assert processor.get_callback_count() == 0
    print("  [OK] Cleared all callbacks")
    
    # Process frame
    buffer = processor.get_next_available_buffer()
    buffer[:] = np.random.randint(0, 65535, buffer.shape, dtype=np.uint16)
    processor.process(buffer)
    
    time.sleep(0.2)
    
    # Verify
    assert counts['cb1'] == 0, "CB1 should not be called"
    assert counts['cb2'] == 0, "CB2 should not be called"
    
    print("  [OK] No callbacks called after clear")
    print("  PASSED\n")
    return True


def test_data_integrity():
    """Test that all callbacks receive same data"""
    print("TEST 4: Data Integrity")
    print("  Testing that all callbacks receive same data...")
    
    processor = ope.Processor(ope.Backend.CPU)
    processor.set_input_parameters(1024, 256, 1, ope.DataType.UINT16)
    processor.initialize()
    
    # Storage for copied data
    data_copies = {'cb1': None, 'cb2': None, 'cb3': None}
    
    def make_callback(name):
        def callback(data):
            data_copies[name] = data.copy()  # Must copy!
        return callback
    
    processor.add_output_callback(make_callback('cb1'))
    processor.add_output_callback(make_callback('cb2'))
    processor.add_output_callback(make_callback('cb3'))
    
    # Process frame
    buffer = processor.get_next_available_buffer()
    buffer[:] = np.random.randint(0, 65535, buffer.shape, dtype=np.uint16)
    processor.process(buffer)
    
    time.sleep(0.2)
    
    # Verify all received data
    assert data_copies['cb1'] is not None, "CB1 should receive data"
    assert data_copies['cb2'] is not None, "CB2 should receive data"
    assert data_copies['cb3'] is not None, "CB3 should receive data"
    
    print(f"  [OK] All callbacks received data (shape: {data_copies['cb1'].shape})")
    
    # Verify data is identical
    assert np.array_equal(data_copies['cb1'], data_copies['cb2']), "CB1 and CB2 data should match"
    assert np.array_equal(data_copies['cb2'], data_copies['cb3']), "CB2 and CB3 data should match"
    
    print("  [OK] All callbacks received identical data")
    print("  PASSED\n")
    return True


def test_multiple_frames():
    """Test multiple frames with multiple callbacks"""
    print("TEST 5: Multiple Frames")
    print("  Testing multiple frames with multiple callbacks...")
    
    processor = ope.Processor(ope.Backend.CPU)
    processor.set_input_parameters(1024, 256, 1, ope.DataType.UINT16)
    processor.initialize()
    
    NUM_FRAMES = 10
    counts = {'cb1': 0, 'cb2': 0}
    
    processor.add_output_callback(lambda data: counts.update({'cb1': counts['cb1'] + 1}))
    processor.add_output_callback(lambda data: counts.update({'cb2': counts['cb2'] + 1}))
    
    # Process multiple frames
    for i in range(NUM_FRAMES):
        buffer = processor.get_next_available_buffer()
        buffer[:] = np.random.randint(0, 65535, buffer.shape, dtype=np.uint16)
        processor.process(buffer)
    
    time.sleep(0.3)
    
    # Verify
    assert counts['cb1'] == NUM_FRAMES, f"CB1 count={counts['cb1']}, expected {NUM_FRAMES}"
    assert counts['cb2'] == NUM_FRAMES, f"CB2 count={counts['cb2']}, expected {NUM_FRAMES}"
    
    print(f"  [OK] Both callbacks called {NUM_FRAMES} times")
    print("  PASSED\n")
    return True


def test_legacy_api():
    """Test that legacy set_callback still works"""
    print("TEST 6: Legacy API Compatibility")
    print("  Testing set_callback() still works...")
    
    processor = ope.Processor(ope.Backend.CPU)
    processor.set_input_parameters(1024, 256, 1, ope.DataType.UINT16)
    processor.initialize()
    
    count = [0]  # Use list for mutability in lambda
    
    # Use legacy API
    processor.set_callback(lambda data: count.__setitem__(0, count[0] + 1))
    
    assert processor.get_callback_count() == 1
    print("  [OK] set_callback() registered 1 callback")
    
    # Process frame
    buffer = processor.get_next_available_buffer()
    buffer[:] = np.random.randint(0, 65535, buffer.shape, dtype=np.uint16)
    processor.process(buffer)
    
    time.sleep(0.2)
    
    assert count[0] == 1, f"Callback count={count[0]}, expected 1"
    
    print("  [OK] Legacy callback called correctly")
    print("  PASSED\n")
    return True


def main():
    print("=" * 50)
    print("Multi-Consumer Python Tests")
    print("=" * 50)
    print()
    
    passed = 0
    total = 0
    
    tests = [
        test_basic_multi_consumer,
        test_remove_callback,
        test_clear_callbacks,
        test_data_integrity,
        test_multiple_frames,
        test_legacy_api
    ]
    
    for test in tests:
        total += 1
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("[OK] ALL TESTS PASSED!")
    else:
        print("[FAIL] SOME TESTS FAILED!")
    print("=" * 50)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
