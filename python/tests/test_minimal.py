"""
Minimal test - just try to create and initialize
"""

import sys

print("Step 1: Import module...")
sys.stdout.flush()

try:
    import octproengine as ope
    print("PASSED - Imported")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)

print()
print("Step 2: Create CPU processor...")
sys.stdout.flush()

try:
    proc = ope.Processor(ope.Backend.CPU)
    print("PASSED - Created")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 3: Set parameters...")
sys.stdout.flush()

try:
    proc.set_input_parameters(512, 256, 1, ope.DataType.UINT16)
    print("PASSED - Parameters set")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 4: Initialize...")
sys.stdout.flush()

try:
    proc.initialize()
    print("PASSED - Initialized")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 5: Set callback (lambda)...")
sys.stdout.flush()

try:
    proc.set_callback(lambda x: None)
    print("PASSED - Callback set")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 6: Call get_next_available_buffer()...")
print("         (Will hang here if there's an issue (VS2022 pybind11 mutex bug!))")
sys.stdout.flush()

try:
    buffer = proc.get_next_available_buffer()
    print("PASSED - GOT BUFFER!")
    print(f"     Shape: {buffer.shape}")
    print(f"     Dtype: {buffer.dtype}")
    sys.stdout.flush()
except KeyboardInterrupt:
    print()
    print("Interrupted by Ctrl+C")
    print()
    print("The call to get_next_available_buffer() is blocking forever.")
    print("This means either:")
    print("  1. The freeBuffersQueue is empty (no buffers added)")
    print("  2. The condition variable wait() is not being notified")
    print("  3. There's a deadlock with the GIL or another lock")
    sys.exit(1)
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 7: Stop processor...")
sys.stdout.flush()

try:
    proc.stop()
    print("PASSED - Stopped")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("SUCCESS - All PASSED ")
print("=" * 60)

sys.exit(0)
