import sys
import os
import numpy as np
import octproengine as ope


def test_device_enumeration():
    """Test device enumeration functionality."""
    print("Testing device enumeration...")
    print()

    # CUDA devices
    if ope.BackendUtils.is_cuda_available():
        print("CUDA Devices:")
        cuda_devices = ope.BackendUtils.get_cuda_devices()
        for device in cuda_devices:
            print(f"  Device {device.id}: {device.name}")
            print(f"    Total Memory: {device.total_memory / (1024**2):.0f} MB")
            print(f"    Available Memory: {device.available_memory / (1024**2):.0f} MB")
            print(f"    Compute Capability: {device.compute_capability_major}.{device.compute_capability_minor}")
        print()
    else:
        print("  CUDA not available")
        print()

    # OpenCL devices
    if ope.BackendUtils.is_opencl_available():
        print("OpenCL Devices:")
        opencl_devices = ope.BackendUtils.get_opencl_devices()
        for device in opencl_devices:
            print(f"  Device {device.id}: {device.name}")
            print(f"    Vendor: {device.vendor_name}")
            print(f"    Version: {device.device_version}")
            print(f"    Total Memory: {device.total_memory / (1024**2):.0f} MB")
        print()
    else:
        print("  OpenCL not available")
        print()

    # CPU info
    if ope.BackendUtils.is_cpu_available():
        print("CPU Info:")
        cpu_info = ope.BackendUtils.get_cpu_info()
        print(f"  {cpu_info.name}")
        print(f"  {cpu_info.device_version}")
        print()
    else:
        print("  CPU backend not available")
        print()


def test_backend_configuration():
    """Test backend configuration creation and usage."""
    print("Testing backend configuration...")
    print()

    # Test CUDA configuration
    if ope.BackendUtils.is_cuda_available():
        print("Testing CUDA configuration:")

        # Create CUDA config
        cuda_config = ope.CudaConfig()
        cuda_config.device_id = 0
        cuda_config.enable_zero_copy = False

        print(f"  Created: {cuda_config}")
        print(f"  Backend type: {cuda_config.get_backend_type()}")
        print(f"  Is valid: {cuda_config.is_valid()}")
        print(f"  String representation: {cuda_config.to_string()}")
        print()

    # Test OpenCL configuration
    if ope.BackendUtils.is_opencl_available():
        print("Testing OpenCL configuration:")

        # Create OpenCL config
        opencl_config = ope.OpenCLConfig()
        opencl_config.platform_id = 0
        opencl_config.device_id = 0
        opencl_config.prefer_gpu = True

        print(f"  Created: {opencl_config}")
        print(f"  Backend type: {opencl_config.get_backend_type()}")
        print(f"  Is valid: {opencl_config.is_valid()}")
        print(f"  String representation: {opencl_config.to_string()}")
        print()

    # Test CPU configuration
    if ope.BackendUtils.is_cpu_available():
        print("Testing CPU configuration:")

        # Create CPU config
        cpu_config = ope.CpuConfig()
        cpu_config.num_threads = 4
        cpu_config.enable_simd = True

        print(f"  Created: {cpu_config}")
        print(f"  Backend type: {cpu_config.get_backend_type()}")
        print(f"  Is valid: {cpu_config.is_valid()}")
        print(f"  String representation: {cpu_config.to_string()}")
        print()


def test_processor_integration():
    """Test processor integration with backend configuration."""
    print("Testing processor integration...")
    print()

    # Find an available backend
    if ope.BackendUtils.is_cuda_available():
        test_backend = ope.Backend.CUDA
    elif ope.BackendUtils.is_opencl_available():
        test_backend = ope.Backend.OPENCL
    elif ope.BackendUtils.is_cpu_available():
        test_backend = ope.Backend.CPU
    else:
        print("No backends available!")
        return False

    try:
        # Create processor
        processor = ope.Processor(test_backend)
        print(f"Created processor with backend: {processor.get_backend()}")

        # Get current configuration
        current_config = processor.get_backend_config()
        if current_config:
            print(f"Current config: {current_config.to_string()}")

        # Test switching backends via config
        if test_backend != ope.Backend.CPU and ope.BackendUtils.is_cpu_available():
            print("\nSwitching to CPU backend via config...")
            cpu_config = ope.CpuConfig()
            cpu_config.num_threads = 2
            processor.set_backend_config(cpu_config)

            new_config = processor.get_backend_config()
            if new_config and new_config.get_backend_type() == ope.Backend.CPU:
                print(f"[OK] Successfully switched to CPU: {new_config.to_string()}")
            else:
                print("[FAILED] Backend switch failed")
                return False

        # Test switching back to original backend
        if test_backend == ope.Backend.CUDA:
            print("\nSwitching back to CUDA via config...")
            cuda_config = ope.CudaConfig()
            cuda_config.device_id = 0
            processor.set_backend_config(cuda_config)

            new_config = processor.get_backend_config()
            if new_config and new_config.get_backend_type() == ope.Backend.CUDA:
                print(f"[OK] Successfully switched to CUDA: {new_config.to_string()}")
            else:
                print("[FAILED] Backend switch failed")
                return False

        print()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_save_load_configuration():
    """Test save/load backend configuration to/from file."""
    print("Testing save/load configuration...")
    print()

    config_file = "test_backend_config_python.ini"

    try:
        # Create processor with some configuration
        if ope.BackendUtils.is_cuda_available():
            processor = ope.Processor(ope.Backend.CUDA)
            cuda_config = ope.CudaConfig()
            cuda_config.device_id = 0
            cuda_config.enable_zero_copy = True
            processor.set_backend_config(cuda_config)
        elif ope.BackendUtils.is_opencl_available():
            processor = ope.Processor(ope.Backend.OPENCL)
            opencl_config = ope.OpenCLConfig()
            opencl_config.platform_id = 0
            opencl_config.device_id = 0
            processor.set_backend_config(opencl_config)
        else:
            processor = ope.Processor(ope.Backend.CPU)
            cpu_config = ope.CpuConfig()
            cpu_config.num_threads = 4
            processor.set_backend_config(cpu_config)

        processor.set_num_buffers(4)

        # Save configuration
        processor.save_backend_config_to_file(config_file)
        print(f"Saved configuration to {config_file}")

        # Create new processor and load configuration
        processor2 = ope.Processor(ope.Backend.CPU)  # Start with different backend
        processor2.load_backend_config_from_file(config_file)

        # Verify configuration loaded correctly
        loaded_config = processor2.get_backend_config()
        if loaded_config:
            print(f"Loaded configuration: {loaded_config.to_string()}")
            if loaded_config.get_backend_type() == processor.get_backend():
                print("[OK] Configuration loaded correctly")
            else:
                print("[FAILED] Backend type mismatch after load")
                return False

            if processor2.get_num_buffers() == 4:
                print("[OK] Buffer settings preserved")
            else:
                print("[FAILED] Buffer settings not preserved")
                return False
        else:
            print("[FAILED] Could not get loaded configuration")
            return False

        # Cleanup
        if os.path.exists(config_file):
            os.remove(config_file)

        print()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        # Cleanup on error
        if os.path.exists(config_file):
            os.remove(config_file)
        return False


def main():
    """Run all backend configuration tests."""
    print("=" * 40)
    print("Backend Configuration API Test")
    print("=" * 40)
    print()

    # Run tests
    test_device_enumeration()
    test_backend_configuration()

    test_passed = True

    if not test_processor_integration():
        test_passed = False

    if not test_save_load_configuration():
        test_passed = False

    # Summary
    print("=" * 40)
    if test_passed:
        print("TEST PASSED")
        print("Backend configuration API working correctly")
    else:
        print("TEST FAILED")
        print("Some tests did not pass")
    print("=" * 40)

    return 0 if test_passed else 1


if __name__ == "__main__":
    sys.exit(main())