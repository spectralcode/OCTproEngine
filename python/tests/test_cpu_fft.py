"""CPU batched IFFT: independent rows, thread limits, and concurrent processors."""
import threading

import numpy as np
import octproengine as ope


def make_processor(threads, length, ascans, bscans):
    proc = ope.Processor(ope.Backend.CPU)
    config = ope.CpuConfig()
    config.num_threads = threads
    proc.set_backend_config(config)
    proc.set_input_parameters(length, ascans, bscans, ope.DataType.UINT16)
    proc.enable_log_scaling(False)
    proc.set_grayscale_range(0.0, 1.0)
    proc.set_signal_multiplicator_and_addend(1.0, 0.0)
    proc.enable_dispersion_compensation(True)
    proc.enable_windowing(True)
    proc.initialize()
    phase = np.linspace(-2.0, 1.0, length, dtype=np.float32)
    window = np.linspace(0.25, 1.0, length, dtype=np.float32)
    proc.set_custom_dispersion_curve(phase)
    proc.set_custom_window_curve(window)
    return proc, phase, window


def check_outputs(processors, geometry, rng):
    pending = []
    submissions = []
    try:
        for (proc, phase, window), (length, ascans, bscans) in zip(processors, geometry):
            data = rng.integers(1, 32, (bscans, ascans, length), dtype=np.uint16)
            spectrum = data.astype(np.float64) * window * np.exp(1j * phase.astype(np.float64))
            reference = np.abs(np.fft.ifft(spectrum, axis=-1) * length)[..., :length // 2] / (length // 2)
            done, output = threading.Event(), []

            def callback(values, _, output=output, done=done):
                output.append(np.copy(values))
                done.set()

            callback_id = proc.add_output_callback(callback)
            pending.append((proc, callback_id, done, output, reference))
            buffer = proc.get_next_available_buffer()
            buffer[:] = data.reshape(buffer.shape)
            submissions.append((proc, buffer))

        # Prepare both inputs first, then submit without reference calculations
        # between calls so the independent processors can exercise the shared pool.
        for proc, buffer in submissions:
            proc.process(buffer)

        for proc, _, done, output, reference in pending:
            assert done.wait(10), "CPU FFT output timed out"
            np.testing.assert_allclose(output[0].reshape(reference.shape), reference,
                                       atol=1e-5, rtol=1e-5)
    finally:
        for proc, callback_id, *_ in pending:
            proc.remove_output_callback(callback_id)


def main():
    if not ope.BackendUtils.is_cpu_available():
        print("SKIP: CPU backend unavailable")
        return
    rng = np.random.default_rng(9412)
    # Non-power-of-two lengths, uneven row counts, and multiple B-scans exercise
    # both transform strides and reuse of the B-scan workspace.
    for threads in (1, 2, 4, 0):
        for shape in ((30, 17, 3), (1536, 19, 2)):
            processor = make_processor(threads, *shape)
            for _ in range(2):
                check_outputs([processor], [shape], rng)
            del processor
    shapes = [(2048, 17, 2), (512, 33, 3)]
    processors = [make_processor(threads, *shape)
                  for threads, shape in zip((0, 2), shapes)]
    for _ in range(8):
        check_outputs(processors, shapes, rng)
    processors.clear()
    print("PASS: 32 CPU FFT outputs match NumPy across thread limits and concurrent processors")


if __name__ == "__main__":
    main()
