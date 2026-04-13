"""
Starter example: load raw OCT data from disk, process it, and display it.

This script is intentionally small and meant as a starting point for custom
development. To move from file-based processing to live processing, replace
get_next_frame() with your own acquisition code.
"""

from pathlib import Path
import threading

import matplotlib.pyplot as plt
import numpy as np
import octproengine as ope


# ---------------------------------------------------------------------------
# Edit these settings first
# ---------------------------------------------------------------------------

# Input source
# Set this to your raw OCT file before running the example.
# Example: Path(r"C:\data\my_oct_frame.raw")
RAW_FILE_PATH = None

# Optional: load a saved processor INI file instead of using the inline
# processing settings below.
CONFIG_FILE = None

# Optional: load a custom resampling LUT from CSV.
# Leave as None to use the polynomial resampling coefficients below.
RESAMPLING_LUT_FILE = None

# Backend
BACKEND = ope.Backend.CPU

# Input data parameters
SIGNAL_LENGTH = 2048
ASCANS_PER_BSCAN = 1024
BSCANS_PER_BUFFER = 1
DATA_TYPE = ope.DataType.UINT16

# Processing parameters
ENABLE_RESAMPLING = False
INTERPOLATION_METHOD = ope.InterpolationMethod.CUBIC
RESAMPLING_COEFFICIENTS = [0.5, 2048.0, -100.0, 50.0]

ENABLE_WINDOWING = True
WINDOW_TYPE = ope.WindowType.HANN
WINDOW_CENTER = 0.5
WINDOW_FILL_FACTOR = 0.95

ENABLE_DISPERSION = False
DISPERSION_COEFFICIENTS = [0.0, 0.0, 1.0, -3.0]
DISPERSION_FACTOR = 1.0

ENABLE_LOG_SCALING = True
GRAYSCALE_MIN = 30.0
GRAYSCALE_MAX = 100.0

# Display / run control
DISPLAY_BSCAN_INDEX = 0
FRAMES_TO_PROCESS = 1
TIMEOUT_SECONDS = 5.0


def to_numpy_dtype(data_type):
    mapping = {
        ope.DataType.UINT8: np.uint8,
        ope.DataType.UINT16: np.uint16,
        ope.DataType.UINT32: np.uint32,
        ope.DataType.UINT64: np.uint64,
        ope.DataType.INT8: np.int8,
        ope.DataType.INT16: np.int16,
        ope.DataType.INT32: np.int32,
        ope.DataType.INT64: np.int64,
        ope.DataType.FLOAT32: np.float32,
        ope.DataType.FLOAT64: np.float64,
        ope.DataType.COMPLEX_FLOAT32: np.complex64,
        ope.DataType.COMPLEX_FLOAT64: np.complex128,
    }
    if data_type not in mapping:
        raise ValueError(f"Unsupported input data type: {data_type}")
    return mapping[data_type]


def configure_processor(processor):
    if CONFIG_FILE is not None:
        processor.load_config(str(CONFIG_FILE))
    else:
        processor.set_input_parameters(
            signal_length=SIGNAL_LENGTH,
            ascans_per_bscan=ASCANS_PER_BSCAN,
            bscans_per_buffer=BSCANS_PER_BUFFER,
            data_type=DATA_TYPE,
        )

        processor.enable_resampling(ENABLE_RESAMPLING)
        if ENABLE_RESAMPLING:
            processor.set_interpolation_method(INTERPOLATION_METHOD)
            processor.set_resampling_coefficients(RESAMPLING_COEFFICIENTS)

        processor.enable_windowing(ENABLE_WINDOWING)
        if ENABLE_WINDOWING:
            processor.set_window_parameters(
                window_type=WINDOW_TYPE,
                center_position=WINDOW_CENTER,
                fill_factor=WINDOW_FILL_FACTOR,
            )

        processor.enable_dispersion_compensation(ENABLE_DISPERSION)
        if ENABLE_DISPERSION:
            processor.set_dispersion_coefficients(
                DISPERSION_COEFFICIENTS,
                factor=DISPERSION_FACTOR,
            )

        processor.enable_log_scaling(ENABLE_LOG_SCALING)
        processor.set_grayscale_range(min=GRAYSCALE_MIN, max=GRAYSCALE_MAX)

    if RESAMPLING_LUT_FILE is not None:
        ok = processor.config.loadResamplingLutFromFile(str(RESAMPLING_LUT_FILE))
        if not ok:
            raise RuntimeError(f"Failed to load resampling LUT: {RESAMPLING_LUT_FILE}")
        processor.enable_resampling(True)
        processor.use_custom_resampling_curve(True)


def get_effective_input_settings(processor):
    data_params = processor.config.dataParams
    return {
        "signal_length": data_params.signalLength,
        "ascans_per_bscan": data_params.ascansPerBscan,
        "bscans_per_buffer": data_params.bscansPerBuffer,
        "data_type": data_params.inputDataType,
    }


def load_frame_from_file(file_path, expected_samples, numpy_dtype):
    if file_path is None:
        raise ValueError(
            "Set RAW_FILE_PATH to your raw OCT file before running this example."
        )

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    raw = np.fromfile(file_path, dtype=numpy_dtype)
    if raw.size != expected_samples:
        expected_bytes = expected_samples * np.dtype(numpy_dtype).itemsize
        actual_bytes = raw.size * np.dtype(numpy_dtype).itemsize
        raise ValueError(
            "Input file size does not match the configured dimensions.\n"
            f"Expected: {expected_samples} samples ({expected_bytes} bytes)\n"
            f"Actual:   {raw.size} samples ({actual_bytes} bytes)"
        )
    return raw


def get_next_frame(raw_frame):
    """Replace this with your DAQ / camera readout for live processing."""
    return raw_frame


def update_display(image_artist, bscan, frame_index):
    if image_artist is None:
        plt.ion()
        figure, axis = plt.subplots(figsize=(10, 6))
        image_artist = axis.imshow(
            bscan.T,
            cmap="gray",
            aspect="auto",
            origin="upper",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(f"Processed OCT B-scan (frame {frame_index + 1})")
        axis.set_xlabel("A-scan")
        axis.set_ylabel("Depth")
        figure.tight_layout()
        plt.show(block=False)
    else:
        image_artist.set_data(bscan.T)
        image_artist.axes.set_title(f"Processed OCT B-scan (frame {frame_index + 1})")
        image_artist.figure.canvas.draw_idle()

    plt.pause(0.001)
    return image_artist


def main():
    print("=" * 50)
    print("OCTproEngine: Raw File Processing Starter")
    print("=" * 50)
    print("Replace get_next_frame() when moving to live acquisition.")
    print()

    output_ready = threading.Event()
    latest_output = {"bscan": None}

    with ope.Processor(BACKEND) as processor:
        configure_processor(processor)

        settings = get_effective_input_settings(processor)
        signal_length = settings["signal_length"]
        ascans_per_bscan = settings["ascans_per_bscan"]
        bscans_per_buffer = settings["bscans_per_buffer"]
        input_dtype = settings["data_type"]

        if DISPLAY_BSCAN_INDEX < 0 or DISPLAY_BSCAN_INDEX >= bscans_per_buffer:
            raise ValueError(
                f"DISPLAY_BSCAN_INDEX must be between 0 and {bscans_per_buffer - 1}"
            )

        numpy_dtype = to_numpy_dtype(input_dtype)
        expected_samples = signal_length * ascans_per_bscan * bscans_per_buffer
        raw_frame = load_frame_from_file(RAW_FILE_PATH, expected_samples, numpy_dtype)

        print(f"Backend: {BACKEND}")
        print(
            "Input: "
            f"{signal_length} samples x {ascans_per_bscan} A-scans x {bscans_per_buffer} B-scans"
        )
        print(f"Data type: {input_dtype}")
        print(f"Input file: {RAW_FILE_PATH}")
        if CONFIG_FILE is not None:
            print(f"Config file: {CONFIG_FILE}")
        if RESAMPLING_LUT_FILE is not None:
            print(f"Resampling LUT: {RESAMPLING_LUT_FILE}")
        print()

        def on_output(output_array, buffer_id):
            latest_output["bscan"] = output_array[DISPLAY_BSCAN_INDEX].copy()
            output_ready.set()

        processor.add_output_callback(on_output)
        processor.initialize()

        image_artist = None

        for frame_index in range(FRAMES_TO_PROCESS):
            frame = get_next_frame(raw_frame)
            output_ready.clear()

            buffer = processor.get_next_available_buffer()
            buffer[:] = np.asarray(frame, dtype=numpy_dtype).reshape(buffer.shape)
            processor.process(buffer)

            if not output_ready.wait(timeout=TIMEOUT_SECONDS):
                raise TimeoutError(
                    f"Timed out waiting for processed output after {TIMEOUT_SECONDS:.1f} seconds"
                )

            image_artist = update_display(image_artist, latest_output["bscan"], frame_index)

        if image_artist is not None:
            plt.ioff()
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
