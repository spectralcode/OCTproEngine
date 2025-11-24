#	Read the OpenCL kernel source file and split to avoid MSVC string length limit
#	Split at line 370 (before Dispersion Compensation section)

cmake_policy(SET CMP0007 NEW)

file(STRINGS ${INPUT_FILE} KERNEL_LINES)
list(LENGTH KERNEL_LINES LINE_COUNT)

#	First part: lines 0-369 (370 lines)
set(PART1 "")
set(LINE_NUM 0)
foreach(line IN LISTS KERNEL_LINES)
	if(LINE_NUM LESS 370)
		string(APPEND PART1 "${line}\n")
	endif()
	math(EXPR LINE_NUM "${LINE_NUM} + 1")
endforeach()

#	Second part: lines 370+
set(PART2 "")
set(LINE_NUM 0)
foreach(line IN LISTS KERNEL_LINES)
	if(NOT LINE_NUM LESS 370)
		string(APPEND PART2 "${line}\n")
	endif()
	math(EXPR LINE_NUM "${LINE_NUM} + 1")
endforeach()

#	Generate the C++ source file using raw string literals (split into 2 parts)
file(WRITE ${OUTPUT_FILE}
"/**
**  This file is part of OCTproEngine.
**  Auto-generated embedded OpenCL kernel source
**  DO NOT EDIT - Generated from opencl_kernels.cl
**/

#include \"opencl_kernels.h\"
#include <string>

namespace ope {
namespace opencl {

// Kernel name constants
const char* KERNEL_INPUT_TO_COMPLEX = \"inputToCufftComplex\";
const char* KERNEL_INPUT_TO_COMPLEX_BITSHIFT = \"inputToCufftComplex_and_bitshift\";
const char* KERNEL_ROLLING_AVERAGE_BACKGROUND_REMOVAL = \"rollingAverageBackgroundRemoval\";
const char* KERNEL_KLINEARIZATION = \"klinearization\";
const char* KERNEL_KLINEARIZATION_QUADRATIC = \"klinearizationQuadratic\";
const char* KERNEL_KLINEARIZATION_CUBIC = \"klinearizationCubic\";
const char* KERNEL_KLINEARIZATION_LANCZOS = \"klinearizationLanczos\";
const char* KERNEL_WINDOWING = \"windowing\";
const char* KERNEL_KLINEARIZATION_AND_WINDOWING = \"klinearizationAndWindowing\";
const char* KERNEL_KLINEARIZATION_CUBIC_AND_WINDOWING = \"klinearizationCubicAndWindowing\";
const char* KERNEL_KLINEARIZATION_LANCZOS_AND_WINDOWING = \"klinearizationLanczosAndWindowing\";
const char* KERNEL_FILL_DISPERSIVE_PHASE = \"fillDispersivePhase\";
const char* KERNEL_DISPERSION_COMPENSATION = \"dispersionCompensation\";
const char* KERNEL_DISPERSION_COMPENSATION_AND_WINDOWING = \"dispersionCompensationAndWindowing\";
const char* KERNEL_KLINEARIZATION_AND_WINDOWING_AND_DISPERSION = \"klinearizationAndWindowingAndDispersionCompensation\";
const char* KERNEL_KLINEARIZATION_CUBIC_AND_WINDOWING_AND_DISPERSION = \"klinearizationCubicAndWindowingAndDispersionCompensation\";
const char* KERNEL_KLINEARIZATION_LANCZOS_AND_WINDOWING_AND_DISPERSION = \"klinearizationLanczosAndWindowingAndDispersionCompensation\";
const char* KERNEL_POST_PROCESS_TRUNCATE_LOG = \"postProcessTruncateLog\";
const char* KERNEL_POST_PROCESS_TRUNCATE_LIN = \"postProcessTruncateLin\";
const char* KERNEL_GET_MINIMUM_VARIANCE_MEAN = \"getMinimumVarianceMean\";
const char* KERNEL_MEAN_ALINE_SUBTRACTION = \"meanALineSubtraction\";
const char* KERNEL_BSCAN_FLIP = \"bscanFlip\";
const char* KERNEL_FILL_SINUSOIDAL_SCAN_CURVE = \"fillSinusoidalScanCorrectionCurve\";
const char* KERNEL_SINUSOIDAL_SCAN_CORRECTION = \"sinusoidalScanCorrection\";
const char* KERNEL_GET_POST_PROCESS_BACKGROUND = \"getPostProcessBackground\";
const char* KERNEL_POST_PROCESS_BACKGROUND_SUBTRACTION = \"postProcessBackgroundSubtraction\";

// OpenCL kernel source embedded as raw string literal
// Split into parts to avoid MSVC string length limit
const char* getKernelSource() {
	static std::string kernelSource =
		R\"OPENCL_KERNELS(${PART1})OPENCL_KERNELS\"
		R\"OPENCL_KERNELS(${PART2})OPENCL_KERNELS\";
	return kernelSource.c_str();
}

} // namespace opencl
} // namespace ope
")

message(STATUS "Generated ${OUTPUT_FILE} from ${INPUT_FILE}")
