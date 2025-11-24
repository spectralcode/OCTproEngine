#ifndef OPE_OPENCL_KERNELS_H
#define OPE_OPENCL_KERNELS_H

#include <CL/cl.h>

namespace ope {
namespace opencl {

// ============================================
// Kernel Names (matching those in .cl file)
// ============================================

// Input conversion
extern const char* KERNEL_INPUT_TO_COMPLEX;
extern const char* KERNEL_INPUT_TO_COMPLEX_BITSHIFT;

// Background removal
extern const char* KERNEL_ROLLING_AVERAGE_BACKGROUND_REMOVAL;

// K-linearization
extern const char* KERNEL_KLINEARIZATION;
extern const char* KERNEL_KLINEARIZATION_QUADRATIC;
extern const char* KERNEL_KLINEARIZATION_CUBIC;
extern const char* KERNEL_KLINEARIZATION_LANCZOS;

// Windowing
extern const char* KERNEL_WINDOWING;

// K-linearization + Windowing
extern const char* KERNEL_KLINEARIZATION_AND_WINDOWING;
extern const char* KERNEL_KLINEARIZATION_CUBIC_AND_WINDOWING;
extern const char* KERNEL_KLINEARIZATION_LANCZOS_AND_WINDOWING;

// Dispersion compensation
extern const char* KERNEL_FILL_DISPERSIVE_PHASE;
extern const char* KERNEL_DISPERSION_COMPENSATION;
extern const char* KERNEL_DISPERSION_COMPENSATION_AND_WINDOWING;

// K-linearization + Windowing + Dispersion
extern const char* KERNEL_KLINEARIZATION_AND_WINDOWING_AND_DISPERSION;
extern const char* KERNEL_KLINEARIZATION_CUBIC_AND_WINDOWING_AND_DISPERSION;
extern const char* KERNEL_KLINEARIZATION_LANCZOS_AND_WINDOWING_AND_DISPERSION;

// Post-processing
extern const char* KERNEL_POST_PROCESS_TRUNCATE_LOG;
extern const char* KERNEL_POST_PROCESS_TRUNCATE_LIN;

// Fixed-pattern noise
extern const char* KERNEL_GET_MINIMUM_VARIANCE_MEAN;
extern const char* KERNEL_MEAN_ALINE_SUBTRACTION;

// B-scan operations
extern const char* KERNEL_BSCAN_FLIP;

// Sinusoidal scan correction
extern const char* KERNEL_FILL_SINUSOIDAL_SCAN_CURVE;
extern const char* KERNEL_SINUSOIDAL_SCAN_CORRECTION;

// Post-process background
extern const char* KERNEL_GET_POST_PROCESS_BACKGROUND;
extern const char* KERNEL_POST_PROCESS_BACKGROUND_SUBTRACTION;

// ============================================
// OpenCL Kernel Source Code
// ============================================

// This returns the complete OpenCL kernel source code as a string
const char* getKernelSource();

} // namespace opencl
} // namespace ope

#endif // OPE_OPENCL_KERNELS_H
