# CudaConfig.cmake
# CUDA Toolkit Selection + Architecture Configuration Module
#
# This module selects the CUDA toolkit/compiler (when requested) and configures
# CUDA compilation architectures and flags based on platform and user preferences.
# It should be included from CMakeLists.txt files that build CUDA code.
#
# Usage:
#	# In your top-level CMakeLists.txt (when BUILD_CUDA is ON):
#	include(cmake/CudaConfig.cmake)
#
# Toolkit selection (optional, per-build):
#	# Exact install root (preferred for custom paths)
#	-DCUDA_HOME="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
#	# Version-only (portable across machines)
#	-DCUDA_VERSION=12.2
#	# Env vars also honored: CUDA_HOME, CUDAToolkit_ROOT, CUDA_PATH
#
# Override architectures:
#	cmake .. -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;90-virtual"
#	or legacy form:
#	-DCUDA_ARCH_LIST="75;86;90"
#
# CUDA Architecture Configuration:
# See: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
#
# Other common options:
#	-DCUDA_FAST_MATH=ON
#	-DCUDA_ENABLE_WARNINGS=ON
#	-DCUDA_EXTRA_NVCC_FLAGS="--expt-relaxed-constexpr;-lineinfo"
#	-DCUDA_RDC=ON
#	-DCUDA_SEPARABLE_COMPILATION=ON

# --------------------------
# 1) Selection inputs
# --------------------------
# Preferred explicit root and version
set(CUDA_HOME "" CACHE PATH "Optional CUDA root (e.g. C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2 or /usr/local/cuda-12.2)")
set(CUDA_VERSION "" CACHE STRING "Optional exact CUDA Toolkit version (e.g. 11.8, 12.2)")

# Use environment defaults if cache vars are not set
if(NOT CUDA_HOME AND DEFINED ENV{CUDA_HOME})
	set(CUDA_HOME "$ENV{CUDA_HOME}")
endif()
if(NOT CUDAToolkit_ROOT AND DEFINED ENV{CUDAToolkit_ROOT})
	set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}")
endif()
if(NOT CUDAToolkit_ROOT AND DEFINED ENV{CUDA_PATH})
	set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}")
endif()
if(NOT CUDA_VERSION AND DEFINED ENV{CUDA_VERSION})
	set(CUDA_VERSION "$ENV{CUDA_VERSION}")
endif()

# If CUDA_HOME is provided, prefer its nvcc and treat as root
if(CUDA_HOME)
	if(WIN32)
		set(_nvcc "${CUDA_HOME}/bin/nvcc.exe")
	else()
		set(_nvcc "${CUDA_HOME}/bin/nvcc")
	endif()
	if(EXISTS "${_nvcc}")
		# Must be set before enabling the CUDA language
		set(CMAKE_CUDA_COMPILER "${_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
	endif()
	set(CUDAToolkit_ROOT "${CUDA_HOME}" CACHE PATH "" FORCE)
endif()

# If still not chosen, respect PATH order: first 'nvcc' wins
if(NOT CMAKE_CUDA_COMPILER)
	find_program(_nvcc_on_path nvcc)
	if(_nvcc_on_path)
		set(CMAKE_CUDA_COMPILER "${_nvcc_on_path}" CACHE FILEPATH "CUDA compiler (from PATH)" FORCE)
	endif()
endif()

# --------------------------
# 2) Enable & find toolkit
# --------------------------
enable_language(CUDA)

if(CUDA_VERSION)
	find_package(CUDAToolkit ${CUDA_VERSION} EXACT REQUIRED)
else()
	find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_CUDA_STANDARD 14)
add_compile_definitions(BUILD_WITH_CUDA)
message(STATUS "Building with CUDA support")

# --------------------------
# 3) User-facing CUDA options
# --------------------------
# These are simple, optional knobs developers can set per-build.

# Preserve flexible arch control via either CMAKE_CUDA_ARCHITECTURES (preferred) or CUDA_ARCH_LIST (legacy gencode flags).
#	If the user sets CMAKE_CUDA_ARCHITECTURES themselves, we do not override it.
#	If they don't, but set CUDA_ARCH_LIST, we honor it by adding -gencode flags.
#	Otherwise, we apply the default preset block below and set CMAKE_CUDA_ARCHITECTURES accordingly.

# Legacy-style list (e.g. "86;89;90"); translated to -gencode flags when used.
set(CUDA_ARCH_LIST "" CACHE STRING "SM architectures to target (e.g. 75;86;89;90)")

# Cuda options and flags
option(CUDA_FAST_MATH "Enable --use_fast_math for CUDA" ON)
option(CUDA_ENABLE_WARNINGS "Enable extra CUDA warnings" OFF)
set(CUDA_EXTRA_NVCC_FLAGS "" CACHE STRING "Extra flags passed to NVCC (semicolon-separated)")
option(CUDA_RDC "Enable relocatable device code (CUDA_RESOLVE_DEVICE_SYMBOLS)" ON)
option(CUDA_SEPARABLE_COMPILATION "Enable separable compilation (CUDA_SEPARABLE_COMPILATION)" ON)

# If the user did not set CMAKE_CUDA_ARCHITECTURES, apply the default preset.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CUDA_ARCH_LIST)
	# No explicit architectures were provided; apply your preset block and messages.

	# === Option 1: Maximum compatibility with CUDA 11.0+ ===
	# Uncomment for older GPUs (Maxwell and newer)
	# Native binaries for common GPUs + PTX for forward compatibility
	# set(CUDA_ARCH_OPTION1 52-real 60-real 61-real 70-real 75-real 80-real 86-virtual)
	
	# === Option 2: Maximum compatibility with CUDA 12.x ===
	# Uncomment for CUDA 12.x with broader support
	# set(CUDA_ARCH_OPTION2 52-real 60-real 61-real 70-real 75-real 80-real 86-real 87-real 89-real 90-virtual)
	
	# === Option 3: CUDA 13.0+ (Default - Modern GPUs) ===
	# Turing (RTX 20xx), Ampere (RTX 30xx), Ada Lovelace (RTX 40xx), Hopper (H100)
	# Native binaries for modern GPUs + PTX for future GPUs
	set(CUDA_ARCH_DEFAULT 75-real 80-real 86-real 87-real 89-real 90-virtual)
	
	# Use the selected option (default: Option 3)
	# To change: uncomment one of the options above and change CUDA_ARCH_DEFAULT to CUDA_ARCH_OPTIONX
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_DEFAULT} PARENT_SCOPE)
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_DEFAULT} CACHE STRING "" FORCE)

	message(STATUS "Using default desktop CUDA architectures: ${CUDA_ARCH_DEFAULT}")
	message(STATUS "To change, set CMAKE_CUDA_ARCHITECTURES when running cmake")
	message(STATUS "  Example: cmake .. -DCMAKE_CUDA_ARCHITECTURES=\"75-real;86-real;90-virtual\"")
endif()

# If user provided CUDA_ARCH_LIST, honor it with -gencode flags.
if(CUDA_ARCH_LIST)
	separate_arguments(_cuda_arches NATIVE_COMMAND "${CUDA_ARCH_LIST}")
	foreach(sm IN LISTS _cuda_arches)
		add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${sm},code=sm_${sm}>)
	endforeach()
endif()

# Warnings
if(CUDA_ENABLE_WARNINGS)
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
endif()

# Fast math
if(CUDA_FAST_MATH)
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

# Extra NVCC flags
if(CUDA_EXTRA_NVCC_FLAGS)
	separate_arguments(_nvcc_extra NATIVE_COMMAND "${CUDA_EXTRA_NVCC_FLAGS}")
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:${_nvcc_extra}>)
endif()

