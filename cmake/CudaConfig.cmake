# CudaConfig.cmake
# CUDA Architecture Configuration Module
# 
# This module configures CUDA compilation architectures based on platform
# and user preferences. It should be included from CMakeLists.txt files
# that build CUDA code.
#
# Usage:
#   include(cmake/CudaConfig.cmake)
#
# Override architectures:
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;86;89"
#
# See: CUDA_ARCHITECTURE_GUIDE.md for detailed documentation

# ========================================
# CUDA Architecture Configuration
# ========================================
# Change these flags according to your GPU and CUDA version
# See: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
#
# IMPORTANT: For distributed libraries, we use:
#   XX-real = generates native binary (SASS) for sm_XX
#   XX-virtual = generates PTX for compute_XX (forward compatibility)
#
# The "-virtual" suffix on the highest architecture ensures future GPUs
# can JIT-compile the PTX code even if they don't have native binaries.
# This is equivalent to: -gencode=arch=compute_XX,code=compute_XX

function(configure_cuda_architectures)
	if(NOT BUILD_CUDA)
		message(STATUS "CUDA disabled - skipping architecture configuration")
		return()
	endif()
	
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		# Detect Jetson (ARM architecture)
		if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
			message(STATUS "Detected ARM64 architecture - assuming Jetson platform")
			# Jetson Nano: sm_53, Jetson Xavier: sm_72, Jetson Orin: sm_87
			# Note: 87-virtual provides forward compatibility for future Jetson GPUs
			set(CMAKE_CUDA_ARCHITECTURES 53-real 72-real 87-virtual PARENT_SCOPE)
			message(STATUS "Using Jetson CUDA architectures: 53-real 72-real 87-virtual")
		else()
			# Desktop/Server platforms
			
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
			
			message(STATUS "Using default desktop CUDA architectures: ${CUDA_ARCH_DEFAULT}")
			message(STATUS "To change, set CMAKE_CUDA_ARCHITECTURES when running cmake")
			message(STATUS "  Example: cmake .. -DCMAKE_CUDA_ARCHITECTURES=\"75-real;86-real;90-virtual\"")
		endif()
		
		# Architecture reference:
		# sm_52: Maxwell (GTX 9xx series)
		# sm_60/61: Pascal (GTX 10xx, P100)
		# sm_70: Volta (V100)
		# sm_75: Turing (RTX 20xx, T4)
		# sm_80: Ampere (A100, RTX 30xx)
		# sm_86: Ampere (RTX 3060, RTX 3070, RTX 3080)
		# sm_87: Ampere (Jetson Orin)
		# sm_89: Ada Lovelace (RTX 40xx, L40)
		# sm_90: Hopper (H100)
		#
		# Suffixes:
		# -real: Generates native binary (SASS) = -gencode=arch=compute_XX,code=sm_XX
		# -virtual: Generates PTX only = -gencode=arch=compute_XX,code=compute_XX
		# (no suffix): Generates both native and PTX
	else()
		message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
	endif()
endfunction()

# Call the configuration function
configure_cuda_architectures()

# Additional CUDA compilation flags can be set here
if(BUILD_CUDA)
	# Enable separable compilation for better code organization
	set(CMAKE_CUDA_SEPARABLE_COMPILATION ON PARENT_SCOPE)
	
	# Suppress some common warnings
	if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
		# Add CUDA-specific flags here if needed
		# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr" PARENT_SCOPE)
	endif()
endif()
