cmake_minimum_required(VERSION 3.18)

# Auto generate version.h from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION)
string(STRIP "${PROJECT_VERSION}" PROJECT_VERSION)

project(octproengine VERSION ${PROJECT_VERSION} LANGUAGES CXX)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/version.h"
	@ONLY
)

# Options
option(BUILD_CUDA "Build with CUDA support" ON)
option(BUILD_CPU "Build with CPU backend" ON)
option(BUILD_OPENCL "Build with OpenCL backend" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_TOOLS "Build optional ProcessorTools (Recorder, etc.)" ON)
option(FFTW3_AUTO_DOWNLOAD "Automatically download FFTW3 to thirdparty/fftw if not found (Windows only)" ON)
option(CLFFT_AUTO_DOWNLOAD "Automatically download clFFT to thirdparty/clfft if not found (Windows only)" ON)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Workaround for Visual Studio 2022 + pybind11 mutex bug
# see: https://github.com/microsoft/STL/issues/4875
if(MSVC)
	add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Validate backend selection
if(NOT BUILD_CUDA AND NOT BUILD_CPU AND NOT BUILD_OPENCL)
	message(FATAL_ERROR "At least one backend must be enabled (BUILD_CUDA, BUILD_CPU, or BUILD_OPENCL)")
endif()

# Add cmake module path for custom Find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find dependencies
if(BUILD_CUDA)
	include(cmake/CudaConfig.cmake)
else()
	message(STATUS "Building without CUDA")
endif()

# Find FFTW3 (only needed for CPU backend)
if(BUILD_CPU)
	find_package(FFTW3 REQUIRED)
else()
	message(STATUS "Building without CPU backend - FFTW3 not required")
endif()

# Find OpenCL and clFFT (only needed for OpenCL backend)
if(BUILD_OPENCL)
	# Help CMake find OpenCL in CUDA installation
	if(WIN32 AND NOT DEFINED OpenCL_ROOT AND NOT DEFINED ENV{OpenCL_ROOT})
		# Check common CUDA installation locations
		set(CUDA_SEARCH_PATHS
			"$ENV{CUDA_PATH}"
			"$ENV{CUDA_HOME}"
			"$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA"
			"$ENV{ProgramW6432}/NVIDIA GPU Computing Toolkit/CUDA"
		)
		
		foreach(SEARCH_PATH ${CUDA_SEARCH_PATHS})
			if(EXISTS "${SEARCH_PATH}")
				file(GLOB CUDA_VERSIONS "${SEARCH_PATH}/v*")
				if(CUDA_VERSIONS)
					list(SORT CUDA_VERSIONS)
					list(REVERSE CUDA_VERSIONS)
					
					# Find first version with valid OpenCL files
					foreach(CUDA_VER_PATH ${CUDA_VERSIONS})
						# Normalize path to use forward slashes
						file(TO_CMAKE_PATH "${CUDA_VER_PATH}" CUDA_VER_PATH_NORMALIZED)
						
						if(EXISTS "${CUDA_VER_PATH_NORMALIZED}/include/CL/opencl.h")
							if(EXISTS "${CUDA_VER_PATH_NORMALIZED}/lib/x64/OpenCL.lib" OR EXISTS "${CUDA_VER_PATH_NORMALIZED}/lib/OpenCL.lib")
								set(OpenCL_INCLUDE_DIR "${CUDA_VER_PATH_NORMALIZED}/include")
								if(EXISTS "${CUDA_VER_PATH_NORMALIZED}/lib/x64/OpenCL.lib")
									set(OpenCL_LIBRARY "${CUDA_VER_PATH_NORMALIZED}/lib/x64/OpenCL.lib")
								else()
									set(OpenCL_LIBRARY "${CUDA_VER_PATH_NORMALIZED}/lib/OpenCL.lib")
								endif()
								message(STATUS "Found OpenCL in CUDA: ${CUDA_VER_PATH_NORMALIZED}")
								break()
							endif()
						endif()
					endforeach()
					
					if(DEFINED OpenCL_INCLUDE_DIR)
						break()
					endif()
				endif()
			endif()
		endforeach()
	endif()
	
	find_package(OpenCL REQUIRED)
	message(STATUS "Found OpenCL: ${OpenCL_LIBRARIES}")
	
	find_package(clFFT)
	if(clFFT_FOUND)
		message(STATUS "clFFT found - OpenCL backend enabled")
	else()
		message(WARNING "clFFT not found - disabling OpenCL backend")
		message(STATUS "Note: Auto-download is ${CLFFT_AUTO_DOWNLOAD}")
		set(BUILD_OPENCL OFF CACHE BOOL "Build with OpenCL backend" FORCE)
	endif()
else()
	message(STATUS "Building without OpenCL backend")
endif()

# Include DLL copying helper for Windows
include(cmake/CopyDLLs.cmake)

# octproengine src folder
add_subdirectory(src)

# Subdirectories
if(BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()

if(BUILD_PYTHON)
	add_subdirectory(python)
endif()

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# print summary
message(STATUS "")
message(STATUS "=== OCTproEngine Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA backend: ${BUILD_CUDA}")
message(STATUS "CPU backend: ${BUILD_CPU}")
message(STATUS "OpenCL backend: ${BUILD_OPENCL}")
message(STATUS "Python bindings: ${BUILD_PYTHON}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build tools: ${BUILD_TOOLS}")
message(STATUS "===================================")
message(STATUS "")