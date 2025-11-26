# OpenCLHelper.cmake
# Helper to find OpenCL in CUDA installations on Windows
# This allows CMake's FindOpenCL to locate OpenCL headers and libraries
# that are bundled with NVIDIA CUDA Toolkit

if(WIN32 AND NOT DEFINED OpenCL_ROOT AND NOT DEFINED ENV{OpenCL_ROOT})
	# Common CUDA installation locations
	set(CUDA_SEARCH_PATHS
		"$ENV{CUDA_PATH}"
		"$ENV{CUDA_HOME}"
		"$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA"
		"$ENV{ProgramW6432}/NVIDIA GPU Computing Toolkit/CUDA"
	)
	
	foreach(SEARCH_PATH ${CUDA_SEARCH_PATHS})
		if(EXISTS "${SEARCH_PATH}")
			# Find all CUDA versions (v11.0, v12.0, etc.)
			file(GLOB CUDA_VERSIONS "${SEARCH_PATH}/v*")
			if(CUDA_VERSIONS)
				# Sort and get the latest version
				list(SORT CUDA_VERSIONS)
				list(REVERSE CUDA_VERSIONS)
				
				# Find first version with valid OpenCL files
				foreach(CUDA_VER_PATH ${CUDA_VERSIONS})
					# Normalize path to use forward slashes for CMake
					file(TO_CMAKE_PATH "${CUDA_VER_PATH}" CUDA_VER_PATH_NORMALIZED)
					
					# Check if this CUDA version has OpenCL headers
					if(EXISTS "${CUDA_VER_PATH_NORMALIZED}/include/CL/opencl.h")
						# Check for OpenCL library (x64 or default location)
						if(EXISTS "${CUDA_VER_PATH_NORMALIZED}/lib/x64/OpenCL.lib" OR 
						   EXISTS "${CUDA_VER_PATH_NORMALIZED}/lib/OpenCL.lib")
							# Set variables for CMake's FindOpenCL module
							set(OpenCL_INCLUDE_DIR "${CUDA_VER_PATH_NORMALIZED}/include" CACHE PATH "OpenCL include directory")
							if(EXISTS "${CUDA_VER_PATH_NORMALIZED}/lib/x64/OpenCL.lib")
								set(OpenCL_LIBRARY "${CUDA_VER_PATH_NORMALIZED}/lib/x64/OpenCL.lib" CACHE FILEPATH "OpenCL library")
							else()
								set(OpenCL_LIBRARY "${CUDA_VER_PATH_NORMALIZED}/lib/OpenCL.lib" CACHE FILEPATH "OpenCL library")
							endif()
							message(STATUS "Found OpenCL in CUDA: ${CUDA_VER_PATH_NORMALIZED}")
							return()
						endif()
					endif()
				endforeach()
			endif()
		endif()
	endforeach()
endif()
