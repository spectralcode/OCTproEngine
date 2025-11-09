# Set minimum CMake version and policies
cmake_minimum_required(VERSION 3.15)

# Set CMake policies for compatibility with newer pybind11
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)  # Only interpret if() arguments as variables or keywords when unquoted
endif()
if(POLICY CMP0057)
    cmake_policy(SET CMP0057 NEW)  # Support new IN_LIST if() operator
endif()

# Project declaration
project(octproengine_python)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
	# If not found, try to use pip-installed pybind11
	execute_process(
		COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
		OUTPUT_VARIABLE pybind11_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE pybind11_RESULT
	)
	
	if(pybind11_RESULT EQUAL 0)
		message(STATUS "Found pybind11 via pip: ${pybind11_DIR}")
		find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
	else()
		message(FATAL_ERROR 
			"pybind11 not found. Install it with:\n"
			"  pip install pybind11\n"
			"or\n"
			"  apt-get install pybind11-dev (on Ubuntu/Debian)")
	endif()
endif()

# Create Python module with unique name
pybind11_add_module(octproengine_python 
	octproengine_bindings.cpp
)

# Link against the main C++ library
target_link_libraries(octproengine_python PRIVATE octproengine)

# Include directories
target_include_directories(octproengine_python PRIVATE 
	${CMAKE_SOURCE_DIR}/include
	${Python3_INCLUDE_DIRS}
)

# Set output name to "octproengine" (without the _python suffix)
set_target_properties(octproengine_python PROPERTIES
	OUTPUT_NAME "octproengine"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>"
)

# Windows: Copy all required DLLs to the Python module directory
if(WIN32)
	# Copy octproengine.dll
	add_custom_command(TARGET octproengine_python POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"$<TARGET_FILE:octproengine>"
			"$<TARGET_FILE_DIR:octproengine_python>"
		COMMENT "Copying octproengine.dll to Python module directory"
	)
	
	# Copy FFTW DLL (single precision)
	set(_fftw_dll "")
	if(DEFINED FFTW3_ROOT AND EXISTS "${FFTW3_ROOT}/libfftw3f-3.dll")
		set(_fftw_dll "${FFTW3_ROOT}/libfftw3f-3.dll")
	elseif(EXISTS "${CMAKE_SOURCE_DIR}/src/thirdparty/fftw/libfftw3f-3.dll")
		set(_fftw_dll "${CMAKE_SOURCE_DIR}/src/thirdparty/fftw/libfftw3f-3.dll")
	endif()
	
	if(_fftw_dll)
		add_custom_command(TARGET octproengine_python POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${_fftw_dll}"
				"$<TARGET_FILE_DIR:octproengine_python>"
			COMMENT "Copying FFTW DLL to Python module directory"
		)
	else()
		message(WARNING "FFTW DLL (libfftw3f-3.dll) not found; Python module may fail to import at runtime.")
	endif()
	
	# Copy CUDA DLLs if CUDA is enabled
	if(BUILD_CUDA)
		# Find CUDA DLLs
		find_file(CUDA_CUDART_DLL 
			NAMES cudart64_12.dll cudart64_11.dll cudart64_10.dll
			PATHS "${CUDAToolkit_BIN_DIR}"
			NO_DEFAULT_PATH
		)
		find_file(CUDA_CUFFT_DLL 
			NAMES cufft64_11.dll cufft64_10.dll cufft64_12.dll
			PATHS "${CUDAToolkit_BIN_DIR}"
			NO_DEFAULT_PATH
		)
		
		if(CUDA_CUDART_DLL)
			add_custom_command(TARGET octproengine_python POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${CUDA_CUDART_DLL}"
					"$<TARGET_FILE_DIR:octproengine_python>"
				COMMENT "Copying CUDA runtime DLL to Python module directory"
			)
		endif()
		
		if(CUDA_CUFFT_DLL)
			add_custom_command(TARGET octproengine_python POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${CUDA_CUFFT_DLL}"
					"$<TARGET_FILE_DIR:octproengine_python>"
				COMMENT "Copying CUDA FFT DLL to Python module directory"
			)
		endif()
	endif()
endif()

# Linux: Set RPATH so the module can find shared libraries
if(UNIX AND NOT APPLE)
	set_target_properties(octproengine_python PROPERTIES
		INSTALL_RPATH "$ORIGIN"
		BUILD_WITH_INSTALL_RPATH TRUE
	)
endif()

# Installation
install(TARGETS octproengine_python
	LIBRARY DESTINATION ${Python3_SITELIB}
)

message(STATUS "Python bindings configured:")
message(STATUS "  Python version: ${Python3_VERSION}")
message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")