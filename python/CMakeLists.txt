# Set minimum CMake version and policies
cmake_minimum_required(VERSION 3.15)

if(POLICY CMP0054)
	cmake_policy(SET CMP0054 NEW)  # Only interpret if() arguments as variables or keywords when unquoted
endif()
if(POLICY CMP0057)
	cmake_policy(SET CMP0057 NEW)  # Support new IN_LIST if() operator
endif()
if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW) # LIBRARY DESTINATION is used for both SHARED and MODULE targets
endif()

# Project declaration
project(octproengine_python)

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
if(WIN32 AND COMMAND copy_required_dlls)
	copy_required_dlls(octproengine_python)
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