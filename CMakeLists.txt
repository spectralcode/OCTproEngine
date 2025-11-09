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
option(BUILD_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Workaround for Visual Studio 2022 + pybind11 mutex bug
# see: https://github.com/microsoft/STL/issues/4875
if(MSVC)
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# Add cmake module path for custom Find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find dependencies
if(BUILD_CUDA)
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
	set(CMAKE_CUDA_STANDARD 14)
	add_compile_definitions(BUILD_WITH_CUDA)
	message(STATUS "Building with CUDA support")
else()
	message(STATUS "Building CPU-only (CUDA disabled)")
endif()

# Find FFTW3
find_package(FFTW3 REQUIRED)

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
message(STATUS "CUDA support: ${BUILD_CUDA}")
message(STATUS "Python bindings: ${BUILD_PYTHON}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "===================================")
message(STATUS "")