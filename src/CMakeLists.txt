# ========================================
# Source Files
# ========================================

set(CORE_SOURCES
	core/types.cpp
	core/iobuffer.cpp
	core/processor.cpp
	core/processorconfiguration.cpp
	core/backendconfig.cpp
	core/callback_manager.cpp
	core/buffer_manager.cpp
	core/output_buffer_manager.cpp
	core/processortool.cpp
)


set(TOOLS_SOURCES)
if(BUILD_TOOLS)
	list(APPEND TOOLS_SOURCES tools/recorder.cpp)
endif()

set(CPU_SOURCES
	backends/cpu/cpu_backend.cpp
)


set(CUDA_SOURCES
	backends/cuda/cuda_backend.cu
	backends/cuda/cuda_kernels.cu
)


# ========================================
# OpenCL Kernel Embedding
# ========================================
if(BUILD_OPENCL)
	# Auto-generate opencl_kernels_source.cpp from opencl_kernels.cl
	add_custom_command(
		OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_kernels_source.cpp
		COMMAND ${CMAKE_COMMAND}
			-DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_kernels.cl
			-DOUTPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_kernels_source.cpp
			-P ${CMAKE_SOURCE_DIR}/cmake/EmbedOpenCLKernels.cmake
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/backends/opencl/opencl_kernels.cl
		COMMENT "Embedding OpenCL kernels from opencl_kernels.cl"
		VERBATIM
	)
endif()

set(OPENCL_SOURCES
	backends/opencl/opencl_backend.cpp
	backends/opencl/opencl_kernels_source.cpp
)


set(VULKAN_SOURCES
	backends/vulkan/vulkan_backend.cpp
)


# ========================================
# Build Library
# ========================================

# Determine which sources to include
set(BACKEND_SOURCES)
if(BUILD_CPU)
	list(APPEND BACKEND_SOURCES ${CPU_SOURCES})
endif()

if(BUILD_CUDA)
	list(APPEND BACKEND_SOURCES ${CUDA_SOURCES})
endif()

if(BUILD_OPENCL)
	list(APPEND BACKEND_SOURCES ${OPENCL_SOURCES})
endif()

if(BUILD_VULKAN)
	list(APPEND BACKEND_SOURCES ${VULKAN_SOURCES})
endif()

# Create library
add_library(octproengine SHARED
	${CORE_SOURCES}
	${TOOLS_SOURCES}
	${BACKEND_SOURCES}
)


# Link libraries
set(LINK_LIBRARIES)

if(BUILD_CPU)
	list(APPEND LINK_LIBRARIES FFTW3::fftw3f)
endif()

if(BUILD_CUDA)
	list(APPEND LINK_LIBRARIES CUDA::cudart CUDA::cufft)
endif()

if(BUILD_OPENCL)
	list(APPEND LINK_LIBRARIES OpenCL::OpenCL VkFFT::VkFFT)
endif()

if(BUILD_VULKAN)
	# Find glslang libraries (required by VkFFT for runtime shader compilation)
	find_library(GLSLANG_LIB NAMES glslang
		PATHS
			${Vulkan_INCLUDE_DIRS}/../Lib
			${Vulkan_INCLUDE_DIRS}/../lib
			/usr/lib
			/usr/local/lib
	)
	find_library(SPIRV_LIB NAMES SPIRV
		PATHS
			${Vulkan_INCLUDE_DIRS}/../Lib
			${Vulkan_INCLUDE_DIRS}/../lib
			/usr/lib
			/usr/local/lib
	)

	list(APPEND LINK_LIBRARIES Vulkan::Vulkan VkFFT::VkFFT ${SHADERC_LIB})
	if(GLSLANG_LIB)
		list(APPEND LINK_LIBRARIES ${GLSLANG_LIB})
		message(STATUS "Found glslang library: ${GLSLANG_LIB}")
	endif()
	if(SPIRV_LIB)
		list(APPEND LINK_LIBRARIES ${SPIRV_LIB})
		message(STATUS "Found SPIRV library: ${SPIRV_LIB}")
	endif()

	# Add glslang include directory for VkFFT
	target_include_directories(octproengine PRIVATE ${GLSLANG_INCLUDE_DIRS})
endif()

target_link_libraries(octproengine PUBLIC ${LINK_LIBRARIES})

# CUDA-specific properties
if(BUILD_CUDA)
	set_target_properties(octproengine PROPERTIES
		CUDA_SEPARABLE_COMPILATION ${CUDA_SEPARABLE_COMPILATION}
		CUDA_RESOLVE_DEVICE_SYMBOLS ${CUDA_RDC}
	)
endif()

if(UNIX)
	set_target_properties(octproengine PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ========================================
# Include Directories
# ========================================

target_include_directories(octproengine PUBLIC
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)

# ========================================
# Library Properties
# ========================================

# Set version
set_target_properties(octproengine PROPERTIES
	VERSION ${PROJECT_VERSION}
	SOVERSION 1
)


# Define BUILD_OPE_LIBRARY when building the DLL (enables dllexport)
target_compile_definitions(octproengine PRIVATE
	BUILD_OPE_LIBRARY
)


# Define backend availability macros
if(BUILD_CUDA)
	target_compile_definitions(octproengine PUBLIC OPE_CUDA_AVAILABLE)
	message(STATUS "Defining OPE_CUDA_AVAILABLE")
endif()

if(BUILD_CPU)
	target_compile_definitions(octproengine PUBLIC OPE_CPU_AVAILABLE)
	message(STATUS "Defining OPE_CPU_AVAILABLE")
endif()

if(BUILD_OPENCL)
	target_compile_definitions(octproengine PUBLIC OPE_OPENCL_AVAILABLE)
	message(STATUS "Defining OPE_OPENCL_AVAILABLE")
endif()

if(BUILD_VULKAN)
	target_compile_definitions(octproengine PUBLIC OPE_VULKAN_AVAILABLE)
	message(STATUS "Defining OPE_VULKAN_AVAILABLE")
endif()

# Windows-specific settings
if(WIN32)
	target_compile_definitions(octproengine PRIVATE
		NOMINMAX
		_CRT_SECURE_NO_WARNINGS
	)
endif()

# Linux/GCC-specific settings - set default symbol visibility
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	target_compile_options(octproengine PRIVATE
		-fvisibility=hidden
	)
endif()

# ========================================
# Copy Vulkan Shaders to Build Directory
# ========================================
if(BUILD_VULKAN)
	# Copy Vulkan shaders to build directory (next to DLL/so)
	add_custom_command(TARGET octproengine POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory
			$<TARGET_FILE_DIR:octproengine>/shaders
		COMMAND ${CMAKE_COMMAND} -E copy_directory
			${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/shaders
			$<TARGET_FILE_DIR:octproengine>/shaders
		COMMENT "Copying Vulkan shaders to build directory"
	)

	# Copy shaders to test directory
	if(BUILD_TESTS)
		add_custom_command(TARGET octproengine POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory
				${CMAKE_BINARY_DIR}/tests/$<CONFIG>/shaders
			COMMAND ${CMAKE_COMMAND} -E copy_directory
				${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/shaders
				${CMAKE_BINARY_DIR}/tests/$<CONFIG>/shaders
			COMMENT "Copying Vulkan shaders to test directory"
		)
	endif()

	# Copy shaders to examples directory
	if(BUILD_EXAMPLES)
		add_custom_command(TARGET octproengine POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory
				${CMAKE_BINARY_DIR}/examples/$<CONFIG>/shaders
			COMMAND ${CMAKE_COMMAND} -E copy_directory
				${CMAKE_CURRENT_SOURCE_DIR}/backends/vulkan/shaders
				${CMAKE_BINARY_DIR}/examples/$<CONFIG>/shaders
			COMMENT "Copying Vulkan shaders to examples directory"
		)
	endif()
endif()

# ========================================
# Install Rules (Phase 5)
# ========================================

install(TARGETS octproengine
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
)


install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
	DESTINATION include/octproengine
	FILES_MATCHING PATTERN "*.h"
)
