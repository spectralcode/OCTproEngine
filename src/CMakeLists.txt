# ========================================
# Source Files
# ========================================

set(CORE_SOURCES
	core/types.cpp
	core/iobuffer.cpp
	core/processor.cpp
	core/processorconfiguration.cpp
	core/callback_manager.cpp
	core/cudautils.cpp
)

set(CPU_SOURCES
	backends/cpu/cpu_backend.cpp
)

set(CUDA_SOURCES
	backends/cuda/cuda_backend.cu
	backends/cuda/cuda_kernels.cu
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

# Create library
add_library(octproengine SHARED
	${CORE_SOURCES}
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