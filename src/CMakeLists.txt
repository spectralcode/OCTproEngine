# ========================================
# Source Files
# ========================================

set(CORE_SOURCES
	core/types.cpp
	core/iobuffer.cpp
	core/processor.cpp
	core/processorconfiguration.cpp
	core/callback_manager.cpp
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

if(BUILD_CUDA)
	# Build library with CUDA support
	add_library(octproengine SHARED
		${CORE_SOURCES}
		${CPU_SOURCES}
		${CUDA_SOURCES}
	)
	
	target_link_libraries(octproengine PUBLIC
		CUDA::cudart
		CUDA::cufft
		FFTW3::fftw3f
	)
	
	# CUDA-specific properties
	set_target_properties(octproengine PROPERTIES
		CUDA_SEPARABLE_COMPILATION ${CUDA_SEPARABLE_COMPILATION}
		CUDA_RESOLVE_DEVICE_SYMBOLS ${CUDA_RDC}
	)
	if(UNIX)
		set_target_properties(octproengine PROPERTIES POSITION_INDEPENDENT_CODE ON)
	endif()	
else()
	# CPU-only build
	add_library(octproengine SHARED
		${CORE_SOURCES}
		${CPU_SOURCES}
	)
	
	target_link_libraries(octproengine PUBLIC
		FFTW3::fftw3f
	)
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