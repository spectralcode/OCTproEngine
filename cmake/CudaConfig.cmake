# For custom configuration:
#
# Select specific CUDA version with specific architecture
# for windows: 
#	cmake -S .. -B . -G "Visual Studio 17 2022" -T cuda=12.9 -DCMAKE_CUDA_ARCHITECTURES="60-real;61-virtual"
#
# Override target architectures:
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;90-virtual"
#	See: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
#
# Configure build options:
#   cmake .. -DCUDA_FAST_MATH=OFF
#   cmake .. -DCUDA_RDC=ON -DCUDA_SEPARABLE_COMPILATION=OFF
#   cmake .. -DCUDA_EXTRA_NVCC_FLAGS="--expt-relaxed-constexpr;-lineinfo"


# Detect whether the user already provided a value (via -D / presets / toolchain)
get_property(_user_set CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY TYPE)
set(_user_provided FALSE)
if(_user_set)
	set(_user_provided TRUE)
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDAToolkit 11.0 REQUIRED)

# If the user did not provide cuda architecture, apply defaults
# todo: maybe limit default archs for faster dev builds; full list only needed for release wheels
if(NOT _user_provided OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
	unset(CMAKE_CUDA_ARCHITECTURES CACHE)
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
		# Jetson Nano:
		set(CMAKE_CUDA_ARCHITECTURES "53-real;53-virtual") #todo: likely wrong for other models other than Jetsons Nano, needs proper detection
	elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
		# CUDA 13.0+:
		set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;87-real;89-real;90-real;100-real;120-real;120-virtual")
	elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
		# CUDA 12.x:
		set(CMAKE_CUDA_ARCHITECTURES "52-real;60-real;61-real;70-real;75-real;80-real;86-real;87-real;89-real;90-real;100-real;120-real;120-virtual")
	else()
		# CUDA 11.x:
		set(CMAKE_CUDA_ARCHITECTURES "52-real;60-real;61-real;70-real;75-real;80-real;86-real;86-virtual")
	endif()

	message(STATUS "Using project defaults: ${CMAKE_CUDA_ARCHITECTURES}")
else()
	message(STATUS "Using user-provided architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()


# Options
option(CUDA_FAST_MATH "Enable --use_fast_math" ON)
option(CUDA_RDC "Enable relocatable device code" ON)
option(CUDA_SEPARABLE_COMPILATION "Enable separable compilation" ON)
option(CUDA_ENABLE_WARNINGS "Enable extra CUDA warnings" OFF)


if(CUDA_FAST_MATH)
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

if(CUDA_ENABLE_WARNINGS)
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
endif()


# Extra NVCC flags (e.g., --expt-relaxed-constexpr, -lineinfo, --expt-extended-lambda)
set(CUDA_EXTRA_NVCC_FLAGS "" CACHE STRING "Extra NVCC flags (semicolon-separated)")
if(CUDA_EXTRA_NVCC_FLAGS)
	separate_arguments(_nvcc_extra NATIVE_COMMAND "${CUDA_EXTRA_NVCC_FLAGS}")
	add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:${_nvcc_extra}>)
endif()