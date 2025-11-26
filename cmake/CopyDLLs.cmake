# Function to copy all required DLLs to a specific directory
# Usage: copy_required_dlls_to_dir(target_name destination_dir)
function(copy_required_dlls_to_dir TARGET_NAME DEST_DIR)
	if(NOT WIN32)
		return()
	endif()

	# Copy octproengine.dll
	add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			$<TARGET_FILE:octproengine>
			${DEST_DIR}
		COMMENT "Copying octproengine.dll"
	)

	# Copy FFTW DLL if building with CPU backend
	if(BUILD_CPU AND FFTW3_DLL)
		add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${FFTW3_DLL}"
				${DEST_DIR}
			COMMENT "Copying FFTW DLL"
		)
	endif()

	# Copy CUDA DLLs if CUDA is enabled
	if(BUILD_CUDA)
		# Find CUDA DLLs (cached, so only searched once)
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
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${CUDA_CUDART_DLL}"
					${DEST_DIR}
				COMMENT "Copying CUDA runtime DLL"
			)
		endif()

		if(CUDA_CUFFT_DLL)
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${CUDA_CUFFT_DLL}"
					${DEST_DIR}
				COMMENT "Copying CUDA FFT DLL"
			)
		endif()
	endif()

	# Copy OpenCL DLL if OpenCL is enabled
	if(BUILD_OPENCL)
		# Find OpenCL DLL (cached, so only searched once)
		find_file(OPENCL_DLL
			NAMES OpenCL.dll
			PATHS
				"${CMAKE_SOURCE_DIR}/thirdparty/opencl/bin"
				"$ENV{CUDA_PATH}/bin"
				"C:/Windows/System32"
		)

		if(OPENCL_DLL)
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${OPENCL_DLL}"
					${DEST_DIR}
				COMMENT "Copying OpenCL DLL"
			)
		endif()
		# VkFFT is header-only, no DLL needed
	endif()
endfunction()

# Function to copy all required DLLs to a target's directory
# Usage: copy_required_dlls(target_name)
function(copy_required_dlls TARGET_NAME)
	if(NOT WIN32)
		return()
	endif()

	# Copy octproengine.dll
	add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			$<TARGET_FILE:octproengine>
			$<TARGET_FILE_DIR:${TARGET_NAME}>
		COMMENT "Copying octproengine.dll to ${TARGET_NAME} directory"
	)

	# Copy FFTW DLL if building with CPU backend
	if(BUILD_CPU AND FFTW3_DLL)
		add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${FFTW3_DLL}"
				$<TARGET_FILE_DIR:${TARGET_NAME}>
			COMMENT "Copying FFTW DLL to ${TARGET_NAME} directory"
		)
	endif()

	# Copy CUDA DLLs if CUDA is enabled
	if(BUILD_CUDA)
		# Find CUDA DLLs (cached, so only searched once)
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
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${CUDA_CUDART_DLL}"
					$<TARGET_FILE_DIR:${TARGET_NAME}>
				COMMENT "Copying CUDA runtime DLL to ${TARGET_NAME} directory"
			)
		endif()

		if(CUDA_CUFFT_DLL)
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${CUDA_CUFFT_DLL}"
					$<TARGET_FILE_DIR:${TARGET_NAME}>
				COMMENT "Copying CUDA FFT DLL to ${TARGET_NAME} directory"
			)
		endif()
	endif()

	# Copy OpenCL DLL if OpenCL is enabled
	if(BUILD_OPENCL)
		# Find OpenCL DLL (cached, so only searched once)
		find_file(OPENCL_DLL
			NAMES OpenCL.dll
			PATHS
				"${CMAKE_SOURCE_DIR}/thirdparty/opencl/bin"
				"$ENV{CUDA_PATH}/bin"
				"C:/Windows/System32"
		)

		if(OPENCL_DLL)
			add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${OPENCL_DLL}"
					$<TARGET_FILE_DIR:${TARGET_NAME}>
				COMMENT "Copying OpenCL DLL to ${TARGET_NAME} directory"
			)
		endif()
	endif()
endfunction()
