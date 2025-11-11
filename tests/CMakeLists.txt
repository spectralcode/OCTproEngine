function(add_ope_test TEST_NAME SOURCE_FILE)
	# Create executable
	add_executable(${TEST_NAME} ${SOURCE_FILE})
	
	# Link to octproengine library
	target_link_libraries(${TEST_NAME} PRIVATE octproengine)
	
	# Register with CTest
	add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
	
	# On Windows, make sure this test depends on DLL copying
	if(WIN32)
		add_dependencies(${TEST_NAME} copy_test_dlls)
	endif()
endfunction()

# ========================================
# Windows: Copy DLLs to test directory
# ========================================

if(WIN32)
	# Create a custom target that copies DLLs
	add_custom_target(copy_test_dlls ALL
		COMMENT "Copying DLLs to test directory..."
	)
	
	# Make sure the output directory exists first
	add_custom_command(TARGET copy_test_dlls PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
		COMMENT "Ensuring test output directory exists"
	)

	# Copy octproengine.dll
	add_custom_command(TARGET copy_test_dlls POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			$<TARGET_FILE:octproengine>
			${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
		COMMENT "Copying octproengine.dll"
	)
	
	# Copy FFTW3 DLL from thirdparty folder
	set(FFTW_DLL_PATH "${CMAKE_SOURCE_DIR}/src/thirdparty/fftw/libfftw3f-3.dll")
	
	if(EXISTS "${FFTW_DLL_PATH}")
		add_custom_command(TARGET copy_test_dlls POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${FFTW_DLL_PATH}"
				${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
			COMMENT "Copying libfftw3f-3.dll from thirdparty"
		)
	else()
		message(WARNING "FFTW DLL not found at: ${FFTW_DLL_PATH}")
	endif()
	
	# Make copy_test_dlls depend on octproengine being built
	add_dependencies(copy_test_dlls octproengine)
endif()

# ========================================
# Define Tests
# ========================================

add_ope_test(test_backend_output_comparison test_backend_output_comparison.cpp)
add_ope_test(test_performance_benchmark test_performance_benchmark.cpp)
add_ope_test(test_curve_functionality test_curve_functionality.cpp)
add_ope_test(test_setconfig test_setconfig.cpp)
add_ope_test(test_multi_consumer test_multi_consumer.cpp)

# ========================================
# Copy test data to build directory
# ========================================
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_data)
	file(COPY ${CMAKE_SOURCE_DIR}/tests/test_data
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()