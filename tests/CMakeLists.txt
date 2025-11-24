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

	# Copy all required DLLs to test directory
	copy_required_dlls_to_dir(copy_test_dlls ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)

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
add_ope_test(test_cuda_api test_cuda_api.cpp)
add_ope_test(test_buffer_id_and_raw_callbacks test_buffer_id_and_raw_callbacks.cpp)

# Recorder test (requires BUILD_TOOLS=ON)
if(BUILD_TOOLS)
	add_executable(test_recorder test_recorder.cpp)
	target_link_libraries(test_recorder PRIVATE octproengine)
	add_test(NAME test_recorder COMMAND test_recorder)

	if(WIN32)
		add_dependencies(test_recorder copy_test_dlls)
	endif()

	message(STATUS "Recorder test enabled (BUILD_TOOLS=ON)")
endif()

# ========================================
# Copy test data to build directory
# ========================================
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_data)
	file(COPY ${CMAKE_SOURCE_DIR}/tests/test_data
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()