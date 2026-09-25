set(OPE_ENABLED_TEST_BACKENDS)
foreach(BACKEND CUDA CPU OPENCL VULKAN)
	if(BUILD_${BACKEND})
		list(APPEND OPE_ENABLED_TEST_BACKENDS ${BACKEND})
	endif()
endforeach()
if(NOT OPE_ENABLED_TEST_BACKENDS)
	message(FATAL_ERROR "Tests require at least one enabled backend")
endif()
list(GET OPE_ENABLED_TEST_BACKENDS 0 OPE_DEFAULT_TEST_BACKEND)
set(OPE_TEST_BACKENDS "${OPE_DEFAULT_TEST_BACKEND}" CACHE STRING "Test backends (semicolon-separated: CUDA;CPU;OPENCL;VULKAN)")
option(OPE_TEST_EXTENDED "Register long stress tests, backend comparisons and benchmarks" OFF)

if(NOT OPE_TEST_BACKENDS)
	message(FATAL_ERROR "OPE_TEST_BACKENDS must contain at least one enabled backend")
endif()
foreach(BACKEND IN LISTS OPE_TEST_BACKENDS)
	if(NOT BACKEND IN_LIST OPE_ENABLED_TEST_BACKENDS)
		message(FATAL_ERROR "OPE_TEST_BACKENDS contains '${BACKEND}'; enabled backends: ${OPE_ENABLED_TEST_BACKENDS}")
	endif()
endforeach()
list(REMOVE_DUPLICATES OPE_TEST_BACKENDS)
message(STATUS "Test backends: ${OPE_TEST_BACKENDS}; extended tests: ${OPE_TEST_EXTENDED}")

function(add_ope_test TEST_NAME SOURCE_FILE)
	cmake_parse_arguments(TEST "BACKENDS;EXTENDED" "TIMEOUT;LABEL" "" ${ARGN})
	# Create executable (extra arguments are compiled into the test, e.g. internal sources)
	add_executable(${TEST_NAME} ${SOURCE_FILE} ${TEST_UNPARSED_ARGUMENTS})
	
	# Link to octproengine library
	target_link_libraries(${TEST_NAME} PRIVATE octproengine)

	# On Windows, make sure this test depends on DLL copying
	if(WIN32)
		add_dependencies(${TEST_NAME} copy_test_dlls)
	endif()

	# Keep executables available for direct runs, even when CTest excludes them.
	if(TEST_EXTENDED AND NOT OPE_TEST_EXTENDED)
		return()
	endif()

	# Register with CTest
	set(TEST_NAMES)
	if(TEST_BACKENDS)
		foreach(BACKEND IN LISTS OPE_TEST_BACKENDS)
			add_test(NAME ${TEST_NAME}_${BACKEND} COMMAND ${TEST_NAME} ${BACKEND})
			list(APPEND TEST_NAMES ${TEST_NAME}_${BACKEND})
		endforeach()
		# Backend variants may write the same test files.
		set_tests_properties(${TEST_NAMES} PROPERTIES RESOURCE_LOCK ${TEST_NAME})
	else()
		add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
		set(TEST_NAMES ${TEST_NAME})
	endif()
	set_tests_properties(${TEST_NAMES} PROPERTIES SKIP_RETURN_CODE 77)
	if(TEST_TIMEOUT)
		set_tests_properties(${TEST_NAMES} PROPERTIES TIMEOUT ${TEST_TIMEOUT})
	endif()
	if(TEST_LABEL)
		set_tests_properties(${TEST_NAMES} PROPERTIES LABELS ${TEST_LABEL})
	endif()

	if(TEST_EXTENDED)
		set_property(TEST ${TEST_NAMES} APPEND PROPERTY LABELS extended)
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

add_ope_test(test_backend_output_comparison test_backend_output_comparison.cpp EXTENDED)
add_ope_test(test_performance_benchmark test_performance_benchmark.cpp BACKENDS EXTENDED LABEL perf)
add_ope_test(test_curve_functionality test_curve_functionality.cpp)
if(BUILD_CPU)
	# Compile the internal backend directly, as with the BufferManager tests below.
	add_ope_test(test_cpu_backend_curves test_cpu_backend_curves.cpp ${CMAKE_SOURCE_DIR}/src/backends/cpu/cpu_backend.cpp)
	target_include_directories(test_cpu_backend_curves PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/thirdparty/pocketfft)
	set_tests_properties(test_cpu_backend_curves PROPERTIES TIMEOUT 60)
endif()
add_ope_test(test_config test_config.cpp)
add_ope_test(test_csv_helper test_csv_helper.cpp)
add_ope_test(test_setconfig_propagation test_setconfig_propagation.cpp)
add_ope_test(test_background_frame test_background_frame.cpp BACKENDS TIMEOUT 300)
if(OPE_TEST_EXTENDED)
	add_test(NAME test_background_frame_cross_backend COMMAND test_background_frame --cross-backend)
	set_tests_properties(test_background_frame_cross_backend PROPERTIES
		SKIP_RETURN_CODE 77 TIMEOUT 300 RESOURCE_LOCK test_background_frame LABELS extended)
endif()
add_ope_test(test_frame_correction test_frame_correction.cpp BACKENDS TIMEOUT 120)
add_ope_test(test_multi_consumer test_multi_consumer.cpp BACKENDS TIMEOUT 300)
add_ope_test(test_output_buffer_ownership test_output_buffer_ownership.cpp BACKENDS TIMEOUT 30)
add_ope_test(test_buffer_id_and_raw_callbacks test_buffer_id_and_raw_callbacks.cpp BACKENDS TIMEOUT 300)
add_ope_test(test_backend_state_transfer test_backend_state_transfer.cpp EXTENDED)
add_ope_test(test_backend_config test_backend_config.cpp)
add_ope_test(test_zerocopy_runtime test_zerocopy_runtime.cpp)
add_ope_test(test_zerocopy_performance test_zerocopy_performance.cpp EXTENDED LABEL perf)
add_ope_test(test_buffer_ordering_content test_buffer_ordering_content.cpp BACKENDS EXTENDED TIMEOUT 1200)
add_ope_test(test_input_consumer test_input_consumer.cpp BACKENDS TIMEOUT 300)
add_ope_test(test_processor_teardown test_processor_teardown.cpp BACKENDS TIMEOUT 300)
# Pure performance benchmarks: run with "ctest -L perf", excluded from the
# functional suite via "ctest -LE perf"
add_ope_test(test_consumer_performance_impact test_consumer_performance_impact.cpp BACKENDS EXTENDED LABEL perf)
add_ope_test(test_dual_processor_performance test_dual_processor_performance.cpp BACKENDS EXTENDED TIMEOUT 600 LABEL perf)

# Standalone BufferManager stress tests: compile buffer_manager.cpp directly
# because BufferManager is internal and not exported from the library
add_ope_test(test_consumer_churn_stress test_consumer_churn_stress.cpp ${CMAKE_SOURCE_DIR}/src/core/buffer_manager.cpp EXTENDED TIMEOUT 120)
target_include_directories(test_consumer_churn_stress PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_ope_test(test_consumer_lifecycle test_consumer_lifecycle.cpp ${CMAKE_SOURCE_DIR}/src/core/buffer_manager.cpp)
target_include_directories(test_consumer_lifecycle PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_consumer_lifecycle PROPERTIES TIMEOUT 120)

add_ope_test(test_consumer_policies test_consumer_policies.cpp ${CMAKE_SOURCE_DIR}/src/core/buffer_manager.cpp)
target_include_directories(test_consumer_policies PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_tests_properties(test_consumer_policies PROPERTIES TIMEOUT 120)

# Recorder test (requires BUILD_TOOLS=ON)
if(BUILD_TOOLS)
	add_ope_test(test_recorder test_recorder.cpp BACKENDS TIMEOUT 300)

	message(STATUS "Recorder test enabled (BUILD_TOOLS=ON)")
endif()

# ========================================
# Copy test data to build directory
# ========================================
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_data)
	file(COPY ${CMAKE_SOURCE_DIR}/tests/test_data
		DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
