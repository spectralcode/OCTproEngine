# Examples CMakeLists.txt

# Basic console example (no GUI dependencies)
add_executable(example_basic example_basic.cpp)
target_link_libraries(example_basic PRIVATE octproengine)
copy_required_dlls(example_basic)

# OCTproViewer with ImGui (optional)
option(BUILD_OCT_VIEWER "Build interactive OCTproViewer with ImGui" OFF)

if(BUILD_OCT_VIEWER)
	message(STATUS "============================================")
	message(STATUS "Building OCTproViewer with ImGui")
	message(STATUS "============================================")

	# Check if this is first-time setup
	if(NOT EXISTS ${CMAKE_SOURCE_DIR}/thirdparty/glfw/CMakeLists.txt OR
	   NOT EXISTS ${CMAKE_SOURCE_DIR}/thirdparty/imgui/imgui.cpp)
		message(STATUS "First-time setup: Dependencies will be downloaded automatically")
		message(STATUS "This is a one-time download (~3 MB total)")
		message(STATUS "")
	endif()

	# Enable FetchContent for automatic dependency downloads
	include(FetchContent)

	# Set the download directory to thirdparty folder
	set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/thirdparty CACHE PATH "Directory for FetchContent downloads")

	# ============================================================================
	# GLFW - Cross-platform window and input library
	# ============================================================================
	message(STATUS "Checking GLFW dependency...")

	# First, try to find system-installed GLFW
	find_package(glfw3 QUIET)

	if(glfw3_FOUND)
		message(STATUS "  GLFW found via system package")
	elseif(EXISTS ${CMAKE_SOURCE_DIR}/thirdparty/glfw/CMakeLists.txt)
		message(STATUS "  GLFW found in thirdparty folder, using existing installation")
		set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
		set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
		set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
		set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
		add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/glfw ${CMAKE_BINARY_DIR}/glfw)
	else()
		message(STATUS "  GLFW not found, downloading automatically (ZIP archive, ~1 MB)...")
		FetchContent_Declare(
			glfw
			URL https://github.com/glfw/glfw/archive/refs/tags/3.4.zip
			SOURCE_DIR ${CMAKE_SOURCE_DIR}/thirdparty/glfw
			DOWNLOAD_EXTRACT_TIMESTAMP TRUE
		)

		set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
		set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
		set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
		set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

		FetchContent_MakeAvailable(glfw)
		message(STATUS "  GLFW downloaded successfully to thirdparty/glfw")
	endif()

	# ============================================================================
	# ImGui - Dear ImGui graphical user interface library
	# ============================================================================
	message(STATUS "Checking ImGui dependency...")

	# Check if ImGui already exists
	if(EXISTS ${CMAKE_SOURCE_DIR}/thirdparty/imgui/imgui.cpp)
		message(STATUS "  ImGui found in thirdparty folder, using existing installation")
		set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/thirdparty/imgui)
	else()
		message(STATUS "  ImGui not found, downloading automatically (ZIP archive, ~2 MB)...")
		FetchContent_Declare(
			imgui
			URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.5.zip
			SOURCE_DIR ${CMAKE_SOURCE_DIR}/thirdparty/imgui
			DOWNLOAD_EXTRACT_TIMESTAMP TRUE
		)

		FetchContent_MakeAvailable(imgui)

		# Wait for download to complete and verify files exist
		if(NOT EXISTS ${CMAKE_SOURCE_DIR}/thirdparty/imgui/imgui.cpp)
			message(FATAL_ERROR "ImGui download failed - imgui.cpp not found after download")
		endif()

		set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/thirdparty/imgui)
		message(STATUS "  ImGui downloaded successfully to thirdparty/imgui")
	endif()

	# ImGui sources (Dear ImGui doesn't have CMakeLists, so we add files manually)
	set(IMGUI_SOURCES
		${IMGUI_DIR}/imgui.cpp
		${IMGUI_DIR}/imgui_draw.cpp
		${IMGUI_DIR}/imgui_tables.cpp
		${IMGUI_DIR}/imgui_widgets.cpp
		${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
		${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
	)

	# Create octproviewer executable
	add_executable(octproviewer octproviewer.cpp ${IMGUI_SOURCES})

	# Include directories
	target_include_directories(octproviewer PRIVATE
		${IMGUI_DIR}
		${IMGUI_DIR}/backends
		${CMAKE_SOURCE_DIR}/include
	)

	# Link libraries (platform-specific OpenGL)
	target_link_libraries(octproviewer PRIVATE
		octproengine
		glfw
	)

	# Platform-specific OpenGL linking
	if(WIN32)
		target_link_libraries(octproviewer PRIVATE opengl32)
	else()
		# Linux/Unix/MacOS
		find_package(OpenGL REQUIRED)
		target_link_libraries(octproviewer PRIVATE OpenGL::GL)
	endif()

	# Windows: Copy DLLs to executable directory
	copy_required_dlls(octproviewer)

	message(STATUS "")
	message(STATUS "============================================")
	message(STATUS "OCTproViewer Configuration Complete")
	message(STATUS "============================================")
	message(STATUS "Build with: cmake --build . --config Release --target octproviewer")
	if(WIN32)
		message(STATUS "Executable: ${CMAKE_BINARY_DIR}/examples/Release/octproviewer.exe")
	else()
		message(STATUS "Executable: ${CMAKE_BINARY_DIR}/examples/octproviewer")
	endif()
	message(STATUS "============================================")
	message(STATUS "")
else()
	message(STATUS "OCTproViewer disabled (use -DBUILD_OCT_VIEWER=ON to enable)")
endif()

# CUDA device selection example
add_executable(example_cuda_device_selection example_cuda_device_selection.cpp)
target_link_libraries(example_cuda_device_selection PRIVATE octproengine)
copy_required_dlls(example_cuda_device_selection)

# Recorder example (requires BUILD_TOOLS=ON)
if(BUILD_TOOLS)
	add_executable(example_recorder example_recorder.cpp)
	target_link_libraries(example_recorder PRIVATE octproengine)
	copy_required_dlls(example_recorder)

	message(STATUS "Recorder example enabled (BUILD_TOOLS=ON)")
else()
	message(STATUS "Recorder example disabled (use -DBUILD_TOOLS=ON to enable)")
endif()
