# FindVulkanSDK.cmake
# Automatically locates Vulkan SDK on Windows without requiring CMAKE_PREFIX_PATH
#
# This module helps find the Vulkan SDK by:
# 1. Checking the VULKAN_SDK environment variable
# 2. Searching common Windows installation paths
# 3. Auto-detecting the most recent version if multiple are installed

if(WIN32 AND NOT Vulkan_FOUND)
	# Check environment variable first
	if(DEFINED ENV{VULKAN_SDK})
		list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
		message(STATUS "Using VULKAN_SDK from environment: $ENV{VULKAN_SDK}")
	else()
		# Try common Vulkan SDK installation paths
		file(GLOB VULKAN_SDK_PATHS "C:/VulkanSDK/*")
		if(VULKAN_SDK_PATHS)
			# Get the most recent version (last in sorted list)
			list(SORT VULKAN_SDK_PATHS)
			list(REVERSE VULKAN_SDK_PATHS)
			list(GET VULKAN_SDK_PATHS 0 VULKAN_SDK_PATH)
			list(APPEND CMAKE_PREFIX_PATH "${VULKAN_SDK_PATH}")
			message(STATUS "Auto-detected Vulkan SDK: ${VULKAN_SDK_PATH}")
		endif()
	endif()
endif()
