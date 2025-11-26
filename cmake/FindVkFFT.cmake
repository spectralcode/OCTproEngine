# FindVkFFT.cmake
# Find VkFFT library (header-only)
# Creates target VkFFT::VkFFT
#
# Search order:
#	1. ${CMAKE_SOURCE_DIR}/thirdparty/vkfft/VkFFT-1.3.4 (project-local)
#	2. If not found and VKFFT_AUTO_DOWNLOAD=ON -> download from GitHub and extract

set(VKFFT_VERSION "1.3.4")
set(VKFFT_THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty/vkfft")
set(VKFFT_EXPECTED_DIR "${VKFFT_THIRDPARTY_DIR}/VkFFT-${VKFFT_VERSION}")


# Check if VkFFT exists in thirdparty folder
if(NOT EXISTS "${VKFFT_EXPECTED_DIR}/vkFFT/vkFFT.h")
	if(VKFFT_AUTO_DOWNLOAD)
		message(STATUS "")
		message(STATUS "========================================")
		message(STATUS "VkFFT not found in ${VKFFT_EXPECTED_DIR}")
		message(STATUS "Downloading VkFFT ${VKFFT_VERSION} from GitHub...")
		message(STATUS "========================================")
		message(STATUS "")

		# Create thirdparty directory if it does not exist
		file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/thirdparty")
		file(MAKE_DIRECTORY "${VKFFT_THIRDPARTY_DIR}")

		# Download VkFFT source archive
		set(_vkfft_zip "${VKFFT_THIRDPARTY_DIR}/VkFFT-${VKFFT_VERSION}.zip")
		file(DOWNLOAD
			"https://github.com/DTolm/VkFFT/archive/refs/tags/v${VKFFT_VERSION}.zip"
			"${_vkfft_zip}"
			SHOW_PROGRESS
			STATUS DOWNLOAD_STATUS
			TIMEOUT 60
		)

		list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
		if(NOT STATUS_CODE EQUAL 0)
			list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
			message(WARNING
				"Failed to download VkFFT: ${ERROR_MESSAGE}\n"
				"You can download manually from:\n"
				"  https://github.com/DTolm/VkFFT/releases/tag/v${VKFFT_VERSION}\n"
				"and extract to: ${VKFFT_EXPECTED_DIR}"
			)
			set(VKFFT_FOUND FALSE)
			return()
		endif()

		# Extract ZIP
		message(STATUS "Extracting VkFFT...")
		execute_process(
			COMMAND ${CMAKE_COMMAND} -E tar xf "${_vkfft_zip}"
			WORKING_DIRECTORY "${VKFFT_THIRDPARTY_DIR}"
			RESULT_VARIABLE EXTRACT_RESULT
		)

		if(NOT EXTRACT_RESULT EQUAL 0)
			message(WARNING "Failed to extract VkFFT archive: ${_vkfft_zip}")
			set(VKFFT_FOUND FALSE)
			return()
		endif()

		# Clean up zip file
		file(REMOVE "${_vkfft_zip}")

		message(STATUS "")
		message(STATUS "VkFFT ${VKFFT_VERSION} downloaded and extracted to ${VKFFT_EXPECTED_DIR}")
		message(STATUS "")
	else()
		message(STATUS "VkFFT not found and VKFFT_AUTO_DOWNLOAD is OFF.")
	endif()
else()
	message(STATUS "Found VkFFT in thirdparty folder: ${VKFFT_EXPECTED_DIR}")
endif()

# ========================================
# Find VkFFT header
# ========================================

find_path(VKFFT_INCLUDE_DIR
	NAMES vkFFT.h
	PATHS
		"${VKFFT_EXPECTED_DIR}/vkFFT"
		"${VKFFT_THIRDPARTY_DIR}/VkFFT-${VKFFT_VERSION}/vkFFT"
	NO_DEFAULT_PATH
)

# ========================================
# Handle results
# ========================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VkFFT
	REQUIRED_VARS VKFFT_INCLUDE_DIR
	VERSION_VAR VKFFT_VERSION
)

# Create imported target (header-only interface library)
if(VKFFT_FOUND AND NOT TARGET VkFFT::VkFFT)
	add_library(VkFFT::VkFFT INTERFACE IMPORTED)
	set_target_properties(VkFFT::VkFFT PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${VKFFT_INCLUDE_DIR}/.."
	)

	message(STATUS "Found VkFFT ${VKFFT_VERSION} (header-only):")
	message(STATUS "  Include dir: ${VKFFT_INCLUDE_DIR}")
endif()

if(NOT VKFFT_FOUND)
	message(STATUS "")
	message(STATUS "========================================")
	message(STATUS "VkFFT not found!")
	message(STATUS "========================================")
	if(VKFFT_AUTO_DOWNLOAD)
		message(STATUS "Auto-download is enabled but VkFFT still could not be found.")
		message(STATUS "Please download manually from:")
		message(STATUS "  https://github.com/DTolm/VkFFT/releases/tag/v${VKFFT_VERSION}")
		message(STATUS "  Extract to: ${VKFFT_EXPECTED_DIR}")
		message(STATUS "  Then re-run CMake.")
	else()
		message(STATUS "")
		message(STATUS "Option 1 - Auto-download (default):")
		message(STATUS "  cmake -S . -B build -DVKFFT_AUTO_DOWNLOAD=ON")
		message(STATUS "")
		message(STATUS "Option 2 - Manual download:")
		message(STATUS "  Download: https://github.com/DTolm/VkFFT/releases/tag/v${VKFFT_VERSION}")
		message(STATUS "  Extract to: ${VKFFT_EXPECTED_DIR}")
		message(STATUS "  Then re-run CMake.")
		message(STATUS "")
	endif()
	message(STATUS "========================================")
	message(STATUS "")
endif()

mark_as_advanced(VKFFT_INCLUDE_DIR)
