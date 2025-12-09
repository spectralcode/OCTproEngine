# FindReaderWriterQueue.cmake
# Find moodycamel::readerwriterqueue library (header-only)
# Creates target ReaderWriterQueue::ReaderWriterQueue
#
# Search order:
#	1. ${CMAKE_SOURCE_DIR}/thirdparty/readerwriterqueue (project-local)
#	2. If not found and READERWRITERQUEUE_AUTO_DOWNLOAD=ON -> download from GitHub

set(RWQ_VERSION "1.0.6")
set(RWQ_THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty/readerwriterqueue")

# Check if readerwriterqueue exists in thirdparty folder
if(NOT EXISTS "${RWQ_THIRDPARTY_DIR}/readerwriterqueue.h")
	if(READERWRITERQUEUE_AUTO_DOWNLOAD)
		message(STATUS "")
		message(STATUS "========================================")
		message(STATUS "readerwriterqueue not found in ${RWQ_THIRDPARTY_DIR}")
		message(STATUS "Downloading readerwriterqueue ${RWQ_VERSION} from GitHub...")
		message(STATUS "========================================")
		message(STATUS "")

		# Create thirdparty directory if it does not exist
		file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/thirdparty")
		file(MAKE_DIRECTORY "${RWQ_THIRDPARTY_DIR}")

		# Download header files directly (it's just 2 files)
		file(DOWNLOAD
			"https://raw.githubusercontent.com/cameron314/readerwriterqueue/v${RWQ_VERSION}/readerwriterqueue.h"
			"${RWQ_THIRDPARTY_DIR}/readerwriterqueue.h"
			SHOW_PROGRESS
			STATUS DOWNLOAD_STATUS
			TIMEOUT 30
		)

		list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
		if(NOT STATUS_CODE EQUAL 0)
			list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
			message(WARNING "Failed to download readerwriterqueue.h: ${ERROR_MESSAGE}")
			set(ReaderWriterQueue_FOUND FALSE)
			return()
		endif()

		file(DOWNLOAD
			"https://raw.githubusercontent.com/cameron314/readerwriterqueue/v${RWQ_VERSION}/atomicops.h"
			"${RWQ_THIRDPARTY_DIR}/atomicops.h"
			SHOW_PROGRESS
			STATUS DOWNLOAD_STATUS
			TIMEOUT 30
		)

		list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
		if(NOT STATUS_CODE EQUAL 0)
			list(GET DOWNLOAD_STATUS 1 ERROR_MESSAGE)
			message(WARNING "Failed to download atomicops.h: ${ERROR_MESSAGE}")
			set(ReaderWriterQueue_FOUND FALSE)
			return()
		endif()

		message(STATUS "")
		message(STATUS "readerwriterqueue ${RWQ_VERSION} downloaded to ${RWQ_THIRDPARTY_DIR}")
		message(STATUS "")
	else()
		message(STATUS "readerwriterqueue not found and READERWRITERQUEUE_AUTO_DOWNLOAD is OFF.")
	endif()
else()
	message(STATUS "Found readerwriterqueue in thirdparty folder: ${RWQ_THIRDPARTY_DIR}")
endif()

# ========================================
# Find readerwriterqueue header
# ========================================

find_path(READERWRITERQUEUE_INCLUDE_DIR
	NAMES readerwriterqueue.h
	PATHS "${RWQ_THIRDPARTY_DIR}"
	NO_DEFAULT_PATH
)

# ========================================
# Handle results
# ========================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ReaderWriterQueue
	REQUIRED_VARS READERWRITERQUEUE_INCLUDE_DIR
	VERSION_VAR RWQ_VERSION
)

# Create imported target (header-only interface library)
if(ReaderWriterQueue_FOUND AND NOT TARGET ReaderWriterQueue::ReaderWriterQueue)
	add_library(ReaderWriterQueue::ReaderWriterQueue INTERFACE IMPORTED)
	set_target_properties(ReaderWriterQueue::ReaderWriterQueue PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${READERWRITERQUEUE_INCLUDE_DIR}"
	)

	message(STATUS "Found readerwriterqueue ${RWQ_VERSION} (header-only):")
	message(STATUS "  Include dir: ${READERWRITERQUEUE_INCLUDE_DIR}")
endif()

mark_as_advanced(READERWRITERQUEUE_INCLUDE_DIR)
