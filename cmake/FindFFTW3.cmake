# FindFFTW3.cmake
# Find FFTW3 library (single precision version)
# Creates target FFTW3::fftw3f
#
# Search order:
#	1. ${CMAKE_SOURCE_DIR}/thirdparty/fftw (project-local)
#	2. User-specified paths (FFTW3_ROOT, FFTW3_DIR, env vars)
#	3. System paths
#	4. If not found and FFTW3_AUTO_DOWNLOAD=ON -> download to thirdparty/fftw and generate .lib (Windows only)

set(FFTW3_THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty/fftw")

# ========================================
# Auto-download FFTW3 if user explicitly allows it (Windows only)
# ========================================
if(WIN32)
	# Check if FFTW exists in thirdparty folder 
	if(NOT EXISTS "${FFTW3_THIRDPARTY_DIR}/libfftw3f-3.dll")
		if(FFTW3_AUTO_DOWNLOAD)
			message(STATUS "")
			message(STATUS "========================================")
			message(STATUS "FFTW3 not found in ${FFTW3_THIRDPARTY_DIR}")
			message(STATUS "Downloading FFTW3 3.3.5 (64-bit DLLs) from https://fftw.org...")
			message(STATUS "========================================")
			message(STATUS "")
			
			# Create thirdparty directory if it does not exist
			file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/thirdparty")
			file(MAKE_DIRECTORY "${FFTW3_THIRDPARTY_DIR}")
			
			# Download FFTW
			set(_fftw_zip "${FFTW3_THIRDPARTY_DIR}/fftw3.zip")
			file(DOWNLOAD 
				"https://fftw.org/pub/fftw/fftw-3.3.5-dll64.zip"
				"${_fftw_zip}"
				SHOW_PROGRESS
				STATUS DOWNLOAD_STATUS
			)
			
			list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
			if(NOT STATUS_CODE EQUAL 0)
				message(FATAL_ERROR
					"Failed to download FFTW3. Please download manually from\n"
					"  https://www.fftw.org/download.html\n"
					"and extract to: ${FFTW3_THIRDPARTY_DIR}\n"
					"Then generate .lib files as described in the FFTW documentation."
				)
			endif()
			
			# Extract ZIP (flat layout: fftw3.h, libfftw3f-3.dll, ...)
			message(STATUS "Extracting FFTW3...")
			execute_process(
				COMMAND ${CMAKE_COMMAND} -E tar xf "${_fftw_zip}"
				WORKING_DIRECTORY "${FFTW3_THIRDPARTY_DIR}"
				RESULT_VARIABLE EXTRACT_RESULT
			)
			
			if(NOT EXTRACT_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to extract FFTW3 archive: ${_fftw_zip}")
			endif()
			
			# Clean up zip file
			file(REMOVE "${_fftw_zip}")
			
			message(STATUS "")
			message(STATUS "FFTW3 downloaded and extracted to ${FFTW3_THIRDPARTY_DIR}")
			message(STATUS "")
		else()
			message(STATUS "FFTW3 not found and FFTW3_AUTO_DOWNLOAD is OFF.")
			message(STATUS "The CPU backend requires FFTW3. Please use one of the options below:")
		endif()
	else()
		message(STATUS "Found FFTW3 in thirdparty folder: ${FFTW3_THIRDPARTY_DIR}")
	endif()
endif()

# ========================================
# Search for FFTW3
# ========================================

# Search paths - prioritize thirdparty, then user-specified, then system
set(FFTW3_SEARCH_PATHS
	${FFTW3_THIRDPARTY_DIR}
	${FFTW3_ROOT}
	$ENV{FFTW3_DIR}
	$ENV{FFTW3_ROOT}
	"C:/Program Files/fftw"
	"C:/fftw"
	/usr
	/usr/local
	/opt/local
)

# Find include directory
# Note: Windows FFTW zip extracts files directly (no include/ subdirectory)
find_path(FFTW3_INCLUDE_DIR
	NAMES fftw3.h
	PATHS ${FFTW3_SEARCH_PATHS}
	PATH_SUFFIXES include ""
	NO_DEFAULT_PATH
)

# If not found in specified paths, try system-wide search
if(NOT FFTW3_INCLUDE_DIR)
	find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
endif()

# Find library (single precision version: fftw3f)
find_library(FFTW3_LIBRARY
	NAMES fftw3f libfftw3f-3 fftw3f-3
	PATHS ${FFTW3_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64 bin ""
	NO_DEFAULT_PATH
)

# If not found in specified paths, try system-wide search
if(NOT FFTW3_LIBRARY)
	find_library(FFTW3_LIBRARY NAMES fftw3f libfftw3f-3 fftw3f-3)
endif()

# On Windows, also locate the DLL and (MSVC only) optionally generate .lib from .def
if(WIN32)
	# Find the DLL
	find_file(FFTW3_DLL
		NAMES libfftw3f-3.dll fftw3f.dll
		PATHS ${FFTW3_SEARCH_PATHS}
		PATH_SUFFIXES lib lib64 bin ""
		NO_DEFAULT_PATH
	)
	
	if(NOT FFTW3_DLL)
		find_file(FFTW3_DLL NAMES libfftw3f-3.dll fftw3f.dll)
	endif()
	
	# If we have a DLL but no library yet, try to generate import lib from .def (MSVC only)
	if(FFTW3_DLL AND NOT FFTW3_LIBRARY)
		get_filename_component(FFTW3_DIR "${FFTW3_DLL}" DIRECTORY)
		set(FFTW3_DEF "${FFTW3_DIR}/libfftw3f-3.def")
		set(FFTW3_LIB "${FFTW3_DIR}/libfftw3f-3.lib")
		
		if(EXISTS "${FFTW3_DEF}" AND NOT EXISTS "${FFTW3_LIB}")
			if(MSVC)
				find_program(LIB_EXECUTABLE lib)
				if(LIB_EXECUTABLE)
					message(STATUS "Generating import library from .def using ${LIB_EXECUTABLE}...")
					execute_process(
						COMMAND ${LIB_EXECUTABLE} "/def:${FFTW3_DEF}" "/out:${FFTW3_LIB}" "/machine:x64"
						WORKING_DIRECTORY "${FFTW3_DIR}"
						RESULT_VARIABLE LIB_RESULT
					)
					
					if(LIB_RESULT EQUAL 0)
						message(STATUS "Generated ${FFTW3_LIB}")
					else()
						message(WARNING
							"Failed to generate import library from .def file.\n"
							"Run CMake from a Visual Studio Developer Command Prompt\n"
							"or create the import library manually with:\n"
							"  lib /def:libfftw3f-3.def /machine:x64 /out:libfftw3f-3.lib"
						)
					endif()
				else()
					message(WARNING
						"lib.exe not found. Cannot generate import library from .def file.\n"
						"Run CMake from a Visual Studio Developer Command Prompt\n"
						"or create the import library manually as described on the FFTW website."
					)
				endif()
			else()
				message(STATUS
					"Not using MSVC. Import library generation from .def is skipped.\n"
					"You can usually link directly against the DLL with MinGW."
				)
			endif()
		endif()
		
		# If the .lib now exists, use it as FFTW3_LIBRARY
		if(EXISTS "${FFTW3_LIB}")
			set(FFTW3_LIBRARY "${FFTW3_LIB}")
		endif()
	endif()
endif()

# ========================================
# Handle results
# ========================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
	REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR
)

# Create imported target
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3f)
	add_library(FFTW3::fftw3f UNKNOWN IMPORTED)
	set_target_properties(FFTW3::fftw3f PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
		IMPORTED_LOCATION "${FFTW3_LIBRARY}"
	)
	# On Windows, store the DLL path as a property to be used for copying to output dir
	if(WIN32 AND FFTW3_DLL)
		set_property(TARGET FFTW3::fftw3f PROPERTY FFTW3_DLL "${FFTW3_DLL}")
	endif()

	message(STATUS "Found FFTW3 (single precision):")
	message(STATUS "  Include dir: ${FFTW3_INCLUDE_DIR}")
	message(STATUS "  Library:     ${FFTW3_LIBRARY}")
	if(WIN32 AND FFTW3_DLL)
		message(STATUS "  DLL:         ${FFTW3_DLL}")
	endif()
endif()

if(NOT FFTW3_FOUND)
	message(STATUS "")
	message(STATUS "========================================")
	message(STATUS "FFTW3 not found!")
	message(STATUS "========================================")
	if(WIN32)
		if(FFTW3_AUTO_DOWNLOAD)
			message(STATUS "Auto-download is enabled but FFTW3 still could not be found.")
			message(STATUS "Please download manually from:")
			message(STATUS "  https://www.fftw.org/download.html")
			message(STATUS "  Extract to: ${FFTW3_THIRDPARTY_DIR}")
			message(STATUS "  Generate .lib files as described in the FFTW documentation.")
			message(STATUS "  Then re-run CMake.")
		else()
			message(STATUS "")
			message(STATUS "Option 1 - Auto-download (Windows only):")
			message(STATUS "  cmake -S . -B build -DFFTW3_AUTO_DOWNLOAD=ON")
			message(STATUS "")
			message(STATUS "Option 2 - Manual download and install:")
			message(STATUS "  Download: https://www.fftw.org/download.htmlp")
			message(STATUS "  Extract to: ${FFTW3_THIRDPARTY_DIR}")
			message(STATUS "  Generate .lib files as described in the FFTW documentation.")
			message(STATUS "  Then re-run CMake.")
			message(STATUS "")
		endif()
	elseif(UNIX)
		message(STATUS "")
		message(STATUS "To install FFTW3:")
		message(STATUS "  Ubuntu / Debian:   sudo apt-get install libfftw3-dev")
		message(STATUS "  Fedora / RHEL:     sudo dnf install fftw-devel")
		message(STATUS "")
	endif()
	message(STATUS "========================================")
	message(STATUS "")
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY FFTW3_DLL)
