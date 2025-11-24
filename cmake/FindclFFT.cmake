# FindclFFT.cmake
# Find clFFT library (OpenCL FFT library)
# Creates target clFFT::clFFT
#
# Search order:
#	1. ${CMAKE_SOURCE_DIR}/thirdparty/clfft (project-local)
#	2. User-specified paths (clFFT_ROOT, clFFT_DIR, env vars)
#	3. System paths

set(clFFT_THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty/clfft")

# ========================================
# Search for clFFT
# ========================================

# Search paths - prioritize thirdparty, then user-specified, then system
set(clFFT_SEARCH_PATHS
	${clFFT_THIRDPARTY_DIR}
	${clFFT_ROOT}
	$ENV{clFFT_DIR}
	$ENV{clFFT_ROOT}
	"C:/Program Files/clFFT"
	"C:/clFFT"
	/usr
	/usr/local
	/opt/local
)

# Find include directory
find_path(clFFT_INCLUDE_DIR
	NAMES clFFT.h
	PATHS ${clFFT_SEARCH_PATHS}
	PATH_SUFFIXES include include/clFFT
	NO_DEFAULT_PATH
)

# If not found in specified paths, try system-wide search
if(NOT clFFT_INCLUDE_DIR)
	find_path(clFFT_INCLUDE_DIR NAMES clFFT.h PATH_SUFFIXES include/clFFT include)
endif()

# Find library
find_library(clFFT_LIBRARY
	NAMES clFFT clfft
	PATHS ${clFFT_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64 bin lib/import lib64/import
	NO_DEFAULT_PATH
)

# If not found in specified paths, try system-wide search
if(NOT clFFT_LIBRARY)
	find_library(clFFT_LIBRARY NAMES clFFT clfft)
endif()

# On Windows, also locate the DLL
if(WIN32)
	# Find the DLL
	find_file(clFFT_DLL
		NAMES clFFT.dll clfft.dll
		PATHS ${clFFT_SEARCH_PATHS}
		PATH_SUFFIXES lib lib64 bin bin/Release bin/Debug
		NO_DEFAULT_PATH
	)

	if(NOT clFFT_DLL)
		find_file(clFFT_DLL NAMES clFFT.dll clfft.dll)
	endif()
endif()

# ========================================
# Handle results
# ========================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(clFFT
	REQUIRED_VARS clFFT_LIBRARY clFFT_INCLUDE_DIR
)

# Create imported target
if(clFFT_FOUND AND NOT TARGET clFFT::clFFT)
	add_library(clFFT::clFFT UNKNOWN IMPORTED)
	set_target_properties(clFFT::clFFT PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${clFFT_INCLUDE_DIR}"
		IMPORTED_LOCATION "${clFFT_LIBRARY}"
	)
	# On Windows, store the DLL path as a property to be used for copying to output dir
	if(WIN32 AND clFFT_DLL)
		set_property(TARGET clFFT::clFFT PROPERTY clFFT_DLL "${clFFT_DLL}")
	endif()

	message(STATUS "Found clFFT:")
	message(STATUS "  Include dir: ${clFFT_INCLUDE_DIR}")
	message(STATUS "  Library:     ${clFFT_LIBRARY}")
	if(WIN32 AND clFFT_DLL)
		message(STATUS "  DLL:         ${clFFT_DLL}")
	endif()
endif()

if(NOT clFFT_FOUND)
	message(STATUS "")
	message(STATUS "========================================")
	message(STATUS "clFFT not found!")
	message(STATUS "========================================")
	if(WIN32)
		message(STATUS "")
		message(STATUS "To install clFFT on Windows:")
		message(STATUS "  1. Download from: https://github.com/clMathLibraries/clFFT/releases")
		message(STATUS "  2. Extract to: ${clFFT_THIRDPARTY_DIR}")
		message(STATUS "     or to: C:/Program Files/clFFT")
		message(STATUS "  3. Re-run CMake")
		message(STATUS "")
		message(STATUS "Or set clFFT_ROOT to point to your clFFT installation:")
		message(STATUS "  cmake -S . -B build -DclFFT_ROOT=<path-to-clfft>")
		message(STATUS "")
	elseif(UNIX)
		message(STATUS "")
		message(STATUS "To install clFFT:")
		message(STATUS "  Ubuntu / Debian:   sudo apt-get install libclfft-dev")
		message(STATUS "  Fedora / RHEL:     sudo dnf install clFFT-devel")
		message(STATUS "  Build from source: https://github.com/clMathLibraries/clFFT")
		message(STATUS "")
	endif()
	message(STATUS "========================================")
	message(STATUS "")
endif()

mark_as_advanced(clFFT_INCLUDE_DIR clFFT_LIBRARY clFFT_DLL)
