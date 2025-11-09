# FindFFTW3.cmake
# Find FFTW3 library (single precision version)
# Creates target FFTW3::fftw3f
#
# Variables set by this module:
#   FFTW3_FOUND        - True if FFTW3 was found
#   FFTW3_INCLUDE_DIR  - Include directories for FFTW3
#   FFTW3_LIBRARY      - Library to link

# Search paths
set(FFTW3_SEARCH_PATHS
	# Windows common locations
	"C:/qt_projects/octproengine/src/thirdparty/fftw"
	"C:/Program Files/fftw"
	"C:/fftw"
	# Linux common locations
	/usr
	/usr/local
	# Environment variable
	$ENV{FFTW3_DIR}
)

# Find include directory
find_path(FFTW3_INCLUDE_DIR
	NAMES fftw3.h
	PATHS ${FFTW3_SEARCH_PATHS}
	PATH_SUFFIXES include
)

# Find library (single precision version: fftw3f)
find_library(FFTW3_LIBRARY
	NAMES fftw3f libfftw3f-3 fftw3f-3
	PATHS ${FFTW3_SEARCH_PATHS}
	PATH_SUFFIXES lib lib64
)

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
	REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR
	VERSION_VAR FFTW3_VERSION
)

# Create imported target
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3f)
	add_library(FFTW3::fftw3f UNKNOWN IMPORTED)
	set_target_properties(FFTW3::fftw3f PROPERTIES
		IMPORTED_LOCATION "${FFTW3_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
	)
	message(STATUS "Found FFTW3: ${FFTW3_LIBRARY}")
endif()

# Mark variables as advanced
mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY)
