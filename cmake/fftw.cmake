find_path(FFTW3_INCLUDES fftw3.h)
mark_as_advanced(FFTW3_INCLUDES)

# Find the FFTW libraries
find_library(FFTW3_LIB fftw3)
mark_as_advanced(FFTW3_LIB)

# If float support is enabled, find the single-precision FFTW libraries.
if(Float)
  find_library(FFTW3F_LIB fftw3f)
  mark_as_advanced(FFTW3F_LIB)
endif()

set(FFTW3_LIBRARIES ${FFTW3_LIB})
if(Float)
  list(APPEND FFTW3_LIBRARIES ${FFTW3F_LIB})
endif()

# Check that we found everything we needed
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 DEFAULT_MSG FFTW3_INCLUDES
                                  FFTW3_LIBRARIES)

if(FFTW3_FOUND)
  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDES})
else()
  message(
    FATAL_ERROR
      "FFTW3 not found. ParaFaFT requires FFTW3 to be installed and accessible."
  )
endif()
