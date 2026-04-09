find_path(FFTW3_INCLUDES fftw3.h HINTS ${PARAFAFT_FFTW_INCLUDE_DIR})
mark_as_advanced(FFTW3_INCLUDES)

# Find the FFTW libraries
find_library(FFTW3_LIB fftw3 HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3_LIB)

# Find optional threaded FFTW libraries Priority: libfftw3_threads (POSIX
# threads) > fftw3_omp (OpenMP) > serial
find_library(FFTW3_THREADS_LIB fftw3_threads HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3_THREADS_LIB)

find_library(FFTW3_OMP_LIB fftw3_omp HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3_OMP_LIB)

# If float support is enabled, find the single-precision FFTW libraries.
if(FLOAT)
  find_library(FFTW3F_LIB fftw3f HINTS ${PARAFAFT_FFTW_LIB_DIR})
  mark_as_advanced(FFTW3F_LIB)
  # Also find threaded versions if available
  find_library(FFTW3F_THREADS_LIB fftw3f_threads HINTS ${PARAFAFT_FFTW_LIB_DIR})
  mark_as_advanced(FFTW3F_THREADS_LIB)
  find_library(FFTW3F_OMP_LIB fftw3f_omp HINTS ${PARAFAFT_FFTW_LIB_DIR})
  mark_as_advanced(FFTW3F_OMP_LIB)
endif()

set(FFTW3_LIBRARIES ${FFTW3_LIB})
if(FLOAT)
  list(APPEND FFTW3_LIBRARIES ${FFTW3F_LIB})
endif()

# And finally, check that we found everything we needed
set(_FFTW_REQUIRED_VARS FFTW3_INCLUDES FFTW3_LIB)
if(FLOAT)
  list(APPEND _FFTW_REQUIRED_VARS FFTW3F_LIB)
endif()

# Check that we found everything we needed
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 DEFAULT_MSG ${_FFTW_REQUIRED_VARS})

if(FFTW3_FOUND)
  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDES})
else()
  message(
    FATAL_ERROR
      "FFTW3 not found. ParaFaFT requires FFTW3 to be installed and accessible."
  )
endif()
