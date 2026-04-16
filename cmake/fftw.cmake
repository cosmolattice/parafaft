find_path(FFTW3_INCLUDES fftw3.h HINTS ${PARAFAFT_FFTW_INCLUDE_DIR})
mark_as_advanced(FFTW3_INCLUDES)

# Find the FFTW libraries (double precision — required)
find_library(FFTW3_LIB fftw3 HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3_LIB)

# Find optional threaded FFTW libraries. Priority: libfftw3_threads (POSIX
# threads) > fftw3_omp (OpenMP) > serial.
find_library(FFTW3_THREADS_LIB fftw3_threads HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3_THREADS_LIB)

find_library(FFTW3_OMP_LIB fftw3_omp HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3_OMP_LIB)

# Find single-precision FFTW libraries (optional: enables FFTWBackend<float>).
find_library(FFTW3F_LIB fftw3f HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3F_LIB)
find_library(FFTW3F_THREADS_LIB fftw3f_threads HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3F_THREADS_LIB)
find_library(FFTW3F_OMP_LIB fftw3f_omp HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3F_OMP_LIB)

set(FFTW3_LIBRARIES ${FFTW3_LIB})

# Verify that the required double-precision bits were found. FFTW3F is
# optional — the top-level CMakeLists consults PARAFAFT_FFTW3F_AVAILABLE
# to decide whether to link it and enable FFTWBackend<float>.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 DEFAULT_MSG FFTW3_INCLUDES FFTW3_LIB)

if(FFTW3_FOUND)
  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDES})
else()
  message(
    FATAL_ERROR
      "FFTW3 not found. ParaFaFT requires FFTW3 to be installed and accessible."
  )
endif()

if(FFTW3F_LIB)
  set(PARAFAFT_FFTW3F_AVAILABLE TRUE)
else()
  set(PARAFAFT_FFTW3F_AVAILABLE FALSE)
endif()
