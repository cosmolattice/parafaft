# cmake/cufftmp.cmake
# Find cuFFTMp and NVSHMEM for multi-GPU distributed FFT support.
#
# cuFFTMp ships with the NVIDIA HPC SDK (not the standard CUDA Toolkit).
# It requires NVSHMEM for GPU-to-GPU communication.
#
# Search order:
#   1. CUFFTMP_HOME / NVSHMEM_HOME environment or CMake variables
#   2. HPC SDK math_libs path (auto-detected from NVHPC_ROOT or standard paths)
#
# Outputs:
#   CUFFTMP_FOUND        - TRUE if cuFFTMp and NVSHMEM were found
#   CUFFTMP_INCLUDE_DIR  - cuFFTMp header directory
#   CUFFTMP_LIBRARY      - path to libcufftMp.so
#   NVSHMEM_INCLUDE_DIR  - NVSHMEM header directory
#   NVSHMEM_LIBRARY      - path to libnvshmem_host.so
#
# Creates imported targets:
#   cufftMp::cufftMp     - cuFFTMp library
#   nvshmem::host        - NVSHMEM host library

if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
  add_compile_options(-cudalib=cufftmp)
endif()

set(CUFFTMP_HOME "$ENV{CUFFTMP_HOME}" CACHE PATH "Path to cuFFTMp installation")
set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" CACHE PATH "Path to NVSHMEM installation")

# Build search paths from HPC SDK if not explicitly set
set(_cufftmp_search_paths)
set(_nvshmem_search_paths)

if(CUFFTMP_HOME)
  list(APPEND _cufftmp_search_paths "${CUFFTMP_HOME}")
endif()
if(NVSHMEM_HOME)
  list(APPEND _nvshmem_search_paths "${NVSHMEM_HOME}")
endif()

# Auto-detect HPC SDK paths
foreach(_base /opt/nvidia/hpc_sdk/Linux_x86_64 /opt/nvidia/hpc_sdk/Linux_aarch64)
  if(IS_DIRECTORY "${_base}")
    file(GLOB _versions "${_base}/*")
    foreach(_ver ${_versions})
      if(IS_DIRECTORY "${_ver}/math_libs")
        list(APPEND _cufftmp_search_paths "${_ver}/math_libs")
      endif()
      if(IS_DIRECTORY "${_ver}/comm_libs/nvshmem")
        list(APPEND _nvshmem_search_paths "${_ver}/comm_libs/nvshmem")
      endif()
    endforeach()
  endif()
endforeach()

# Find cuFFTMp header
find_path(CUFFTMP_INCLUDE_DIR
  NAMES cufftMp.h
  HINTS ${_cufftmp_search_paths}
  PATH_SUFFIXES include include/cufftmp
)

# Find cuFFTMp library
find_library(CUFFTMP_LIBRARY
  NAMES cufftMp
  HINTS ${_cufftmp_search_paths}
  PATH_SUFFIXES lib64 lib
)

# Find NVSHMEM header
find_path(NVSHMEM_INCLUDE_DIR
  NAMES nvshmem.h
  HINTS ${_nvshmem_search_paths}
  PATH_SUFFIXES include
)

# Find NVSHMEM host library
find_library(NVSHMEM_LIBRARY
  NAMES nvshmem_host
  HINTS ${_nvshmem_search_paths}
  PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUFFTMP
  REQUIRED_VARS CUFFTMP_LIBRARY CUFFTMP_INCLUDE_DIR NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIR
  FAIL_MESSAGE "cuFFTMp not found. Set CUFFTMP_HOME and NVSHMEM_HOME, or install via NVIDIA HPC SDK."
)

if(CUFFTMP_FOUND)
  message(STATUS "Found cuFFTMp: ${CUFFTMP_LIBRARY}")

  if(NOT TARGET cufftMp::cufftMp)
    add_library(cufftMp::cufftMp SHARED IMPORTED)
    set_target_properties(cufftMp::cufftMp PROPERTIES
      IMPORTED_LOCATION "${CUFFTMP_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUFFTMP_INCLUDE_DIR}"
    )
  endif()

  if(NOT TARGET nvshmem::host)
    add_library(nvshmem::host SHARED IMPORTED)
    set_target_properties(nvshmem::host PROPERTIES
      IMPORTED_LOCATION "${NVSHMEM_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(CUFFTMP_INCLUDE_DIR CUFFTMP_LIBRARY NVSHMEM_INCLUDE_DIR NVSHMEM_LIBRARY)
endif()
