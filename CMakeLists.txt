cmake_minimum_required(VERSION 3.19)
project(
  parafaft
  VERSION 1.0.0
  DESCRIPTION "Parallel FFT library using pencil decomposition"
  LANGUAGES CXX)

# Options
option(PARAFAFT_CUDA "Enable CUDA support" OFF)
option(PARAFAFT_HIP "Enable HIP support" OFF)
option(PARAFAFT_TEST "Build tests" OFF)
option(PARAFAFT_BENCHMARK "Build benchmarks" OFF)
option(PARAFAFT_GPU_ALLTOALLW
       "Use MPI_Alltoallw with derived types for GPU backends" OFF)
option(PARAFAFT_FFTW_THREADS
       "Use libfftw3_threads for threaded FFTW (preferred)" ON)
option(PARAFAFT_FFTW_OMP "Use fftw3_omp for OpenMP-based threaded FFTW" ON)
set(PARAFAFT_FFTW_LIB_DIR
    ""
    CACHE
      PATH
      "Path to FFTW libraries installation (optional, can also be found via CMAKE_PREFIX_PATH)"
)
set(PARAFAFT_FFTW_INCLUDE_DIR
    ""
    CACHE
      PATH
      "Path to FFTW headers installation (optional, can also be found via CMAKE_PREFIX_PATH)"
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA support
if(PARAFAFT_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA support enabled")

    # Set CUDA architectures if not specified
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    message(STATUS "CUDA architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(FATAL_ERROR "CUDA support requested but no CUDA compiler found.")
  endif()
endif()

# HIP support
if(PARAFAFT_HIP)
  include(CheckLanguage)
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    # first, add cmake module path for HIP
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    enable_language(HIP)
    find_package(HIP REQUIRED)
    find_package(hipfft REQUIRED)
    message(STATUS "HIP support enabled")
  else()
    message(FATAL_ERROR "HIP support requested but no HIP compiler found.")
  endif()
endif()

# Find required packages
find_package(MPI REQUIRED)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/fftw.cmake)

# Determine threading strategy for FFTW Priority: libfftw3_threads (POSIX) >
# fftw3_omp (OpenMP) > serial
set(PARAFAFT_FFTW_THREADING
    "SERIAL"
    CACHE STRING "FFTW threading mode: Threads, OpenMP, or Serial")

if(PARAFAFT_FFTW_THREADS AND FFTW3_THREADS_LIB)
  set(PARAFAFT_FFTW_THREADING "Threads")
  message(STATUS "FFTW threading: Using libfftw3_threads")
elseif(PARAFAFT_FFTW_OMP AND FFTW3_OMP_LIB)
  set(PARAFAFT_FFTW_THREADING "OpenMP")
  message(STATUS "FFTW threading: Using fftw3_omp")
  # Need OpenMP for fftw3_omp
  find_package(OpenMP REQUIRED)
else()
  set(PARAFAFT_FFTW_THREADING "Serial")
  message(STATUS "FFTW threading: Serial (no threaded libraries found)")
endif()

# Create header-only library target
add_library(parafaft INTERFACE)
add_library(parafaft::parafaft ALIAS parafaft)

# Specify include directories
target_include_directories(
  parafaft INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                     $<INSTALL_INTERFACE:include>)

# Link dependencies Order matters for static linking: threaded libs must come
# before main fftw3
set(PARAFAFT_LINK_LIBRARIES MPI::MPI_CXX)

# Add threading libraries first (they depend on fftw3)
if(PARAFAFT_FFTW_THREADING STREQUAL "Threads")
  list(APPEND PARAFAFT_LINK_LIBRARIES ${FFTW3_THREADS_LIB})
  message(STATUS "FFTW: Linking with libfftw3_threads")
elseif(PARAFAFT_FFTW_THREADING STREQUAL "OpenMP")
  list(APPEND PARAFAFT_LINK_LIBRARIES ${FFTW3_OMP_LIB} OpenMP::OpenMP_CXX)
  message(STATUS "FFTW: Linking with fftw3_omp and OpenMP")
endif()

# Add main FFTW library last (threaded libs depend on it)
list(APPEND PARAFAFT_LINK_LIBRARIES ${FFTW3_LIBRARIES})

# Expose threading mode to code via preprocessor
if(PARAFAFT_FFTW_THREADING STREQUAL "Threads")
  target_compile_definitions(parafaft INTERFACE PARAFAFT_FFTW_THREADS)
elseif(PARAFAFT_FFTW_THREADING STREQUAL "OpenMP")
  target_compile_definitions(parafaft INTERFACE PARAFAFT_FFTW_OMP)
endif()

target_link_libraries(parafaft INTERFACE ${PARAFAFT_LINK_LIBRARIES})

target_include_directories(parafaft
                           INTERFACE $<BUILD_INTERFACE:${FFTW3_INCLUDE_DIRS}>)

# Add CUDA dependencies if enabled
if(PARAFAFT_CUDA AND CMAKE_CUDA_COMPILER)
  target_link_libraries(parafaft INTERFACE CUDA::cufft CUDA::cudart)
  target_compile_definitions(parafaft INTERFACE PARAFAFT_CUDA_ENABLED)
  set_property(TARGET parafaft PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Add HIP dependencies if enabled
if(PARAFAFT_HIP AND CMAKE_HIP_COMPILER)
  target_link_libraries(parafaft INTERFACE hip::hipfft hip::host)
  target_compile_definitions(parafaft INTERFACE PARAFAFT_HIP_ENABLED)
endif()

# GPU alltoallw override (opt-in: use MPI derived types on GPU instead of manual
# packing)
if(PARAFAFT_GPU_ALLTOALLW)
  target_compile_definitions(parafaft INTERFACE PARAFAFT_GPU_ALLTOALLW=1)
endif()

# Add test subdirectory if enabled
if(PARAFAFT_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

# Add benchmark subdirectory if enabled
if(PARAFAFT_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS parafaft EXPORT parafaftTargets)

install(FILES parafaft_generic.hpp parafaft_r2c.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/)

install(
  DIRECTORY backend
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  FILES_MATCHING
  PATTERN "*.hpp")

install(
  EXPORT parafaftTargets
  FILE parafaftTargets.cmake
  NAMESPACE parafaft::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parafaft)

# Generate config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/parafaftConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parafaft)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parafaft)
