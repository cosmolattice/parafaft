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
option(PARAFAFT_BENCH "Build benchmarks" OFF)
option(PARAFAFT_GPU_ALLTOALLW
       "Use MPI_Alltoallw with derived types for GPU backends" OFF)
option(PARAFAFT_CUFFTMP
       "Enable cuFFTMp multi-GPU support (requires NVIDIA HPC SDK + NVSHMEM)" OFF)
option(PARAFAFT_FFTW_THREADS
       "Use libfftw3_threads for threaded FFTW (preferred)" ON)
option(PARAFAFT_FFTW_OMP "Use fftw3_omp for OpenMP-based threaded FFTW" ON)
option(
  PARAFAFT_BATCH_PARALLEL
  "Parallelise each stage across its batch of 1D transforms instead of using FFTW's internal threading"
  ON)
set(PARAFAFT_BATCH_PARALLEL_BACKEND
    "Auto"
    CACHE STRING
          "Executor for batch parallelism: Auto, OpenMP, Threads, or Serial")
set_property(CACHE PARAFAFT_BATCH_PARALLEL_BACKEND
             PROPERTY STRINGS Auto OpenMP Threads Serial)
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
  if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
    message(STATUS "CUDA support enabled via NVHPC")
    find_package(CUDAToolkit REQUIRED)
    # nvc++ compiles .cu natively — tell CMake to treat them as CXX
    set(PARAFAFT_NVHPC_CUDA TRUE)
  else()
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

  # Downstream-facing helper (parafaft_enable_cuda_sources) lives in
  # cmake/parafaftHelpers.cmake so it can be installed/exported alongside
  # parafaftConfig.cmake. Include it in-tree so tests and benchmarks can
  # use it too.
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/parafaftHelpers.cmake)
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

# Optional: single-precision FFTW (fftw3f) — enables FFTWBackend<float>.
# Link only when the library was found at configure time; otherwise the
# float trait specialization stays undefined and FFTWBackend<float> is
# unavailable to consumers.
if(PARAFAFT_FFTW3F_AVAILABLE)
  # Match the double-precision threading choice: if we're using fftw3_threads
  # (or fftw3_omp) for double, link the corresponding fftw3f_threads /
  # fftw3f_omp for consistency. Fall through silently if the matching
  # threaded variant isn't installed.
  if(PARAFAFT_FFTW_THREADING STREQUAL "Threads" AND FFTW3F_THREADS_LIB)
    list(APPEND PARAFAFT_LINK_LIBRARIES ${FFTW3F_THREADS_LIB})
    message(STATUS "FFTW3F: Linking with libfftw3f_threads")
  elseif(PARAFAFT_FFTW_THREADING STREQUAL "OpenMP" AND FFTW3F_OMP_LIB)
    list(APPEND PARAFAFT_LINK_LIBRARIES ${FFTW3F_OMP_LIB})
    message(STATUS "FFTW3F: Linking with libfftw3f_omp")
  endif()
  list(APPEND PARAFAFT_LINK_LIBRARIES ${FFTW3F_LIB})
  target_compile_definitions(parafaft INTERFACE PARAFAFT_FFTW3F_AVAILABLE)
  message(STATUS "FFTW3F: single-precision FFTW found — FFTWBackend<float> enabled")
else()
  message(STATUS "FFTW3F: libfftw3f not found — FFTWBackend<float> will be unavailable")
endif()

# Expose threading mode to code via preprocessor
if(PARAFAFT_FFTW_THREADING STREQUAL "Threads")
  target_compile_definitions(parafaft INTERFACE PARAFAFT_FFTW_THREADS)
elseif(PARAFAFT_FFTW_THREADING STREQUAL "OpenMP")
  target_compile_definitions(parafaft INTERFACE PARAFAFT_FFTW_OMP)
endif()

# Batch parallelism: ParaFaFT splits each stage's batch of 1D transforms across
# threads and runs serial FFTW plans per chunk. Pick the executor.
#
# Auto prefers OpenMP (cheaper barriers, and OMP_PROC_BIND/OMP_PLACES then
# actually pin our workers) and falls back to a std::thread pool, which needs no
# dependency beyond libpthread.
if(PARAFAFT_BATCH_PARALLEL)
  set(PARAFAFT_PARALLEL_RESOLVED "${PARAFAFT_BATCH_PARALLEL_BACKEND}")

  if(PARAFAFT_PARALLEL_RESOLVED STREQUAL "Auto")
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
      set(PARAFAFT_PARALLEL_RESOLVED "OpenMP")
    else()
      set(PARAFAFT_PARALLEL_RESOLVED "Threads")
    endif()
  endif()

  if(PARAFAFT_PARALLEL_RESOLVED STREQUAL "OpenMP")
    find_package(OpenMP REQUIRED)
    target_compile_definitions(parafaft INTERFACE PARAFAFT_BATCH_PARALLEL
                                                  PARAFAFT_PARALLEL_OMP)
    target_link_libraries(parafaft INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "Batch parallelism: OpenMP executor")
  elseif(PARAFAFT_PARALLEL_RESOLVED STREQUAL "Threads")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_compile_definitions(parafaft INTERFACE PARAFAFT_BATCH_PARALLEL
                                                  PARAFAFT_PARALLEL_THREADS)
    target_link_libraries(parafaft INTERFACE Threads::Threads)
    message(STATUS "Batch parallelism: std::thread pool executor (pthreads)")
  else()
    target_compile_definitions(parafaft INTERFACE PARAFAFT_BATCH_PARALLEL)
    message(STATUS "Batch parallelism: serial executor (single chunk)")
  endif()
else()
  message(
    STATUS "Batch parallelism: disabled — using FFTW's internal threading")
endif()

target_link_libraries(parafaft INTERFACE ${PARAFAFT_LINK_LIBRARIES})

target_include_directories(parafaft
                           INTERFACE $<BUILD_INTERFACE:${FFTW3_INCLUDE_DIRS}>)

# Add CUDA dependencies if enabled.
#
# Compile/link flags that are NVHPC-specific are propagated via the
# INTERFACE target wrapped in $<$<CXX_COMPILER_ID:NVHPC>:…>. This way a
# consumer using clang or gcc against an installed ParaFaFT sees no
# NVHPC-only flags, while an NVHPC consumer gets -cuda (and
# -cudalib=cufftmp when appropriate) automatically.
if(PARAFAFT_CUDA)
  if(PARAFAFT_CUFFTMP)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cufftmp.cmake)
    if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
      target_compile_definitions(parafaft INTERFACE
        PARAFAFT_CUDA_ENABLED PARAFAFT_CUFFTMP_ENABLED)
      target_compile_options(parafaft INTERFACE
        $<$<CXX_COMPILER_ID:NVHPC>:-cuda -cudalib=cufftmp>)
      target_link_options(parafaft INTERFACE
        $<$<CXX_COMPILER_ID:NVHPC>:-cuda -cudalib=cufftmp>)
      # `-cudalib=cufftmp` is supposed to add the math_libs include path, but
      # that resolution can fail in downstream builds where another CUDA
      # toolkit module or nvcc-style flags are present. Attach the include
      # dir explicitly so consumers see cufftMp.h regardless.
      if(CUFFTMP_INCLUDE_DIR)
        target_include_directories(parafaft INTERFACE
          $<BUILD_INTERFACE:${CUFFTMP_INCLUDE_DIR}>)
      else()
        message(WARNING
          "cuFFTMp: cufftMp.h not located for NVHPC build. Relying on "
          "-cudalib=cufftmp to supply the include path. If downstream "
          "consumers fail with 'cannot open source file \"cufftMp.h\"', set "
          "CUFFTMP_HOME to your HPC SDK's math_libs/<ver> directory.")
      endif()
      if(NVSHMEM_INCLUDE_DIR)
        target_include_directories(parafaft INTERFACE
          $<BUILD_INTERFACE:${NVSHMEM_INCLUDE_DIR}>)
      endif()
      message(STATUS "cuFFTMp support enabled (NVHPC)")
    elseif(CUFFTMP_FOUND)
      # cuFFTMp is a superset of cuFFT — link it INSTEAD of CUDA::cufft
      target_link_libraries(parafaft INTERFACE
        cufftMp::cufftMp nvshmem::host CUDA::cuda_driver CUDA::cudart)
      target_compile_definitions(parafaft INTERFACE
        PARAFAFT_CUDA_ENABLED PARAFAFT_CUFFTMP_ENABLED)
      message(STATUS "cuFFTMp support enabled (replaces cuFFT)")
    else()
      message(FATAL_ERROR "PARAFAFT_CUFFTMP=ON but cuFFTMp/NVSHMEM not found. "
              "Set CUFFTMP_HOME and NVSHMEM_HOME or install NVIDIA HPC SDK.")
    endif()
  else()
    if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
      target_compile_definitions(parafaft INTERFACE PARAFAFT_CUDA_ENABLED)
      target_compile_options(parafaft INTERFACE
        $<$<CXX_COMPILER_ID:NVHPC>:-cuda>)
      target_link_options(parafaft INTERFACE
        $<$<CXX_COMPILER_ID:NVHPC>:-cuda>)
    else()
      target_link_libraries(parafaft INTERFACE CUDA::cufft CUDA::cudart)
      target_compile_definitions(parafaft INTERFACE PARAFAFT_CUDA_ENABLED)
    endif()
  endif()
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
if(PARAFAFT_BENCH)
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
              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/parafaftHelpers.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parafaft)
