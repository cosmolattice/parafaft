find_package(OpenMP REQUIRED)

find_library(FFTW3_MPI_LIB fftw3_mpi HINTS ${PARAFAFT_FFTW_LIB_DIR})
mark_as_advanced(FFTW3_MPI_LIB)

if(NOT FFTW3_MPI_LIB)
  message(
    FATAL_ERROR
      "fftw3_mpi is required for benchmarks but was not found. Please install FFTW with MPI support."
  )
endif()

set(BENCHMARK_LINK_LIBS parafaft::parafaft)

if(FFTW3_MPI_LIB)
  list(APPEND BENCHMARK_LINK_LIBS ${FFTW3_MPI_LIB})
  message(STATUS "FFTW MPI support enabled for benchmarks")
endif()

if(PARAFAFT_FFTW_THREADING STREQUAL "Threads")
  if(FFTW3_THREADS_LIB)
    list(APPEND BENCHMARK_LINK_LIBS ${FFTW3_THREADS_LIB})
  endif()
elseif(PARAFAFT_FFTW_THREADING STREQUAL "OpenMP")
  if(FFTW3_OMP_LIB)
    list(APPEND BENCHMARK_LINK_LIBS ${FFTW3_OMP_LIB} OpenMP::OpenMP_CXX)
  endif()
endif()

add_executable(bench_c2c bench_c2c.cpp)
target_link_libraries(bench_c2c ${BENCHMARK_LINK_LIBS} OpenMP::OpenMP_CXX)
set_target_properties(bench_c2c PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${CMAKE_BINARY_DIR}/bench)

add_executable(bench_r2c bench_r2c.cpp)
target_link_libraries(bench_r2c ${BENCHMARK_LINK_LIBS} OpenMP::OpenMP_CXX)
set_target_properties(bench_r2c PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${CMAKE_BINARY_DIR}/bench)

if(FFTW3_MPI_LIB)
  target_compile_definitions(bench_c2c PRIVATE PARAFAFT_FFTW_MPI)
  target_compile_definitions(bench_r2c PRIVATE PARAFAFT_FFTW_MPI)
endif()

if(PARAFAFT_FFTW_THREADING STREQUAL "Threads")
  if(FFTW3_THREADS_LIB)
    target_compile_definitions(bench_c2c PRIVATE PARAFAFT_FFTW_THREADS)
    target_compile_definitions(bench_r2c PRIVATE PARAFAFT_FFTW_THREADS)
  endif()
elseif(PARAFAFT_FFTW_THREADING STREQUAL "OpenMP")
  if(FFTW3_OMP_LIB)
    target_compile_definitions(bench_c2c PRIVATE PARAFAFT_FFTW_OMP)
    target_compile_definitions(bench_r2c PRIVATE PARAFAFT_FFTW_OMP)
  endif()
endif()

if(PARAFAFT_CUDA)
  add_executable(bench_c2c_cuda bench_c2c.cu)
  target_link_libraries(bench_c2c_cuda PRIVATE parafaft::parafaft MPI::MPI_CXX)
  set_target_properties(bench_c2c_cuda PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
    CUDA_SEPARABLE_COMPILATION ON)
  parafaft_enable_cuda_sources(bench_c2c_cuda)

  add_executable(bench_r2c_cuda bench_r2c.cu)
  target_link_libraries(bench_r2c_cuda PRIVATE parafaft::parafaft MPI::MPI_CXX)
  set_target_properties(bench_r2c_cuda PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
    CUDA_SEPARABLE_COMPILATION ON)
  parafaft_enable_cuda_sources(bench_r2c_cuda)
endif()

if(PARAFAFT_CUDA AND PARAFAFT_CUFFTMP)
  add_executable(bench_c2c_cufftmp bench_c2c_cufftmp.cu)
  target_link_libraries(bench_c2c_cufftmp PRIVATE parafaft::parafaft MPI::MPI_CXX)
  set_target_properties(bench_c2c_cufftmp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
    CUDA_SEPARABLE_COMPILATION ON)
  parafaft_enable_cuda_sources(bench_c2c_cufftmp)

  add_executable(bench_r2c_cufftmp bench_r2c_cufftmp.cu)
  target_link_libraries(bench_r2c_cufftmp PRIVATE parafaft::parafaft MPI::MPI_CXX)
  set_target_properties(bench_r2c_cufftmp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
    CUDA_SEPARABLE_COMPILATION ON)
  parafaft_enable_cuda_sources(bench_r2c_cufftmp)
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bench)
