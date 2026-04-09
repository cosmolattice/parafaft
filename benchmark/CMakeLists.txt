find_package(OpenMP)

find_library(FFTW3_MPI_LIB fftw3_mpi HINTS ${FFTW_DIR}/lib)
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
target_link_libraries(bench_c2c ${BENCHMARK_LINK_LIBS})
set_target_properties(bench_c2c PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${CMAKE_BINARY_DIR}/bench)

add_executable(bench_r2c bench_r2c.cpp)
target_link_libraries(bench_r2c ${BENCHMARK_LINK_LIBS})
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

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bench)
