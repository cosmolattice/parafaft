# ##############################################################################
# FFTW Backend Tests
# ##############################################################################

# C2C MPI Tests
add_executable(test_fftw_c2c_nd c2c/test_fftw_c2c_nd.cpp)
target_link_libraries(test_fftw_c2c_nd parafaft::parafaft)
set_target_properties(test_fftw_c2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/c2c)

add_executable(test_fftw_c2c_roundtrip_nd c2c/test_fftw_c2c_roundtrip_nd.cpp)
target_link_libraries(test_fftw_c2c_roundtrip_nd parafaft::parafaft)
set_target_properties(
  test_fftw_c2c_roundtrip_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                        ${CMAKE_BINARY_DIR}/c2c)

# R2C MPI Tests
add_executable(test_fftw_r2c_nd r2c/test_fftw_r2c_nd.cpp)
target_link_libraries(test_fftw_r2c_nd parafaft::parafaft)
set_target_properties(test_fftw_r2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/r2c)

add_executable(test_fftw_r2c_roundtrip_nd r2c/test_fftw_r2c_roundtrip_nd.cpp)
target_link_libraries(test_fftw_r2c_roundtrip_nd parafaft::parafaft)
set_target_properties(
  test_fftw_r2c_roundtrip_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                        ${CMAKE_BINARY_DIR}/r2c)

add_executable(test_fftw_r2c_padded_roundtrip_nd
               r2c/test_fftw_r2c_padded_roundtrip_nd.cpp)
target_link_libraries(test_fftw_r2c_padded_roundtrip_nd parafaft::parafaft)
set_target_properties(
  test_fftw_r2c_padded_roundtrip_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               ${CMAKE_BINARY_DIR}/r2c)

add_executable(test_fftw_r2c_padded_nd r2c/test_fftw_r2c_padded_nd.cpp)
target_link_libraries(test_fftw_r2c_padded_nd parafaft::parafaft)
set_target_properties(
  test_fftw_r2c_padded_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                     ${CMAKE_BINARY_DIR}/r2c)

# Unit Tests
add_executable(test_fftw_backend unit/test_fftw_backend.cpp)
target_link_libraries(test_fftw_backend parafaft::parafaft)
set_target_properties(test_fftw_backend PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                   ${CMAKE_BINARY_DIR}/unit)

# Backend-independent exchange geometry arithmetic (64-bit counts and
# displacements, single-partition identity). Host-only, no GPU required.
add_executable(test_exchange_geometry unit/test_exchange_geometry.cpp)
target_link_libraries(test_exchange_geometry parafaft::parafaft)
set_target_properties(test_exchange_geometry PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                        ${CMAKE_BINARY_DIR}/unit)

# Single-precision smoke test (only when libfftw3f was found)
if(PARAFAFT_FFTW3F_AVAILABLE)
  add_executable(test_fftw_c2c_nd_float c2c/test_fftw_c2c_nd_float.cpp)
  target_link_libraries(test_fftw_c2c_nd_float parafaft::parafaft)
  set_target_properties(
    test_fftw_c2c_nd_float PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                      ${CMAKE_BINARY_DIR}/c2c)
endif()

# ##############################################################################
# cuFFT Backend Tests
# ##############################################################################

if(PARAFAFT_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
  endif()

  # C2C MPI Tests
  add_executable(test_cufft_c2c_nd c2c/test_cufft_c2c_nd.cu)
  target_link_libraries(test_cufft_c2c_nd parafaft::parafaft)
  set_target_properties(test_cufft_c2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     ${CMAKE_BINARY_DIR}/c2c)
  parafaft_enable_cuda_sources(test_cufft_c2c_nd)

  # R2C MPI Tests
  add_executable(test_cufft_r2c_nd r2c/test_cufft_r2c_nd.cu)
  target_link_libraries(test_cufft_r2c_nd parafaft::parafaft)
  set_target_properties(test_cufft_r2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     ${CMAKE_BINARY_DIR}/r2c)
  parafaft_enable_cuda_sources(test_cufft_r2c_nd)

  add_executable(test_cufft_r2c_padded_nd r2c/test_cufft_r2c_padded_nd.cu)
  target_link_libraries(test_cufft_r2c_padded_nd parafaft::parafaft)
  set_target_properties(
    test_cufft_r2c_padded_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                        ${CMAKE_BINARY_DIR}/r2c)
  parafaft_enable_cuda_sources(test_cufft_r2c_padded_nd)

  # Unit Tests (cuFFT Backend)
  add_executable(test_cuvector unit/test_cuvector.cu)
  target_link_libraries(test_cuvector CUDA::cudart)
  set_target_properties(test_cuvector PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR}/unit)
  parafaft_enable_cuda_sources(test_cuvector)

  add_executable(test_cufft_backend unit/test_cufft_backend.cu)
  target_link_libraries(test_cufft_backend parafaft::parafaft)
  set_target_properties(test_cufft_backend PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                      ${CMAKE_BINARY_DIR}/unit)
  parafaft_enable_cuda_sources(test_cufft_backend)

  # Single-precision C2C smoke test
  add_executable(test_cufft_c2c_nd_float c2c/test_cufft_c2c_nd_float.cu)
  target_link_libraries(test_cufft_c2c_nd_float parafaft::parafaft)
  set_target_properties(
    test_cufft_c2c_nd_float PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                       ${CMAKE_BINARY_DIR}/c2c)
  parafaft_enable_cuda_sources(test_cufft_c2c_nd_float)
endif()

# ##############################################################################
# hipFFT Backend Tests
# ##############################################################################

if(PARAFAFT_HIP)
  # C2C MPI Tests
  add_executable(test_hipfft_c2c_nd c2c/test_hipfft_c2c_nd.hip)
  target_link_libraries(test_hipfft_c2c_nd parafaft::parafaft)
  set_target_properties(test_hipfft_c2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                      ${CMAKE_BINARY_DIR}/c2c)

  # R2C MPI Tests
  add_executable(test_hipfft_r2c_nd r2c/test_hipfft_r2c_nd.hip)
  target_link_libraries(test_hipfft_r2c_nd parafaft::parafaft)
  set_target_properties(test_hipfft_r2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                      ${CMAKE_BINARY_DIR}/r2c)

  add_executable(test_hipfft_r2c_padded_nd r2c/test_hipfft_r2c_padded_nd.hip)
  target_link_libraries(test_hipfft_r2c_padded_nd parafaft::parafaft)
  set_target_properties(
    test_hipfft_r2c_padded_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                         ${CMAKE_BINARY_DIR}/r2c)

  # Unit Tests (hipFFT Backend)
  add_executable(test_hipvector unit/test_hipvector.hip)
  target_link_libraries(test_hipvector hip::host)
  set_target_properties(test_hipvector PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/unit)

  add_executable(test_hipfft_backend unit/test_hipfft_backend.hip)
  target_link_libraries(test_hipfft_backend parafaft::parafaft)
  set_target_properties(test_hipfft_backend PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                       ${CMAKE_BINARY_DIR}/unit)

  # Single-precision C2C smoke test
  add_executable(test_hipfft_c2c_nd_float c2c/test_hipfft_c2c_nd_float.hip)
  target_link_libraries(test_hipfft_c2c_nd_float parafaft::parafaft)
  set_target_properties(
    test_hipfft_c2c_nd_float PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                        ${CMAKE_BINARY_DIR}/c2c)
endif()

# ##############################################################################

# Create directories for output files
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/c2c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/r2c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/unit)

# Clean target - CMake handles this automatically with 'make clean' or
# equivalent

# Help target (using 'show-help' instead of reserved 'help' name)
add_custom_target(
  show-help
  COMMENT "Displaying help information"
  COMMAND echo "Parallel FFT Test Suite"
  COMMAND echo ""
  COMMAND echo "Build targets:"
  COMMAND echo "  cmake --build .         - Build all tests"
  COMMAND echo "  cmake --build . --target clean - Remove all built files"
  COMMAND echo ""
  COMMAND echo "Run targets:"
  COMMAND echo "  cmake --build . --target test   - Run tests"
  COMMAND echo ""
  COMMAND echo "Other targets:"
  COMMAND echo
          "  cmake --build . --target show-help     - Display this help message"
  COMMAND echo ""
  COMMAND echo "Alternatively, you can use make syntax if using Make generator:"
  COMMAND echo "  make all, make test, make show-help, etc.")

# Enable testing
enable_testing()

function(add_serial_tests test_name test_file test_build)
  add_test(NAME ${test_name}
           COMMAND bash -c
                   "make ${test_build} && ${CMAKE_BINARY_DIR}/${test_file}")
endfunction()

function(add_mpi_tests test_name test_file test_build)
  add_test(
    NAME ${test_name}_np1
    COMMAND
      bash -c
      "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file}")
  add_test(
    NAME ${test_name}_np4
    COMMAND
      bash -c
      "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file}")
  add_test(
    NAME ${test_name}_np8
    COMMAND
      bash -c
      "make ${test_build} && mpirun -np 8 ${CMAKE_BINARY_DIR}/${test_file}")
endfunction()

function(add_mpi_tests_varN_varD test_name test_file test_build)
  # Loop over shapes: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial
  foreach(S 0 1 2)
    # 2D tests
    add_test(
      NAME ${test_name}_np1_N32_D2_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 32 2 ${S}"
    )
    add_test(
      NAME ${test_name}_np4_N32_D2_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 32 2 ${S}"
    )
    # 3D tests
    add_test(
      NAME ${test_name}_np1_N32_D3_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 32 3 ${S}"
    )
    add_test(
      NAME ${test_name}_np4_N32_D3_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 32 3 ${S}"
    )
    # 4D tests (smaller N)
    add_test(
      NAME ${test_name}_np1_N16_D4_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 16 4 ${S}"
    )
    add_test(
      NAME ${test_name}_np4_N16_D4_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 16 4 ${S}"
    )
    # 5D tests (even smaller N)
    add_test(
      NAME ${test_name}_np1_N8_D5_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 8 5 ${S}"
    )
    add_test(
      NAME ${test_name}_np4_N8_D5_S${S}
      COMMAND
        bash -c
        "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 8 5 ${S}"
    )
  endforeach()
endfunction()

# Threaded smoke tests.
#
# The batch-parallel path splits each stage's batch across OMP_NUM_THREADS
# chunks, so the thread count changes which code runs — the default tests
# (1 thread, single chunk) never exercise it. The counts below cover the three
# partition cases: an even split, an uneven split (remainder chunks), and more
# threads than transforms (empty chunks / null plans).
#
# Small grids on purpose: these check correctness of the partitioning, not
# performance.
function(add_threaded_smoke_tests test_name test_file test_build)
  foreach(THREADS 2 3 8)
    add_test(
      NAME ${test_name}_np4_threads${THREADS}
      COMMAND
        bash -c
        "make ${test_build} && OMP_NUM_THREADS=${THREADS} mpirun -np 4 --oversubscribe ${CMAKE_BINARY_DIR}/${test_file} 16 3 0"
    )
  endforeach()
  # Tiny grid where the batch is smaller than the thread count on some stages.
  add_test(
    NAME ${test_name}_np4_threads8_tiny
    COMMAND
      bash -c
      "make ${test_build} && OMP_NUM_THREADS=8 mpirun -np 4 --oversubscribe ${CMAKE_BINARY_DIR}/${test_file} 4 3 0"
  )
endfunction()

# Add tests to CTest
add_serial_tests(fftw_unit_backend unit/test_fftw_backend test_fftw_backend)
add_serial_tests(unit_exchange_geometry unit/test_exchange_geometry
                 test_exchange_geometry)

# MPI tests
add_mpi_tests_varn_vard(fftw_c2c_nd c2c/test_fftw_c2c_nd test_fftw_c2c_nd)
add_mpi_tests_varn_vard(fftw_c2c_roundtrip_nd c2c/test_fftw_c2c_roundtrip_nd
                        test_fftw_c2c_roundtrip_nd)

add_mpi_tests_varn_vard(fftw_r2c_nd r2c/test_fftw_r2c_nd test_fftw_r2c_nd)
add_mpi_tests_varn_vard(fftw_r2c_roundtrip_nd r2c/test_fftw_r2c_roundtrip_nd
                        test_fftw_r2c_roundtrip_nd)
add_mpi_tests_varn_vard(fftw_r2c_padded_nd r2c/test_fftw_r2c_padded_nd
                        test_fftw_r2c_padded_nd)
add_mpi_tests_varn_vard(
  fftw_r2c_padded_roundtrip_nd r2c/test_fftw_r2c_padded_roundtrip_nd
  test_fftw_r2c_padded_roundtrip_nd)

if(PARAFAFT_FFTW3F_AVAILABLE)
  add_mpi_tests_varn_vard(fftw_c2c_nd_float c2c/test_fftw_c2c_nd_float
                          test_fftw_c2c_nd_float)
endif()

# Multi-threaded runs of the reference-comparison tests (see above). Both the
# C2C and R2C stage plans are chunked, and R2C additionally chunks the
# in-place R2C/C2R plans, so cover both.
add_threaded_smoke_tests(fftw_c2c_nd c2c/test_fftw_c2c_nd test_fftw_c2c_nd)
add_threaded_smoke_tests(fftw_r2c_nd r2c/test_fftw_r2c_nd test_fftw_r2c_nd)
add_threaded_smoke_tests(fftw_r2c_padded_roundtrip_nd
                         r2c/test_fftw_r2c_padded_roundtrip_nd
                         test_fftw_r2c_padded_roundtrip_nd)

# cuFFT tests (only added if CUDA is enabled)
if(PARAFAFT_CUDA)
  add_serial_tests(cufft_unit_backend unit/test_cufft_backend
                   test_cufft_backend)
  add_serial_tests(cufft_unit_cuvector unit/test_cuvector test_cuvector)

  add_mpi_tests_varn_vard(cufft_c2c_nd c2c/test_cufft_c2c_nd test_cufft_c2c_nd)

  add_mpi_tests_varn_vard(cufft_r2c_nd r2c/test_cufft_r2c_nd test_cufft_r2c_nd)
  add_mpi_tests_varn_vard(cufft_r2c_padded_nd r2c/test_cufft_r2c_padded_nd
                          test_cufft_r2c_padded_nd)

  add_mpi_tests_varn_vard(cufft_c2c_nd_float c2c/test_cufft_c2c_nd_float
                          test_cufft_c2c_nd_float)
endif()

# cuFFTMp tests (only added if CUDA + cuFFTMp enabled)
if(PARAFAFT_CUDA AND PARAFAFT_CUFFTMP)
  add_executable(test_cufftmp_c2c_nd c2c/test_cufftmp_c2c_nd.cu)
  target_link_libraries(test_cufftmp_c2c_nd parafaft::parafaft)
  set_target_properties(test_cufftmp_c2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                       ${CMAKE_BINARY_DIR}/c2c)
  parafaft_enable_cuda_sources(test_cufftmp_c2c_nd)

  add_executable(test_cufftmp_r2c_nd r2c/test_cufftmp_r2c_nd.cu)
  target_link_libraries(test_cufftmp_r2c_nd parafaft::parafaft)
  set_target_properties(test_cufftmp_r2c_nd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                       ${CMAKE_BINARY_DIR}/r2c)
  parafaft_enable_cuda_sources(test_cufftmp_r2c_nd)

  # cuFFTMp only supports D=2,3. Test with np=1 and np=2.
  foreach(S 0 1 2)
    foreach(D_VAL 2 3)
      add_test(
        NAME cufftmp_c2c_nd_np1_N32_D${D_VAL}_S${S}
        COMMAND bash -c
          "make test_cufftmp_c2c_nd && mpirun -np 1 ${CMAKE_BINARY_DIR}/c2c/test_cufftmp_c2c_nd 32 ${D_VAL} ${S}"
      )
      add_test(
        NAME cufftmp_c2c_nd_np2_N32_D${D_VAL}_S${S}
        COMMAND bash -c
          "make test_cufftmp_c2c_nd && mpirun -np 2 ${CMAKE_BINARY_DIR}/c2c/test_cufftmp_c2c_nd 32 ${D_VAL} ${S}"
      )
      add_test(
        NAME cufftmp_r2c_nd_np1_N32_D${D_VAL}_S${S}
        COMMAND bash -c
          "make test_cufftmp_r2c_nd && mpirun -np 1 ${CMAKE_BINARY_DIR}/r2c/test_cufftmp_r2c_nd 32 ${D_VAL} ${S}"
      )
      add_test(
        NAME cufftmp_r2c_nd_np2_N32_D${D_VAL}_S${S}
        COMMAND bash -c
          "make test_cufftmp_r2c_nd && mpirun -np 2 ${CMAKE_BINARY_DIR}/r2c/test_cufftmp_r2c_nd 32 ${D_VAL} ${S}"
      )
    endforeach()
  endforeach()
endif()

# hipFFT tests (only added if HIP is enabled)
if(PARAFAFT_HIP)
  add_serial_tests(hipfft_unit_backend unit/test_hipfft_backend
                   test_hipfft_backend)
  add_serial_tests(hipfft_unit_hipvector unit/test_hipvector test_hipvector)

  add_mpi_tests_varn_vard(hipfft_c2c_nd c2c/test_hipfft_c2c_nd
                          test_hipfft_c2c_nd)

  add_mpi_tests_varn_vard(hipfft_r2c_nd r2c/test_hipfft_r2c_nd
                          test_hipfft_r2c_nd)
  add_mpi_tests_varn_vard(hipfft_r2c_padded_nd r2c/test_hipfft_r2c_padded_nd
                          test_hipfft_r2c_padded_nd)

  add_mpi_tests_varn_vard(hipfft_c2c_nd_float c2c/test_hipfft_c2c_nd_float
                          test_hipfft_c2c_nd_float)
endif()
