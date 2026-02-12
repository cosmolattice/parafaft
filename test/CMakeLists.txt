# ##############################################################################
# FFTW Backend Tests
# ##############################################################################

# C2C MPI Tests
add_executable(test_fftw_c2c_nd c2c/test_fftw_c2c_nd.cpp)
target_link_libraries(test_fftw_c2c_nd parafaft::parafaft)
set_target_properties(test_fftw_c2c_nd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_fftw_c2c_roundtrip_nd c2c/test_fftw_c2c_roundtrip_nd.cpp)
target_link_libraries(test_fftw_c2c_roundtrip_nd parafaft::parafaft)
set_target_properties(test_fftw_c2c_roundtrip_nd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

# R2C MPI Tests
add_executable(test_fftw_r2c_nd r2c/test_fftw_r2c_nd.cpp)
target_link_libraries(test_fftw_r2c_nd parafaft::parafaft)
set_target_properties(test_fftw_r2c_nd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_r2c_roundtrip_nd r2c/test_fftw_r2c_roundtrip_nd.cpp)
target_link_libraries(test_fftw_r2c_roundtrip_nd parafaft::parafaft)
set_target_properties(test_fftw_r2c_roundtrip_nd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_r2c_padded_roundtrip_nd r2c/test_fftw_r2c_padded_roundtrip_nd.cpp)
target_link_libraries(test_fftw_r2c_padded_roundtrip_nd parafaft::parafaft)
set_target_properties(test_fftw_r2c_padded_roundtrip_nd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_r2c_padded_nd r2c/test_fftw_r2c_padded_nd.cpp)
target_link_libraries(test_fftw_r2c_padded_nd parafaft::parafaft)
set_target_properties(test_fftw_r2c_padded_nd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

# Unit Tests
add_executable(test_fftw_backend unit/test_fftw_backend.cpp)
target_link_libraries(test_fftw_backend parafaft::parafaft)
set_target_properties(test_fftw_backend PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit
)

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
    set_target_properties(test_cufft_c2c_nd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    # R2C MPI Tests
    add_executable(test_cufft_r2c_nd r2c/test_cufft_r2c_nd.cu)
    target_link_libraries(test_cufft_r2c_nd parafaft::parafaft)
    set_target_properties(test_cufft_r2c_nd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_cufft_r2c_padded_nd r2c/test_cufft_r2c_padded_nd.cu)
    target_link_libraries(test_cufft_r2c_padded_nd parafaft::parafaft)
    set_target_properties(test_cufft_r2c_padded_nd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    # Unit Tests (cuFFT Backend)
    add_executable(test_cuvector unit/test_cuvector.cu)
    target_link_libraries(test_cuvector CUDA::cudart)
    set_target_properties(test_cuvector PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit
    )

    add_executable(test_cufft_backend unit/test_cufft_backend.cu)
    target_link_libraries(test_cufft_backend parafaft::parafaft)
    set_target_properties(test_cufft_backend PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit
    )
endif()

# ##############################################################################
# hipFFT Backend Tests
# ##############################################################################

if(PARAFAFT_HIP)
    # C2C MPI Tests
    add_executable(test_hipfft_c2c_nd c2c/test_hipfft_c2c_nd.hip)
    target_link_libraries(test_hipfft_c2c_nd parafaft::parafaft)
    set_target_properties(test_hipfft_c2c_nd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    # R2C MPI Tests
    add_executable(test_hipfft_r2c_nd r2c/test_hipfft_r2c_nd.hip)
    target_link_libraries(test_hipfft_r2c_nd parafaft::parafaft)
    set_target_properties(test_hipfft_r2c_nd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_hipfft_r2c_padded_nd r2c/test_hipfft_r2c_padded_nd.hip)
    target_link_libraries(test_hipfft_r2c_padded_nd parafaft::parafaft)
    set_target_properties(test_hipfft_r2c_padded_nd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    # Unit Tests (hipFFT Backend)
    add_executable(test_hipvector unit/test_hipvector.hip)
    target_link_libraries(test_hipvector hip::host)
    set_target_properties(test_hipvector PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit
    )

    add_executable(test_hipfft_backend unit/test_hipfft_backend.hip)
    target_link_libraries(test_hipfft_backend parafaft::parafaft)
    set_target_properties(test_hipfft_backend PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit
    )
endif()

# Serial Reference Tests (using standard g++ compiler)
add_executable(reference_serial_3d reference/reference_serial_3d.cpp)
target_link_libraries(reference_serial_3d parafaft::parafaft)
set_target_properties(reference_serial_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/reference
)

add_executable(reference_serial_r2c_3d reference/reference_serial_r2c_3d.cpp)
target_link_libraries(reference_serial_r2c_3d parafaft::parafaft)
set_target_properties(reference_serial_r2c_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/reference
)

# ##############################################################################

# Create directories for output files
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/c2c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/r2c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/unit)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/reference)

# Compare MPI output with serial reference
add_custom_target(compare
    COMMENT "Comparing MPI vs Serial Output"
    COMMAND echo "=== Comparing MPI vs Serial Output ==="
    COMMAND ${CMAKE_BINARY_DIR}/reference/reference_serial_3d > ${CMAKE_BINARY_DIR}/reference/serial_output.txt
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_fftw_c2c_roundtrip_nd > ${CMAKE_BINARY_DIR}/c2c/mpi_output.txt
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_outputs.py ${CMAKE_BINARY_DIR}/reference/serial_output.txt ${CMAKE_BINARY_DIR}/c2c/mpi_output.txt
    DEPENDS reference_serial_3d test_fftw_c2c_roundtrip_nd
)

# Clean target - CMake handles this automatically with 'make clean' or equivalent

# Help target (using 'show-help' instead of reserved 'help' name)
add_custom_target(show-help
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
    COMMAND echo "  cmake --build . --target show-help     - Display this help message"
    COMMAND echo ""
    COMMAND echo "Alternatively, you can use make syntax if using Make generator:"
    COMMAND echo "  make all, make test, make show-help, etc."
)

# Enable testing
enable_testing()

function(add_serial_tests test_name test_file test_build)
    add_test(NAME ${test_name} COMMAND 
        bash -c "make ${test_build} && ${CMAKE_BINARY_DIR}/${test_file}")
endfunction()

function(add_mpi_tests test_name test_file test_build)
    add_test(NAME ${test_name}_np1 COMMAND 
        bash -c "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file}")
    add_test(NAME ${test_name}_np4 COMMAND 
        bash -c "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file}")
    add_test(NAME ${test_name}_np8 COMMAND 
        bash -c "make ${test_build} && mpirun -np 8 ${CMAKE_BINARY_DIR}/${test_file}")
endfunction()

function(add_mpi_tests_varN_varD test_name test_file test_build)
    # Loop over shapes: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial
    foreach(S 0 1 2)
        # 2D tests
        add_test(NAME ${test_name}_np1_N32_D2_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 32 2 ${S}")
        add_test(NAME ${test_name}_np4_N32_D2_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 32 2 ${S}")
        # 3D tests
        add_test(NAME ${test_name}_np1_N32_D3_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 32 3 ${S}")
        add_test(NAME ${test_name}_np4_N32_D3_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 32 3 ${S}")
        # 4D tests (smaller N)
        add_test(NAME ${test_name}_np1_N16_D4_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 16 4 ${S}")
        add_test(NAME ${test_name}_np4_N16_D4_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 16 4 ${S}")
        # 5D tests (even smaller N)
        add_test(NAME ${test_name}_np1_N8_D5_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 8 5 ${S}")
        add_test(NAME ${test_name}_np4_N8_D5_S${S} COMMAND 
            bash -c "make ${test_build} && mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 8 5 ${S}")
    endforeach()
endfunction()

# Add tests to CTest
add_serial_tests(fftw_unit_backend unit/test_fftw_backend test_fftw_backend)

# MPI tests
add_mpi_tests_varN_varD(fftw_c2c_nd c2c/test_fftw_c2c_nd test_fftw_c2c_nd)
add_mpi_tests_varN_varD(fftw_c2c_roundtrip_nd c2c/test_fftw_c2c_roundtrip_nd test_fftw_c2c_roundtrip_nd)

add_mpi_tests_varN_varD(fftw_r2c_nd r2c/test_fftw_r2c_nd test_fftw_r2c_nd)
add_mpi_tests_varN_varD(fftw_r2c_roundtrip_nd r2c/test_fftw_r2c_roundtrip_nd test_fftw_r2c_roundtrip_nd)
add_mpi_tests_varN_varD(fftw_r2c_padded_nd r2c/test_fftw_r2c_padded_nd test_fftw_r2c_padded_nd)
add_mpi_tests_varN_varD(fftw_r2c_padded_roundtrip_nd r2c/test_fftw_r2c_padded_roundtrip_nd test_fftw_r2c_padded_roundtrip_nd)

# cuFFT tests (only added if CUDA is enabled)
if(PARAFAFT_CUDA)
    add_serial_tests(cufft_unit_backend unit/test_cufft_backend test_cufft_backend)
    add_serial_tests(cufft_unit_cuvector unit/test_cuvector test_cuvector)

    add_mpi_tests_varN_varD(cufft_c2c_nd c2c/test_cufft_c2c_nd test_cufft_c2c_nd)

    add_mpi_tests_varN_varD(cufft_r2c_nd r2c/test_cufft_r2c_nd test_cufft_r2c_nd)
    add_mpi_tests_varN_varD(cufft_r2c_padded_nd r2c/test_cufft_r2c_padded_nd test_cufft_r2c_padded_nd)
endif()

# hipFFT tests (only added if HIP is enabled)
if(PARAFAFT_HIP)
    add_serial_tests(hipfft_unit_backend unit/test_hipfft_backend test_hipfft_backend)
    add_serial_tests(hipfft_unit_hipvector unit/test_hipvector test_hipvector)

    add_mpi_tests_varN_varD(hipfft_c2c_nd c2c/test_hipfft_c2c_nd test_hipfft_c2c_nd)

    add_mpi_tests_varN_varD(hipfft_r2c_nd r2c/test_hipfft_r2c_nd test_hipfft_r2c_nd)
    add_mpi_tests_varN_varD(hipfft_r2c_padded_nd r2c/test_hipfft_r2c_padded_nd test_hipfft_r2c_padded_nd)
endif()
