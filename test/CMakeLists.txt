# ##############################################################################
# FFTW Backend Tests
# ##############################################################################

# C2C MPI Tests
add_executable(test_fftw_mpi_constant_32 c2c/test_fftw_mpi_constant_32.cpp)
target_link_libraries(test_fftw_mpi_constant_32 parafaft::parafaft)
set_target_properties(test_fftw_mpi_constant_32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_fftw_mpi_c2c_gaussian_roundtrip c2c/test_fftw_mpi_c2c_gaussian_roundtrip.cpp)
target_link_libraries(test_fftw_mpi_c2c_gaussian_roundtrip parafaft::parafaft)
set_target_properties(test_fftw_mpi_c2c_gaussian_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_fftw_mpi_8cubed c2c/test_fftw_mpi_8cubed.cpp)
target_link_libraries(test_fftw_mpi_8cubed parafaft::parafaft)
set_target_properties(test_fftw_mpi_8cubed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_fftw_mpi_4d_roundtrip c2c/test_fftw_mpi_4d_roundtrip.cpp)
target_link_libraries(test_fftw_mpi_4d_roundtrip parafaft::parafaft)
set_target_properties(test_fftw_mpi_4d_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_fftw_mpi_c2c_gaussian_3d c2c/test_fftw_mpi_c2c_gaussian_3d.cpp)
target_link_libraries(test_fftw_mpi_c2c_gaussian_3d parafaft::parafaft)
set_target_properties(test_fftw_mpi_c2c_gaussian_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_fftw_mpi_c2c_gaussian_2d c2c/test_fftw_mpi_c2c_gaussian_2d.cpp)
target_link_libraries(test_fftw_mpi_c2c_gaussian_2d parafaft::parafaft)
set_target_properties(test_fftw_mpi_c2c_gaussian_2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_fftw_mpi_c2c_gaussian_4d c2c/test_fftw_mpi_c2c_gaussian_4d.cpp)
target_link_libraries(test_fftw_mpi_c2c_gaussian_4d parafaft::parafaft)
set_target_properties(test_fftw_mpi_c2c_gaussian_4d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

# R2C MPI Tests
add_executable(test_fftw_mpi_r2c_gaussian_2d r2c/test_fftw_mpi_r2c_gaussian_2d.cpp)
target_link_libraries(test_fftw_mpi_r2c_gaussian_2d parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_gaussian_2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_mpi_r2c_gaussian_3d r2c/test_fftw_mpi_r2c_gaussian_3d.cpp)
target_link_libraries(test_fftw_mpi_r2c_gaussian_3d parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_gaussian_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_mpi_r2c_gaussian_4d r2c/test_fftw_mpi_r2c_gaussian_4d.cpp)
target_link_libraries(test_fftw_mpi_r2c_gaussian_4d parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_gaussian_4d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_mpi_r2c_roundtrip r2c/test_fftw_mpi_r2c_roundtrip.cpp)
target_link_libraries(test_fftw_mpi_r2c_roundtrip parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_mpi_r2c_padded_roundtrip r2c/test_fftw_mpi_r2c_padded_roundtrip.cpp)
target_link_libraries(test_fftw_mpi_r2c_padded_roundtrip parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_padded_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_mpi_r2c_padded_gaussian_2d r2c/test_fftw_mpi_r2c_padded_gaussian_2d.cpp)
target_link_libraries(test_fftw_mpi_r2c_padded_gaussian_2d parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_padded_gaussian_2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_mpi_r2c_padded_gaussian_3d r2c/test_fftw_mpi_r2c_padded_gaussian_3d.cpp)
target_link_libraries(test_fftw_mpi_r2c_padded_gaussian_3d parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_padded_gaussian_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_fftw_mpi_r2c_padded_gaussian_4d r2c/test_fftw_mpi_r2c_padded_gaussian_4d.cpp)
target_link_libraries(test_fftw_mpi_r2c_padded_gaussian_4d parafaft::parafaft)
set_target_properties(test_fftw_mpi_r2c_padded_gaussian_4d PROPERTIES
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
    add_executable(test_cufft_mpi_c2c_gaussian_2d c2c/test_cufft_mpi_c2c_gaussian_2d.cu)
    target_link_libraries(test_cufft_mpi_c2c_gaussian_2d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_c2c_gaussian_2d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    add_executable(test_cufft_mpi_c2c_gaussian_3d c2c/test_cufft_mpi_c2c_gaussian_3d.cu)
    target_link_libraries(test_cufft_mpi_c2c_gaussian_3d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_c2c_gaussian_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    add_executable(test_cufft_mpi_c2c_gaussian_4d c2c/test_cufft_mpi_c2c_gaussian_4d.cu)
    target_link_libraries(test_cufft_mpi_c2c_gaussian_4d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_c2c_gaussian_4d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    # R2C MPI Tests
    add_executable(test_cufft_mpi_r2c_padded_gaussian_2d r2c/test_cufft_mpi_r2c_padded_gaussian_2d.cu)
    target_link_libraries(test_cufft_mpi_r2c_padded_gaussian_2d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_r2c_padded_gaussian_2d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_cufft_mpi_r2c_padded_gaussian_3d r2c/test_cufft_mpi_r2c_padded_gaussian_3d.cu)
    target_link_libraries(test_cufft_mpi_r2c_padded_gaussian_3d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_r2c_padded_gaussian_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_cufft_mpi_r2c_padded_gaussian_4d r2c/test_cufft_mpi_r2c_padded_gaussian_4d.cu)
    target_link_libraries(test_cufft_mpi_r2c_padded_gaussian_4d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_r2c_padded_gaussian_4d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_cufft_mpi_r2c_gaussian_2d r2c/test_cufft_mpi_r2c_gaussian_2d.cu)
    target_link_libraries(test_cufft_mpi_r2c_gaussian_2d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_r2c_gaussian_2d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_cufft_mpi_r2c_gaussian_3d r2c/test_cufft_mpi_r2c_gaussian_3d.cu)
    target_link_libraries(test_cufft_mpi_r2c_gaussian_3d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_r2c_gaussian_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_cufft_mpi_r2c_gaussian_4d r2c/test_cufft_mpi_r2c_gaussian_4d.cu)
    target_link_libraries(test_cufft_mpi_r2c_gaussian_4d parafaft::parafaft)
    set_target_properties(test_cufft_mpi_r2c_gaussian_4d PROPERTIES
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
    add_executable(test_hipfft_mpi_c2c_gaussian_2d c2c/test_hipfft_mpi_c2c_gaussian_2d.hip)
    target_link_libraries(test_hipfft_mpi_c2c_gaussian_2d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_c2c_gaussian_2d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    add_executable(test_hipfft_mpi_c2c_gaussian_3d c2c/test_hipfft_mpi_c2c_gaussian_3d.hip)
    target_link_libraries(test_hipfft_mpi_c2c_gaussian_3d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_c2c_gaussian_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    add_executable(test_hipfft_mpi_c2c_gaussian_4d c2c/test_hipfft_mpi_c2c_gaussian_4d.hip)
    target_link_libraries(test_hipfft_mpi_c2c_gaussian_4d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_c2c_gaussian_4d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
    )

    # R2C MPI Tests
    add_executable(test_hipfft_mpi_r2c_padded_gaussian_2d r2c/test_hipfft_mpi_r2c_padded_gaussian_2d.hip)
    target_link_libraries(test_hipfft_mpi_r2c_padded_gaussian_2d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_r2c_padded_gaussian_2d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_hipfft_mpi_r2c_padded_gaussian_3d r2c/test_hipfft_mpi_r2c_padded_gaussian_3d.hip)
    target_link_libraries(test_hipfft_mpi_r2c_padded_gaussian_3d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_r2c_padded_gaussian_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_hipfft_mpi_r2c_padded_gaussian_4d r2c/test_hipfft_mpi_r2c_padded_gaussian_4d.hip)
    target_link_libraries(test_hipfft_mpi_r2c_padded_gaussian_4d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_r2c_padded_gaussian_4d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_hipfft_mpi_r2c_gaussian_2d r2c/test_hipfft_mpi_r2c_gaussian_2d.hip)
    target_link_libraries(test_hipfft_mpi_r2c_gaussian_2d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_r2c_gaussian_2d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_hipfft_mpi_r2c_gaussian_3d r2c/test_hipfft_mpi_r2c_gaussian_3d.hip)
    target_link_libraries(test_hipfft_mpi_r2c_gaussian_3d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_r2c_gaussian_3d PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
    )

    add_executable(test_hipfft_mpi_r2c_gaussian_4d r2c/test_hipfft_mpi_r2c_gaussian_4d.hip)
    target_link_libraries(test_hipfft_mpi_r2c_gaussian_4d parafaft::parafaft)
    set_target_properties(test_hipfft_mpi_r2c_gaussian_4d PROPERTIES
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
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_fftw_mpi_c2c_gaussian_roundtrip > ${CMAKE_BINARY_DIR}/c2c/mpi_output.txt
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_outputs.py ${CMAKE_BINARY_DIR}/reference/serial_output.txt ${CMAKE_BINARY_DIR}/c2c/mpi_output.txt
    DEPENDS reference_serial_3d test_fftw_mpi_c2c_gaussian_roundtrip
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
    add_test(NAME ${test_name} COMMAND ${CMAKE_BINARY_DIR}/${test_file})
    # Make sure the test depends on the executable being built
    set_tests_properties(${test_name} PROPERTIES DEPENDS ${test_build})
endfunction()

function(add_mpi_tests test_name test_file test_build)
    add_test(NAME ${test_name}_np1 COMMAND mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file})
    add_test(NAME ${test_name}_np4 COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file})
    add_test(NAME ${test_name}_np8 COMMAND mpirun -np 8 ${CMAKE_BINARY_DIR}/${test_file})
    # Make sure the test depends on the executable being built
    set_tests_properties(${test_name}_np1 PROPERTIES DEPENDS ${test_build})
    set_tests_properties(${test_name}_np4 PROPERTIES DEPENDS ${test_build})
    set_tests_properties(${test_name}_np8 PROPERTIES DEPENDS ${test_build})
endfunction()

function(add_mpi_tests_varN test_name test_file test_build)
    add_test(NAME ${test_name}_np1_N8 COMMAND mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 8)
    add_test(NAME ${test_name}_np1_N32 COMMAND mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 32)
    add_test(NAME ${test_name}_np1_N64 COMMAND mpirun -np 1 ${CMAKE_BINARY_DIR}/${test_file} 64)
    add_test(NAME ${test_name}_np4_N8 COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 8)
    add_test(NAME ${test_name}_np4_N32 COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 32)
    add_test(NAME ${test_name}_np4_N64 COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/${test_file} 64)
    # Make sure the test depends on the executable being built
    set_tests_properties(${test_name}_np1_N8 PROPERTIES DEPENDS ${test_build})
    set_tests_properties(${test_name}_np1_N32 PROPERTIES DEPENDS ${test_build})
    set_tests_properties(${test_name}_np1_N64 PROPERTIES DEPENDS ${test_build})
    set_tests_properties(${test_name}_np4_N8 PROPERTIES DEPENDS ${test_build})
    set_tests_properties(${test_name}_np4_N32 PROPERTIES DEPENDS ${test_build})
    set_tests_properties(${test_name}_np4_N64 PROPERTIES DEPENDS ${test_build})
endfunction()

# Add tests to CTest
add_serial_tests(fftw_unit_backend unit/test_fftw_backend test_fftw_backend)

# MPI tests
add_mpi_tests(fftw_c2c_constant_32 c2c/test_fftw_mpi_constant_32 test_fftw_mpi_constant_32)
add_mpi_tests(fftw_c2c_8cubed c2c/test_fftw_mpi_8cubed test_fftw_mpi_8cubed)
add_mpi_tests(fftw_c2c_4d_roundtrip c2c/test_fftw_mpi_4d_roundtrip test_fftw_mpi_4d_roundtrip)
add_mpi_tests_varN(fftw_c2c_gaussian_roundtrip c2c/test_fftw_mpi_c2c_gaussian_roundtrip test_fftw_mpi_c2c_gaussian_roundtrip)
add_mpi_tests_varN(fftw_c2c_gaussian_2d c2c/test_fftw_mpi_c2c_gaussian_2d test_fftw_mpi_c2c_gaussian_2d)
add_mpi_tests_varN(fftw_c2c_gaussian_3d c2c/test_fftw_mpi_c2c_gaussian_3d test_fftw_mpi_c2c_gaussian_3d)
add_mpi_tests_varN(fftw_c2c_gaussian_4d c2c/test_fftw_mpi_c2c_gaussian_4d test_fftw_mpi_c2c_gaussian_4d)

add_mpi_tests_varN(fftw_r2c_gaussian_2d r2c/test_fftw_mpi_r2c_gaussian_2d test_fftw_mpi_r2c_gaussian_2d)
add_mpi_tests_varN(fftw_r2c_gaussian_3d r2c/test_fftw_mpi_r2c_gaussian_3d test_fftw_mpi_r2c_gaussian_3d)
add_mpi_tests_varN(fftw_r2c_gaussian_4d r2c/test_fftw_mpi_r2c_gaussian_4d test_fftw_mpi_r2c_gaussian_4d)
add_mpi_tests_varN(fftw_r2c_roundtrip r2c/test_fftw_mpi_r2c_roundtrip test_fftw_mpi_r2c_roundtrip)
add_mpi_tests_varN(fftw_r2c_padded_gaussian_2d r2c/test_fftw_mpi_r2c_padded_gaussian_2d test_fftw_mpi_r2c_padded_gaussian_2d)
add_mpi_tests_varN(fftw_r2c_padded_gaussian_3d r2c/test_fftw_mpi_r2c_padded_gaussian_3d test_fftw_mpi_r2c_padded_gaussian_3d)
add_mpi_tests_varN(fftw_r2c_padded_gaussian_4d r2c/test_fftw_mpi_r2c_padded_gaussian_4d test_fftw_mpi_r2c_padded_gaussian_4d)
add_mpi_tests_varN(fftw_r2c_padded_roundtrip r2c/test_fftw_mpi_r2c_padded_roundtrip test_fftw_mpi_r2c_padded_roundtrip)

# cuFFT tests (only added if CUDA is enabled)
if(PARAFAFT_CUDA)
    add_serial_tests(cufft_unit_backend unit/test_cufft_backend test_cufft_backend)
    add_serial_tests(cufft_unit_cuvector unit/test_cuvector test_cuvector)

    add_mpi_tests_varN(cufft_c2c_gaussian_2d c2c/test_cufft_mpi_c2c_gaussian_2d test_cufft_mpi_c2c_gaussian_2d)
    add_mpi_tests_varN(cufft_c2c_gaussian_3d c2c/test_cufft_mpi_c2c_gaussian_3d test_cufft_mpi_c2c_gaussian_3d)
    add_mpi_tests_varN(cufft_c2c_gaussian_4d c2c/test_cufft_mpi_c2c_gaussian_4d test_cufft_mpi_c2c_gaussian_4d)

    add_mpi_tests_varN(cufft_r2c_gaussian_2d r2c/test_cufft_mpi_r2c_gaussian_2d test_cufft_mpi_r2c_gaussian_2d)
    add_mpi_tests_varN(cufft_r2c_gaussian_3d r2c/test_cufft_mpi_r2c_gaussian_3d test_cufft_mpi_r2c_gaussian_3d)
    add_mpi_tests_varN(cufft_r2c_gaussian_4d r2c/test_cufft_mpi_r2c_gaussian_4d test_cufft_mpi_r2c_gaussian_4d)
    add_mpi_tests_varN(cufft_r2c_padded_gaussian_2d r2c/test_cufft_mpi_r2c_padded_gaussian_2d test_cufft_mpi_r2c_padded_gaussian_2d)
    add_mpi_tests_varN(cufft_r2c_padded_gaussian_3d r2c/test_cufft_mpi_r2c_padded_gaussian_3d test_cufft_mpi_r2c_padded_gaussian_3d)
    add_mpi_tests_varN(cufft_r2c_padded_gaussian_4d r2c/test_cufft_mpi_r2c_padded_gaussian_4d test_cufft_mpi_r2c_padded_gaussian_4d)
endif()

# hipFFT tests (only added if HIP is enabled)
if(PARAFAFT_HIP)
    add_serial_tests(hipfft_unit_backend unit/test_hipfft_backend test_hipfft_backend)
    add_serial_tests(hipfft_unit_hipvector unit/test_hipvector test_hipvector)

    add_mpi_tests_varN(hipfft_c2c_gaussian_2d c2c/test_hipfft_mpi_c2c_gaussian_2d test_hipfft_mpi_c2c_gaussian_2d)
    add_mpi_tests_varN(hipfft_c2c_gaussian_3d c2c/test_hipfft_mpi_c2c_gaussian_3d test_hipfft_mpi_c2c_gaussian_3d)
    add_mpi_tests_varN(hipfft_c2c_gaussian_4d c2c/test_hipfft_mpi_c2c_gaussian_4d test_hipfft_mpi_c2c_gaussian_4d)

    add_mpi_tests_varN(hipfft_r2c_gaussian_2d r2c/test_hipfft_mpi_r2c_gaussian_2d test_hipfft_mpi_r2c_gaussian_2d)
    add_mpi_tests_varN(hipfft_r2c_gaussian_3d r2c/test_hipfft_mpi_r2c_gaussian_3d test_hipfft_mpi_r2c_gaussian_3d)
    add_mpi_tests_varN(hipfft_r2c_gaussian_4d r2c/test_hipfft_mpi_r2c_gaussian_4d test_hipfft_mpi_r2c_gaussian_4d)
    add_mpi_tests_varN(hipfft_r2c_padded_gaussian_2d r2c/test_hipfft_mpi_r2c_padded_gaussian_2d test_hipfft_mpi_r2c_padded_gaussian_2d)
    add_mpi_tests_varN(hipfft_r2c_padded_gaussian_3d r2c/test_hipfft_mpi_r2c_padded_gaussian_3d test_hipfft_mpi_r2c_padded_gaussian_3d)
    add_mpi_tests_varN(hipfft_r2c_padded_gaussian_4d r2c/test_hipfft_mpi_r2c_padded_gaussian_4d test_hipfft_mpi_r2c_padded_gaussian_4d)
endif()
