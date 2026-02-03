cmake_minimum_required(VERSION 3.10)
project(ParallelFFTTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)

include(CheckLanguage)
option(CUDA "Build with CUDA support" OFF)
if(CUDA)
    check_language(CUDA)
    if(!CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA support requested but no CUDA compiler found.")
    endif()
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})
    # if CMAKE_CUDA_ARCHITECTURES is not set, set it to native
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
else()
    message(STATUS "Building without CUDA support")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    /opt/homebrew/include
    ${MPI_INCLUDE_PATH}
    ${FFTW3_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    /opt/homebrew/lib
    /usr/lib/aarch64-linux-gnu
    ${FFTW3_LIBRARY_DIRS}
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")

# Link flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")

# Define dependencies for MPI tests
set(MPI_GENERIC_DEPS 
    ../parafaft_generic.hpp 
    ../backend/fftw3/fft_backend_fftw.hpp
)

set(MPI_R2C_DEPS 
    ../parafaft_r2c.hpp 
    ../backend/fftw3/fft_backend_fftw.hpp
)

set(UNIT_DEPS 
    ../backend/fft_backend.hpp 
    ../backend/fftw3/fft_backend_fftw.hpp
)

# C2C MPI Tests
add_executable(test_mpi_constant_32 c2c/test_mpi_constant_32.cpp)
target_link_libraries(test_mpi_constant_32 ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_mpi_constant_32 PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_mpi_constant_32 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_mpi_gaussian_roundtrip c2c/test_mpi_gaussian_roundtrip.cpp)
target_link_libraries(test_mpi_gaussian_roundtrip ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_mpi_gaussian_roundtrip PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_mpi_gaussian_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_8cubed c2c/test_8cubed.cpp)
target_link_libraries(test_8cubed ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_8cubed PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_8cubed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

add_executable(test_4d_roundtrip c2c/test_4d_roundtrip.cpp)
target_link_libraries(test_4d_roundtrip ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_4d_roundtrip PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_4d_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/c2c
)

# R2C MPI Tests
add_executable(test_mpi_r2c_gaussian r2c/test_mpi_r2c_gaussian.cpp)
target_link_libraries(test_mpi_r2c_gaussian ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_mpi_r2c_gaussian PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_mpi_r2c_gaussian PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_mpi_r2c_roundtrip r2c/test_mpi_r2c_roundtrip.cpp)
target_link_libraries(test_mpi_r2c_roundtrip ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_mpi_r2c_roundtrip PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_mpi_r2c_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_mpi_r2c_padded_roundtrip r2c/test_mpi_r2c_padded_roundtrip.cpp)
target_link_libraries(test_mpi_r2c_padded_roundtrip ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_mpi_r2c_padded_roundtrip PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_mpi_r2c_padded_roundtrip PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

add_executable(test_mpi_r2c_padded_gaussian r2c/test_mpi_r2c_padded_gaussian.cpp)
target_link_libraries(test_mpi_r2c_padded_gaussian ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_mpi_r2c_padded_gaussian PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_mpi_r2c_padded_gaussian PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
)

# cufft tests
if(CUDA)
    add_executable(test_mpi_r2c_roundtrip_cufft r2c/test_mpi_r2c_roundtrip_cufft.cu)
    target_link_libraries(test_mpi_r2c_roundtrip_cufft ${MPI_LIBRARIES} ${FFTW3_LIBRARIES} ${CUDA_LIBRARIES} CUDA::cufft_static 
    ${CMAKE_DL_LIBS})
    target_compile_options(test_mpi_r2c_roundtrip_cufft PRIVATE ${MPI_COMPILE_FLAGS})

    set_target_properties(test_mpi_r2c_roundtrip_cufft PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/r2c
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Unit Tests (Backend)
add_executable(test_fftw_backend unit/test_fftw_backend.cpp)
target_link_libraries(test_fftw_backend ${MPI_LIBRARIES} ${FFTW3_LIBRARIES})
target_compile_options(test_fftw_backend PRIVATE ${MPI_COMPILE_FLAGS})
set_target_properties(test_fftw_backend PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit
)
if(CUDA)
    add_executable(test_cufft_backend unit/test_cufft_backend_r2c.cu)
    target_link_libraries(test_cufft_backend ${MPI_LIBRARIES} ${FFTW3_LIBRARIES} ${CUDA_LIBRARIES} CUDA::cufft_static 
    ${CMAKE_DL_LIBS})
    target_compile_options(test_cufft_backend PRIVATE ${MPI_COMPILE_FLAGS})
    set_target_properties(test_cufft_backend PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Serial Reference Tests (using standard g++ compiler)
add_executable(reference_serial_3d reference/reference_serial_3d.cpp)
target_link_libraries(reference_serial_3d ${FFTW3_LIBRARIES})
set_target_properties(reference_serial_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/reference
)

add_executable(reference_serial_r2c_3d reference/reference_serial_r2c_3d.cpp)
target_link_libraries(reference_serial_r2c_3d ${FFTW3_LIBRARIES})
set_target_properties(reference_serial_r2c_3d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/reference
)

# Define test groups
set(C2C_TESTS 
    test_mpi_constant_32 
    test_mpi_gaussian_roundtrip 
    test_8cubed 
    test_4d_roundtrip
)

set(R2C_TESTS 
    test_mpi_r2c_gaussian 
    test_mpi_r2c_roundtrip 
    test_mpi_r2c_padded_roundtrip 
    test_mpi_r2c_padded_gaussian
)

set(UNIT_TESTS test_fftw_backend)

set(REFERENCE_TESTS 
    reference_serial_3d 
    reference_serial_r2c_3d
)

# Create directories for output files
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/c2c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/r2c)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/unit)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/reference)

# Custom targets for running tests
add_custom_target(run-c2c
    COMMENT "Running C2C Tests"
    COMMAND echo "=== Running C2C Tests ==="
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_mpi_constant_32
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_mpi_gaussian_roundtrip
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_8cubed
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_4d_roundtrip
    DEPENDS ${C2C_TESTS}
)

add_custom_target(run-r2c
    COMMENT "Running R2C Tests"
    COMMAND echo "=== Running R2C Tests ==="
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_gaussian
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_roundtrip
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_padded_roundtrip
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_padded_gaussian
    DEPENDS ${R2C_TESTS}
)

add_custom_target(run-unit
    COMMENT "Running Unit Tests"
    COMMAND echo "=== Running Unit Tests ==="
    COMMAND ${CMAKE_BINARY_DIR}/unit/test_fftw_backend
    DEPENDS ${UNIT_TESTS}
)

add_custom_target(run-reference
    COMMENT "Running Reference Tests"
    COMMAND echo "=== Running Reference Tests ==="
    COMMAND ${CMAKE_BINARY_DIR}/reference/reference_serial_3d
    COMMAND ${CMAKE_BINARY_DIR}/reference/reference_serial_r2c_3d
    DEPENDS ${REFERENCE_TESTS}
)

add_custom_target(run-all
    COMMENT "Running All Tests"
    DEPENDS run-unit run-reference run-c2c run-r2c
)

# Compare MPI output with serial reference
add_custom_target(compare
    COMMENT "Comparing MPI vs Serial Output"
    COMMAND echo "=== Comparing MPI vs Serial Output ==="
    COMMAND ${CMAKE_BINARY_DIR}/reference/reference_serial_3d > ${CMAKE_BINARY_DIR}/reference/serial_output.txt
    COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_mpi_gaussian_roundtrip > ${CMAKE_BINARY_DIR}/c2c/mpi_output.txt
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_outputs.py ${CMAKE_BINARY_DIR}/reference/serial_output.txt ${CMAKE_BINARY_DIR}/c2c/mpi_output.txt
    DEPENDS reference_serial_3d test_mpi_gaussian_roundtrip
)

# Clean target - CMake handles this automatically with 'make clean' or equivalent

# Help target (using 'show-help' instead of reserved 'help' name)
add_custom_target(show-help
    COMMENT "Displaying help information"
    COMMAND echo "Parallel FFT Test Suite - CMake Version"
    COMMAND echo ""
    COMMAND echo "Build targets:"
    COMMAND echo "  cmake --build .         - Build all tests"
    COMMAND echo "  cmake --build . --target clean - Remove all built files"
    COMMAND echo ""
    COMMAND echo "Run targets:"
    COMMAND echo "  cmake --build . --target run-unit      - Run backend unit tests"
    COMMAND echo "  cmake --build . --target run-reference - Run serial reference tests"
    COMMAND echo "  cmake --build . --target run-c2c       - Run C2C MPI tests"
    COMMAND echo "  cmake --build . --target run-r2c       - Run R2C MPI tests"
    COMMAND echo "  cmake --build . --target run-all       - Run all tests"
    COMMAND echo "  cmake --build . --target compare       - Compare MPI vs serial output"
    COMMAND echo "  cmake --build . --target show-help     - Display this help message"
    COMMAND echo ""
    COMMAND echo "Alternatively, you can use make syntax if using Make generator:"
    COMMAND echo "  make all, make run-unit, make run-c2c, make show-help, etc."
)

# Enable testing
enable_testing()

# Add tests to CTest
add_test(NAME unit_tests COMMAND ${CMAKE_BINARY_DIR}/unit/test_fftw_backend)
add_test(NAME reference_3d COMMAND ${CMAKE_BINARY_DIR}/reference/reference_serial_3d)
add_test(NAME reference_r2c_3d COMMAND ${CMAKE_BINARY_DIR}/reference/reference_serial_r2c_3d)

# MPI tests
add_test(NAME c2c_constant_32 COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_mpi_constant_32)
add_test(NAME c2c_gaussian_roundtrip COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_mpi_gaussian_roundtrip)
add_test(NAME c2c_8cubed COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_8cubed)
add_test(NAME c2c_4d_roundtrip COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/c2c/test_4d_roundtrip)

add_test(NAME r2c_gaussian COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_gaussian)
add_test(NAME r2c_roundtrip COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_roundtrip)
add_test(NAME r2c_padded_roundtrip COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_padded_roundtrip)
add_test(NAME r2c_padded_gaussian COMMAND mpirun -np 4 ${CMAKE_BINARY_DIR}/r2c/test_mpi_r2c_padded_gaussian)
