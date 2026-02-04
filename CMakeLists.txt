cmake_minimum_required(VERSION 3.19)
project(parafaft
    VERSION 1.0.0
    DESCRIPTION "Parallel FFT library using pencil decomposition"
    LANGUAGES CXX
)

# Options
option(PARAFAFT_CUDA "Enable CUDA support" OFF)
option(PARAFAFT_HIP "Enable HIP support" OFF)
option(PARAFAFT_TEST "Build tests" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
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
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)

# Create header-only library target
add_library(parafaft INTERFACE)
add_library(parafaft::parafaft ALIAS parafaft)

# Specify include directories
target_include_directories(parafaft INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(parafaft INTERFACE
    MPI::MPI_CXX
    ${FFTW3_LIBRARIES}
)

target_include_directories(parafaft INTERFACE
    ${FFTW3_INCLUDE_DIRS}
)

# Add CUDA dependencies if enabled
if(PARAFAFT_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(parafaft INTERFACE
        CUDA::cufft
        CUDA::cudart
    )
    target_compile_definitions(parafaft INTERFACE PARAFAFT_CUDA_ENABLED)
    set_property(TARGET parafaft PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Add HIP dependencies if enabled
if(PARAFAFT_HIP AND CMAKE_HIP_COMPILER)
    target_link_libraries(parafaft INTERFACE
        hip::hipfft
        hip::host
    )
    target_compile_definitions(parafaft INTERFACE PARAFAFT_HIP_ENABLED)
endif()

# Add test subdirectory if enabled
if(PARAFAFT_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS parafaft
    EXPORT parafaftTargets
)

install(FILES
    parafaft_generic.hpp
    parafaft_r2c.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parafaft
)

install(DIRECTORY backend
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/parafaft
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT parafaftTargets
    FILE parafaftTargets.cmake
    NAMESPACE parafaft::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parafaft
)

# Generate config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/parafaftConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parafaft
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/parafaftConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parafaft
)
