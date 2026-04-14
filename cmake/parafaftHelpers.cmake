# parafaftHelpers.cmake
#
# Downstream-facing helpers for ParaFaFT users.
#
# parafaft_enable_cuda_sources(<target>)
#   Tag .cu files in <target> as LANGUAGE CXX so NVHPC's nvc++ compiles
#   them. No-op when the CXX compiler is not NVHPC — in that case CMake's
#   enable_language(CUDA) already dispatches .cu files to nvcc/clang.
#
#   Only needed when the downstream target uses the .cu file extension
#   with NVHPC. Users can also simply name their sources .cpp; nvc++
#   compiles them identically when -cuda is active (ParaFaFT propagates
#   -cuda through its INTERFACE target, so that flag is already there).

function(parafaft_enable_cuda_sources target)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
    return()
  endif()
  get_target_property(_sources ${target} SOURCES)
  foreach(_src ${_sources})
    if(_src MATCHES "\\.cu$")
      set_source_files_properties(${_src} PROPERTIES LANGUAGE CXX)
    endif()
  endforeach()
endfunction()
