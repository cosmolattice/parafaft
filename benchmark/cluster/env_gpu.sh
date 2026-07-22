#!/usr/bin/env bash
# =============================================================================
# ParaFaFT — GPU (cuFFT) benchmark environment + build helper for PC2 Noctua 2.
#
# Usage (on a login node or in an interactive GPU job):
#     source benchmark/cluster/env_gpu.sh
#     build_gpu                # build build_gpu/bench/bench_r2c_cuda
#
# For the cuFFTMp baseline, USE_CUFFTMP must be set BEFORE sourcing this file —
# it gates the NVHPC module load that provides cuFFTMp + NVSHMEM:
#     USE_CUFFTMP=1 source benchmark/cluster/env_gpu.sh
#     build_gpu                # also builds bench_r2c_cufftmp
#
# The SLURM scripts (strong_scaling_gpu.slurm, weak_scaling_gpu.slurm) source
# this file to load runtime modules and set the CUDA-aware MPI flag.
# =============================================================================

_pf_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PARAFAFT_ROOT="$(cd "${_pf_here}/../.." && pwd)"
# Separate build dir per mode. The two modes differ in loaded modules (NVHPC) and
# in a cached CMake option, and a shared dir gets both wrong: the PARAFAFT_CUFFTMP=ON
# entry is sticky (a later plain build_gpu does not reset it to OFF, so it still
# demands cuFFTMp without NVHPC loaded), and objects would be relinked under a
# different module environment than they were compiled in.
if [ "${USE_CUFFTMP:-0}" = "1" ]; then
  export PARAFAFT_GPU_BUILD="${PARAFAFT_ROOT}/build_gpu_cufftmp"
else
  export PARAFAFT_GPU_BUILD="${PARAFAFT_ROOT}/build_gpu"
fi

# --- Modules -----------------------------------------------------------------
# TODO(user): resolve exact strings with
#   module spider CUDA ; module spider OpenMPI
# g++ comes from the GCC toolchain bundled with the OpenMPI module (used as the
# nvcc host compiler). Load a CUDA-AWARE OpenMPI (needed for multi-GPU redist).
module reset 2>/dev/null || module purge 2>/dev/null || true

module load devel/CMake/3.31.3-GCCcore-14.2.0
module load mpi/OpenMPI/5.0.7-GCC-14.2.0
module load numlib/FFTW.MPI/3.3.10-gompi-2025a
module load data/HDF5/1.14.6-gompi-2025a
module load system/CUDA/12.8.0
module load lib/UCX-CUDA/1.18.0-GCCcore-14.2.0-CUDA-12.8.0

# cuFFTMp is NOT part of the CUDA toolkit — it ships with the NVIDIA HPC SDK
# (math_libs/*/cufftmp) and needs NVSHMEM (comm_libs/nvshmem) alongside it.
# Load NVHPC only when the baseline is wanted, so the ParaFaFT-only path keeps
# the leaner module set. cmake/cufftmp.cmake picks the install up via NVHPC_ROOT.
if [ "${USE_CUFFTMP:-0}" = "1" ]; then
  module load compiler/NVHPC/25.3-CUDA-12.8.0
  export NVHPC_ROOT="${EBROOTNVHPC:?NVHPC module loaded but EBROOTNVHPC unset — check module name}"

  # The EasyBuild NVHPC layout nests the SDK under Linux_x86_64/<ver>/, so the
  # bare ${NVHPC_ROOT}/math_libs that cmake/cufftmp.cmake probes does not exist
  # and the finder reports CUFFTMP_INCLUDE_DIR NOTFOUND. Locate the real dirs
  # and export them so the finder (via *_HOME) and build_gpu (via -I) pick up.
  #
  # cuFFTMp ships DROP-IN cufft.h / cufftXt.h in math_libs/<ver>/include/cufftmp;
  # that dir MUST precede the regular CUDA include, else cufftMp.h aborts with
  # "cuFFT and cuFFTMp version mismatch". build_gpu prepends it via -I.
  _pf_mp_h="$(find "${NVHPC_ROOT}/Linux_x86_64" -path '*/cufftmp/cufftMp.h' 2>/dev/null | head -1)"
  _pf_shmem_h="$(find "${NVHPC_ROOT}/Linux_x86_64" -name nvshmem.h 2>/dev/null | head -1)"
  if [ -n "${_pf_mp_h}" ]; then
    export PARAFAFT_CUFFTMP_INCLUDE="${_pf_mp_h%/*}"                    # .../include/cufftmp
    export CUFFTMP_HOME="${PARAFAFT_CUFFTMP_INCLUDE%/include/cufftmp}"  # .../math_libs/<ver>
    echo ">> cuFFTMp include: ${PARAFAFT_CUFFTMP_INCLUDE}"
  else
    echo ">> WARN: cufftMp.h not found under ${NVHPC_ROOT}/Linux_x86_64"
  fi
  if [ -n "${_pf_shmem_h}" ]; then
    export PARAFAFT_NVSHMEM_INCLUDE="${_pf_shmem_h%/*}"                 # .../nvshmem/include
    export NVSHMEM_HOME="${PARAFAFT_NVSHMEM_INCLUDE%/include}"          # .../nvshmem
    echo ">> NVSHMEM include: ${PARAFAFT_NVSHMEM_INCLUDE}"
  else
    echo ">> WARN: nvshmem.h not found under ${NVHPC_ROOT}/Linux_x86_64"
  fi
fi

module list 2>&1 || true

# Enable CUDA-aware collectives in OpenMPI (lets device pointers pass through MPI).
export OMPI_MCA_opal_cuda_support=1
export OMPI_CXX=g++

build_gpu() {
  local extra=()
  local targets=(bench_r2c_cuda)

  # Compiler + CUDA-language flags differ by mode.
  #   * Plain cuFFT build: g++ as nvcc's host compiler. CMake's enable_language(CUDA)
  #     dispatches .cu files to nvcc, so we pass the nvcc host/arch flags.
  #   * cuFFTMp build: MUST use nvc++. cuFFTMp is only cleanly linkable through
  #     NVHPC's `-cudalib=cufftmp`, and CMakeLists takes that path ONLY when
  #     CMAKE_CXX_COMPILER_ID is NVHPC. With g++ it falls back to hunting for
  #     libcufftMp.so / libnvshmem_host.so and dies with "cuFFTMp/NVSHMEM not
  #     found". Under nvc++ the CUDA language is not enabled (nvc++ compiles .cu
  #     as CXX via -cuda), so the nvcc host/arch flags do not apply.
  local cxx=g++
  local mode_flags=(-DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_CUDA_ARCHITECTURES=80)

  if [ "${USE_CUFFTMP:-0}" = "1" ]; then
    extra+=(-DPARAFAFT_CUFFTMP=ON)
    targets+=(bench_r2c_cufftmp)
    cxx=nvc++                 # from the NVHPC module loaded above (on PATH)
    # Prepend the cuFFTMp drop-in include (and NVSHMEM) so <cufft.h> resolves to
    # the cuFFTMp variant ahead of the regular CUDA include — otherwise
    # cufftMp.h's version guard aborts the build. CMAKE_CXX_FLAGS lands early on
    # the compile line, ahead of nvc++'s -cudalib=cufftmp implicit paths.
    local mp_I=""
    [ -n "${PARAFAFT_CUFFTMP_INCLUDE:-}" ] && mp_I="-I${PARAFAFT_CUFFTMP_INCLUDE}"
    [ -n "${PARAFAFT_NVSHMEM_INCLUDE:-}" ] && mp_I="${mp_I:+${mp_I} }-I${PARAFAFT_NVSHMEM_INCLUDE}"
    mode_flags=(-DCMAKE_CXX_FLAGS="${mp_I}")   # nvcc host/arch flags inert on NVHPC path
    echo ">> cuFFTMp baseline ENABLED (USE_CUFFTMP=1) — building with nvc++; extra includes: ${mp_I:-<none>}"
  else
    extra+=(-DPARAFAFT_CUFFTMP=OFF)   # explicit: never inherit a stale cached ON
  fi

  echo ">> Configuring GPU build in ${PARAFAFT_GPU_BUILD} (CXX=${cxx})"
  cmake -S "${PARAFAFT_ROOT}" -B "${PARAFAFT_GPU_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAFAFT_BENCH=ON \
    -DPARAFAFT_CUDA=ON \
    -DCMAKE_CXX_COMPILER="${cxx}" \
    "${mode_flags[@]}" \
    "${extra[@]}" \
    || return 1

  echo ">> Building: ${targets[*]}"
  cmake --build "${PARAFAFT_GPU_BUILD}" --target "${targets[@]}" -j || return 1
  echo ">> Done: ${PARAFAFT_GPU_BUILD}/bench/{${targets[*]// /,}}"
}
