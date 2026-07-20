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
fi

module list 2>&1 || true

# Enable CUDA-aware collectives in OpenMPI (lets device pointers pass through MPI).
export OMPI_MCA_opal_cuda_support=1
export OMPI_CXX=g++

build_gpu() {
  local extra=()
  local targets=(bench_r2c_cuda)

  # cuFFTMp baseline is opt-in: it needs cuFFTMp + NVSHMEM, and the CMake option
  # hard-errors if they are not found. Enable with USE_CUFFTMP=1.
  if [ "${USE_CUFFTMP:-0}" = "1" ]; then
    extra+=(-DPARAFAFT_CUFFTMP=ON)
    targets+=(bench_r2c_cufftmp)
    echo ">> cuFFTMp baseline ENABLED (USE_CUFFTMP=1)"
  else
    extra+=(-DPARAFAFT_CUFFTMP=OFF)   # explicit: never inherit a stale cached ON
  fi

  echo ">> Configuring GPU build in ${PARAFAFT_GPU_BUILD}"
  cmake -S "${PARAFAFT_ROOT}" -B "${PARAFAFT_GPU_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAFAFT_BENCH=ON \
    -DPARAFAFT_CUDA=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    "${extra[@]}" \
    || return 1

  echo ">> Building: ${targets[*]}"
  cmake --build "${PARAFAFT_GPU_BUILD}" --target "${targets[@]}" -j || return 1
  echo ">> Done: ${PARAFAFT_GPU_BUILD}/bench/{${targets[*]// /,}}"
}
