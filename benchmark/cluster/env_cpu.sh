#!/usr/bin/env bash
# =============================================================================
# ParaFaFT — CPU benchmark environment + build helper for PC2 Noctua 2.
#
# Usage (on a login node, from anywhere):
#     source benchmark/cluster/env_cpu.sh
#     build_cpu                # configure + build build_cpu/bench/bench_r2c
#
# The SLURM scripts (strong_scaling_cpu.slurm, weak_scaling_cpu.slurm) also
# source this file to load the runtime modules.
# =============================================================================

# --- Repo root (this file lives in <repo>/benchmark/cluster/) ----------------
_pf_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PARAFAFT_ROOT="$(cd "${_pf_here}/../.." && pwd)"
export PARAFAFT_CPU_BUILD="${PARAFAFT_ROOT}/build_cpu"

# --- Modules -----------------------------------------------------------------
# TODO(user): resolve the EXACT module strings on first login with
#   module spider OpenMPI ; module spider FFTW ; module spider FFTW.MPI
# g++ comes from the GCC toolchain bundled with the OpenMPI module below, so no
# separate compiler module is needed. OpenMPI is preferred over Intel MPI here
# (Intel MPI 2021.6-2021.9 has a documented collective-hang bug).
module reset 2>/dev/null || module purge 2>/dev/null || true

module load devel/CMake/3.31.3-GCCcore-14.2.0
module load mpi/OpenMPI/5.0.7-GCC-14.2.0
module load numlib/FFTW.MPI/3.3.10-gompi-2025a
module load data/HDF5/1.14.6-gompi-2025a

module list 2>&1 || true

# Make OpenMPI's C++ wrapper use g++, in case find_package(MPI) probes mpicxx.
export OMPI_CXX=g++

build_cpu() {
  # EasyBuild exports $EBROOTFFTW and $EBROOTFFTWMPI once the modules are loaded.
  # fftw3 (+ threads/omp) live under $EBROOTFFTW, the fftw3_mpi wrappers under
  # $EBROOTFFTWMPI — feed BOTH to CMAKE_PREFIX_PATH so find_library() resolves
  # everything. Guarded with :+ so an unset var is simply skipped.
  local prefix=""
  [ -n "${EBROOTFFTW:-}" ]    && prefix="${EBROOTFFTW}"
  [ -n "${EBROOTFFTWMPI:-}" ] && prefix="${prefix:+${prefix};}${EBROOTFFTWMPI}"

  echo ">> Configuring CPU build in ${PARAFAFT_CPU_BUILD}"
  cmake -S "${PARAFAFT_ROOT}" -B "${PARAFAFT_CPU_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAFAFT_BENCH=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    ${prefix:+-DCMAKE_PREFIX_PATH="${prefix}"} \
    ${EBROOTFFTW:+-DPARAFAFT_FFTW_LIB_DIR="${EBROOTFFTW}/lib"} \
    ${EBROOTFFTW:+-DPARAFAFT_FFTW_INCLUDE_DIR="${EBROOTFFTW}/include"} \
    || return 1

  echo ">> Building bench_r2c"
  cmake --build "${PARAFAFT_CPU_BUILD}" --target bench_r2c -j || return 1
  echo ">> Done: ${PARAFAFT_CPU_BUILD}/bench/bench_r2c"
}
