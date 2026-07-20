#!/usr/bin/env bash
# =============================================================================
# ParaFaFT — single-point CPU benchmark job (3D R2C, FFTW backend).
# NOT submitted directly — the submit_*_cpu.sh launchers sbatch it once per
# sweep point, passing resources (--nodes/--ntasks-per-node/--cpus-per-task/
# --time/-J) and the PF_* variables on the command line.
# Each job also times FFTW-MPI (fftw_mean/fftw_std columns) for comparison.
# =============================================================================
#SBATCH -p normal
#SBATCH --output=parafaft-%x-%j.out

set -euo pipefail
: "${PF_N:?PF_N required}"; : "${PF_ITERS:?PF_ITERS required}"; : "${PF_TAG:?PF_TAG required}"

RANKS_PER_NODE="${PF_RANKS_PER_NODE:?PF_RANKS_PER_NODE required}"
# Threads per rank come from what SLURM actually allocated (set by the launcher's
# --cpus-per-task), not from a hardcoded core count. bench_r2c honours
# OMP_NUM_THREADS regardless of whether FFTW is linked against fftw3_omp or
# fftw3_threads, so this is the single source of truth for the thread count.
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPUS_PER_TASK}"
export OMP_PROC_BIND=close OMP_PLACES=cores

export PARAFAFT_ROOT="${PARAFAFT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PARAFAFT_ROOT}/benchmark/cluster/env_cpu.sh"

BIN="${PARAFAFT_CPU_BUILD}/bench/bench_r2c"
[ -x "$BIN" ] || { echo "ERROR: $BIN missing — run the launcher (it builds once up front)"; exit 1; }

NODES="${SLURM_JOB_NUM_NODES}"
TASKS=$(( NODES * RANKS_PER_NODE ))

# Unique per-job working dir so concurrent jobs never clobber each other's CSV.
WORK="${PARAFAFT_ROOT}/benchmark/cluster/results/${PF_TAG}/N${PF_N}_n${NODES}_${SLURM_JOB_ID}"
mkdir -p "$WORK"; cd "$WORK"

echo "=== ${PF_TAG}: N=${PF_N} nodes=${NODES} ranks=${TASKS} threads=${OMP_NUM_THREADS} iters=${PF_ITERS} ==="
srun --ntasks="${TASKS}" --ntasks-per-node="${RANKS_PER_NODE}" --cpus-per-task="${CPUS_PER_TASK}" \
     "$BIN" "${PF_N}" 3 "${PF_ITERS}"
echo ">> wrote ${WORK}/bench_r2c.csv"
