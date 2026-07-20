#!/usr/bin/env bash
# =============================================================================
# ParaFaFT — single-point GPU benchmark job (3D R2C, cuFFT backend).
# NOT submitted directly — the submit_*_gpu.sh launchers sbatch it once per
# sweep point, passing resources (--nodes/--ntasks-per-node/--gres/--time/-J)
# and the PF_* variables. Set USE_CUFFTMP=1 to also run the cuFFTMp baseline.
# =============================================================================
#SBATCH -p gpu
#SBATCH --output=parafaft-%x-%j.out

set -euo pipefail
: "${PF_N:?PF_N required}"; : "${PF_ITERS:?PF_ITERS required}"
: "${PF_GPUS:?PF_GPUS required}"; : "${PF_TAG:?PF_TAG required}"

export PARAFAFT_ROOT="${PARAFAFT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PARAFAFT_ROOT}/benchmark/cluster/env_gpu.sh"

BIN="${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cuda"
[ -x "$BIN" ] || { echo "ERROR: $BIN missing — run the launcher (it builds once up front)"; exit 1; }

G="${PF_GPUS}"
TPN=$(( G < 4 ? G : 4 ))          # ranks (=GPUs) per node; bench binds local_rank % num_devices

WORK="${PARAFAFT_ROOT}/benchmark/cluster/results/${PF_TAG}/N${PF_N}_g${G}_${SLURM_JOB_ID}"
mkdir -p "$WORK"; cd "$WORK"

echo "=== ${PF_TAG}: N=${PF_N} gpus=${G} nodes=${SLURM_JOB_NUM_NODES} tpn=${TPN} iters=${PF_ITERS} ==="
srun --ntasks="${G}" --ntasks-per-node="${TPN}" "$BIN" "${PF_N}" 3 "${PF_ITERS}"

MP="${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cufftmp"
if [ "${USE_CUFFTMP:-0}" = "1" ]; then
  if [ -x "$MP" ]; then
    echo "--- cuFFTMp baseline ---"
    srun --ntasks="${G}" --ntasks-per-node="${TPN}" "$MP" "${PF_N}" 3 "${PF_ITERS}"
  else
    echo "WARN: USE_CUFFTMP=1 but $MP missing; skipping cuFFTMp"
  fi
fi
echo ">> wrote ${WORK}/"
