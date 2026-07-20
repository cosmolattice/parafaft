#!/usr/bin/env bash
# =============================================================================
# Launch STRONG-scaling GPU jobs — one INDEPENDENT SLURM job per (grid, GPUs)
# point. 1 rank/GPU; nodes = ceil(GPUs/4). Short per-point jobs schedule far
# more easily on the small `gpu` partition than one 8-node reservation.
#
#     ./submit_strong_gpu.sh                 # ParaFaFT only
#     USE_CUFFTMP=1 ./submit_strong_gpu.sh   # also run the cuFFTMp baseline
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
source ./env_gpu.sh

USE_CUFFTMP="${USE_CUFFTMP:-0}"

# Every binary this sweep will run must exist, else the jobs start and silently
# skip what is missing. Checking only bench_r2c_cuda would let a pre-existing
# ParaFaFT-only build satisfy the guard and produce a sweep with no baseline.
NEEDED=("${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cuda")
[ "$USE_CUFFTMP" = "1" ] && NEEDED+=("${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cufftmp")

missing() { for b in "${NEEDED[@]}"; do [ -x "$b" ] || return 0; done; return 1; }
missing && build_gpu
missing && { echo "ERROR: build failed — missing: $(for b in "${NEEDED[@]}"; do [ -x "$b" ] || echo -n "$b "; done)"; exit 1; }
mkdir -p results/logs

# "N GPUS TIME ITERS"  — N=1024 (~26 GB) fits one 40 GB A100, spans 1->32 GPUs.
POINTS=(
  "1024 1 00:30:00 50" "1024 2 00:20:00 50" "1024 4 00:15:00 50"
  "1024 8 00:15:00 50" "1024 16 00:15:00 50" "1024 32 00:15:00 50"
  # --- optional larger grid (~87 GB, needs >=3 GPUs): ---
  # "1536 4 00:30:00 50" "1536 8 00:20:00 50" "1536 16 00:15:00 50" "1536 32 00:15:00 50"
)

for pt in "${POINTS[@]}"; do
  read -r N G TIME ITERS <<< "$pt"
  NODES=$(( (G + 3) / 4 )); TPN=$(( G < 4 ? G : 4 ))
  sbatch -J "strgpu-N${N}-g${G}" \
    --nodes="${NODES}" --ntasks-per-node="${TPN}" --gres=gpu:a100:"${TPN}" --cpus-per-task=16 \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_GPUS="${G}",PF_TAG=strong_gpu,USE_CUFFTMP="${USE_CUFFTMP}" \
    job_gpu.slurm
done
echo ">> submitted ${#POINTS[@]} strong-GPU jobs (USE_CUFFTMP=${USE_CUFFTMP}). Watch: squeue -u \$USER"
echo ">> when done: ./collect.sh"
