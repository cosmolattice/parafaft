#!/usr/bin/env bash
# =============================================================================
# Launch WEAK-scaling GPU jobs — one INDEPENDENT SLURM job per point.
# Constant work/GPU: base 512^3 on 1 GPU, N = round_even(512*nGPU^(1/3)).
#
#     ./submit_weak_gpu.sh
#     USE_CUFFTMP=1 ./submit_weak_gpu.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
source ./env_gpu.sh

USE_CUFFTMP="${USE_CUFFTMP:-0}"

# See submit_strong_gpu.sh: guard on every binary the sweep runs, not just the
# ParaFaFT one, so a stale ParaFaFT-only build cannot yield a baseline-less sweep.
NEEDED=("${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cuda")
[ "$USE_CUFFTMP" = "1" ] && NEEDED+=("${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cufftmp")

missing() { for b in "${NEEDED[@]}"; do [ -x "$b" ] || return 0; done; return 1; }
missing && build_gpu
missing && { echo "ERROR: build failed — missing: $(for b in "${NEEDED[@]}"; do [ -x "$b" ] || echo -n "$b "; done)"; exit 1; }
mkdir -p results/logs

# "GPUS N TIME ITERS"  — ~1.34e8 points/GPU throughout.
POINTS=(
  "1 512 00:20:00 50" "2 646 00:20:00 50" "4 814 00:20:00 50"
  "8 1024 00:20:00 50" "16 1290 00:20:00 50" "32 1626 00:25:00 50"
)

for pt in "${POINTS[@]}"; do
  read -r G N TIME ITERS <<< "$pt"
  NODES=$(( (G + 3) / 4 )); TPN=$(( G < 4 ? G : 4 ))
  sbatch -J "weakgpu-N${N}-g${G}" \
    --nodes="${NODES}" --ntasks-per-node="${TPN}" --gres=gpu:a100:"${TPN}" --cpus-per-task=16 \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_GPUS="${G}",PF_TAG=weak_gpu,USE_CUFFTMP="${USE_CUFFTMP}" \
    job_gpu.slurm
done
echo ">> submitted ${#POINTS[@]} weak-GPU jobs (USE_CUFFTMP=${USE_CUFFTMP}). Watch: squeue -u \$USER"
echo ">> when done: ./collect.sh"
