#!/usr/bin/env bash
# =============================================================================
# Launch STRONG-scaling GPU jobs for the LARGE grids (N=1536, N=2048).
#
# Companion to submit_strong_gpu.sh, which covers N=1024 from 1 GPU. These
# grids do not fit on a small number of GPUs, so each sweep starts at the
# smallest GPU count whose per-GPU memory fits, and doubles from there.
# PF_TAG is strong_gpu, the same as the N=1024 sweep, so collect.sh merges
# everything into one CSV and plot.py draws one curve per N (marker = grid N).
#
#     ./submit_strong_gpu_large.sh              # submit the sweep
#     DRY_RUN=1 ./submit_strong_gpu_large.sh    # print the plan, submit nothing
#     MEM_BUDGET_MB=30000 ./submit_strong_gpu_large.sh   # stricter memory cap
#
# ---------------------------------------------------------------------------
# Memory model (why the sweeps start where they do)
# ---------------------------------------------------------------------------
# Per rank the CUDA backend holds three device buffers, each about A/P bytes,
# where A = N^2 * (N/2+1) * 16 is the whole complex grid and P the GPU count:
#
#     d_data (user buffer)  +  scratch_b_ (ping-pong)  +  pack_buffer_
#
# On top of that cuFFT keeps plan work areas worth roughly 1.6x the local
# array, so the real footprint is about 4.6 * (A/P).
#
# That factor is measured, not guessed. At N=1024 on one GPU the run got past
# plan creation and both internal buffers (17.2 GB) and then failed allocating
# d_data (8.6 GB) on a 40 GB card, which puts the work areas near 13.7 GB for
# an 8.6 GB local array. The same model says N=1024 on 2 GPUs sits at 50% of
# the card, which is the configuration that has always worked.
#
#     N=1536:  A = 29.0 GB   ->  4 GPUs = 33.4 GB/GPU (85%)   <- start here
#     N=2048:  A = 68.8 GB   -> 16 GPUs = 19.8 GB/GPU (50%)   <- start here
#
# N=2048 on 8 GPUs would need 8.60 GB per rank for the local array, the exact
# figure that already ran out of memory at N=1024 on one GPU, so it is left
# out rather than burned as a failed job.
#
# The estimate is recomputed below for every point and any point over
# MEM_BUDGET_MB is skipped with a message, so a mistaken sweep costs nothing.
# If a point does run out anyway, the benchmark now reports the allocation
# that failed and the free/total memory instead of a misleading CUDA error.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

# DRY_RUN prints the plan and its memory estimates without touching the module
# system or the build, so the sweep can be sanity-checked off the cluster.
DRY_RUN="${DRY_RUN:-0}"
[ "$DRY_RUN" = "1" ] || source ./env_gpu.sh

# Per-GPU memory budget. Deliberately below the 40 GB the card advertises: the
# one measured failure (N=1024 on 1 GPU) sat at ~100% of nominal, so nominal is
# where it breaks, not where it fits. ~85% keeps a margin for the fact that the
# 1.6x cuFFT work-area factor is an estimate from a single data point.
MEM_BUDGET_MB="${MEM_BUDGET_MB:-34000}"

if [ "$DRY_RUN" != "1" ]; then
  BIN="${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cuda"
  [ -x "$BIN" ] || build_gpu
  [ -x "$BIN" ] || { echo "ERROR: build failed — missing $BIN"; exit 1; }
  mkdir -p results/logs
fi

# "N GPUS TIME ITERS". Walltimes are ceilings; the jobs exit as soon as the
# sweep point finishes, and setup (host fill + cuFFT planning) dominates.
POINTS=(
  # N=1536 — 4 GPUs is one whole node, so this point is also the largest
  # grid measured without any inter-node traffic.
  "1536  4 00:30:00 20"
  "1536  8 00:20:00 20"
  "1536 16 00:20:00 20"
  "1536 32 00:20:00 20"

  # N=2048 — 16 GPUs (4 nodes) is the smallest count that fits.
  "2048 16 00:40:00 20"
  "2048 32 00:30:00 20"
  "2048 64 00:30:00 20"   # 16 nodes: by far the priciest point, drop if tight

  # "1536 64 00:20:00 20"
)

# Estimated per-GPU footprint in MiB, as a multiple of the local array A/P
# (A = N^2*(N/2+1)*16 bytes): three device buffers plus ~1.6x cuFFT work area.
# On a single rank nothing redistributes, so the pack buffer is not allocated
# and only two buffers are live — the reason N=1024 now fits on one GPU.
est_mem_mb() {
  local N=$1 P=$2
  local A=$(( N * N * (N / 2 + 1) * 16 ))
  local factor10=46          # 3 buffers + 1.6 work area
  [ "$P" -eq 1 ] && factor10=36   # 2 buffers + 1.6 work area
  echo $(( factor10 * A / (10 * P) / 1048576 ))
}

echo ">> memory budget: ${MEM_BUDGET_MB} MiB/GPU (override with MEM_BUDGET_MB=)"
submitted=0
skipped=0
for pt in "${POINTS[@]}"; do
  read -r N G TIME ITERS <<< "$pt"
  NODES=$(( (G + 3) / 4 )); TPN=$(( G < 4 ? G : 4 ))
  MEM=$(est_mem_mb "$N" "$G")
  PCT=$(( 100 * MEM / MEM_BUDGET_MB ))

  if [ "$MEM" -gt "$MEM_BUDGET_MB" ]; then
    echo "   SKIP N=${N} g=${G}: needs ~${MEM} MiB/GPU (${PCT}% of budget) — too large"
    skipped=$(( skipped + 1 ))
    continue
  fi
  echo "   N=${N} g=${G} nodes=${NODES}: ~${MEM} MiB/GPU (${PCT}% of budget)"
  [ "$DRY_RUN" = "1" ] && continue

  sbatch -J "strgpu-N${N}-g${G}" \
    --nodes="${NODES}" --ntasks-per-node="${TPN}" --gres=gpu:a100:"${TPN}" --cpus-per-task=16 \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_GPUS="${G}",PF_TAG=strong_gpu,USE_CUFFTMP=0 \
    job_gpu.slurm
  submitted=$(( submitted + 1 ))
done

if [ "$DRY_RUN" = "1" ]; then
  echo ">> dry run: nothing submitted (${skipped} point(s) would be skipped)"
else
  echo ">> submitted ${submitted} large-grid strong-GPU jobs (${skipped} skipped)."
  echo ">> watch: squeue -u \$USER    then: ./collect.sh && python3 plot.py"
fi
