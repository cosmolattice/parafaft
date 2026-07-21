#!/usr/bin/env bash
# =============================================================================
# Launch the cuFFTMp COMPARISON — one INDEPENDENT SLURM job per (grid, GPUs)
# point, at EXACTLY the points where a ParaFaFT+cuFFT datapoint already exists
# (results/strong_gpu__bench_r2c_cuda.csv). Every job runs ParaFaFT+cuFFT and
# then the cuFFTMp baseline back-to-back in the same allocation, so the two
# curves are measured on the same nodes/GPUs — a clean matched comparison.
#
#     ./submit_strong_gpu_cufftmp.sh              # submit the sweep
#     DRY_RUN=1 ./submit_strong_gpu_cufftmp.sh    # print the plan, submit nothing
#
# ---------------------------------------------------------------------------
# Why USE_CUFFTMP=1 is exported BEFORE sourcing env_gpu.sh
# ---------------------------------------------------------------------------
# With USE_CUFFTMP=1, env_gpu.sh (a) loads the NVHPC module that provides
# cuFFTMp + NVSHMEM and (b) points PARAFAFT_GPU_BUILD at a SEPARATE build dir,
# build_gpu_cufftmp, configured with -DPARAFAFT_CUFFTMP=ON. That separation is
# deliberate: the plain build_gpu must never carry a sticky CUFFTMP=ON cache.
# The flag is threaded through to job_gpu.slurm via --export so the job sources
# env_gpu.sh into the very same build dir and runs both binaries from it.
#
# ---------------------------------------------------------------------------
# Point list — matches results/strong_gpu__bench_r2c_cuda.csv exactly (10 pts)
# ---------------------------------------------------------------------------
# The intended ParaFaFT sweep was larger; three points never produced data:
#   * N=1024 g=1  — OOM (cudaMemcpyAsync: invalid argument); a 26 GB grid does
#                   not fit one 40 GB A100. Cannot form a matched pair here.
#   * N=1536 g=4  — OOM (cufftPlanMany C2C backward, code 5) planning on 4 GPUs.
#                   Cannot form a matched pair here.
#   * N=2048 g=64 — never submitted (16 nodes, the "drop if tight" point). This
#                   is the only runnable gap; see the commented block below to
#                   add it (it has NO ParaFaFT baseline yet, so this job would
#                   also produce the missing ParaFaFT+cuFFT datapoint).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

DRY_RUN="${DRY_RUN:-0}"

# Must be set BEFORE sourcing env_gpu.sh — it gates the NVHPC module load and
# selects the build_gpu_cufftmp build dir. Exported so job_gpu.slurm inherits it.
export USE_CUFFTMP=1

[ "$DRY_RUN" = "1" ] || source ./env_gpu.sh

if [ "$DRY_RUN" != "1" ]; then
  # Guard on BOTH binaries: a stale ParaFaFT-only build must not satisfy the
  # check and yield a baseline-less sweep.
  NEEDED=(
    "${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cuda"
    "${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cufftmp"
  )
  missing() { for b in "${NEEDED[@]}"; do [ -x "$b" ] || return 0; done; return 1; }
  missing && build_gpu
  missing && { echo "ERROR: build failed — missing: $(for b in "${NEEDED[@]}"; do [ -x "$b" ] || echo -n "$b "; done)"; exit 1; }
  mkdir -p results/logs
fi

# "N GPUS TIME ITERS" — one row per existing ParaFaFT+cuFFT datapoint.
# ITERS mirror the original sweeps (N=1024 -> 50, N>=1536 -> 20). Walltimes are
# bumped vs. the ParaFaFT-only jobs because each job now runs a second binary;
# they are ceilings — the job exits as soon as the point finishes.
POINTS=(
  # N=1024 (~26 GB), from submit_strong_gpu.sh, minus the OOM g=1 point
  "1024  2 00:30:00 50"
  "1024  4 00:25:00 50"
  "1024  8 00:25:00 50"
  "1024 16 00:25:00 50"
  "1024 32 00:25:00 50"

  # N=1536 (~29 GB), from submit_strong_gpu_large.sh, minus the OOM g=4 point
  "1536  8 00:30:00 20"
  "1536 16 00:30:00 20"
  "1536 32 00:30:00 20"

  # N=2048 (~69 GB), from submit_strong_gpu_large.sh
  "2048 16 00:50:00 20"
  "2048 32 00:45:00 20"

  # --- the point that was left out (no ParaFaFT baseline yet; 16 nodes) --------
  # Uncomment to also fill N=2048 g=64. This job produces BOTH the missing
  # ParaFaFT+cuFFT datapoint and its cuFFTMp partner.
  # "2048 64 00:45:00 20"
)

echo ">> cuFFTMp comparison sweep — ${#POINTS[@]} points (USE_CUFFTMP=1)"
submitted=0
for pt in "${POINTS[@]}"; do
  read -r N G TIME ITERS <<< "$pt"
  NODES=$(( (G + 3) / 4 )); TPN=$(( G < 4 ? G : 4 ))
  echo "   N=${N} g=${G} nodes=${NODES} tpn=${TPN} time=${TIME} iters=${ITERS}"
  [ "$DRY_RUN" = "1" ] && continue

  sbatch -J "strgpu-N${N}-g${G}" \
    --nodes="${NODES}" --ntasks-per-node="${TPN}" --gres=gpu:a100:"${TPN}" --cpus-per-task=16 \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_GPUS="${G}",PF_TAG=strong_gpu,USE_CUFFTMP=1 \
    job_gpu.slurm
  submitted=$(( submitted + 1 ))
done

if [ "$DRY_RUN" = "1" ]; then
  echo ">> dry run: nothing submitted"
else
  echo ">> submitted ${submitted} cuFFTMp-comparison jobs. Watch: squeue -u \$USER"
  echo ">> when done: ./collect.sh   (writes results/strong_gpu__bench_r2c_cufftmp.csv)"
fi
