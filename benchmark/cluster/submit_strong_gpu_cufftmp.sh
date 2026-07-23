#!/usr/bin/env bash
# =============================================================================
# Launch the cuFFTMp COMPARISON — one INDEPENDENT SLURM job per (grid, GPUs)
# point still MISSING from results/strong_gpu__bench_r2c_cufftmp.csv.
#
# This is a BACK-FILL run, not the full sweep. Every point below already has its
# ParaFaFT+cuFFT datapoint in results/strong_gpu__bench_r2c_cuda.csv, so the
# jobs run ONLY the cuFFTMp binary (PF_SKIP_PARAFAFT=1). Each binary pays
# ~20-25 s of fixed cost (host fill + H2D + plan + warmup + srun launch)
# independent of iteration count; dropping the redundant ParaFaFT run roughly
# halves the walltime, which is what makes 2-minute limits viable.
#
#     ./submit_strong_gpu_cufftmp.sh              # submit the sweep
#     DRY_RUN=1 ./submit_strong_gpu_cufftmp.sh    # print the plan, submit nothing
#     PF_QOS= ./submit_strong_gpu_cufftmp.sh      # submit at normal priority
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
# What is still missing (cuda CSV has 10 points, cufftmp CSV has 5)
# ---------------------------------------------------------------------------
# Collected already: (4,1024) (8,1024) (16,1024) (16,1536) (16,2048).
# Missing, and therefore submitted below:
#   * N=1024 g=2, N=1536 g=8 — these DID run (jobs 33712214 / 33712219): the
#     ParaFaFT half succeeded, then cuFFTMp aborted in NVSHMEM with
#     "cuMemCreate failed" (CUDA out of memory) while growing the symmetric
#     heap. That is a memory wall, not a time limit, so a plain resubmit will
#     most likely fail again — but it fails in well under a minute, so it costs
#     almost nothing to confirm. Drop these two lines if you need the slots.
#   * N=1024 g=32, N=1536 g=32, N=2048 g=32 — queued but cancelled before they
#     ran. These are the ones that should actually produce data.
#
# Points with no ParaFaFT baseline at all, hence not here:
#   * N=1024 g=1  — OOM (cudaMemcpyAsync: invalid argument); 26 GB on one 40 GB
#                   A100.
#   * N=1536 g=4  — OOM (cufftPlanMany C2C backward, code 5).
#   * N=2048 g=64 — never submitted (16 nodes). If you add it back, it needs
#                   PF_SKIP_PARAFAFT=0 so the job also produces the ParaFaFT
#                   datapoint, and a longer limit (~00:04:00).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# QoS — `express` for very high scheduling priority (PC2 docs, "Quality-of-
# Service (QoS) and Job Priorities"). Worth it here: these are five tiny jobs
# and the whole point is to get them past a queue of 8-node reservations before
# the deadline. Express draws on a PER-USER MONTHLY quota that does not
# replenish until the 1st of next month, so check what is left first:
#
#     pc2status
#
# The back-fill is cheap against that quota — ~0.7 GPU-h expected across all
# five jobs (Slurm accounts elapsed time, not the requested --time), so the
# three 32-GPU points cost far less than a single full-length production run.
#
# Set PF_QOS= (empty) to fall back to normal priority. Do that if sbatch
# rejects the submission with "Invalid qos specification" (quota exhausted or
# express not granted for this account) or a QOSMax*PerJobLimit error — the
# jobs are otherwise identical and will still run, just later.
# ---------------------------------------------------------------------------
PF_QOS="${PF_QOS-express}"
QOS_ARG=()
[ -n "$PF_QOS" ] && QOS_ARG=(--qos="$PF_QOS")

# Must be set BEFORE sourcing env_gpu.sh — it gates the NVHPC module load and
# selects the build_gpu_cufftmp build dir. Exported so job_gpu.slurm inherits it.
export USE_CUFFTMP=1

# Every point below already has its ParaFaFT+cuFFT datapoint, so the jobs run
# the cuFFTMp binary only. Set PF_SKIP_PARAFAFT=0 to restore matched pairs.
export PF_SKIP_PARAFAFT="${PF_SKIP_PARAFAFT:-1}"

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

# "N GPUS TIME ITERS" — one row per MISSING cuFFTMp datapoint.
#
# ITERS: 25 everywhere (down from 50 at N=1024). The timed loop is not what
# costs walltime — see below — so a uniform 25 keeps the error bars respectable
# without lengthening any job appreciably.
#
# TIME: derived from measurements, not padded guesses. Decomposing the three
# completed matched-pair jobs at g=16 (sacct, jobs 337122xx) into timed compute
# = iters x (cuFFT_mean + cuFFTMp_mean) and the remainder:
#
#     N      elapsed   timed compute   fixed cost (2 binaries)
#     1024   1:05      12 s            ~53 s
#     1536   0:57      17 s            ~40 s
#     2048   1:27      38 s            ~49 s
#
# So ~45 s of every job was setup, ~22 s per binary: host fill of the local
# slab (a per-element index loop, the largest term), H2D upload, cuFFT/cuFFTMp
# plan creation, one warmup transform, and the srun launch. Running cuFFTMp
# alone removes one full copy of that.
#
# Per-point budget = single-binary fixed cost + 25 x cuFFTMp_mean, with the
# cuFFTMp mean bounded above by the measured ParaFaFT mean at the same point
# (cuFFTMp ran 0.72-0.81x ParaFaFT at every point measured so far):
#
#     N     g    fill (GB/rank)   est. fixed   25 x t   total
#     1024   2   13.4             ~40 s        ~3 s     ~43 s
#     1024  32    0.84            ~14 s        ~2 s     ~16 s
#     1536   8    3.6             ~20 s       ~14 s     ~34 s
#     1536  32    0.91            ~14 s        ~7 s     ~21 s
#     2048  32    2.15            ~17 s       ~15 s     ~32 s
#
# 00:02:00 covers the worst of these with ~3x headroom, so one limit for all.
# Going lower buys nothing: at this size SLURM backfills on node count, not on
# walltime, and the g=32 points need 8 nodes each either way.
POINTS=(
  # Cancelled before running — these are the ones that should yield data.
  "1024 32 00:02:00 25"
  "1536 32 00:02:00 25"
  "2048 32 00:02:00 25"

  # Ran and died in NVSHMEM with "cuMemCreate failed" (CUDA OOM growing the
  # symmetric heap). Expect a repeat; they abort in <1 min. Drop if slots are
  # tight.
  "1024  2 00:02:00 25"
  "1536  8 00:02:00 25"
)

echo ">> cuFFTMp back-fill — ${#POINTS[@]} missing points (USE_CUFFTMP=1, PF_SKIP_PARAFAFT=${PF_SKIP_PARAFAFT}, qos=${PF_QOS:-<default>})"
submitted=0
for pt in "${POINTS[@]}"; do
  read -r N G TIME ITERS <<< "$pt"
  NODES=$(( (G + 3) / 4 )); TPN=$(( G < 4 ? G : 4 ))
  echo "   N=${N} g=${G} nodes=${NODES} tpn=${TPN} time=${TIME} iters=${ITERS}"
  [ "$DRY_RUN" = "1" ] && continue

  sbatch -J "strgpu-N${N}-g${G}" "${QOS_ARG[@]}" \
    --nodes="${NODES}" --ntasks-per-node="${TPN}" --gres=gpu:a100:"${TPN}" --cpus-per-task=16 \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_GPUS="${G}",PF_TAG=strong_gpu,USE_CUFFTMP=1,PF_SKIP_PARAFAFT="${PF_SKIP_PARAFAFT}" \
    job_gpu.slurm
  submitted=$(( submitted + 1 ))
done

if [ "$DRY_RUN" = "1" ]; then
  echo ">> dry run: nothing submitted"
else
  echo ">> submitted ${submitted} cuFFTMp-comparison jobs. Watch: squeue -u \$USER"
  echo ">> when done: ./collect.sh   (writes results/strong_gpu__bench_r2c_cufftmp.csv)"
fi
