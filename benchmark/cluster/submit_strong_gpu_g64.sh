#!/usr/bin/env bash
# =============================================================================
# Extend every GPU strong-scaling line to 64 GPUs (16 nodes) — one job per N.
# Adds the g=64 point to all three curves (N=1024, 1536, 2048) for BOTH
# backends. This is the top of the sweep and the confirmation point for the
# cuFFTMp multi-node plateau: at g16/g32 (4/8 nodes) cuFFTMp is flat while
# ParaFaFT keeps scaling, so 16 nodes decides whether that holds.
#
#     ./submit_strong_gpu_g64.sh              # submit all three jobs
#     DRY_RUN=1 ./submit_strong_gpu_g64.sh    # print the plan, submit nothing
#     PF_QOS= ./submit_strong_gpu_g64.sh      # submit at normal priority
#
# No point here has a datapoint yet in EITHER CSV, so PF_SKIP_PARAFAFT=0: each
# job runs both binaries back-to-back in one allocation and fills both
# results/strong_gpu__bench_r2c_cuda.csv and ...cufftmp.csv at (64, N).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

DRY_RUN="${DRY_RUN:-0}"

# QoS — express for high scheduling priority. These are 16-node jobs and may
# wait behind large reservations; express draws on the per-user monthly quota
# (check `pc2status`). Set PF_QOS= to fall back to normal priority if sbatch
# rejects it ("Invalid qos specification" / QOSMax*PerJobLimit).
PF_QOS="${PF_QOS-express}"
QOS_ARG=()
[ -n "$PF_QOS" ] && QOS_ARG=(--qos="$PF_QOS")

# Both binaries run — needed to produce the missing ParaFaFT baselines at g64.
export USE_CUFFTMP=1
export PF_SKIP_PARAFAFT=0

[ "$DRY_RUN" = "1" ] || source ./env_gpu.sh

if [ "$DRY_RUN" != "1" ]; then
  NEEDED=(
    "${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cuda"
    "${PARAFAFT_GPU_BUILD}/bench/bench_r2c_cufftmp"
  )
  missing() { for b in "${NEEDED[@]}"; do [ -x "$b" ] || return 0; done; return 1; }
  missing && build_gpu
  missing && { echo "ERROR: build failed — missing: $(for b in "${NEEDED[@]}"; do [ -x "$b" ] || echo -n "$b "; done)"; exit 1; }
  mkdir -p results/logs
fi

# "N GPUS TIME ITERS" — g=64 for each existing N line. ITERS=20 (matches the
# N>=1536 sweeps). All grids fit trivially at 64 A100s: per-GPU real slab is
# ~0.4 GB (N=1024), ~0.45 GB (1536), ~1.1 GB (2048).
#
# TIME: kept short. Each job pays ~22 s/binary fixed cost x2 binaries (host
# fill + H2D + plan + warmup + srun launch) plus a 16-node NVSHMEM bring-up
# (the slowest bootstrap of any point here) plus 20 x (cuFFT + cuFFTMp). At g32
# that compute was 0.15 s (1024) / 0.65 s (1536) / 1.45 s (2048) per iter; g64
# only shrinks it. So real work is ~50 s (1024), ~58 s (1536), ~70 s (2048).
# Limits below give >2x headroom for a possibly slow bring-up; Slurm bills
# elapsed, not the limit, so the margin is free and only affects backfill.
POINTS=(
  "1024 64 00:03:00 20"
  "1536 64 00:03:00 20"
  "2048 64 00:04:00 20"
)

echo ">> g64 completion — ${#POINTS[@]} points, BOTH binaries (qos=${PF_QOS:-<default>})"
submitted=0
for pt in "${POINTS[@]}"; do
  read -r N G TIME ITERS <<< "$pt"
  NODES=$(( (G + 3) / 4 )); TPN=$(( G < 4 ? G : 4 ))
  echo "   N=${N} g=${G} nodes=${NODES} tpn=${TPN} time=${TIME} iters=${ITERS}"
  [ "$DRY_RUN" = "1" ] && continue

  sbatch -J "strgpu-N${N}-g${G}" "${QOS_ARG[@]}" \
    --nodes="${NODES}" --ntasks-per-node="${TPN}" --gres=gpu:a100:"${TPN}" --cpus-per-task=16 \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_GPUS="${G}",PF_TAG=strong_gpu,USE_CUFFTMP=1,PF_SKIP_PARAFAFT=0 \
    job_gpu.slurm
  submitted=$(( submitted + 1 ))
done

if [ "$DRY_RUN" = "1" ]; then
  echo ">> dry run: nothing submitted"
else
  echo ">> submitted ${submitted} g64 jobs. Watch: squeue -u \$USER"
  echo ">> when done: ./collect.sh   (adds (64,N) to both cuda and cufftmp CSVs)"
fi
