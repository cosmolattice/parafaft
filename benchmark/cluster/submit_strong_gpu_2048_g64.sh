#!/usr/bin/env bash
# =============================================================================
# One extra strong-scaling point: N=2048 on 64 GPUs (16 nodes) — the gap at the
# top of the sweep, and the confirmation point for the cuFFTMp multi-node
# plateau. At g16/g32 (4/8 nodes) cuFFTMp is flat (~0.85 s at N=2048) while
# ParaFaFT keeps scaling (1.04 -> 0.60 s); 16 nodes should either extend that
# plateau (cuFFTMp ~0.85, ParaFaFT drops again) or break the story.
#
#     ./submit_strong_gpu_2048_g64.sh              # submit the one job
#     DRY_RUN=1 ./submit_strong_gpu_2048_g64.sh    # print the plan, submit nothing
#     PF_QOS= ./submit_strong_gpu_2048_g64.sh      # submit at normal priority
#
# Unlike submit_strong_gpu_cufftmp.sh, this point has NO ParaFaFT+cuFFT
# datapoint yet, so PF_SKIP_PARAFAFT=0: the job runs BOTH binaries back-to-back
# in one allocation and fills results/strong_gpu__bench_r2c_cuda.csv AND
# results/strong_gpu__bench_r2c_cufftmp.csv at (64, 2048).
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"

DRY_RUN="${DRY_RUN:-0}"

# QoS — express for high scheduling priority. This is a 16-node job, so it may
# wait behind large reservations; express draws on the per-user monthly quota
# (check `pc2status`). Set PF_QOS= to fall back to normal priority if sbatch
# rejects it ("Invalid qos specification" / QOSMax*PerJobLimit).
PF_QOS="${PF_QOS-express}"
QOS_ARG=()
[ -n "$PF_QOS" ] && QOS_ARG=(--qos="$PF_QOS")

# Both binaries run here — needed to produce the missing ParaFaFT baseline.
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

# Single point: N=2048, 64 GPUs, 20 iters (matches the N>=1536 sweeps).
# Walltime: both binaries this time, so ~2x the ~22 s/binary fixed cost, plus
# 16-node NVSHMEM bootstrap (slowest to come up of any point here) and
# 20 x (cuFFT + cuFFTMp) compute. At g32 that compute was 20x(0.60+0.85)=29 s;
# g64 cuFFT should drop (~0.35), cuFFTMp stays ~0.85, so ~24 s. Budget ~90 s of
# real work; 00:04:00 gives >2x headroom for a possibly slow 16-node bring-up.
N=2048
G=64
TIME=00:04:00
ITERS=20

NODES=$(( (G + 3) / 4 )); TPN=$(( G < 4 ? G : 4 ))
echo ">> extra point: N=${N} g=${G} nodes=${NODES} tpn=${TPN} time=${TIME} iters=${ITERS}"
echo "   USE_CUFFTMP=1 PF_SKIP_PARAFAFT=0 (runs BOTH binaries) qos=${PF_QOS:-<default>}"

if [ "$DRY_RUN" = "1" ]; then
  echo ">> dry run: nothing submitted"
  exit 0
fi

sbatch -J "strgpu-N${N}-g${G}" "${QOS_ARG[@]}" \
  --nodes="${NODES}" --ntasks-per-node="${TPN}" --gres=gpu:a100:"${TPN}" --cpus-per-task=16 \
  --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
  --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_GPUS="${G}",PF_TAG=strong_gpu,USE_CUFFTMP=1,PF_SKIP_PARAFAFT=0 \
  job_gpu.slurm

echo ">> submitted. Watch: squeue -u \$USER"
echo ">> when done: ./collect.sh   (adds (64,2048) to both cuda and cufftmp CSVs)"
