#!/usr/bin/env bash
# =============================================================================
# Launch STRONG-scaling CPU jobs for N=2048 ONLY — both configurations:
#   - pure MPI   (128 ranks/node x 1 thread),      PF_TAG=strong_cpu
#   - hybrid     (4 ranks/node   x 32 threads),    PF_TAG=strong_cpu_hybrid
#
# A CPU counterpart to the N=2048 GPU sweep (submit_strong_gpu_large.sh), for a
# like-for-like comparison at the same grid. The two PF_TAGs match the existing
# sweeps, so collect.sh appends these points to results/strong_cpu__bench_r2c.csv
# and results/strong_cpu_hybrid__bench_r2c.csv, and plot.py draws N=2048 as its
# own marker alongside N=1536 / N=4096 — nothing already collected is rerun.
#
#     ./submit_strong_cpu_2048.sh              # submit both sweeps
#     DRY_RUN=1 ./submit_strong_cpu_2048.sh    # print the plan, submit nothing
#     PURE=0 ./submit_strong_cpu_2048.sh       # hybrid only
#     HYBRID=0 ./submit_strong_cpu_2048.sh     # pure MPI only
#
# Memory: the benchmark needs ~24*N^3 B total (parafaft + the FFTW-MPI
# reference it also times). For N=2048 that is ~206 GB, which is exactly the
# per-node ceiling the existing sweeps already run at (4096 on its 8-node
# floor). The sweep therefore starts at 2 nodes (~103 GB/node) for headroom and
# spans to 64 full nodes.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"          # -> benchmark/cluster

DRY_RUN="${DRY_RUN:-0}"
PURE="${PURE:-1}"        # launch the pure-MPI sweep
HYBRID="${HYBRID:-1}"    # launch the hybrid sweep
[ "$DRY_RUN" = "1" ] || source ./env_cpu.sh

CORES_PER_NODE=128       # TODO: must match the node. Earlier runs reported 192 logical CPUs.

if [ "$DRY_RUN" != "1" ]; then
  BIN="${PARAFAFT_CPU_BUILD}/bench/bench_r2c"
  [ -x "$BIN" ] || build_cpu                      # build ONCE up front (jobs never build -> no races)
  [ -x "$BIN" ] || { echo "ERROR: build failed"; exit 1; }
  mkdir -p results/logs
fi

# "NODES TIME ITERS" — fixed grid N=2048, growing node count. Generous walltimes
# at low node counts (where each rank does more work), short at high counts.
POINTS=(
  "2  01:00:00 20"
  "4  00:40:00 20"
  "8  00:30:00 20"
  "16 00:20:00 20"
  "32 00:15:00 20"
  "64 00:15:00 20"
)

N=2048

# Submit one sweep. Args: <ranks_per_node> <tag> <job-prefix>
submit_sweep() {
  local rpn="$1" tag="$2" prefix="$3"
  local cpt=$(( CORES_PER_NODE / rpn ))
  echo ">> ${tag}: ${rpn} ranks/node x ${cpt} threads, N=${N}, nodes 2..64"
  for pt in "${POINTS[@]}"; do
    read -r NODES TIME ITERS <<< "$pt"
    local mem_per_node=$(( 24 * N / 1000 * N * N / NODES / 1000000 ))  # ~GB/node
    echo "   ${tag} N=${N} nodes=${NODES}: ~${mem_per_node} GB/node, walltime ${TIME}"
    [ "$DRY_RUN" = "1" ] && continue
    sbatch -J "${prefix}-N${N}-n${NODES}" \
      --nodes="${NODES}" --ntasks-per-node="${rpn}" --cpus-per-task="${cpt}" \
      --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
      --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_TAG="${tag}",PF_RANKS_PER_NODE="${rpn}" \
      job_cpu.slurm
  done
}

njobs=0
if [ "$PURE" = "1" ]; then
  submit_sweep "${CORES_PER_NODE}" strong_cpu strcpu
  njobs=$(( njobs + ${#POINTS[@]} ))
fi
if [ "$HYBRID" = "1" ]; then
  submit_sweep 4 strong_cpu_hybrid strcpuh
  njobs=$(( njobs + ${#POINTS[@]} ))
fi

if [ "$DRY_RUN" = "1" ]; then
  echo ">> dry run: nothing submitted"
else
  echo ">> submitted ${njobs} N=2048 CPU jobs. Watch: squeue -u \$USER"
  echo ">> when done: ./collect.sh   (appends to strong_cpu / strong_cpu_hybrid CSVs)"
fi
