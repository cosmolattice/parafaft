#!/usr/bin/env bash
# =============================================================================
# Launch STRONG-scaling CPU jobs — HYBRID MPI+OpenMP (4 ranks/node x 32 threads
# = full 128-core node). Same grids/node-counts as submit_strong_cpu.sh so the
# two can be compared at matched node counts (plot.py aligns them by total cores).
#
#     ./submit_strong_cpu_hybrid.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
source ./env_cpu.sh

BIN="${PARAFAFT_CPU_BUILD}/bench/bench_r2c"
[ -x "$BIN" ] || build_cpu                        # same binary as pure MPI; OMP is runtime-controlled
[ -x "$BIN" ] || { echo "ERROR: build failed"; exit 1; }

CORES_PER_NODE=128        # TODO: must match the node. Earlier runs reported 192 logical CPUs.
RANKS_PER_NODE=4          # => OMP_NUM_THREADS = CORES_PER_NODE/4 threads per rank
mkdir -p results/logs

# "N NODES TIME ITERS" — mirror submit_strong_cpu.sh for a like-for-like comparison.
POINTS=(
  "1536 1 00:45:00 50" "1536 2 00:30:00 50" "1536 4 00:20:00 50" "1536 8 00:15:00 50"
  "1536 16 00:15:00 50" "1536 32 00:15:00 50" "1536 64 00:15:00 50"
  "4096 8 01:00:00 20" "4096 16 00:40:00 20" "4096 32 00:30:00 20"
  "4096 64 00:20:00 20" "4096 128 00:15:00 20"
  # "8192 64 01:00:00 10" "8192 128 00:40:00 10"
)

CPT=$(( CORES_PER_NODE / RANKS_PER_NODE ))
for pt in "${POINTS[@]}"; do
  read -r N NODES TIME ITERS <<< "$pt"
  sbatch -J "strcpuh-N${N}-n${NODES}" \
    --nodes="${NODES}" --ntasks-per-node="${RANKS_PER_NODE}" --cpus-per-task="${CPT}" \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_TAG=strong_cpu_hybrid,PF_RANKS_PER_NODE="${RANKS_PER_NODE}" \
    job_cpu.slurm
done
echo ">> submitted ${#POINTS[@]} strong-CPU-hybrid jobs (${RANKS_PER_NODE} ranks/node x ${CPT} threads). squeue -u \$USER"
echo ">> when done: ./collect.sh"
