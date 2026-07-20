#!/usr/bin/env bash
# =============================================================================
# Launch WEAK-scaling CPU jobs — HYBRID MPI+OpenMP (4 ranks/node x 32 threads
# = full 128-core node). Mirrors submit_weak_cpu.sh for a matched comparison.
#
#     ./submit_weak_cpu_hybrid.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
source ./env_cpu.sh

BIN="${PARAFAFT_CPU_BUILD}/bench/bench_r2c"
[ -x "$BIN" ] || build_cpu
[ -x "$BIN" ] || { echo "ERROR: build failed"; exit 1; }

CORES_PER_NODE=128        # TODO: must match the node. Earlier runs reported 192 logical CPUs.
RANKS_PER_NODE=4          # => OMP_NUM_THREADS = CORES_PER_NODE/4 threads per rank
mkdir -p results/logs

# "NODES N TIME ITERS" — mirror submit_weak_cpu.sh.
POINTS=(
  "1 512 00:40:00 30" "2 646 00:40:00 30" "4 814 00:40:00 30" "8 1024 00:40:00 30"
  "16 1290 00:40:00 30" "32 1626 00:45:00 30" "64 2048 00:45:00 30" "128 2580 00:50:00 30"
)

CPT=$(( CORES_PER_NODE / RANKS_PER_NODE ))
for pt in "${POINTS[@]}"; do
  read -r NODES N TIME ITERS <<< "$pt"
  sbatch -J "weakcpuh-N${N}-n${NODES}" \
    --nodes="${NODES}" --ntasks-per-node="${RANKS_PER_NODE}" --cpus-per-task="${CPT}" \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_TAG=weak_cpu_hybrid,PF_RANKS_PER_NODE="${RANKS_PER_NODE}" \
    job_cpu.slurm
done
echo ">> submitted ${#POINTS[@]} weak-CPU-hybrid jobs (${RANKS_PER_NODE} ranks/node x ${CPT} threads). squeue -u \$USER"
echo ">> when done: ./collect.sh"
