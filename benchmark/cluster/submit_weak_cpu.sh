#!/usr/bin/env bash
# =============================================================================
# Launch WEAK-scaling CPU jobs — one INDEPENDENT SLURM job per point.
# Constant work/rank: base 512^3 on 1 node, N = round_even(512*nodes^(1/3)).
# The small-node jobs are cheap to schedule and start without waiting for the
# large-node ones.
#
#     ./submit_weak_cpu.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
source ./env_cpu.sh

BIN="${PARAFAFT_CPU_BUILD}/bench/bench_r2c"
[ -x "$BIN" ] || build_cpu
[ -x "$BIN" ] || { echo "ERROR: build failed"; exit 1; }

CORES_PER_NODE=128                  # TODO: must match the node. Earlier runs reported 192 logical CPUs.
RANKS_PER_NODE=${CORES_PER_NODE}    # pure MPI: 1 rank per core => 1 thread per rank
mkdir -p results/logs

# "NODES N TIME ITERS"  — ~1.05e6 points/rank throughout.
POINTS=(
  "1 512 00:40:00 30" "2 646 00:40:00 30" "4 814 00:40:00 30" "8 1024 00:40:00 30"
  "16 1290 00:40:00 30" "32 1626 00:45:00 30" "64 2048 00:45:00 30" "128 2580 00:50:00 30"
)

CPT=$(( CORES_PER_NODE / RANKS_PER_NODE ))
for pt in "${POINTS[@]}"; do
  read -r NODES N TIME ITERS <<< "$pt"
  sbatch -J "weakcpu-N${N}-n${NODES}" \
    --nodes="${NODES}" --ntasks-per-node="${RANKS_PER_NODE}" --cpus-per-task="${CPT}" \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_TAG=weak_cpu,PF_RANKS_PER_NODE="${RANKS_PER_NODE}" \
    job_cpu.slurm
done
echo ">> submitted ${#POINTS[@]} weak-CPU jobs. Watch: squeue -u \$USER"
echo ">> when done: ./collect.sh"
