#!/usr/bin/env bash
# =============================================================================
# Launch STRONG-scaling CPU jobs — one INDEPENDENT SLURM job per (grid, nodes)
# point. Small-node/large-grid points are the long ones; large-node points are
# short. Submitting them separately lets each schedule on its own (short jobs
# start sooner) instead of waiting for one big allocation.
#
#     ./submit_strong_cpu.sh            # build once, then sbatch every point
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")"          # -> benchmark/cluster
source ./env_cpu.sh

BIN="${PARAFAFT_CPU_BUILD}/bench/bench_r2c"
[ -x "$BIN" ] || build_cpu                        # build ONCE up front (jobs never build -> no races)
[ -x "$BIN" ] || { echo "ERROR: build failed"; exit 1; }

CORES_PER_NODE=128                  # TODO: must match the node. Earlier runs reported 192 logical CPUs.
RANKS_PER_NODE=${CORES_PER_NODE}    # pure MPI: 1 rank per core => 1 thread per rank
mkdir -p results/logs

# "N NODES TIME ITERS"  — fixed grid, growing nodes. Tune TIME to keep jobs short.
# Mem ~24*N^3 B total (~240 GB/node): 1536~0.09TB, 4096~1.7TB(>=8n), 8192~13TB(>=56n).
POINTS=(
  "1536 1 00:45:00 50" "1536 2 00:30:00 50" "1536 4 00:20:00 50" "1536 8 00:15:00 50"
  "1536 16 00:15:00 50" "1536 32 00:15:00 50" "1536 64 00:15:00 50"
  "4096 8 01:00:00 20" "4096 16 00:40:00 20" "4096 32 00:30:00 20"
  "4096 64 00:20:00 20" "4096 128 00:15:00 20"
  # --- optional "hero" ~13 TB (>=56 nodes), uncomment to include: ---
  # "8192 64 01:00:00 10" "8192 128 00:40:00 10"
)

CPT=$(( CORES_PER_NODE / RANKS_PER_NODE ))
for pt in "${POINTS[@]}"; do
  read -r N NODES TIME ITERS <<< "$pt"
  sbatch -J "strcpu-N${N}-n${NODES}" \
    --nodes="${NODES}" --ntasks-per-node="${RANKS_PER_NODE}" --cpus-per-task="${CPT}" \
    --time="${TIME}" --output="results/logs/parafaft-%x-%j.out" \
    --export=ALL,PF_N="${N}",PF_ITERS="${ITERS}",PF_TAG=strong_cpu,PF_RANKS_PER_NODE="${RANKS_PER_NODE}" \
    job_cpu.slurm
done
echo ">> submitted ${#POINTS[@]} strong-CPU jobs. Watch: squeue -u \$USER"
echo ">> when done: ./collect.sh   (merges results/strong_cpu/*/ into results/strong_cpu__bench_r2c.csv)"
