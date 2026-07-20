#!/usr/bin/env bash
# =============================================================================
# Merge the per-job CSVs (one directory per point) into one CSV per study/binary.
# Run after the launched jobs finish:  ./collect.sh
# Produces e.g. results/strong_cpu__bench_r2c.csv, results/strong_gpu__bench_r2c_cuda.csv,
#              results/strong_gpu__bench_r2c_cufftmp.csv, ...
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/results" 2>/dev/null || { echo "no results/ yet"; exit 0; }
shopt -s nullglob

for study_dir in */; do
  study="${study_dir%/}"
  [ "$study" = "logs" ] && continue
  for base in bench_r2c bench_r2c_cuda bench_r2c_cufftmp; do
    files=("$study"/*/"$base".csv)
    [ ${#files[@]} -gt 0 ] || continue
    out="${study}__${base}.csv"
    head -n1 "${files[0]}" > "$out"
    for f in "${files[@]}"; do tail -n +2 "$f" >> "$out"; done
    # sort data rows by the mpi_procs column (field 1) for readability
    { head -n1 "$out"; tail -n +2 "$out" | sort -t, -k1,1n; } > "$out.tmp" && mv "$out.tmp" "$out"
    echo ">> results/${out}  (${#files[@]} points)"
  done
done
