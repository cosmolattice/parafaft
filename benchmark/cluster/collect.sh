#!/usr/bin/env bash
# =============================================================================
# Merge the per-job CSVs (one directory per point) into one CSV per study/binary.
# Run after the launched jobs finish:  ./collect.sh
# Produces e.g. results/strong_cpu__bench_r2c.csv, results/strong_gpu__bench_r2c_cuda.csv,
#              results/strong_gpu__bench_r2c_cufftmp.csv, ...
#
# Duplicate datapoints (same mpi_procs/threads/N measured more than once — e.g. a
# ParaFaFT point re-run as part of a later cuFFTMp comparison sweep) are collapsed
# into ONE row per config, averaging rather than discarding: iteration-weighted
# mean, pooled std (within-run spread + between-run spread), summed iterations.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]:-$0}")/results" 2>/dev/null || { echo "no results/ yet"; exit 0; }
shopt -s nullglob

# avg_dupes <csv> -> collapsed+sorted CSV on stdout.
# Groups rows by their config columns (everything that is not a *_mean/*_std
# measurement or the iterations count) and merges each group:
#   iterations n_i, mean m_i, std s_i  ->  N=Σn_i,  M=Σ(n_i m_i)/N,
#   pooled var = Σ n_i (s_i^2 + m_i^2) / N - M^2   (exact for population std,
#   captures both within-run variance and the spread between the runs' means).
# A one-row group returns unchanged (M=m, pooled var=s^2).
avg_dupes() {
  awk -F, -v OFS=, '
    NR==1 {
      ncol = NF; print
      for (i = 1; i <= NF; i++) {
        nm = $i
        if (nm ~ /_mean$/)       { role[i] = "mean"; p = nm; sub(/_mean$/, "", p); prefix[i] = p }
        else if (nm ~ /_std$/)   { role[i] = "std";  p = nm; sub(/_std$/,  "", p); prefix[i] = p }
        else if (nm == "iterations") { role[i] = "iter"; itercol = i }
        else                     { role[i] = "key" }
      }
      next
    }
    {
      w = (itercol ? $itercol + 0 : 1); if (w <= 0) w = 1
      key = ""
      for (i = 1; i <= ncol; i++) if (role[i] == "key") key = key SUBSEP $i
      if (!(key in seen)) {
        seen[key] = 1; order[++ng] = key
        for (i = 1; i <= ncol; i++) if (role[i] == "key") kv[key, i] = $i
      }
      sumw[key] += w
      delete mrow; delete srow
      for (i = 1; i <= ncol; i++) {
        if (role[i] == "mean")     mrow[prefix[i]] = $i + 0
        else if (role[i] == "std") srow[prefix[i]] = $i + 0
      }
      for (p in mrow) {
        swm[key, p]  += w * mrow[p]
        swms[key, p] += w * (srow[p] * srow[p] + mrow[p] * mrow[p])
      }
    }
    END {
      for (g = 1; g <= ng; g++) {
        key = order[g]; line = ""
        for (i = 1; i <= ncol; i++) {
          if (role[i] == "key")       v = kv[key, i]
          else if (role[i] == "iter") v = sumw[key]
          else if (role[i] == "mean") { p = prefix[i]; v = sprintf("%.6f", swm[key, p] / sumw[key]) }
          else {  # std
            p = prefix[i]; mn = swm[key, p] / sumw[key]
            var = swms[key, p] / sumw[key] - mn * mn; if (var < 0) var = 0
            v = sprintf("%.6f", sqrt(var))
          }
          line = (i == 1 ? v : line OFS v)
        }
        print line
      }
    }
  ' "$1" | { IFS= read -r hdr; printf '%s\n' "$hdr"; sort -t, -k1,1n; }
}

for study_dir in */; do
  study="${study_dir%/}"
  [ "$study" = "logs" ] && continue
  for base in bench_r2c bench_r2c_cuda bench_r2c_cufftmp; do
    files=("$study"/*/"$base".csv)
    [ ${#files[@]} -gt 0 ] || continue
    out="${study}__${base}.csv"
    head -n1 "${files[0]}" > "$out"
    for f in "${files[@]}"; do tail -n +2 "$f" >> "$out"; done
    raw=$(( $(wc -l < "$out") - 1 ))
    # Collapse duplicate configs (averaging) and sort data rows by mpi_procs.
    avg_dupes "$out" > "$out.tmp" && mv "$out.tmp" "$out"
    final=$(( $(wc -l < "$out") - 1 ))
    note=""
    [ "$final" -lt "$raw" ] && note="  (${raw} rows -> ${final} after averaging duplicates)"
    echo ">> results/${out}  (${final} points)${note}"
  done
done
