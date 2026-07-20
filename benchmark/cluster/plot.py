#!/usr/bin/env python3
"""Plot ParaFaFT strong- and weak-scaling results (3D R2C) from the collected CSVs.

Consumes whatever `collect.sh` produced under <results>/:
    strong_cpu__bench_r2c.csv          (ParaFaFT-CPU pure MPI + FFTW-MPI baseline)
    strong_cpu_hybrid__bench_r2c.csv   (ParaFaFT-CPU hybrid MPI+OpenMP + FFTW-MPI)
    weak_cpu__bench_r2c.csv / weak_cpu_hybrid__bench_r2c.csv
    strong_gpu__bench_r2c_cuda.csv     (ParaFaFT-GPU)
    strong_gpu__bench_r2c_cufftmp.csv  (cuFFTMp baseline)
    weak_gpu__bench_r2c_cuda.csv / weak_gpu__bench_r2c_cufftmp.csv

Missing files are skipped, so partial data still plots. Emits two figures
(strong_scaling.{png,pdf}, weak_scaling.{png,pdf}) into <results>/plots/.

Visual encoding (three orthogonal channels):
    color     = configuration : CPU (pure MPI) / CPU (hybrid) / GPU
    linestyle = method        : solid = ParaFaFT, dashed = baseline (FFTW-MPI / cuFFTMp)
    marker    = grid size N

Points whose std/mean exceeds MAX_REL_STD are dropped (and reported on stderr) —
a single noisy run otherwise draws an error bar spanning the whole log axis.

The x-axis is COMPUTE NODES: CPU series use (mpi_procs * threads) / CORES_PER_NODE so
pure MPI and hybrid line up at matched hardware, and GPU series count 1 GPU as 1 node.
A secondary top axis restates the CPU side in cores.

    python3 plot.py [results_dir]        # default: <script_dir>/results

Requires: matplotlib, numpy.
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Time is one forward + one backward transform (seconds).
CPU_COLOR = "#1f77b4"     # pure MPI
HYBRID_COLOR = "#2ca02c"  # hybrid MPI+OpenMP
GPU_COLOR = "#d62728"
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]  # assigned per distinct N

# Points whose std exceeds this fraction of the mean are dropped: on a log axis a
# single run that hit a noisy node produces an error bar spanning the whole figure
# and swamps everything else. Such a point carries no usable timing signal anyway.
MAX_REL_STD = 0.5

# Physical cores per CPU node — must match CORES_PER_NODE in the submit_*.sh scripts
# (the nodes report 192 logical CPUs, but jobs were launched with 128 ranks/node).
# The x-axis is NODES, so CPU cores are divided by this and 1 GPU counts as 1 node;
# that puts both backends on a common "how much hardware did this cost" scale.
CORES_PER_NODE = 128


def load(path):
    """Read a collected CSV into a dict of float columns (empty dict if absent)."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    order = np.argsort(cols["mpi_procs"])
    return {k: v[order] for k, v in cols.items()}


def xunits(data):
    """x = compute nodes. CPU tables carry 'threads' -> ranks*threads/CORES_PER_NODE;
    GPU tables have no 'threads' -> #GPUs, counted as one node per GPU."""
    if "threads" in data:
        return data["mpi_procs"] * data["threads"] / CORES_PER_NODE
    return data["mpi_procs"]


def denoise(x, t, e, tag=""):
    """Drop points whose relative std exceeds MAX_REL_STD (reporting each one)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(t > 0, e / t, 0.0)
    keep = rel <= MAX_REL_STD
    if tag:  # tag="" silences the report for repeat calls on already-reported data
        for xi, ti, ri in zip(x[~keep], t[~keep], rel[~keep]):
            print(f"   dropped {tag} x={xi:g}: mean={ti:.4g}s, std/mean={ri:.2f}", file=sys.stderr)
    return x[keep], t[keep], e[keep]


def groups_by_N(data, tcol, scol, tag=""):
    """Split a table into one (N, x, time, err) group per grid size, minus noisy points."""
    x = xunits(data)
    err = data.get(scol, np.zeros_like(data[tcol]))
    out = []
    for N in sorted(set(data["N"].tolist())):
        m = data["N"] == N
        xs, ts, es = denoise(x[m], data[tcol][m], err[m], f"{tag} N={int(N)}")
        if len(xs):
            out.append((int(N), xs, ts, es))
    return out


def marker_map(results, keys):
    all_N = set()
    for k in keys:
        d = results.get(k)
        if d:
            all_N.update(int(n) for n in d["N"])
    return {N: MARKERS[i % len(MARKERS)] for i, N in enumerate(sorted(all_N))}


# --- styling shared by both plots --------------------------------------------
PARA = dict(ls="-", lw=1.8, capsize=3, ms=6)
BASE = dict(ls="--", lw=1.2, alpha=0.55, capsize=2, ms=5)


def _fmt(v):
    """Plain integer when the value is one, else a short decimal (e.g. 0.5 nodes)."""
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:g}"


def cores_axis(ax):
    """Label the nodes axis with plain numbers and add a top axis in CPU cores.

    The cores axis is only meaningful for the CPU series (a GPU node contributes its
    GPUs, not cores), hence the explicit 'CPU' in the label; nodes is the common one.
    Both axes tick at the same powers of two so they read as a single grid, and both
    use plain integers rather than 2^k / decade exponents.
    """
    lo, hi = ax.get_xlim()
    nodes = [2.0 ** k for k in range(int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi))) + 1)]

    ax.set_xticks(nodes)
    ax.set_xticklabels([_fmt(n) for n in nodes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    sec = ax.secondary_xaxis("top", functions=(lambda n: n * CORES_PER_NODE,
                                               lambda c: c / CORES_PER_NODE))
    sec.set_xlabel(f"# CPU cores ({CORES_PER_NODE}/node)", fontsize=9, labelpad=6)
    sec.tick_params(labelsize=8)
    sec.set_xticks([n * CORES_PER_NODE for n in nodes])
    sec.set_xticklabels([_fmt(n * CORES_PER_NODE) for n in nodes])
    sec.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(lo, hi)  # set_xticks can widen the view; pin it back
    return sec


def key_legends(ax, present_configs, nmap, ideal=True):
    """Attach compact 'what the channels mean' legends outside the axes."""
    color_h = [Line2D([], [], color=c, lw=2.5, label=l) for c, l in present_configs]
    method_h = [Line2D([], [], color="0.3", ls="-", lw=1.8, label="ParaFaFT"),
                Line2D([], [], color="0.3", ls="--", lw=1.2, alpha=0.7,
                       label="FFTW-MPI / cuFFTMp")]
    if ideal:
        method_h.append(Line2D([], [], color="0.5", ls=":", lw=1, label="ideal"))

    def block(handles, y, title):
        leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, y),
                        fontsize=8, title=title)
        leg.get_title().set_fontsize(8)
        ax.add_artist(leg)

    # Stacked blocks: color, then line style, then (strong only) marker.
    block(color_h, 1.0, "color = config")
    block(method_h, 0.72, "line = method")
    if nmap:
        marker_h = [Line2D([], [], color="0.3", lw=0, marker=nmap[N], label=f"N={N}")
                    for N in sorted(nmap)]
        block(marker_h, 0.42, "marker = grid N")


def plot_strong(results, outdir):
    nmap = marker_map(results, ["strong_cpu__bench_r2c", "strong_cpu_hybrid__bench_r2c",
                                "strong_gpu__bench_r2c_cuda"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.08, right=0.73, top=0.92, bottom=0.12)
    ideal_done = [False]

    def ideal(x, t):
        x = np.asarray(x, float)
        ax.plot(x, t[0] * x[0] / x, ls=":", lw=1.0, color="0.55", zorder=0)
        ideal_done[0] = True

    present = []
    for key, color, label, para_col, base_col in [
        ("strong_cpu__bench_r2c", CPU_COLOR, "CPU (pure MPI)", ("parafaft_mean", "parafaft_std"),
         ("fftw_mean", "fftw_std")),
        ("strong_cpu_hybrid__bench_r2c", HYBRID_COLOR, "CPU (hybrid)",
         ("parafaft_mean", "parafaft_std"), ("fftw_mean", "fftw_std")),
        ("strong_gpu__bench_r2c_cuda", GPU_COLOR, "GPU", ("parafaft_mean", "parafaft_std"), None),
    ]:
        d = results.get(key)
        if not d:
            continue
        present.append((color, label))
        for N, x, t, e in groups_by_N(d, *para_col, tag=f"{label} ParaFaFT"):
            ax.errorbar(x, t, yerr=e, marker=nmap[N], color=color, **PARA)
            ideal(x, t)
        if base_col:
            for N, x, t, e in groups_by_N(d, *base_col, tag=f"{label} baseline"):
                ax.errorbar(x, t, yerr=e, marker=nmap[N], color=color, **BASE)

    mp = results.get("strong_gpu__bench_r2c_cufftmp")   # GPU baseline is a separate file
    if mp:
        for N, x, t, e in groups_by_N(mp, "parafaft_mean", "parafaft_std", tag="cuFFTMp"):
            ax.errorbar(x, t, yerr=e, marker=nmap.get(N, "s"), color=GPU_COLOR, **BASE)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("# nodes (CPU: ranks×threads/node; GPU: 1 GPU = 1 node)")
    ax.set_ylabel("time per fwd+bwd transform [s]")
    ax.set_title("ParaFaFT strong scaling (3D R2C)", pad=28)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    cores_axis(ax)
    key_legends(ax, present, nmap)
    _save(fig, outdir, "strong_scaling")


def plot_weak(results, outdir):
    # Each weak point is a different N, so N is not a useful marker channel here;
    # marker just tracks configuration for redundancy with color.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.08, right=0.73, top=0.92, bottom=0.12)

    def clean(data, col, tag):
        t = data[col[0]]
        return denoise(xunits(data), t, data.get(col[1], np.zeros_like(t)), tag)

    def line(data, col, color, marker, style, tag=""):
        if not data:
            return
        x, t, e = clean(data, col, tag)
        ax.errorbar(x, t, yerr=e, marker=marker, color=color, **style)

    para, base = ("parafaft_mean", "parafaft_std"), ("fftw_mean", "fftw_std")
    present = []
    cpu = results.get("weak_cpu__bench_r2c")
    if cpu:
        present.append((CPU_COLOR, "CPU (pure MPI)"))
        line(cpu, para, CPU_COLOR, "o", PARA, "CPU (pure MPI) ParaFaFT")
        line(cpu, base, CPU_COLOR, "o", BASE, "CPU (pure MPI) baseline")
    cpuh = results.get("weak_cpu_hybrid__bench_r2c")
    if cpuh:
        present.append((HYBRID_COLOR, "CPU (hybrid)"))
        line(cpuh, para, HYBRID_COLOR, "^", PARA, "CPU (hybrid) ParaFaFT")
        line(cpuh, base, HYBRID_COLOR, "^", BASE, "CPU (hybrid) baseline")
    gpu = results.get("weak_gpu__bench_r2c_cuda")
    if gpu:
        present.append((GPU_COLOR, "GPU"))
        line(gpu, para, GPU_COLOR, "s", PARA, "GPU ParaFaFT")
        line(results.get("weak_gpu__bench_r2c_cufftmp"), para, GPU_COLOR, "s", BASE, "cuFFTMp")

    # Ideal weak scaling = flat; one faint reference per backend's first kept point.
    for anchor in (cpu, cpuh, gpu):
        if anchor:
            _, t, _ = clean(anchor, para, "")
            if len(t):
                ax.axhline(t[0], ls=":", lw=1.0, color="0.55", zorder=0)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("# nodes (CPU: ranks×threads/node; GPU: 1 GPU = 1 node)"
                  " — grid N grows to keep work/process fixed")
    ax.set_ylabel("time per fwd+bwd transform [s]")
    ax.set_title("ParaFaFT weak scaling (3D R2C)", pad=28)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    cores_axis(ax)
    key_legends(ax, present, nmap=None)
    _save(fig, outdir, "weak_scaling")


def _save(fig, outdir, stem):
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=150)
        print(f">> wrote {path}")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "results")
    if not os.path.isdir(results_dir):
        sys.exit(f"no results dir: {results_dir} (run ./collect.sh first)")

    names = ["strong_cpu__bench_r2c", "strong_cpu_hybrid__bench_r2c",
             "weak_cpu__bench_r2c", "weak_cpu_hybrid__bench_r2c",
             "strong_gpu__bench_r2c_cuda", "strong_gpu__bench_r2c_cufftmp",
             "weak_gpu__bench_r2c_cuda", "weak_gpu__bench_r2c_cufftmp"]
    results = {n: load(os.path.join(results_dir, n + ".csv")) for n in names}
    results = {k: v for k, v in results.items() if v}
    if not results:
        sys.exit(f"no collected CSVs found in {results_dir} (run ./collect.sh)")
    print("loaded:", ", ".join(sorted(results)))

    # A CPU sweep must hold threads/rank fixed; if it varies, the runs were mis-threaded
    # (the old detect_thread_count divided cores by the GLOBAL rank count) and the
    # cores axis collapses onto a vertical line. Flag it rather than plot nonsense.
    for name, d in sorted(results.items()):
        if "threads" in d:
            uniq = sorted({int(t) for t in d["threads"]})
            if len(uniq) > 1:
                print(f"WARNING: {name}: 'threads' varies across runs {uniq} — expected a "
                      "single value. These runs pre-date the detect_thread_count fix and "
                      "must be re-run; the cores axis will look wrong.", file=sys.stderr)

    outdir = os.path.join(results_dir, "plots")
    os.makedirs(outdir, exist_ok=True)
    plot_strong(results, outdir)
    plot_weak(results, outdir)


if __name__ == "__main__":
    main()
