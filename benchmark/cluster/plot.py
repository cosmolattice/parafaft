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

# --- shared paper theme ------------------------------------------------------
# Mirrors plot_style.py of the companion TempLat GPU-scaling figures so the two
# sets sit in one publication with a common look: CVD-safe palette, hairline
# dotted grid and axes, muted dotted "ideal" references, white-haloed markers.
# Kept inline (not imported) so this benchmark plotter stays self-contained;
# if the constants below drift from plot_style.py the two figure sets diverge.
C_IDEAL = "#898781"   # muted reference lines
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK2 = "#52514e"    # tick labels / secondary annotation
C_STEP = "#0b0b0b"    # primary ink
SURFACE = "#ffffff"   # page white == marker-halo colour

PAPER_THEME = {
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 150,
    "font.size": 13,
    "axes.facecolor": SURFACE, "axes.edgecolor": C_AXIS, "axes.linewidth": 0.8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.grid": True, "axes.axisbelow": True, "axes.grid.which": "major",
    "axes.labelcolor": C_STEP, "axes.titlecolor": C_STEP, "axes.titlesize": 14,
    "grid.color": C_GRID, "grid.linewidth": 0.6, "grid.linestyle": ":",
    "xtick.color": C_AXIS, "ytick.color": C_AXIS,
    "xtick.labelcolor": C_INK2, "ytick.labelcolor": C_INK2,
    "legend.frameon": False, "legend.labelcolor": C_INK2, "legend.fontsize": 12,
}

# Time is one r2c + one c2r transform (seconds). Config colours (weak plot):
# categorical blue / aqua from the shared palette, near-black ink for GPU.
CPU_COLOR = "#2a78d6"     # pure MPI
HYBRID_COLOR = "#1baf7a"  # hybrid MPI+OpenMP
GPU_COLOR = C_STEP
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]  # assigned per distinct N


def marker_kw(color, marker="o"):
    """Solid line + white-haloed markers, matching plot_style.marker_kw."""
    return dict(color=color, ls="-", lw=2.0, marker=marker, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.8)


def ideal_kw():
    """Muted dotted reference line, as the companion figures draw ideal curves."""
    return dict(color=C_IDEAL, ls=":", lw=1.6)

# One size for every x tick label (nodes below, CPU cores above) and one for the
# x-axis captions, so the two rows never look like different type. Held a little
# below the body size (13) because the cores row runs to 5-digit numbers.
XTICK_FS = 10
XLABEL_FS = 12

# Points whose std exceeds this fraction of the mean are dropped: on a log axis a
# single run that hit a noisy node produces an error bar spanning the whole figure
# and swamps everything else. Such a point carries no usable timing signal anyway.
MAX_REL_STD = 0.5

# Physical cores per CPU node — must match CORES_PER_NODE in the submit_*.sh scripts
# (the nodes report 192 logical CPUs, but jobs were launched with 128 ranks/node).
# The x-axis is NODES, so CPU cores are divided by this and 1 GPU counts as 1 node;
# that puts both backends on a common "how much hardware did this cost" scale.
CORES_PER_NODE = 128

# GPUs per node. The GPU series crosses from one node to several between this
# value and twice it, which is where inter-node traffic first appears and the
# curve stops being a pure intra-node measurement.
GPUS_PER_NODE = 4


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


def color_map(results, keys):
    """Map each distinct grid size N to a colour along a sequential ramp.

    Faceted-by-config, N carries the colour channel; a sequential (viridis) ramp
    keeps the N ordering legible, and one global map means a colour reads the same
    in every panel. The ramp stops short of the pale yellow end so every line stays
    visible against the white grid.
    """
    all_N = set()
    for k in keys:
        d = results.get(k)
        if d:
            all_N.update(int(n) for n in d["N"])
    Ns = sorted(all_N)
    cmap = plt.get_cmap("viridis")
    pos = np.linspace(0.12, 0.82, len(Ns)) if len(Ns) > 1 else [0.35]
    return {N: cmap(p) for N, p in zip(Ns, pos)}


# --- styling shared by both plots --------------------------------------------
# ParaFaFT: solid, white-haloed markers (marker_kw). Baseline: dashed and
# lighter, the same halo, so a curve and its baseline read as one pair.
PARA = dict(ls="-", lw=2.0, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.8, capsize=2.5)
BASE = dict(ls="--", lw=1.4, ms=6, markeredgecolor=SURFACE, markeredgewidth=1.4,
            alpha=0.8, capsize=2)


def _fmt(v):
    """Plain integer when the value is one, else a short decimal (e.g. 0.5 nodes)."""
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:g}"


def nodes_ticks(ax):
    """Tick the x-axis at powers of two, labelled as plain node counts.

    Returns the node positions so callers that want a second (cores) axis can
    reuse them, keeping both rows of ticks on the same grid.
    """
    lo, hi = ax.get_xlim()
    nodes = [2.0 ** k for k in range(int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi))) + 1)]
    ax.set_xticks(nodes)
    ax.set_xticklabels([_fmt(n) for n in nodes])
    ax.tick_params(axis="x", labelsize=XTICK_FS)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(lo, hi)  # set_xticks can widen the view; pin it back
    return nodes


def cores_axis(ax, show_label=True):
    """Label the nodes axis with plain numbers and add a top axis in CPU cores.

    The cores axis is only meaningful for the CPU series (a GPU node contributes its
    GPUs, not cores), hence the explicit 'CPU' in the label; nodes is the common one.
    Both axes tick at the same powers of two so they read as a single grid, and both
    use plain integers rather than 2^k / decade exponents. Faceted CPU panels share
    an identical top axis, so show_label=False suppresses the redundant caption on
    all but one panel.
    """
    lo, hi = ax.get_xlim()
    nodes = nodes_ticks(ax)

    sec = ax.secondary_xaxis("top", functions=(lambda n: n * CORES_PER_NODE,
                                               lambda c: c / CORES_PER_NODE))
    if show_label:
        sec.set_xlabel(f"CPU cores ({CORES_PER_NODE}/node)", fontsize=XLABEL_FS, labelpad=6)
    sec.tick_params(axis="x", length=3, labelsize=XTICK_FS, colors=C_AXIS, labelcolor=C_INK2)
    sec.set_xticks([n * CORES_PER_NODE for n in nodes])
    sec.set_xticklabels([_fmt(n * CORES_PER_NODE) for n in nodes])
    sec.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    sec.spines["top"].set(color=C_AXIS, linewidth=0.8)
    ax.set_xlim(lo, hi)  # set_xticks can widen the view; pin it back
    return sec


def multinode_divider(ax, x):
    """Mark where the GPU series leaves a single node, between GPUS_PER_NODE
    and twice it, with a full-height rule and a "node boundary" label.

    Drawn floor-to-top like the reference figure. On the old shared axes this had
    to be a short segment on the GPU curve, since the same x meant something else
    for the CPU series; now that the GPU series has its own panel a full axvline
    divides only what it should. Positioned at the geometric mean of the two node
    counts, i.e. the midpoint on the log x-axis.
    """
    lo, hi = GPUS_PER_NODE, 2 * GPUS_PER_NODE
    x = np.asarray(x, float)
    if not (np.any(x <= lo) and np.any(x >= hi)):
        return  # sweep does not bracket the boundary; nothing to mark

    xb = np.sqrt(lo * hi)  # geometric mean of the bracketing node counts
    ax.axvline(xb, color=C_AXIS, lw=0.8, zorder=1)
    ax.annotate(" node boundary →", (xb, 0.02), xycoords=ax.get_xaxis_transform(),
                ha="left", va="bottom", color=C_INK2, fontsize=10)


def key_legends(ax, present_configs, nmap, ideal=True):
    """Attach compact 'what the channels mean' legends to the right of the axes.

    Figure-level (not ax.legend): legends anchored outside the axes are dropped by
    savefig(bbox_inches="tight") when added as axes children, but kept as fig.legends.
    """
    fig = ax.figure
    color_h = [Line2D([], [], color=c, lw=2.5, label=l) for c, l in present_configs]
    method_h = [Line2D([], [], color=C_INK2, ls="-", lw=2.0, label="ParaFaFT"),
                Line2D([], [], color=C_INK2, ls="--", lw=1.4, alpha=0.8,
                       label="FFTW-MPI / cuFFTMp")]
    if ideal:
        method_h.append(Line2D([], [], color=C_IDEAL, ls=":", lw=1.6, label="ideal"))

    def block(handles, y, title):
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.76, y),
                         title=title)
        leg.get_title().set_color(C_STEP)

    # Stacked blocks: config colour, then line style, then (strong only) marker.
    block(color_h, 0.88, "config")
    block(method_h, 0.60, "method")
    if nmap:
        marker_h = [Line2D([], [], color=C_INK2, lw=0, marker=nmap[N], label=f"{N}³")
                    for N in sorted(nmap)]
        block(marker_h, 0.34, "grid $N$")


def panel_label(ax, text):
    """Bold config tag in a white rounded box, tucked into the top-left corner —
    matching plot_style.panel_label of the companion figures."""
    ax.text(0.028, 0.94, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=C_STEP, zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                      edgecolor=C_AXIS, linewidth=0.8))


def ideal_ref(ax, groups):
    """Dotted 1/p slope guide under every ParaFaFT sweep, each anchored at its own
    first point. Unlabelled; the dotted style reads as the ideal reference."""
    for N, x, t, e in groups:
        if len(x) < 2:
            continue
        x = np.asarray(x, float)
        ax.plot(x, t[0] * x[0] / x, zorder=0, **ideal_kw())


def strong_legends(fig, cmap, nmap):
    """One shared legend to the right of the panels: grid N (colour+marker) and
    method (line style). Config is the panel identity now, so it needs no key;
    'ideal' is labelled inline on each panel, so it is absent here too."""
    n_h = [Line2D([], [], color=cmap[N], marker=nmap[N], ls="-", lw=2.0,
                  markeredgecolor=SURFACE, markeredgewidth=1.6, label=f"{N}³")
           for N in sorted(cmap)]
    m_h = [Line2D([], [], color=C_INK2, ls="-", lw=2.0, label="ParaFaFT"),
           Line2D([], [], color=C_INK2, ls="--", lw=1.4, alpha=0.8,
                  label="FFTW-MPI /\ncuFFTMp")]

    l1 = fig.legend(handles=n_h, loc="upper left", bbox_to_anchor=(0.87, 0.84),
                    title="grid $N$")
    l2 = fig.legend(handles=m_h, loc="upper left", bbox_to_anchor=(0.87, 0.42),
                    title="method")
    for leg in (l1, l2):
        leg.get_title().set_color(C_STEP)


def plot_strong(results, outdir):
    """Strong scaling, faceted by configuration.

    One panel per backend (CPU pure-MPI / CPU hybrid / GPU) sharing a log y-axis.
    Within a panel colour = grid N and line style = method (solid ParaFaFT, dashed
    baseline), so a ParaFaFT curve and its baseline sit right on top of each other
    and no line ever crosses between configurations. The combined single-axes
    version overlaid up to ~18 lines in one decade; splitting config into small
    multiples is what makes 'which points belong together' legible.
    """
    keys = ["strong_cpu__bench_r2c", "strong_cpu_hybrid__bench_r2c",
            "strong_gpu__bench_r2c_cuda"]
    nmap = marker_map(results, keys)
    cmap = color_map(results, keys)

    #  key, panel title, baseline (col, err) in the same file, is_cpu
    panels = [
        ("strong_cpu__bench_r2c", "CPU (pure MPI)", ("fftw_mean", "fftw_std"), True),
        ("strong_cpu_hybrid__bench_r2c", "CPU (hybrid)", ("fftw_mean", "fftw_std"), True),
        ("strong_gpu__bench_r2c_cuda", "GPU", None, False),
    ]
    panels = [p for p in panels if results.get(p[0])]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.8), sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.07, right=0.86, top=0.82, bottom=0.15, wspace=0.08)

    para = ("parafaft_mean", "parafaft_std")
    for i, (ax, (key, title, base, is_cpu)) in enumerate(zip(axes, panels)):
        d = results[key]
        para_groups = groups_by_N(d, *para, tag=f"{title} ParaFaFT")
        for N, x, t, e in para_groups:
            ax.errorbar(x, t, yerr=e, marker=nmap[N], color=cmap[N], **PARA)
        if not is_cpu and para_groups:  # GPU node boundary, once per panel
            multinode_divider(ax, np.concatenate([g[1] for g in para_groups]))

        if base:  # in-file baseline (FFTW-MPI)
            for N, x, t, e in groups_by_N(d, *base, tag=f"{title} baseline"):
                ax.errorbar(x, t, yerr=e, marker=nmap[N], color=cmap[N], **BASE)
        else:     # GPU baseline (cuFFTMp) lives in a separate file
            mp = results.get("strong_gpu__bench_r2c_cufftmp")
            if mp:
                for N, x, t, e in groups_by_N(mp, "parafaft_mean", "parafaft_std", tag="cuFFTMp"):
                    ax.errorbar(x, t, yerr=e, marker=nmap.get(N, "s"),
                                color=cmap.get(N, "0.4"), **BASE)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        # Top cores ticks on the CPU panels; the caption is drawn once, figure-
        # centred (below), so it lines up with the bottom '# nodes' label.
        cores_axis(ax, show_label=False) if is_cpu else nodes_ticks(ax)
        # One dotted 1/p slope guide per panel (all N share the -1 slope on log-log),
        # anchored to the widest ParaFaFT sweep and labelled inline rather than in
        # the legend, as the companion figures do.
        ideal_ref(ax, para_groups)
        panel_label(ax, title)

    # Floor the y-axis: the ideal references keep falling past the last measured
    # point; letting them set the limit wastes a third of the figure on empty
    # space. The fastest measurement (GPU at 32) is ~8e-2, so this clips only tails.
    axes[0].set_ylim(bottom=4e-2)
    axes[0].set_ylabel("time per r2c+c2r transform [s]")

    # x captions belong to their own unit: nodes/CPU-cores span the CPU panels,
    # GPUs the GPU panel. Centre each over the panels it describes.
    def span_center(group):
        pos = [ax.get_position() for ax in group]
        return 0.5 * (min(p.x0 for p in pos) + max(p.x1 for p in pos))

    cpu_axes = [ax for ax, p in zip(axes, panels) if p[3]]
    if cpu_axes:
        cx = span_center(cpu_axes)
        fig.text(cx, 0.03, "nodes", ha="center", va="bottom",
                 fontsize=XLABEL_FS, color=C_STEP)
        fig.text(cx, 0.915, f"CPU cores ({CORES_PER_NODE}/node)", ha="center",
                 va="bottom", fontsize=XLABEL_FS, color=C_STEP)
    for ax, p in zip(axes, panels):
        if not p[3]:  # GPU panel: one GPU per rank, so the axis counts GPUs
            fig.text(span_center([ax]), 0.03, "GPUs", ha="center", va="bottom",
                     fontsize=XLABEL_FS, color=C_STEP)

    strong_legends(fig, cmap, nmap)
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
    ax.set_xlabel("nodes  (grid $N$ grows to keep work/process fixed)", fontsize=XLABEL_FS)
    ax.set_ylabel("time per r2c+c2r transform [s]")
    ax.set_title("ParaFaFT weak scaling (3D R2C)", pad=28)
    cores_axis(ax)
    key_legends(ax, present, nmap=None)
    _save(fig, outdir, "weak_scaling")


def _save(fig, outdir, stem):
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        # bbox_inches="tight" expands the saved canvas to include the legends,
        # which sit outside the axes at a fixed point size; without it a smaller
        # figsize crops them (the margin is fractional, the legend text is not).
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f">> wrote {path}")
    plt.close(fig)


def main():
    matplotlib.rcParams.update(PAPER_THEME)  # shared paper look (see PAPER_THEME)
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
