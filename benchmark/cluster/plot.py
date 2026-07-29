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

Visual encoding:
    panel        = configuration : CPU (pure MPI) / CPU (hybrid) / GPU  (strong)
    color+marker = grid size N   (strong; configuration carries colour in weak)
    linestyle    = method        : solid = ParaFaFT, dashed = baseline (FFTW-MPI / cuFFTMp)

Points whose std/mean exceeds MAX_REL_STD are dropped (and reported on stderr) —
a single noisy run otherwise draws an error bar spanning the whole log axis.

Panels are positioned in COMPUTE NODES: CPU series use (mpi_procs * threads) /
CORES_PER_NODE so pure MPI and hybrid line up at matched hardware, and GPU series
count 1 GPU as 1 node. Each panel is then LABELLED in the unit its runs were
launched in — CPU cores / GPUs below the axis — with a secondary top row restating
that in nodes, as the companion TempLat figures do. Note the two node rows are not
the same count at one x: the GPU row counts physical GPUS_PER_NODE-GPU nodes.

Both figures are authored at their FINAL printed size (FULL_W = the paper's
\textwidth) with type sizes in printed points, and saved without bbox_inches=
"tight" — see PAPER_THEME and the geometry constants below. They go into the
paper as a bare \includegraphics{...} inside a figure*, with no width=.

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

# ------------------------------------------------------ paper geometry ------
# As in plot_style.py: figures are authored at their FINAL printed size, so they
# enter the paper with a bare \includegraphics{...} — no width=, no rescaling,
# hence identical type size in every figure. Measured from draft/paper.tex
# (cas-dc, a4paper): \columnwidth = 238.254 pt, \textwidth = 494.509 pt, at
# TeX's 72.27 pt/in. A `figure*` spans FULL_W, a single-column `figure` COL_W.
COL_W = 238.25444 / 72.27    # 3.297 in — one column
FULL_W = 494.50888 / 72.27   # 6.842 in — full text width (figure*)

# Type sizes, in pt, as printed — the paper sets captions at 9 pt, footnotesize
# at 8 pt. Everything in the figures derives from these three. Previously the
# figure was drawn at 13 pt on a 13.3 in canvas and shrunk to 1.05\linewidth in
# LaTeX (a factor ~0.54), which landed the labels at ~7 pt and the ticks at
# ~5.4 pt — smaller than anything else on the page, and different per figure.
FS_LABEL = 8    # axis labels, legend entries
FS_TICK = 7     # tick labels, panel tags
FS_CORES = 6    # top cores row — a step down so its 5-digit labels fit one line
FS_SMALL = 6.5  # secondary annotations (node boundary)
FS_TITLE = 9    # titles, matching the caption

PAPER_THEME = {
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 150,
    # NOT "tight": tight cropping trims each figure to its own content, so two
    # figures declared at the same figsize come out at different aspect ratios
    # and no longer align when set at equal width in LaTeX. Keeping the declared
    # canvas exact makes figsize the single source of truth.
    "savefig.bbox": "standard",
    "font.size": FS_LABEL,
    "axes.facecolor": SURFACE, "axes.edgecolor": C_AXIS, "axes.linewidth": 0.6,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.grid": True, "axes.axisbelow": True, "axes.grid.which": "major",
    "axes.labelsize": FS_LABEL,
    "axes.labelcolor": C_STEP, "axes.titlecolor": C_STEP, "axes.titlesize": FS_TITLE,
    "grid.color": C_GRID, "grid.linewidth": 0.5, "grid.linestyle": ":",
    "xtick.color": C_AXIS, "ytick.color": C_AXIS,
    "xtick.labelcolor": C_INK2, "ytick.labelcolor": C_INK2,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "legend.labelcolor": C_INK2, "legend.fontsize": FS_TICK,
    "legend.title_fontsize": FS_TICK,
    "lines.linewidth": 1.2, "lines.markersize": 3.4,
}

# Time is one r2c + one c2r transform (seconds). Config colours (weak plot):
# categorical blue / aqua from the shared palette, near-black ink for GPU.
CPU_COLOR = "#2a78d6"     # pure MPI
HYBRID_COLOR = "#1baf7a"  # hybrid MPI+OpenMP
GPU_COLOR = C_STEP
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]  # assigned per distinct N


def marker_kw(color, marker="o"):
    """Solid line + white-haloed markers, matching plot_style.marker_kw."""
    return dict(color=color, ls="-", lw=1.2, marker=marker, ms=3.4,
                markeredgecolor=SURFACE, markeredgewidth=0.9)


def ideal_kw():
    """Muted dotted reference line, as the companion figures draw ideal curves."""
    return dict(color=C_IDEAL, ls=":", lw=1.0)

# One size for every x tick label (nodes below, CPU cores above) and one for the
# x-axis captions, so the two rows never look like different type.
XTICK_FS = FS_TICK
XLABEL_FS = FS_LABEL

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
PARA = dict(ls="-", lw=1.2, ms=3.4, markeredgecolor=SURFACE, markeredgewidth=0.9, capsize=1.5)
BASE = dict(ls="--", lw=0.9, ms=2.9, markeredgecolor=SURFACE, markeredgewidth=0.8,
            alpha=0.8, capsize=1.2)


# Sub-node counts read better as vulgar fractions than as "0.25" — the companion
# TempLat figures label their quarter- and half-node points this way.
_FRACTIONS = {0.125: "⅛", 0.25: "¼", 0.5: "½", 0.75: "¾"}


def _fmt(v):
    """Plain integer when the value is one, a vulgar fraction for the common
    sub-unit counts, else a short decimal."""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    for q, s in _FRACTIONS.items():
        if abs(v - q) < 1e-9:
            return s
    return f"{v:g}"


def dual_axis(ax, bottom, top):
    """Tick x at powers of two and label those same positions twice: a primary row
    below the axis and a secondary one above it.

    `bottom` and `top` are each (factor, fontsize) — the label is the x value times
    factor, so one axis carries the unit the runs were launched in (CPU cores, GPUs)
    and the other restates it in nodes. This is the layout of the companion TempLat
    scaling figures: the hardware unit below, nodes above.

    Both rows sit on the same tick positions, so they read as a single grid, and both
    use plain integers rather than 2^k exponents. The CPU cores row runs to five
    digits ("16384") on a column-width panel where the ticks are ~a fifth of an inch
    apart, which is why the caller sets it a step smaller (FS_CORES).
    """
    lo, hi = ax.get_xlim()
    ticks = [2.0 ** k for k in range(int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi))) + 1)]

    factor, fs = bottom
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt(t * factor) for t in ticks])
    ax.tick_params(axis="x", labelsize=fs)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # Identity transform: the secondary axis shares the parent's data coordinates
    # and only the LABELS are rescaled, so the two rows cannot drift apart.
    factor, fs = top
    sec = ax.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
    sec.set_xticks(ticks)
    sec.set_xticklabels([_fmt(t * factor) for t in ticks])
    # length=2 and the smaller label size are what plot_style.nodes_top_axis uses:
    # the secondary row is set a step below the primary one so it reads as a
    # restatement rather than a second, competing axis.
    sec.tick_params(axis="x", length=2, labelsize=fs, colors=C_AXIS, labelcolor=C_INK2)
    sec.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    sec.spines["top"].set(color=C_AXIS, linewidth=0.6)
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
    ax.axvline(xb, color=C_AXIS, lw=0.6, zorder=1)
    # Label at the TOP of the panel: the GPU data maxes around 1 s, so the whole
    # upper part of the axis is empty, whereas the bottom is where the sub-1e-1
    # extension, its tick labels and the "GPUs" caption all live.
    ax.annotate(" node boundary →", (xb, 0.97), xycoords=ax.get_xaxis_transform(),
                ha="left", va="top", color=C_INK2, fontsize=FS_SMALL)


def key_legends(ax, present_configs, nmap, ideal=True):
    """Attach compact 'what the channels mean' legends to the right of the axes.

    Figure-level (not ax.legend): legends anchored outside the axes are dropped by
    savefig(bbox_inches="tight") when added as axes children, but kept as fig.legends.
    """
    fig = ax.figure
    color_h = [Line2D([], [], color=c, lw=1.4, label=l) for c, l in present_configs]
    method_h = [Line2D([], [], color=C_INK2, ls="-", lw=1.2, label="ParaFaFT"),
                Line2D([], [], color=C_INK2, ls="--", lw=0.9, alpha=0.8,
                       label="FFTW-MPI / cuFFTMp")]
    if ideal:
        method_h.append(Line2D([], [], color=C_IDEAL, ls=":", lw=1.0, label="ideal"))

    def block(handles, y, title):
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.755, y),
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
    ax.text(0.028, 0.965, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_TICK, fontweight="bold", color=C_STEP, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE,
                      edgecolor=C_AXIS, linewidth=0.6))


def ideal_ref(ax, groups):
    """Dotted 1/p slope guide under every ParaFaFT sweep, each anchored at its own
    first point. Unlabelled; the dotted style reads as the ideal reference."""
    for N, x, t, e in groups:
        if len(x) < 2:
            continue
        x = np.asarray(x, float)
        ax.plot(x, t[0] * x[0] / x, zorder=0, **ideal_kw())


def strong_legends(fig, cmap, nmap, anchors, y_center, title_col):
    """Two legend blocks in the strip under the CPU panels, each captioned on the
    LEFT of its entries:

        grid N    1024³  2048³        method    ParaFaFT
                  1536³  4096³                  FFTW-MPI / cuFFTMp

    matplotlib only ever sets a legend title *above* its entries, so the caption
    is drawn as figure text and the legend itself is titleless, anchored
    `title_col` to the caption's right. Both are centred on the same `y_center`,
    which is what puts the caption level with the middle of its two rows.

    Config is the panel identity now, so it needs no key, and 'ideal' is read off
    the dotted style inline, so it is absent here too.

    `anchors` are the left edges of the two captions; all three positional
    arguments are figure coords.
    """
    n_h = [Line2D([], [], color=cmap[N], marker=nmap[N], ls="-", lw=1.2,
                  markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"{N}³")
           for N in sorted(cmap)]
    m_h = [Line2D([], [], color=C_INK2, ls="-", lw=1.2, label="ParaFaFT"),
           Line2D([], [], color=C_INK2, ls="--", lw=0.9, alpha=0.8,
                  label="FFTW-MPI / cuFFTMp")]

    # borderpad=0: with the caption outside, the handles must start exactly at
    # the anchor, or the two blocks no longer share one indent.
    common = dict(handletextpad=0.5, borderaxespad=0, borderpad=0.0,
                  labelspacing=0.6, columnspacing=1.1)
    blocks = [("grid $N$", n_h, 2, 1.4), ("method", m_h, 1, 1.7)]
    for x, (title, handles, ncol, hlen) in zip(anchors, blocks):
        fig.legend(handles=handles, loc="center left", ncol=ncol,
                   bbox_to_anchor=(x + title_col, y_center),
                   handlelength=hlen, **common)
        fig.text(x, y_center, title, ha="left", va="center",
                 fontsize=FS_TICK, color=C_STEP)


# --- strong-scaling figure geometry (inches, on the FULL_W canvas) -----------
# The panels are placed by hand rather than by subplots(): the three share a
# common log-y *scale* (same inches per decade) and a common x range, but the GPU
# panel is drawn taller so it reaches further down — to where the cuFFTMp point at
# 4 GPUs lives — while the CPU panels floor at Y_AXIS. Each group carries its x
# axis on its own floor, so the GPU ticks/labels sit at the bottom of the GPU
# panel, one tick+caption row lower than the CPU ones. The strip that frees up
# under the (shorter) CPU panels carries the legend.
S_FIG_H = 2.8      # total canvas height; ~1/4 of \textheight, so two still float
S_M_LEFT = 0.52     # y label + "10^-1"-wide tick labels
S_M_RIGHT = 0.07    # half of the last x tick label
S_M_TOP = 0.31      # the FS_SMALL nodes tick row + its caption
S_GAP = 0.09        # between panels
S_TICK_ROW = 0.19   # x tick labels below the shared axis line
S_CAP_ROW = 0.17    # the "nodes"/"GPUs" caption, below the tick labels
S_LEG_TITLE = 0.46  # width reserved for the "grid N" / "method" caption column
# The two-row legend block (~0.38 in) is not reserved: it lives in the strip the
# taller GPU panel frees under the CPU captions, whose height falls out of the
# geometry below. Y_BOT_GPU sets it — push the GPU floor up and it shrinks.
Y_AXIS = 1e-1       # the CPU floor, and where the CPU x axis is drawn
Y_BOT_GPU = 2.2e-2  # the GPU floor: its own x axis, this far below the CPU one.
                    # Low enough that the 4-GPU cuFFTMp point clears the spine now
                    # that the spine is drawn there rather than a decade above.
Y_HEADROOM = 1.8    # slack above the slowest point, for the panel tags
X_HI_GPU = 64 + 32     # the GPU panel stops just past its last point (64 GPUs) rather
                    # than carrying the CPU panels' empty run-out to 128 nodes.


def plot_strong(results, outdir):
    """Strong scaling, faceted by configuration.

    One panel per backend (CPU pure-MPI / CPU hybrid / GPU) on a common log-y
    scale and a common x scale — one GPU counts as one node, as in the text, so
    equal x means equal hardware in every panel. The GPU panel is truncated at
    X_HI_GPU, since its sweep stops at 64 GPUs while the CPU one runs on. The CPU
    panels floor at y = 1e-1; the GPU panel runs half a decade further down, to
    keep the sub-1e-1 cuFFTMp point in view, and carries its x axis on its own
    floor rather than on the CPU one. Inches per decade are shared, so a given
    time still sits at the same height in all three.
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

    # Gather every curve first: the panel geometry follows from the data extent,
    # so nothing can be drawn before the axes rectangles are known.
    para = ("parafaft_mean", "parafaft_std")
    drawn = []  # (title, is_cpu, ParaFaFT groups, baseline groups)
    for key, title, base, is_cpu in panels:
        d = results[key]
        pg = groups_by_N(d, *para, tag=f"{title} ParaFaFT")
        if base:  # in-file baseline (FFTW-MPI)
            bg = groups_by_N(d, *base, tag=f"{title} baseline")
        else:     # GPU baseline (cuFFTMp) lives in a separate file
            mp = results.get("strong_gpu__bench_r2c_cufftmp")
            bg = groups_by_N(mp, *para, tag="cuFFTMp") if mp else []
        drawn.append((title, is_cpu, pg, bg))

    every = [g for _, _, pg, bg in drawn for g in pg + bg]
    ytop = Y_HEADROOM * max(float(np.max(t + e)) for _, _, t, e in every)
    xlo = min(float(np.min(x)) for _, x, _, _ in every)
    xhi = max(float(np.max(x)) for _, x, _, _ in every)
    xlim = (xlo / 1.35, xhi * 1.35)

    fig = plt.figure(figsize=(FULL_W, S_FIG_H))
    fx, fy = lambda v: v / FULL_W, lambda v: v / S_FIG_H  # inches -> figure coords
    n = len(drawn)
    w = (FULL_W - S_M_LEFT - S_M_RIGHT - (n - 1) * S_GAP) / n

    # Vertical layout, in inches from the canvas floor upward. The GPU panel is
    # the tall one and sets the geometry: it runs from just above the bottom
    # tick+caption rows (which are its own x axis) up to `top`. The CPU floor then
    # follows from the shared log scale — Y_AXIS is half a decade above the GPU
    # floor, so h ∝ #decades and a given time is at the same height everywhere —
    # and the rows below the CPU floor plus the leftover strip carry the legend.
    top = S_FIG_H - S_M_TOP
    gpu_axis_y = S_CAP_ROW + S_TICK_ROW      # GPU x-axis line height
    h_gpu = top - gpu_axis_y
    h_unit = h_gpu / np.log10(ytop / Y_BOT_GPU)  # inches per decade, common to all
    h_cpu = h_unit * np.log10(ytop / Y_AXIS)
    axis_y = top - h_cpu                     # CPU x-axis line height

    axes = []
    for i, (title, is_cpu, pg, bg) in enumerate(drawn):
        h = h_cpu if is_cpu else h_gpu
        ax = fig.add_axes([fx(S_M_LEFT + i * (w + S_GAP)), fy(top - h), fx(w), fy(h)])
        axes.append(ax)

        for N, x, t, e in pg:
            ax.errorbar(x, t, yerr=e, marker=nmap[N], color=cmap[N], **PARA)
        for N, x, t, e in bg:
            ax.errorbar(x, t, yerr=e, marker=nmap.get(N, "s"),
                        color=cmap.get(N, "0.4"), **BASE)
        if not is_cpu and pg:  # GPU node boundary, once per panel
            multinode_divider(ax, np.concatenate([g[1] for g in pg]))

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        # Shared x range, except that the GPU panel is cut just past its last point:
        # its sweep ends at 64 GPUs, so the CPU run-out beyond that is empty space.
        ax.set_xlim(xlim[0], xlim[1] if is_cpu else min(xlim[1], X_HI_GPU))
        ax.set_ylim(Y_AXIS if is_cpu else Y_BOT_GPU, ytop)
        # The unit each configuration was launched in goes below the axis, nodes
        # above: CPU cores / nodes, GPUs / nodes. Captions are drawn once per
        # group, figure-centred (below), rather than per panel.
        if is_cpu:
            dual_axis(ax, bottom=(CORES_PER_NODE, FS_CORES), top=(1.0, FS_SMALL))
        else:
            dual_axis(ax, bottom=(1.0, FS_TICK), top=(1.0 / GPUS_PER_NODE, FS_SMALL))
        # One dotted 1/p slope guide per panel (all N share the -1 slope on log-log),
        # anchored to each sweep's own first point and labelled inline rather than
        # in the legend, as the companion figures do.
        ideal_ref(ax, pg)
        panel_label(ax, title)
        if i:
            ax.tick_params(axis="y", labelleft=False)

    axes[0].set_ylabel("time per r2c+c2r transform [s]")

    # x captions belong to their own unit: CPU cores span the CPU panels, GPUs the
    # GPU panel, and each group restates itself in nodes on top. Centre each over
    # the panels it describes; both rows sit at one height across the figure.
    #
    # The two 'nodes' rows are NOT the same count at the same x: the panels are
    # aligned by the paper's 1 GPU = 1 CPU node convention, whereas the GPU top row
    # counts physical 4-GPU nodes. Hence a caption per group rather than one shared
    # one, each naming the ratio on the row that carries the launch unit.
    def span_center(group):
        pos = [ax.get_position() for ax in group]
        return 0.5 * (min(p.x0 for p in pos) + max(p.x1 for p in pos))

    top_cap_y = S_FIG_H - 0.115              # baseline of the "nodes" row
    cpu_axes = [ax for ax, d in zip(axes, drawn) if d[1]]

    # The secondary caption is set at FS_SMALL like its tick row, matching
    # plot_style.nodes_top_axis; the primary one keeps the axis-label size. Each
    # group's caption hangs a tick row under ITS OWN axis line, so the GPU one
    # follows the GPU axis down while both 'nodes' rows stay on the top edge.
    def captions(group, bottom_text, group_axis_y):
        cx = span_center(group)
        for y, text, fs in ((group_axis_y - S_TICK_ROW - S_CAP_ROW, bottom_text, XLABEL_FS),
                            (top_cap_y, "nodes", FS_SMALL)):
            fig.text(cx, fy(y), text, ha="center", va="bottom",
                     fontsize=fs, color=C_STEP)

    if cpu_axes:
        captions(cpu_axes, f"CPU cores ({CORES_PER_NODE}/node)", axis_y)
    for ax, d in zip(axes, drawn):
        if not d[1]:  # GPU panel: one GPU per rank, so the axis counts GPUs
            captions([ax], "GPUs", gpu_axis_y)

    # Legend strip: the space below the CPU caption row that the taller GPU panel
    # frees. One block per CPU panel, each starting at its panel's left edge, so
    # the two captions sit on the same grid as the panels above them. Centre it in
    # whatever that strip turns out to be rather than in a fixed band, since the
    # CPU floor now floats with the data extent.
    if cpu_axes:
        xs = [ax.get_position().x0 for ax in cpu_axes]
        if len(xs) < 2:  # a single CPU panel still has to hold both blocks
            p = cpu_axes[0].get_position()
            xs = [p.x0, p.x0 + 0.5 * p.width]
        leg_top = axis_y - S_TICK_ROW - S_CAP_ROW
        strong_legends(fig, cmap, nmap, xs, fy(leg_top / 2), fx(S_LEG_TITLE))
    _save(fig, outdir, "strong_scaling")


def plot_weak(results, outdir):
    # Each weak point is a different N, so N is not a useful marker channel here;
    # marker just tracks configuration for redundancy with color.
    # Full text width (figure*) at final print size, as the strong figure: one
    # panel plus the three key blocks, which need the right ~quarter of the
    # canvas. Flat enough that two such figures still float on a page.
    fig, ax = plt.subplots(figsize=(FULL_W, 2.75))
    fig.subplots_adjust(left=0.085, right=0.745, top=0.845, bottom=0.155)

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
    ax.set_xlabel(f"CPU cores ({CORES_PER_NODE}/node)  —  grid $N$ grows to keep "
                  "work/process fixed", fontsize=XLABEL_FS)
    ax.set_ylabel("time per r2c+c2r transform [s]")
    # No in-figure title: at final print size the space above the panel belongs
    # to the nodes axis, and what the figure shows is the caption's job — the
    # strong-scaling figure is titleless for the same reason.
    # Same row order as the strong figure: launch unit below, nodes above.
    sec = dual_axis(ax, bottom=(CORES_PER_NODE, FS_CORES), top=(1.0, FS_SMALL))
    sec.set_xlabel("nodes", fontsize=FS_SMALL, labelpad=3)
    key_legends(ax, present, nmap=None)
    _save(fig, outdir, "weak_scaling")


def _save(fig, outdir, stem):
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        # No bbox_inches="tight": the canvas IS the printed size (see PAPER_THEME),
        # everything — legends included — is laid out inside it, and cropping would
        # make the delivered width depend on the labels rather than on figsize.
        fig.savefig(path, dpi=300)
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
