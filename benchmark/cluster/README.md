# ParaFaFT scaling benchmarks — PC2 Noctua 2

Strong- and weak-scaling benchmarks for ParaFaFT (3D, R2C), split into **CPU**
(`normal`, FFTW backend) and **NVIDIA GPU** (`gpu`, cuFFT backend).

Each sweep point is submitted as its **own independent SLURM job**, sized and
time-limited to that point. Short jobs (many nodes, or small weak-scaling points)
schedule and start without waiting for the long ones — much easier to get compute
time than one big reservation.

Every run also compares ParaFaFT against a reference **distributed** FFT:

| Study | ParaFaFT | Baseline (same run) |
|-------|----------|---------------------|
| CPU   | FFTW     | **FFTW-MPI** (slab) — built in, always on |
| GPU   | cuFFT    | **cuFFTMp** (NVIDIA multi-GPU) — optional, `USE_CUFFTMP=1` |

## Files

| File | Role |
|------|------|
| `env_cpu.sh` / `env_gpu.sh` | Modules + `build_cpu` / `build_gpu` helpers |
| `submit_strong_cpu.sh`, `submit_weak_cpu.sh` | **Launchers** (pure MPI, 128 ranks/node) — build once, then `sbatch` one job per point |
| `submit_strong_cpu_hybrid.sh`, `submit_weak_cpu_hybrid.sh` | Same sweeps, **hybrid MPI+OpenMP** (4 ranks/node × 32 threads) |
| `submit_strong_gpu.sh`, `submit_weak_gpu.sh` | Launchers for GPU |
| `job_cpu.slurm` / `job_gpu.slurm` | Single-point job **templates** (not submitted by hand) |
| `collect.sh` | Merge per-job CSVs into one CSV per study |
| `results/` | Output (per-point subdirs + logs), created on first run |

## 0. One-time setup (login node)

The `module load` lines in `env_cpu.sh` / `env_gpu.sh` are already filled in for
Noctua 2 — double-check them against `module spider <name>` if a load fails
(FFTW.MPI transitively loads FFTW, so `$EBROOTFFTW` and `$EBROOTFFTWMPI` are both
set for CMake). No SLURM account is required on this system.

You don't need to build manually — each launcher builds once up front. To build
explicitly anyway:

```bash
source benchmark/cluster/env_cpu.sh && build_cpu     # -> build_cpu/bench/bench_r2c
source benchmark/cluster/env_gpu.sh && build_gpu     # -> build_gpu/bench/bench_r2c_cuda
USE_CUFFTMP=1 build_gpu                               # + bench_r2c_cufftmp (needs NVSHMEM)
```

## 1. Launch (from `benchmark/cluster/`)

Each launcher builds the binary once, then submits one job per point:

```bash
cd benchmark/cluster
./submit_strong_cpu.sh
./submit_weak_cpu.sh
./submit_strong_gpu.sh
./submit_weak_gpu.sh
# GPU with the cuFFTMp baseline in each job:
USE_CUFFTMP=1 ./submit_strong_gpu.sh
# hybrid MPI+OpenMP CPU runs (4 ranks/node x 32 threads), for comparison:
./submit_strong_cpu_hybrid.sh
./submit_weak_cpu_hybrid.sh
```

The pure-MPI and hybrid CPU sweeps use identical grids and node counts, so they
compare directly at matched hardware. `plot.py` places CPU series on the x-axis
by **total cores** (`mpi_procs × threads`), so pure (128×1) and hybrid (4×32)
line up at the same node count. Change the thread count by editing
`RANKS_PER_NODE` at the top of a hybrid launcher (e.g. `2` → 64 threads/rank).

Watch them with `squeue -u $USER`. Each point is a separate job id, so failures
or requeues are isolated to one point.

## 2. Config knobs (top of each `submit_*.sh`)

- `POINTS=(...)` — the sweep, one entry per job, `"... TIME ITERS"`:
  - strong CPU: `"N NODES TIME ITERS"` · weak CPU: `"NODES N TIME ITERS"`
  - strong GPU: `"N GPUS TIME ITERS"` · weak GPU: `"GPUS N TIME ITERS"`
  - `TIME` is that job's walltime — keep it tight so short jobs schedule fast.
  - Optional hero/larger grids are pre-written but commented out.
- `RANKS_PER_NODE` (CPU) — `128` = pure MPI (default). Set e.g. `8` for hybrid
  MPI+OpenMP (auto `--cpus-per-task=16`, `OMP_NUM_THREADS=16`).
- `USE_CUFFTMP=1` (GPU) — also run cuFFTMp in every job.

GPU resources are derived per point: `nodes = ceil(GPUs/4)`,
`--ntasks-per-node = min(GPUs,4)`, `--gres=gpu:a100:<that>`. Ranks self-select
their GPU (`local_rank % num_devices`), so steps are intentionally **not**
GPU-bound.

## 3. Collect

Each job writes `bench_r2c*.csv` into its own
`results/<study>/N…_<n|g>…_<jobid>/` dir (no clobber between concurrent jobs).
After the jobs finish:

```bash
./collect.sh
```

produces one merged, `mpi_procs`-sorted CSV per study/binary, e.g.:

- `results/strong_cpu__bench_r2c.csv`
  — `mpi_procs,threads,N,parafaft_mean,parafaft_std,fftw_mean,fftw_std,iterations`
  (plot `parafaft_mean` vs `fftw_mean` against `mpi_procs`).
- `results/weak_cpu__bench_r2c.csv` (plot against `N`).
- `results/strong_gpu__bench_r2c_cuda.csv` (+ `…__bench_r2c_cufftmp.csv` if enabled).

All times are seconds for **one forward + one backward** transform.

## 4. Plot

```bash
python3 plot.py            # reads results/, writes results/plots/{strong,weak}_scaling.{png,pdf}
python3 plot.py <dir>      # or point it at another results dir
```

Auto-detects whichever collected CSVs exist (partial data is fine) and emits a
**strong** (log-log, with 1/P ideal references) and a **weak** figure (log-log,
flat = perfect). Three orthogonal visual channels keep the many lines readable:

- **color** = configuration — CPU pure MPI (blue), CPU hybrid (green), GPU (red)
- **line style** = method — solid ParaFaFT, dashed baseline (FFTW-MPI / cuFFTMp)
- **marker** = grid size N (strong plot)

A compact key legend explains the channels. Requires `matplotlib` and `numpy`.

## Files table addendum

| File | Role |
|------|------|
| `collect.sh` | Merge per-job CSVs into one CSV per study |
| `plot.py` | Strong + weak scaling figures from the merged CSVs |

## Notes

- GPU runs need **CUDA-aware MPI**; `env_gpu.sh` sets `OMPI_MCA_opal_cuda_support=1`.
- Peak memory (3D R2C) is `~16–24·N³` bytes total across ranks; the ParaFaFT and
  FFTW-MPI buffers don't coexist, so this bounds the minimum node/GPU count
  (N=4096 needs ≥8 nodes, N=8192 ≥56 nodes).
- Rebuilding: delete `build_cpu/` or `build_gpu/` and re-run a launcher.
