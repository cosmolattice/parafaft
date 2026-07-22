# GPU exchange optimizations (cuFFT backend)

Context: the pencil-decomposition GPU path (`ParaFaFT_R2C` / `ParaFaFT_C2C`
with `CuFFTBackend`) is ~3× slower than NVIDIA cuFFTMp on a single node
(4×A100, NVLink) and ~1.4× slower at 2–4 nodes. The gap is **not** primarily
decomposition geometry — for the FFT itself, cuFFTMp's slab does *one* global
transpose while our pencil does *two*, and slab is communication-optimal at
≤ N ranks. The remaining single-node gap is per-exchange overhead in
`exchange_hybrid` / `exchange_packed` (`parafaft_common.hpp`): explicit pack
kernels, host-blocking synchronization, and no compute/comm overlap. cuFFTMp
does the reshape with device-initiated NVSHMEM puts — no host round-trip, no
explicit pack.

These three changes attack that overhead **without abandoning pencils**.
Ordered by impact. All require building and running on the PC2 GPU checkout —
the CUDA path cannot be compiled or validated on the local (CPU-only) mount, so
each must be tested there (correctness: the existing R2C/C2C tests at np=4,8;
performance: `bench_r2c_cuda`).

---

## 1. Remove the per-exchange host `sync()` + `MPI_Barrier` (highest impact)

**Status: NOT a deletion — a design change. Do not simply remove the fences.**

`exchange_hybrid` currently issues, per call, two host-blocking
synchronization points that are *load-bearing* for the shared `pack_buf`:

- `backend.sync()` + `MPI_Barrier(comm)` at
  `parafaft_common.hpp:732–735` — every rank must finish packing before any
  peer reads its `pack_buf` over IPC.
- `backend.sync()` + `MPI_Barrier(comm)` at `:832–841` — every peer must
  finish reading a rank's `pack_buf` before the next exchange overwrites it.

Each is a full `cudaStreamSynchronize` (host waits for the GPU) followed by a
collective barrier. On one node the transfer over NVLink is cheap, so these
two host round-trips dominate the exchange time. There are two redistributions
per forward transform (D−1=2 for 3D) and again on backward → ~4 sync+barrier
pairs per step.

They cannot be removed as-is: dropping either races the pack-buffer producer
against its IPC consumers (verified by walking the cross-call sequence — a rank
finishing its reads late collides with a faster rank overwriting the buffer in
the next exchange).

### Safe transformation: double-buffer `pack_buf`

Allocate **two** pack buffers and alternate per exchange (ping-pong). Then
exchange N+1 packs into buffer B while peers may still be reading buffer A from
exchange N — no reuse hazard. This lets the **final** `sync()` + `MPI_Barrier`
(`:832–841`) be dropped entirely: the barrier at the *start* of the next
exchange on the same buffer (`:735`, two exchanges later) already provides a
global fence between any write of buffer A and the previous reads of buffer A.
The unpack (`:844`) is stream-ordered after the P2P reads, so it needs no host
sync of its own once the barrier is gone.

Net: **~1 host sync + 1 collective barrier removed per exchange** (halves the
host-side serialization), keeping full correctness.

Touch points:
- `parafaft_r2c.hpp` / `parafaft_c2c.hpp`: allocate `pack_buffer_[2]`, pass the
  alternating buffer into `exchange_hybrid`, track parity across the
  forward/backward stage loops.
- `setup_p2p()`: the IPC-handle `MPI_Allgather` must exchange handles for
  **both** buffers (peers need the pointer for whichever buffer this exchange
  uses). ⚠️ This is the same collective that currently deadlocks multi-node —
  extend it carefully and keep the collective call count identical across
  ranks.
- `exchange_hybrid`: remove `:832` sync + `:841` barrier; the caller guarantees
  buffer alternation.

### Alternative: CUDA IPC-event ordering

Instead of double-buffering, exchange `cudaIpcEventHandle_t`s in `setup_p2p()`
and have a reader wait on the producer's "pack done" event on the stream
(`cudaStreamWaitEvent`) rather than a host barrier — device-side signaling,
closer to what NVSHMEM does. Removes the host round-trip without a second
buffer, but is a larger change to the backend (event create/record/IPC) and to
`setup_p2p`. Prefer double-buffering first; revisit events if the barrier still
shows up in profiles.

### Validation
- Correctness: R2C and C2C tests at np=4 and np=8 must stay bit-identical
  (the removed synchronization guards data movement, not values, so any diff is
  a real race).
- Perf: `bench_r2c_cuda` at 4 GPUs (single node) — this is where the win should
  be largest; expect the single-node gap vs cuFFTMp to shrink from ~3×.

---

## 2. Overlap pack / transfer / unpack with FFT compute

The exchange is bulk-synchronous: FFT all → pack all → transfer all → unpack
all → FFT. cuFFTMp pipelines the reshape with the batched 1D FFTs. We could
hide much of the pack/unpack by overlapping it with the adjacent stage FFT on
the same or a second CUDA stream:

- Enqueue the unpack of exchange N and the FFT of stage N+1 so the unpack
  overlaps independent work.
- Split the pack into per-peer chunks and start transfers for early peers while
  later peers are still packing (partial overlap of pack and transfer).
- Requires stream/event dependency tracking rather than the current single
  ordered stream + full `sync()`. Best done *after* change 1, since removing the
  host barriers is what frees the stream to overlap.

Impact: moderate; compounds with 1. Lower priority because it needs the
synchronization restructure from 1 first.

---

## 3. Slab (1D-grid) path for 3D at low rank counts

For a standalone 3D FFT at ≤ N ranks, a slab decomposition does **one**
transpose vs the pencil's two, moving ~half the aggregate bytes — this is a
large part of cuFFTMp's structural advantage, independent of transport. A
special-case slab path (1D process grid, single redistribution) for 3D would
match cuFFTMp's algorithm at the rank counts we benchmark (≤ 32 GPUs).

Caveats:
- Slab caps parallelism at N ranks and grows communication surface at high G —
  keep the pencil path as the default and select slab only for 3D below a rank
  threshold.
- Only worth it when the FFT is standalone. For the coupled
  lattice-FFT + halo use case (TempLat), slabs wreck halo surface-to-volume, so
  this must be opt-in, not the default.
- Largest code change of the three (new redistribution path + selection
  heuristic); lowest priority. Consider only if 1 + 2 don't close enough of the
  gap.

---

## Not on this list (deliberately)

- **Switching the default to slabs** — rejected: pencils are the right choice
  for the coupled FFT+halo workload; slabs only help a standalone FFT.
- **Replacing transport with NVSHMEM** — that is essentially reimplementing
  cuFFTMp; out of scope for "aligning the pencil path."
- **The multi-node `setup_p2p()` deadlock** — separate, higher-priority bug
  (blocks pencils on GPU entirely past the intra-node case); tracked in the
  TempLat-side `docs/parafaft-hang-bugreport.md`. Change 1 must not regress it.
