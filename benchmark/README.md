# Performance Benchmarks

Benchmarking the MPI parallel FFT implementation against serial FFTW on a 256³ lattice.

## Test Configuration

- **Array size**: 256 × 256 × 256 = 16,777,216 elements
- **Memory (serial)**: 256 MB (single array)
- **Memory (MPI, 4 ranks)**: ~192 MB per rank (3 internal arrays for stages A, B, C)
- **Data**: 3D Gaussian distribution (σ = 16)
- **Compiler flags**: `-O3` optimization
- **MPI configuration**: 4 ranks (2×2 processor grid)

## Results

### Serial FFTW (1 core)

```
Plan creation:    0.002553 seconds
Forward FFT:      0.477149 seconds
Backward FFT:     0.473293 seconds
Total FFT time:   0.950442 seconds
```

### MPI Parallel (4 cores)

```
Setup time:       0.012730 seconds
Forward FFT:      0.156767 seconds
Backward FFT:     0.167254 seconds
Total FFT time:   0.324021 seconds
```

## Performance Comparison

| Metric | Serial | MPI (4 ranks) | Speedup |
|--------|--------|---------------|---------|
| Forward FFT | 0.477 s | 0.157 s | **3.04×** |
| Backward FFT | 0.473 s | 0.167 s | **2.83×** |
| **Total FFT** | **0.950 s** | **0.324 s** | **2.93×** |

## Analysis

- **Overall speedup**: 2.93× on 4 cores (73% parallel efficiency)
- **Forward transform**: Slightly faster than backward (3.04× vs 2.83×)
- **Memory footprint**: Serial uses 256 MB; MPI uses ~192 MB per rank (768 MB total across 4 ranks)
- **Accuracy**: Both achieve machine precision (max error ~5.8e-16)

### Memory Considerations

The MPI implementation uses **3 internal arrays** (arrayA, arrayB, arrayC) to hold data at different stages of the pencil decomposition:
- Stage A: [n0, n1, N2] - distributed in axes 0,1
- Stage B: [n0, N1, n1] - distributed in axes 0,2
- Stage C: [N0, n0, n1] - distributed in axes 1,2

Each array is sized for its respective stage, and all three are needed because the `exchange()` operation requires separate source and destination buffers. This means memory usage per rank is approximately **3× the local data size**.

For the 256³ test with 4 ranks:
- Local data size: 128×128×256 = 4,194,304 elements = 64 MB
- **Actual memory per rank**: ~192 MB (3 arrays)
- **Total across 4 ranks**: ~768 MB (vs 256 MB serial)

### Parallel Efficiency

Parallel efficiency = Speedup / Number of cores = 2.93 / 4 = **73.2%**

This is good efficiency considering:
- Communication overhead from MPI data exchanges (2 global redistributions per transform)
- No local transposes (trades computation for communication)
- Memory overhead for multiple stage buffers

### Expected Scaling

Based on Dalcin et al. (2019), the algorithm:
- Scales well to thousands of cores
- Performance depends on network quality (benefits from fast interconnects)
- Better efficiency expected on larger problems (communication becomes smaller fraction)
- Memory per rank decreases with more ranks (but total memory increases due to 3× factor)

## Running the Benchmarks

```bash
# Build
make all

# Run serial
make run-serial

# Run MPI (4 ranks)
make run-mpi

# Run both
make run-all
```

## Testing Other Configurations

```bash
# Different number of MPI ranks
mpirun -n 8 ./benchmark_mpi   # 8 ranks (more parallelism, less memory per rank)
mpirun -n 16 ./benchmark_mpi  # 16 ranks

# Note: Number of ranks should form a valid processor grid
# For 3D: n_ranks = n0 × n1 where n0, n1 divide the array dimensions
```

## Hardware Information

Run on macOS with:
- Processor: Apple Silicon / Intel (check with `sysctl -n machdep.cpu.brand_string`)
- FFTW3 version: 3.3.10
- OpenMPI version: (check with `mpirun --version`)

---

**Key Takeaway**: The MPI implementation achieves ~3× speedup on 4 cores for a 256³ FFT with 73% parallel efficiency. Memory usage is ~3× higher per rank than the theoretical minimum due to separate stage buffers, but this trade-off enables the algorithm's key feature: no local transposes required.
