# MPI-Parallel Multidimensional FFT

A C++ implementation of multidimensional parallel FFT using MPI, based on the algorithm described in:

**Dalcin, L., Mortensen, M., & Keyes, D. E. (2019). Fast parallel multidimensional FFT using advanced MPI. Journal of Parallel and Distributed Computing.**

## Status

**Fully Working**: Both forward and backward transforms tested and validated

### Complex-to-Complex (C2C) Transforms
- **3D FFT**: Tested with constant, Gaussian, and arbitrary data
- **4D FFT**: Tested with constant and Gaussian data
- **5D & 6D FFT**: Tested with Gaussian distributions
- **Serial vs MPI**: Exact match (verified with comparison tests)
- **Generic Implementation**: Dimension-agnostic template for arbitrary D

### Real-to-Complex (R2C) / Complex-to-Real (C2R) Transforms
- **3D R2C/C2R FFT**: Full roundtrip support (real → complex → real)
- **Grid sizes**: Validated for 8³, 24³, 32³, 100³
- **Roundtrip accuracy**: < 1e-15 error across all configurations
- **Process counts**: Tested with 4 and 8 MPI processes

## Features

- **Dimension-agnostic**: Generic template supports arbitrary dimensions (3D, 4D, 5D, 6D, ...)
- **Three implementations**:
  - `parafaft_generic.hpp`: Dimension-agnostic C2C template (recommended for complex data)
  - `parafaft_r2c.hpp`: R2C/C2R transforms for real-valued data (memory efficient)
  - `parafaft_pencil.hpp`: Specialized 3D and 4D C2C versions (legacy)
- **MPI parallelization**: Distributes up to (D-1) dimensions across processor grids
- **No local transposes**: Uses MPI subarray datatypes with `MPI_Alltoallw` to eliminate local data rearrangements
- **Multiple FFT backends**:
  - **FFTW3**: CPU-based FFT operations (default)
  - **cuFFT**: NVIDIA GPU acceleration via CUDA
  - **hipFFT**: AMD GPU acceleration via ROCm/HIP
- **User-friendly API**: Simple `ParaFaFT<D>` and `ParaFaFT_R2C<D>` template interfaces

## Quick Start

```bash
# Build with CMake
mkdir build && cd build
cmake .. -DPARAFAFT_TEST=ON
make

# Run all tests
ctest

# Run tests with verbose output
ctest -V
```

## Key Algorithm Details

### Global Redistribution Method

Unlike traditional methods that require:
1. Local transpose/remapping operations
2. `MPI_Alltoall` communication of contiguous buffers

This implementation:
1. Uses `MPI_Type_create_subarray` to describe discontiguous memory layouts
2. Performs direct exchange with `MPI_Alltoallw` - **no local transposes needed**

### Decomposition Strategy

- Uses **balanced block-contiguous decomposition** (Equation 9 from paper)
- Creates **(D-1)-dimensional Cartesian processor grids**
- Each axis (except the last) can be distributed across a processor subgroup

### Execution Flow (3D Example)

For a 3D array on a 2D processor grid:

1. **Stage A**: Distributed in axes 0,1; local in axis 2
   - FFT along axis 2 (Z)
   - Redistribute: axis 2 → axis 1
2. **Stage B**: Distributed in axes 0,2; local in axis 1
   - FFT along axis 1 (Y)
   - Redistribute: axis 1 → axis 0
3. **Stage C**: Distributed in axes 1,2; local in axis 0
   - FFT along axis 0 (X)

Total: **3 FFTs** and **2 global redistributions**

### R2C/C2R Data Layout

For R2C transforms, the last axis is reduced from N to N/2+1 in complex space due to Hermitian symmetry:
- **Real input**: `[N₀, N₁, N₂]`
- **Complex output**: `[N₀, N₁, N₂/2+1]`

## Requirements

- MPI implementation (OpenMPI, MPICH, etc.)
- FFTW3 library
- C++14 compatible compiler
- CMake 3.18+

optional:
- CUDA Toolkit (for NVIDIA GPU support)
- ROCm/HIP with hipFFT(for AMD GPU support)

## Building with CMake

```bash
# Basic build
mkdir build && cd build
cmake ..
make

# With CUDA support (NVIDIA GPUs)
cmake .. -DPARAFAFT_CUDA=ON

# With HIP support (AMD GPUs)
cmake .. -DPARAFAFT_HIP=ON

# With tests
cmake .. -DPARAFAFT_TEST=ON

# Both CUDA and tests
cmake .. -DPARAFAFT_CUDA=ON -DPARAFAFT_TEST=ON

# Both HIP and tests
cmake .. -DPARAFAFT_HIP=ON -DPARAFAFT_TEST=ON

# Install the library
cmake --install . --prefix /your/install/path
```

### CMake Options

| Option          | Default | Description                            |
| --------------- | ------- | -------------------------------------- |
| `PARAFAFT_CUDA` | `OFF`   | Enable CUDA/cuFFT backend (NVIDIA GPU) |
| `PARAFAFT_HIP`  | `OFF`   | Enable HIP/hipFFT backend (AMD GPU)    |
| `PARAFAFT_TEST` | `OFF`   | Build the test suite                   |

### Using in Your CMake Project

After installation, use `find_package` to link against parafaft:

```cmake
find_package(parafaft REQUIRED)
target_link_libraries(your_target PRIVATE parafaft::parafaft)
```

Or add as a subdirectory:

```cmake
add_subdirectory(path/to/parafaft)
target_link_libraries(your_target PRIVATE parafaft::parafaft)
```

## Running

```bash
# 3D example (4 MPI ranks)
mpirun -n 4 ./examples/example_3d_pencil

# 4D example (8 MPI ranks)
mpirun -n 8 ./examples/example_4d_pencil

# Run tests
cd build
ctest
```

Note that if you have both CUDA and HIP/ROCm installed, it may be necessary to specify the accelerator backend for MPI. For OpenMPI, one can either use 
```bash
OMPI_MCA_accelerator="rocm" mpirun -n 4 ./your_executable
```
or
```bash
mpirun --mca accelerator rocm -n 4 ./your_executable
```
To enable the rocm accelerator in the entire current session, one can also simply set
```bash
export OMPI_MCA_accelerator="rocm"
```
The corresponding environment variable for CUDA is `cuda`.

## Usage Examples

### Complex-to-Complex (C2C) Transform

```cpp
#include "parafaft_generic.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Works for any dimension!
    int global_shape[3] = {32, 32, 32};
    parafaft::ParaFaFT<3> fft(global_shape);

    // Allocate local data
    int local_size = fft.get_local_size();
    std::vector<std::complex<double>> data(local_size);

    // Initialize data...
    int local_shape[3], global_start[3];
    fft.get_local_shape(local_shape);
    fft.get_global_start(global_start);

    // Forward FFT (in-place)
    fft.forward(data.data());

    // Backward FFT (in-place)
    fft.backward(data.data());

    // Normalize (FFTW convention)
    double scale = 1.0 / (global_shape[0] * global_shape[1] * global_shape[2]);
    for (auto& val : data) val *= scale;

    MPI_Finalize();
    return 0;
}
```

### Real-to-Complex (R2C) / Complex-to-Real (C2R) Transform

```cpp
#include "parafaft_r2c.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int global_shape[3] = {32, 32, 32};
    parafaft::ParaFaFT_R2C<3> fft(global_shape);

    // Get local sizes
    int local_real_size = fft.get_local_real_size();
    int local_complex_size = fft.get_local_complex_size();

    // Allocate arrays
    std::vector<double> real_data(local_real_size);
    std::vector<double> real_result(local_real_size);
    std::vector<std::complex<double>> complex_data(local_complex_size);

    // Initialize real data...
    int local_real_shape[3], real_start[3];
    fft.get_local_real_shape(local_real_shape);
    fft.get_real_global_start(real_start);

    // Forward R2C FFT: real -> complex
    fft.forward(real_data.data(), complex_data.data());

    // ... process in frequency domain ...

    // Backward C2R FFT: complex -> real
    fft.backward(complex_data.data(), real_result.data());

    // Normalize (FFTW convention)
    long long total_size = (long long)global_shape[0] * global_shape[1] * global_shape[2];
    double scale = 1.0 / total_size;
    for (auto& val : real_result) val *= scale;

    MPI_Finalize();
    return 0;
}
```

## Testing

Comprehensive test suite in `test/` directory:

### C2C Tests

#### Specialized Implementation
- **test_mpi_constant_32**: 32³ constant values
- **test_mpi_gaussian_roundtrip**: 32³ Gaussian distribution with file output
- **test_8cubed**: 8³ sequential values
- **test_4d_roundtrip**: 4⁴ constant values
- **reference_serial_3d**: Serial FFTW reference for validation

#### Generic Implementation
*Currently no dedicated generic tests. All specialized tests use the generic implementation internally.*

### R2C/C2R Tests
- **test_mpi_r2c_gaussian**: Forward R2C transform validation
- **test_mpi_r2c_roundtrip**: Full roundtrip (R2C → C2R) accuracy test

Run tests:
```bash
cd build
ctest                # Run all tests
ctest -V             # Verbose output
ctest -R r2c         # Run only R2C tests
ctest -R c2c         # Run only C2C tests
```

See `test/README.md` for detailed test documentation.

## Code Structure

### Main Headers

#### `parafaft_generic.hpp` (Recommended for C2C)
Dimension-agnostic template library for complex-to-complex transforms:
- **`decompose()`**: Balanced block-contiguous decomposition
- **`subarray()`**: MPI subarray datatype creation
- **`exchange()`**: Global redistribution via `MPI_Alltoallw`
- **`ParaFaFT<D>`**: Template class for arbitrary D-dimensional FFT
  - Constructor: Creates (D-1)D processor grid automatically
  - `forward()`/`backward()`: In-place FFT operations
  - `get_local_size()`, `get_local_shape()`, `get_global_start()`: Query functions

**Supports**: 3D, 4D, 5D, 6D, and higher dimensions.

#### `parafaft_r2c.hpp` (Recommended for Real Data)
R2C/C2R transforms for real-valued input data:
- **`ParaFaFT_R2C<D>`**: Template class for real-to-complex FFT
  - `forward(real_input, complex_output)`: R2C transform
  - `backward(complex_input, real_output)`: C2R transform
  - `get_local_real_size()`, `get_local_real_shape()`: Real-space queries
  - `get_local_complex_size()`, `get_local_complex_shape()`: Complex-space queries

**Memory advantage**: Stores only N/2+1 complex values on the last axis instead of N.

#### FFT Backends

##### `fft_backend_fftw.hpp` (CPU)
FFTW3 backend for CPU-based operations:
- Manages FFTW plans for C2C, R2C, and C2R transforms
- Handles batched 1D FFTs with arbitrary strides
- Automatic plan cleanup via RAII

##### `fft_backend_cufft.hpp` (NVIDIA GPU)
cuFFT backend for NVIDIA GPU acceleration:
- Requires CUDA Toolkit and cuFFT library
- Includes `cuvector<T>` device memory wrapper
- Supports C2C, R2C, and C2R in-place transforms
- Enable with `-DPARAFAFT_CUDA=ON`

##### `fft_backend_hipfft.hpp` (AMD GPU)
hipFFT backend for AMD GPU acceleration:
- Requires ROCm and hipFFT library
- Includes `hipvector<T>` device memory wrapper
- Supports C2C, R2C, and C2R in-place transforms
- Enable with `-DPARAFAFT_HIP=ON`

#### `parafaft_pencil.hpp` (Legacy)
Specialized implementations for 3D and 4D C2C transforms:
- Explicit template specializations: `ParaFaFT<3>` and `ParaFaFT<4>`
- Same API as generic version
- **Note**: Generic version is now recommended for all use cases

### Benchmarks
See `benchmark/` directory:
- **benchmark_mpi.cpp**: Specialized 3D implementation benchmark
- **benchmark_generic.cpp**: Generic implementation benchmark
- **compare_implementations.sh**: Automated comparison script
- **PERFORMANCE_RESULTS.md**: Detailed performance analysis

## Performance Considerations

From the paper's benchmarks on Cray XC40:

- **Competitive with P3DFFT, 2DECOMP&FFT**: Often 5-10% faster
- **Better for distributed (inter-node) communication**: Excels when avoiding shared memory
- **Scales well**: Good weak and strong scaling up to thousands of cores

### Trade-offs

**Advantages:**
- Eliminates costly local transpose operations
- Simpler, more maintainable code
- Dimension-agnostic implementation
- Exact match with serial FFTW
- R2C transforms save ~50% memory for real data

**Considerations:**
- `MPI_Alltoallw` may be slower than `MPI_Alltoall` for contiguous data
- Relies on MPI implementation quality for subarray datatype handling

## References

Dalcin, L., Mortensen, M., & Keyes, D. E. (2019). Fast parallel multidimensional FFT using advanced MPI. *Journal of Parallel and Distributed Computing*, 128, 137-150.

## License

This implementation is provided as-is for educational and research purposes.
