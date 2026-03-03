# ParaFaFT: An MPI-Parallel Fast Fourier Transform in arbitrary dimensions with Support for GPUs

ParaFaFT (**Para**llel **Fa**st **F**ourier **T**ransform) provides a C++ implementation of multidimensional parallel FFT using MPI, based on the algorithm described in
[***Dalcin, L., Mortensen, M., & Keyes, D. E. (2019). Fast parallel multidimensional FFT using advanced MPI. Journal of Parallel and Distributed Computing***](https://arxiv.org/abs/1804.09536).

## Features

- **Dimension-agnostic**: Generic template supports arbitrary dimensions $D \geq 2$.
- **C2C and R2C/C2R**:
  - `parafaft_c2c.hpp`: Provides through `ParaFaFT<D>`, complex fourier transforms for arbitrary dimensions.
  - `parafaft_r2c.hpp`: Provides through `ParaFaFT_R2C<D>`, real-to-complex and complex-to-real transforms with memory-efficient layouts.
- **No local transposes**: Uses MPI subarray datatypes with `MPI_Alltoallw` to eliminate local data rearrangements
- **Supported FFT backends**:
  - **FFTW3**: CPU-based FFT operations (default)
  - **cuFFT**: NVIDIA GPU acceleration via CUDA
  - **hipFFT**: AMD GPU acceleration via ROCm/HIP
- **User-friendly API**: Simple `ParaFaFT<D>` and `ParaFaFT_R2C<D>` template interfaces.

## Quick Start

Do a manual build and run the tests:

```bash
# Configure with CMake
mkdir build && cd build
cmake .. -DPARAFAFT_TEST=ON
# Build and run all tests (ParaFaFT is header-only, so no separate library target)
cmake --build . --target all 
ctest
```

For GPU support, see the [Building with CMake](#building-with-cmake) section below.

## Requirements

- MPI implementation (OpenMPI, MPICH, etc.)
- FFTW3 library
- C++14 compatible compiler
- CMake 3.19+

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
find_package(parafaft REQUIRED HINTS /path/to/parafaft/install)
target_link_libraries(your_target PRIVATE parafaft::parafaft)
```

Alternatively, skip the installation and get it directly in your project's CMake using `fetch_content`:

```cmake
# Enable CUDA support if desired, directly in your CMakeLists.txt
set(PARAFAFT_CUDA ON CACHE BOOL "Enable CUDA/cuFFT backend")

include(FetchContent)
FetchContent_Declare(
  ParaFaFT
  GIT_REPOSITORY https://github.com/aflorio2/parafaft.git
  GIT_TAG        main
  )
FetchContent_MakeAvailable(ParaFaFT)

target_link_libraries(your_target PRIVATE parafaft::parafaft)
```


## Usage Examples

### Complex-to-Complex (C2C) Transform

```cpp
#include "parafaft_c2c.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Works for any dimension!
    const int global_shape[3] = {32, 32, 32};
    parafaft::ParaFaFT<3> fft(global_shape);

    // Allocate local data
    const int local_size = fft.get_local_size();
    std::vector<std::complex<double>> data(local_size);

    // Initialize data
    int local_shape[3], global_start[3];
    fft.get_local_shape(local_shape);
    fft.get_global_start(global_start);
    // ...

    // Forward FFT (in-place)
    fft.forward(data.data());

    // Backward FFT (in-place)
    fft.backward(data.data());

    // Normalize (FFTW convention)
    const double scale = 1.0 / (global_shape[0] * global_shape[1] * global_shape[2]);
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

    const int global_shape[3] = {32, 32, 32};
    parafaft::ParaFaFT_R2C<3> fft(global_shape);

    // Get local size, required for R2C transforms. This may be larger than N₀*N₁*(N₂+2) due to additional memory requirements for intermediate stages.
    const int local_padded_size = fft.get_required_output_size();

    // Allocate arrays
    std::vector<double> real_data(local_padded_size);

    // Initialize real data...
    int local_real_shape[3], real_start[3];
    fft.get_local_real_shape(local_real_shape);
    fft.get_real_global_start(real_start);

    // Fill the first N₀*N₁*(N₂+2) elements with your data ...

    // Forward R2C FFT: real -> complex (in-place)
    // Note: the data passed must be padded to the required size for the R2C transform, i.e. the last (smallest) dimension must have N + 2 (real) elements, where the last 2 are padding for the complex output (see also the fftw documentation for R2C transforms)
    fft.forward_in_place(real_data.data());

    // ... process in frequency domain ...

    // Backward C2R FFT: complex -> real
    fft.backward_in_place(real_data.data());

    // Normalize (FFTW convention)
    const double scale = 1.0 / (global_shape[0] * global_shape[1] * global_shape[2]);
    for (auto& val : real_data) val *= scale;

    MPI_Finalize();
    return 0;
}
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

In general, ParaFaFT follows the same memory layout as FFTW for R2C/C2R transforms, ensuring compatibility and efficient memory usage.

## FFT Backends

##### `fft_backend_fftw.hpp` (CPU)
FFTW3 backend for CPU-based operations, always available.

##### `fft_backend_cufft.hpp` (NVIDIA GPU)
cuFFT backend for NVIDIA GPU acceleration:
- Requires CUDA Toolkit and cuFFT library
- Includes `cuvector<T>` device memory wrapper
- Enable with `-DPARAFAFT_CUDA=ON`

##### `fft_backend_hipfft.hpp` (AMD GPU)
hipFFT backend for AMD GPU acceleration:
- Requires ROCm and hipFFT library
- Includes `hipvector<T>` device memory wrapper
- Enable with `-DPARAFAFT_HIP=ON`

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
[Dalcin, L., Mortensen, M., & Keyes, D. E. (2019). Fast parallel multidimensional FFT using advanced MPI. *Journal of Parallel and Distributed Computing*, 128, 137-150.](https://arxiv.org/abs/1804.09536).


## License
This implementation is provided as-is for educational and research purposes.
