# Quick Start Guide

## TL;DR

```bash
# Navigate to directory
cd /path/to/mpifft

# Build and run all tests
cd test
make run-all

# Compare serial vs MPI (validates correctness)
make compare
```

## Building

```bash
# Build examples
mpicxx example_3d_pencil.cpp -o example_3d_pencil -std=c++11 -I/opt/homebrew/include -L/opt/homebrew/lib -lfftw3
mpicxx example_4d_pencil.cpp -o example_4d_pencil -std=c++11 -I/opt/homebrew/include -L/opt/homebrew/lib -lfftw3

# Or build tests
cd test
make all
```

## Running

```bash
# 3D example (4 MPI ranks, 2×2 processor grid)
mpirun -n 4 ./example_3d_pencil

# 4D example (8 MPI ranks, 2×2×2 processor grid)
mpirun -n 8 ./example_4d_pencil

# Run test suite
cd test
make run-all
```

## Basic Usage

```cpp
#include "mpifft_generic.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Create 3D FFT for 32×32×32 array
    int shape[3] = {32, 32, 32};
    mpifft::PencilFFT<3> fft(shape);

    // Allocate local data
    int local_size = fft.get_local_size();
    std::vector<std::complex<double>> data(local_size);

    // Initialize data (see examples for proper initialization)
    // ...

    // Forward transform
    fft.forward(data.data());

    // Backward transform
    fft.backward(data.data());

    // Normalize
    int N = shape[0] * shape[1] * shape[2];
    for (auto& val : data) val /= N;

    MPI_Finalize();
    return 0;
}
```

## What Works

| Feature | Status |
|---------|--------|
| 3D Forward FFT | ✅ Fully working |
| 3D Backward FFT | ✅ Fully working |
| 4D Forward FFT | ✅ Fully working |
| 4D Backward FFT | ✅ Fully working |
| R2C Forward FFT | ✅ Fully working |
| C2R Backward FFT | ✅ Fully working |
| Serial vs MPI | ✅ Exact match |
| Constant data | ✅ Tested |
| Gaussian data | ✅ Tested |
| Arbitrary data | ✅ Tested |

## Testing

```bash
cd test

# Run all tests
make run-all

# Just compare serial vs MPI
make compare

# Individual tests
mpirun -n 4 ./test_mpi_constant_32
mpirun -n 4 ./test_mpi_gaussian_roundtrip
mpirun -n 4 ./test_8cubed
mpirun -n 4 ./test_4d_roundtrip
```

## Common Issues

### "Cannot find fftw3.h"
```bash
# Install FFTW3
brew install fftw        # macOS
apt install libfftw3-dev # Ubuntu
```

Then update include/library paths in your compile command:
```bash
mpicxx ... -I/path/to/fftw/include -L/path/to/fftw/lib -lfftw3
```

### Incorrect results after roundtrip
Make sure to normalize after backward transform:
```cpp
double scale = 1.0 / (N0 * N1 * N2);
for (auto& val : data) val *= scale;
```

### MPI errors at finalization
Ensure FFT object is destroyed before `MPI_Finalize()`:
```cpp
{
    mpifft::PencilFFT<3> fft(shape);
    // Use fft...
} // Destroyed here
MPI_Finalize(); // Now safe
```

## File Overview

| File | Description |
|------|-------------|
| `mpifft_generic.hpp` | Dimension-agnostic C2C transforms |
| `mpifft_r2c.hpp` | R2C/C2R transforms for real data |
| `fft_backend_fftw.hpp` | FFTW backend implementation |
| `example_3d_pencil.cpp` | 3D usage example |
| `example_4d_pencil.cpp` | 4D usage example |
| `test/` | Comprehensive test suite |
| `test/README.md` | Detailed test documentation |

## Documentation

- `README.md` - Complete user guide
- `test/README.md` - Test suite documentation
- `INSTALL.md` - Installation details
- `SUMMARY.md` - Project overview

## Key Points

1. **Both forward and backward transforms work correctly**
2. **MPI results match serial FFTW exactly** (verified with `make compare`)
3. **Use global coordinates** when initializing distributed data
4. **Remember to normalize** after backward transform
5. **Check test/** for working examples

### R2C Transform Example

For real-valued input data, use the R2C transform for better memory efficiency:

```cpp
#include "mpifft_r2c.hpp"

int global_shape[3] = {32, 32, 32};
mpifft::PencilFFT_R2C<3> fft(global_shape);

std::vector<double> real_input(fft.get_local_real_size());
std::vector<std::complex<double>> complex_output(fft.get_local_complex_size());

fft.forward(real_input.data(), complex_output.data());
fft.backward(complex_output.data(), real_input.data());
```

---

**Bottom Line**: Use `mpifft_generic.hpp` for C2C transforms or `mpifft_r2c.hpp` for real-valued data.
