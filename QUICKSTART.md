# Quick Start Guide

## TL;DR

```bash
# Navigate to directory
cd /path/to/parafaft

# Build with CMake
mkdir build && cd build
cmake .. -DPARAFAFT_TEST=ON
make

# Run all tests
ctest
```

## Building with CMake

```bash
# Basic build
mkdir build && cd build
cmake ..
make

# With CUDA support
cmake .. -DPARAFAFT_CUDA=ON

# With HIP support (AMD GPUs)
cmake .. -DPARAFAFT_HIP=ON

# With tests
cmake .. -DPARAFAFT_TEST=ON

# Both CUDA and tests
cmake .. -DPARAFAFT_CUDA=ON -DPARAFAFT_TEST=ON

# Both HIP and tests
cmake .. -DPARAFAFT_HIP=ON -DPARAFAFT_TEST=ON
```

### CMake Options

| Option          | Default | Description                     |
| --------------- | ------- | ------------------------------- |
| `PARAFAFT_CUDA` | `OFF`   | Enable CUDA/cuFFT backend       |
| `PARAFAFT_HIP`  | `OFF`   | Enable HIP/hipFFT backend (AMD) |
| `PARAFAFT_TEST` | `OFF`   | Build the test suite            |

### Using in Your CMake Project

```cmake
# Option 1: After installation
find_package(parafaft REQUIRED)
target_link_libraries(your_target PRIVATE parafaft::parafaft)

# Option 2: As subdirectory
add_subdirectory(path/to/parafaft)
target_link_libraries(your_target PRIVATE parafaft::parafaft)
```

## Running

```bash
# 3D example (4 MPI ranks, 2×2 processor grid)
mpirun -n 4 ./examples/example_3d_pencil

# 4D example (8 MPI ranks, 2×2×2 processor grid)
mpirun -n 8 ./examples/example_4d_pencil

# Run test suite
cd build
ctest
```

## Basic Usage

```cpp
#include "parafaft_generic.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Create 3D FFT for 32×32×32 array
    int shape[3] = {32, 32, 32};
    parafaft::ParaFaFT<3> fft(shape);

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

| Feature          | Status          |
| ---------------- | --------------- |
| 3D Forward FFT   | ✅ Fully working |
| 3D Backward FFT  | ✅ Fully working |
| 4D Forward FFT   | ✅ Fully working |
| 4D Backward FFT  | ✅ Fully working |
| R2C Forward FFT  | ✅ Fully working |
| C2R Backward FFT | ✅ Fully working |
| Serial vs MPI    | ✅ Exact match   |
| Constant data    | ✅ Tested        |
| Gaussian data    | ✅ Tested        |
| Arbitrary data   | ✅ Tested        |

## Testing

```bash
cd build
ctest                    # Run all tests
ctest -V                 # Verbose output
ctest -R r2c             # Run only R2C tests
ctest -R c2c             # Run only C2C tests
```

## Common Issues

### "Cannot find fftw3.h"
```bash
# Install FFTW3
brew install fftw        # macOS
apt install libfftw3-dev # Ubuntu
```

Then update the cmake build:
```bash
cmake .. -DFFTW3_INCLUDE_DIR=/path/to/fftw/include -DFFTW3_LIBRARY=/path/to/fftw/lib/libfftw3.so
make
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
    parafaft::ParaFaFT<3> fft(shape);
    // Use fft...
} // Destroyed here
MPI_Finalize(); // Now safe
```

## File Overview

| File                                    | Description                               |
| --------------------------------------- | ----------------------------------------- |
| `CMakeLists.txt`                        | CMake build configuration                 |
| `parafaft_generic.hpp`                  | Dimension-agnostic C2C transforms         |
| `parafaft_r2c.hpp`                      | R2C/C2R transforms for real data          |
| `backend/fft_backend.hpp`               | Backend interface                         |
| `backend/fftw3/fft_backend_fftw.hpp`    | FFTW backend implementation (CPU)         |
| `backend/cufft/fft_backend_cufft.hpp`   | cuFFT backend implementation (NVIDIA GPU) |
| `backend/hipfft/fft_backend_hipfft.hpp` | hipFFT backend implementation (AMD GPU)   |
| `test/`                                 | Comprehensive test suite                  |
| `test/README.md`                        | Detailed test documentation               |

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
#include "parafaft_r2c.hpp"

int global_shape[3] = {32, 32, 32};
parafaft::ParaFaFT_R2C<3> fft(global_shape);

std::vector<double> real_input(fft.get_local_real_size());
std::vector<std::complex<double>> complex_output(fft.get_local_complex_size());

fft.forward(real_input.data(), complex_output.data());
fft.backward(complex_output.data(), real_input.data());
```

---

**Bottom Line**: Use `parafaft_generic.hpp` for C2C transforms or `parafaft_r2c.hpp` for real-valued data.
