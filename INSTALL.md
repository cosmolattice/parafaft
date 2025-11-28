# Installation and Testing Guide

## Prerequisites

### Required
- **MPI Implementation**: OpenMPI, MPICH, or compatible
- **FFTW3**: Fast Fourier Transform library
- **C++11 Compiler**: g++, clang++, or compatible
- **Python3**: For comparison scripts (optional but recommended)

### Installation

#### macOS (Homebrew)
```bash
brew install fftw
brew install open-mpi
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install libfftw3-dev
sudo apt install libopenmpi-dev
```

#### Other Systems
Install FFTW3 and MPI from your package manager or build from source.

## Verifying Installation

Check that libraries are accessible:

```bash
# Check FFTW
pkg-config --cflags --libs fftw3

# Check MPI
mpirun --version
mpicxx --version
```

## Building Examples

```bash
cd examples
make all
```

This builds both `example_3d_pencil` and `example_4d_pencil`.

## Building Tests

```bash
cd test
make all
```

This builds:
- `test_mpi_constant_32` - 32³ constant values test
- `test_mpi_gaussian_roundtrip` - 32³ Gaussian distribution test
- `test_8cubed` - 8³ sequential values test
- `test_4d_roundtrip` - 4⁴ test
- `test_mpi_r2c_gaussian` - R2C forward transform test
- `test_mpi_r2c_roundtrip` - R2C/C2R roundtrip validation
- `test_fftw_backend` - Backend unit tests
- `reference_serial_3d` - Serial FFTW reference
- `reference_serial_r2c_3d` - Serial R2C reference

## Running Tests

### All Tests
```bash
cd test
make run-all
```

Expected output:
```
=== Running Constant Test (32^3) ===
SUCCESS: All elements are (1,0)!

=== Running Gaussian Roundtrip Test (32^3) ===
Maximum error: 3.224750e-16
✓ PASSED

=== Running 8^3 Sequential Values Test ===
Max error: 0
SUCCESS!

=== Running 4D Test (4^4) ===
SUCCESS: All elements are (1,0)!

All tests completed!
```

### Serial vs MPI Comparison
```bash
cd test
make compare
```

Expected output:
```
=== Comparing Outputs ===
Maximum absolute difference: 0.000000e+00
✓ PASSED: Serial and MPI results match!
```

### Individual Tests
```bash
cd test

# Run specific test (4 MPI ranks)
mpirun -n 4 ./test_mpi_constant_32
mpirun -n 4 ./test_mpi_gaussian_roundtrip
mpirun -n 4 ./test_8cubed
mpirun -n 4 ./test_4d_roundtrip

# Run serial reference (no MPI)
./reference_serial_3d
```

## Running Examples

```bash
# 3D example (requires 4 MPI ranks for 2×2 processor grid)
mpirun -n 4 ./examples/example_3d_pencil

# 4D example (requires 8 MPI ranks for 2×2×2 processor grid)
mpirun -n 8 ./examples/example_4d_pencil
```

## Common Issues

### "Cannot find fftw3.h"
**Problem**: Compiler can't find FFTW headers

**Solution**: Add include path explicitly:
```bash
mpicxx ... -I/path/to/fftw/include -L/path/to/fftw/lib -lfftw3
```

Find your FFTW location:
```bash
# macOS Homebrew
brew --prefix fftw

# Ubuntu/Debian
dpkg -L libfftw3-dev | grep include
```

### "Undefined reference to fftw_*"
**Problem**: Linker can't find FFTW library

**Solution**: Add library path and ensure `-lfftw3` is last:
```bash
mpicxx file.cpp -o file -I/path/include -L/path/lib -lfftw3
```

### "MPI_Init has not been called"
**Problem**: Trying to run MPI program without `mpirun`

**Solution**: Always use `mpirun`:
```bash
mpirun -n 4 ./program    # Correct
./program                # Wrong for MPI programs
```

### Wrong number of processes
**Problem**: Using incorrect number of MPI ranks

**Solution**:
- 3D examples need 4 ranks (2×2 grid)
- 4D examples need 8 ranks (2×2×2 grid)
```bash
mpirun -n 4 ./examples/example_3d_pencil   # Correct
mpirun -n 2 ./examples/example_3d_pencil   # May fail
```

### Tests show errors > 1e-10
**Problem**: Numerical errors in FFT roundtrip

**Status**: This should not happen with the fixed version. If you see this:
1. Make sure you rebuilt everything: `cd test && make clean && make all`
2. Check that you're using FFTW3 (not FFTW2)
3. Report as a bug with your system details

## Implementation Status

### ✅ Fully Working
- **C2C transforms** (`parafaft_generic.hpp`): Forward and backward transforms
- **R2C/C2R transforms** (`parafaft_r2c.hpp`): Real-to-complex and inverse
- **3D/4D FFT**: All tests pass
- **Serial validation**: MPI matches serial FFTW exactly

### Algorithm Implementation
Following Dalcin et al. (2019):
- ✅ Balanced block-contiguous decomposition (Algorithm 1)
- ✅ Subarray datatype creation (Listing 2)
- ✅ Global redistribution via `MPI_Alltoallw` (Listing 3)
- ✅ Cartesian topology setup (Listing 4)
- ✅ Y-axis FFT fix (custom implementation for non-uniform stride)

## Verification

To verify your installation is working correctly:

```bash
cd test
make clean
make all
make compare
```

You should see:
```
✓ PASSED: Serial and MPI results match!
```

If all tests pass, your installation is working correctly!

## Next Steps

1. Read `QUICKSTART.md` for usage guide
2. Look at `examples/example_3d_pencil.cpp` for code examples
3. Check `README.md` for complete documentation
4. See `test/README.md` for test details

---

For troubleshooting, check that:
- FFTW3 is installed (not FFTW2)
- MPI is working (`mpirun -n 2 hostname` should work)
- Include/library paths are correct for your system
