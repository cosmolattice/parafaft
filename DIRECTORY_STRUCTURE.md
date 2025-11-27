# Directory Structure

## Root Files

| File | Description |
|------|-------------|
| `mpifft_generic.hpp` | Primary C2C FFT implementation (dimension-agnostic) |
| `mpifft_r2c.hpp` | R2C/C2R FFT implementation for real data |
| `fft_backend.hpp` | Backend abstraction interface |
| `fft_backend_fftw.hpp` | FFTW backend implementation |
| `example_3d_pencil.cpp` | 3D FFT usage example |
| `example_4d_pencil.cpp` | 4D FFT usage example |
| `Makefile` | Root build file |
| `README.md` | Main documentation |
| `QUICKSTART.md` | Getting started guide |
| `INSTALL.md` | Installation instructions |
| `SUMMARY.md` | Project summary |

## Directories

### test/
Contains test suite for validating FFT implementations.

| File | Description |
|------|-------------|
| `test_mpi_constant_32.cpp` | Basic 32³ constant field test |
| `test_mpi_gaussian_roundtrip.cpp` | 3D Gaussian roundtrip validation |
| `test_8cubed.cpp` | 8³ grid test |
| `test_4d_roundtrip.cpp` | 4D FFT roundtrip test |
| `test_mpi_r2c_gaussian.cpp` | R2C forward transform test |
| `test_mpi_r2c_roundtrip.cpp` | R2C/C2R roundtrip test |
| `test_fftw_backend.cpp` | FFTW backend unit tests |
| `reference_serial_3d.cpp` | Serial 3D reference implementation |
| `reference_serial_r2c_3d.cpp` | Serial R2C reference implementation |

### benchmark/
Performance benchmarking tools.

### include/
External headers (FFTW).

### thoughts/
Research documentation and planning files.

## Build Artifacts (not in repo)

After running `make` in `test/`:
```
test/
├── test_mpi_constant_32          # Executable
├── test_mpi_gaussian_roundtrip   # Executable
├── test_8cubed                   # Executable
├── test_4d_roundtrip             # Executable
├── test_mpi_r2c_gaussian         # Executable
├── test_mpi_r2c_roundtrip        # Executable
├── test_fftw_backend             # Executable
├── reference_serial_3d           # Executable
├── reference_serial_r2c_3d       # Executable
└── *.txt                         # Output files
```

## Typical Workflow

1. **Quick Start**: Read `QUICKSTART.md`
2. **Build**:
   ```bash
   cd test
   make all
   ```
3. **Test**:
   ```bash
   make run-all    # Run all tests
   make compare    # Validate vs serial
   ```
4. **Use**: Include `mpifft_generic.hpp` (C2C) or `mpifft_r2c.hpp` (R2C) in your code
5. **Reference**: Check examples in `example_3d_pencil.cpp`

---

Status: v2.0 (fully functional with R2C/C2R support as of 2025-11-27)
