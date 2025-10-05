# Directory Structure

```
.
├── mpifft_pencil.hpp              # ✅ Template library (3D/4D, fully working)
│
├── example_3d_pencil.cpp          # Template library usage example (3D)
├── example_4d_pencil.cpp          # Template library usage example (4D)
│
├── test/                          # Comprehensive test suite
│   ├── Makefile                   # Build system with 'compare' target
│   ├── README.md                  # Test documentation
│   ├── compare_outputs.py         # Serial vs MPI validation script
│   ├── test_mpi_constant_32.cpp   # 32³ constant values test
│   ├── test_mpi_gaussian_roundtrip.cpp  # 32³ Gaussian test
│   ├── test_8cubed.cpp            # 8³ arbitrary data test
│   ├── test_4d_roundtrip.cpp      # 4⁴ test
│   └── reference_serial_3d.cpp    # Serial FFTW reference
│
├── QUICKSTART.md                  # ← START HERE
├── README.md                      # Complete user guide
├── SUMMARY.md                     # Executive summary
├── INSTALL.md                     # Installation instructions
├── DIRECTORY_STRUCTURE.md         # This file
│
├── Dalcin et al. - 2019 - *.pdf  # Original research paper
│
└── template_wip/                  # Archive of development notes
    ├── README.md                  # WIP documentation
    └── IMPLEMENTATION_NOTES.md    # Technical notes from development
```

## Key Files

### Production Code
- **`mpifft_pencil.hpp`** - Main template library (3D and 4D, fully functional)

### Examples
- `example_3d_pencil.cpp` - 3D usage example
- `example_4d_pencil.cpp` - 4D usage example

### Testing
- `test/` - Complete test suite with Makefile
  - Run `make run-all` to execute all tests
  - Run `make compare` to validate MPI vs serial FFTW

### Documentation
- **`QUICKSTART.md`** - Start here!
- `README.md` - Complete guide
- `SUMMARY.md` - Project summary and status
- `test/README.md` - Detailed test documentation

### Reference
- `template_wip/` - Development history and notes
- `Dalcin et al. - 2019 - *.pdf` - Original research paper

## Build Artifacts (not in repo)

After running `make` in `test/`:
```
test/
├── test_mpi_constant_32          # Executable
├── test_mpi_gaussian_roundtrip   # Executable
├── test_8cubed                   # Executable
├── test_4d_roundtrip             # Executable
├── reference_serial_3d           # Executable
├── gaussian_transformed_mpi.txt  # Output file
├── transformed_gaussian.txt      # Output file
└── *.txt                         # Other output files
```

After building examples in main directory:
```
├── example_3d_pencil             # Executable
├── example_4d_pencil             # Executable
└── *.dSYM/                       # Debug symbols (macOS)
```

Run `make clean` in respective directories to remove these.

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
4. **Use**: Include `mpifft_pencil.hpp` in your code
5. **Reference**: Check examples in `example_3d_pencil.cpp`

## What's New

The library is now **fully functional** with both forward and backward transforms working correctly:
- Y-axis FFT bug fixed (incorrect `dist` parameter)
- All tests pass with exact accuracy
- MPI results match serial FFTW exactly (verified)
- Comprehensive test suite with serial comparison

---

Status: v1.0 (fully functional as of 2025-10-05)
