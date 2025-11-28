# MPI FFT Tests

This directory contains tests for both the specialized and generic (dimension-agnostic) parallel FFT implementations.

## Building Tests

Use the provided Makefile:

```bash
make all              # Build all tests (specialized + generic)
make clean            # Clean all binaries and output files
make run-all          # Build and run all tests
make run-specialized  # Run only specialized implementation tests
make run-generic      # Run only generic implementation tests
make compare          # Run serial and MPI versions, then compare outputs
```

## Test Organization

### Specialized Implementation Tests

These tests use the generic implementation in `parafaft_generic.hpp`:

#### 1. `test_mpi_constant_32.cpp`
- **Size**: 32×32×32
- **Data**: Constant values (1,0)
- **Purpose**: Verify FFT works with constant input (simplest non-trivial case)
- **Expected**: All elements should roundtrip perfectly to (1,0)

#### 2. `test_mpi_gaussian_roundtrip.cpp`
- **Size**: 32×32×32
- **Data**: 3D Gaussian distribution centered in the box
- **Purpose**: Test FFT with smooth, realistic data
- **Outputs**: `gaussian_transformed_mpi.txt` - Transformed Fourier coefficients
- **Expected**: Roundtrip error < 1e-10

#### 3. `test_8cubed.cpp`
- **Size**: 8×8×8
- **Data**: Sequential values (1, 2, 3, ..., 512)
- **Purpose**: Test with arbitrary non-symmetric data
- **Expected**: Perfect roundtrip (error = 0 within machine precision)

#### 4. `test_4d_roundtrip.cpp`
- **Size**: 4×4×4×4
- **Data**: Constant values (1,0)
- **Purpose**: Verify 4D FFT implementation
- **Expected**: All elements should roundtrip perfectly to (1,0)

#### 5. `reference_serial_3d.cpp`
- **Type**: Serial (non-MPI) reference implementation
- **Size**: 32×32×32
- **Data**: 3D Gaussian distribution
- **Purpose**: Generate reference data for validation
- **Outputs**:
  - `initial_gaussian.txt` - Initial Gaussian distribution
  - `transformed_gaussian.txt` - Fourier transform
  - `roundtrip_gaussian.txt` - After forward + backward transform

### Generic Implementation Tests

*No generic implementation tests currently exist. The specialized tests above use the generic
implementation (`parafaft_generic.hpp`) internally.*

To add generic tests that validate higher-dimensional FFTs (5D, 6D), create test files
following the pattern in `test_mpi_gaussian_roundtrip.cpp` and add them to `GENERIC_TESTS`
in the Makefile.

### R2C/C2R Tests

These tests validate the Real-to-Complex transforms in `parafaft_r2c.hpp`:

#### 1. `test_mpi_r2c_gaussian.cpp`
- **Purpose**: R2C forward transform validation
- **Data**: 3D Gaussian distribution
- **Expected**: Transform produces correct complex coefficients

#### 2. `test_mpi_r2c_roundtrip.cpp`
- **Purpose**: Full R2C → C2R roundtrip accuracy test
- **Grid sizes**: 8³, 24³, 32³, 100³
- **Process counts**: 4 and 8 MPI processes
- **Expected**: Roundtrip error < 1e-15

Run with: `bash run_r2c_roundtrip_tests.sh`

### Backend Tests

#### 1. `test_fftw_backend.cpp`
- **Purpose**: Unit tests for the FFTW backend abstraction
- **Tests**: Contiguous arrays, strided arrays, plan reuse, different pointers

## Running Tests

**Important**: All tests should be run from the `test/` directory to ensure output files are created in the correct location.

```bash
cd test  # Make sure you're in the test directory
```

### Run All Tests
```bash
make run-all
```

### Run Specialized Tests Only
```bash
make run-specialized
```

### Run Generic Tests Only
```bash
make run-generic
```

### Run Individual Tests

Specialized tests (3D with 4 ranks, 4D with 4 ranks):
```bash
mpirun -n 4 ./test_mpi_constant_32
mpirun -n 4 ./test_mpi_gaussian_roundtrip
mpirun -n 4 ./test_8cubed
mpirun -n 4 ./test_4d_roundtrip
```

R2C tests:
```bash
# Run full R2C roundtrip test suite
bash run_r2c_roundtrip_tests.sh
```

Backend tests:
```bash
./test_fftw_backend
```

Serial reference:
```bash
./reference_serial_3d
```

## Comparing Serial vs MPI

To verify that the MPI implementation produces identical results to the serial version:

```bash
make compare
```

This will:
1. Run the serial reference (`reference_serial_3d`) to generate `transformed_gaussian.txt`
2. Run the MPI version (`test_mpi_gaussian_roundtrip`) to generate `gaussian_transformed_mpi.txt`
3. Run `compare_outputs.py` to compare the two files element-by-element

Expected result: Maximum absolute difference = 0 (exact match)

## Performance

The generic implementation has **no performance penalty** compared to specialized versions:
- See `../benchmark/PERFORMANCE_RESULTS.md` for detailed benchmarks
- Generic implementation is 0.94-1.01x the speed of specialized (essentially identical)
- Both achieve machine precision accuracy (~10⁻¹⁶)

## Implementation Notes

### Common Features
- All parallel tests use pencil decomposition (Dalcin et al. 2019 algorithm)
- Normalization factor is 1/N where N is total array size
- Tests verify that forward + backward FFT recovers original data
- MPI output is properly reordered from distributed format to match global ordering

### Generic C2C Implementation (`parafaft_generic.hpp`)
- Template-based, works for arbitrary dimensions D
- Automatically handles D-dimensional arrays with (D-1)-dimensional processor grids
- Correctly handles edge cases like non-uniform processor grids (e.g., [2,2,1])
- **Recommended for all C2C (complex-to-complex) transforms**

### R2C/C2R Implementation (`parafaft_r2c.hpp`)
- Real-to-Complex and Complex-to-Real transforms
- Memory efficient: exploits Hermitian symmetry (stores N/2+1 instead of N)
- **Recommended for real-valued input data**

## Key Test Results

All tests pass with errors at machine precision (~10⁻¹⁶):

| Test | Type | Size | Ranks | Max Error | Status |
|------|------|------|-------|-----------|--------|
| test_mpi_constant_32 | C2C | 32³ | 4 | 0 | ✓ PASS |
| test_mpi_gaussian_roundtrip | C2C | 32³ | 4 | ~10⁻¹⁶ | ✓ PASS |
| test_8cubed | C2C | 8³ | 4 | 0 | ✓ PASS |
| test_4d_roundtrip | C2C | 4⁴ | 4 | 0 | ✓ PASS |
| test_mpi_r2c_roundtrip | R2C | 8³-100³ | 4-8 | <10⁻¹⁵ | ✓ PASS |
| test_fftw_backend | Backend | N/A | 1 | ~10⁻¹⁴ | ✓ PASS |
