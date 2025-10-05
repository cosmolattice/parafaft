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

These tests use the dimension-specific implementations in `mpifft_pencil.hpp`:

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

These tests use the dimension-agnostic template implementation in `mpifft_generic.hpp`:

#### 1. `test_generic_3d_gaussian.cpp`
- **Size**: 32×32×32
- **Ranks**: 4
- **Data**: 3D Gaussian distribution (σ=4.0)
- **Purpose**: Validate generic implementation matches specialized version for 3D
- **Outputs**: `gaussian_transformed_generic.txt`
- **Expected**: Roundtrip error < 1e-10, exact match with specialized output

#### 2. `test_generic_4d_gaussian.cpp`
- **Size**: 16×16×16×16 (65,536 elements)
- **Ranks**: 8
- **Data**: 4D Gaussian distribution (σ=3.0)
- **Purpose**: Test generic implementation in 4D
- **Outputs**: `gaussian_transformed_4d.txt`
- **Expected**: Roundtrip error < 1e-10

#### 3. `test_generic_4d_small.cpp`
- **Size**: 4×4×4×4 (256 elements)
- **Ranks**: 4
- **Data**: Constant values (1,0)
- **Purpose**: Test edge case with non-uniform processor grid [2,2,1]
- **Expected**: All elements should roundtrip perfectly to (1,0)
- **Note**: Critical test for validating grid dimension mapping with size-1 dimensions

#### 4. `test_generic_5d_gaussian.cpp`
- **Size**: 8×8×8×8×8 (32,768 elements)
- **Ranks**: 8
- **Data**: 5D Gaussian distribution (σ=2.0)
- **Purpose**: Demonstrate generic implementation in 5D
- **Outputs**: `gaussian_transformed_5d.txt`
- **Expected**: Roundtrip error < 1e-10

#### 5. `test_generic_6d_gaussian.cpp`
- **Size**: 6×6×6×6×6×6 (46,656 elements)
- **Ranks**: 8
- **Data**: 6D Gaussian distribution (σ=1.5)
- **Purpose**: Demonstrate generic implementation in 6D
- **Outputs**: `gaussian_transformed_6d.txt`
- **Expected**: Roundtrip error < 1e-10

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

Generic tests:
```bash
# 3D with 4 ranks
mpirun -n 4 ./test_generic_3d_gaussian

# 4D tests
mpirun -n 8 ./test_generic_4d_gaussian
mpirun -n 4 ./test_generic_4d_small

# 5D and 6D with 8 ranks
mpirun -n 8 ./test_generic_5d_gaussian
mpirun -n 8 ./test_generic_6d_gaussian
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

### Specialized Implementation (`mpifft_pencil.hpp`)
- Explicit template specializations for 3D and 4D
- Processor grid dimensions are (D-1)-dimensional
- Hand-optimized for 3D and 4D use cases

### Generic Implementation (`mpifft_generic.hpp`)
- Template-based, works for arbitrary dimensions D
- Automatically handles D-dimensional arrays with (D-1)-dimensional processor grids
- Correctly handles edge cases like non-uniform processor grids (e.g., [2,2,1])
- Same algorithm and performance as specialized versions
- **Recommended for all new applications**

## Key Test Results

All tests pass with errors at machine precision (~10⁻¹⁶):

| Test | Dimensions | Size | Ranks | Max Error | Status |
|------|-----------|------|-------|-----------|--------|
| Specialized 3D Gaussian | 3D | 32³ | 4 | ~10⁻¹⁶ | ✓ PASS |
| Generic 3D Gaussian | 3D | 32³ | 4 | ~10⁻¹⁶ | ✓ PASS |
| Generic 4D Gaussian | 4D | 16⁴ | 8 | ~10⁻¹⁶ | ✓ PASS |
| Generic 4D Small | 4D | 4⁴ | 4 | 0 | ✓ PASS |
| Generic 5D Gaussian | 5D | 8⁵ | 8 | ~10⁻¹⁶ | ✓ PASS |
| Generic 6D Gaussian | 6D | 6⁶ | 8 | ~10⁻¹⁶ | ✓ PASS |
