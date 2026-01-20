#!/bin/bash
# cuFFT R2C MPI Integration Tests
#
# This script runs the cuFFT backend integration tests with various MPI configurations.
# Requires: CUDA-enabled system with cuFFT, MPI implementation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== cuFFT R2C MPI Integration Tests ==="

# Check if test binary exists
if [ ! -f "./r2c/test_mpi_r2c_roundtrip_cufft" ]; then
    echo "Error: test_mpi_r2c_roundtrip_cufft not found"
    echo "Please build the test first with:"
    echo "  make test_mpi_r2c_roundtrip_cufft"
    exit 1
fi

# Single rank test
echo ""
echo "Running single-rank R2C test..."
mpirun -np 1 ./r2c/test_mpi_r2c_roundtrip_cufft

# 2 ranks test
echo ""
echo "Running 2-rank R2C test..."
mpirun -np 2 ./r2c/test_mpi_r2c_roundtrip_cufft

# 4 ranks test
echo ""
echo "Running 4-rank R2C test..."
mpirun -np 4 ./r2c/test_mpi_r2c_roundtrip_cufft

# Larger grid size test (optional)
if [ "$1" == "--large" ]; then
    echo ""
    echo "Running 4-rank R2C test with 64^3 grid..."
    mpirun -np 4 ./r2c/test_mpi_r2c_roundtrip_cufft 64
fi

echo ""
echo "=== All cuFFT R2C MPI tests passed! ==="
