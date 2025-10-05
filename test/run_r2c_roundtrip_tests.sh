#!/bin/bash

SIZES="8 24 32 100"
PROCS="4 8"

echo "=== R2C Roundtrip FFT Validation ==="
echo ""

for N in $SIZES; do
    for P in $PROCS; do
        echo "Testing N=${N}^3 with ${P} processes..."
        mpirun --allow-run-as-root --oversubscribe -np $P ./test_mpi_r2c_roundtrip $N
        if [ $? -ne 0 ]; then
            echo "FAILED: N=${N}, P=${P}"
            exit 1
        fi
        echo ""
    done
done

echo "=== All R2C roundtrip tests passed! ==="
