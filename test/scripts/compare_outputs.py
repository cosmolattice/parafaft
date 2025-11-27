#!/usr/bin/env python3
import sys

# Read serial output (has header and indices)
serial_data = []
with open('transformed_gaussian.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 5:
            # Format: i0 i1 i2 real imag amplitude
            real = float(parts[3])
            imag = float(parts[4])
            serial_data.append(complex(real, imag))

# Read MPI output (just real and imag per line)
mpi_data = []
with open('gaussian_transformed_mpi.txt', 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2:
            real = float(parts[0])
            imag = float(parts[1])
            mpi_data.append(complex(real, imag))

# Compare
if len(serial_data) != len(mpi_data):
    print(f"ERROR: Different sizes - serial: {len(serial_data)}, MPI: {len(mpi_data)}")
    sys.exit(1)

max_abs_diff = 0.0
max_rel_diff = 0.0
max_idx = 0

for i in range(len(serial_data)):
    abs_diff = abs(mpi_data[i] - serial_data[i])
    magnitude = abs(serial_data[i])
    rel_diff = abs_diff / magnitude if magnitude > 1e-14 else abs_diff

    if abs_diff > max_abs_diff:
        max_abs_diff = abs_diff
        max_rel_diff = rel_diff
        max_idx = i

print(f"Comparing {len(serial_data)} elements...")
print(f"Maximum absolute difference: {max_abs_diff:.6e}")
print(f"Maximum relative difference: {max_rel_diff:.6e}")
print(f"Location of max difference: index {max_idx}")
print(f"  Serial value: {serial_data[max_idx]}")
print(f"  MPI value:    {mpi_data[max_idx]}")

if max_abs_diff < 1e-10:
    print("\n✓ PASSED: Serial and MPI results match!")
    sys.exit(0)
else:
    print("\n✗ FAILED: Results differ beyond tolerance!")
    sys.exit(1)
