# Makefile for parallel FFT tests - reorganized structure
CXX = mpicxx
SERIAL_CXX = g++
CXXFLAGS = -std=c++11 -I.. -I../include -I/opt/homebrew/include
LDFLAGS = -L.. -L/opt/homebrew/lib -L/usr/lib/aarch64-linux-gnu -lfftw3

# Test categories
C2C_TESTS = c2c/test_mpi_constant_32 c2c/test_mpi_gaussian_roundtrip c2c/test_8cubed c2c/test_4d_roundtrip
R2C_TESTS = r2c/test_mpi_r2c_gaussian r2c/test_mpi_r2c_roundtrip
UNIT_TESTS = unit/test_fftw_backend
REFERENCE_TESTS = reference/reference_serial_3d reference/reference_serial_r2c_3d

ALL_TESTS = $(C2C_TESTS) $(R2C_TESTS) $(UNIT_TESTS) $(REFERENCE_TESTS)

.PHONY: all clean run-c2c run-r2c run-unit run-reference run-all help

all: $(ALL_TESTS)

# C2C MPI Tests
c2c/test_mpi_constant_32: c2c/test_mpi_constant_32.cpp ../parafaft_generic.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

c2c/test_mpi_gaussian_roundtrip: c2c/test_mpi_gaussian_roundtrip.cpp ../parafaft_generic.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

c2c/test_8cubed: c2c/test_8cubed.cpp ../parafaft_generic.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

c2c/test_4d_roundtrip: c2c/test_4d_roundtrip.cpp ../parafaft_generic.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# R2C MPI Tests
r2c/test_mpi_r2c_gaussian: r2c/test_mpi_r2c_gaussian.cpp ../parafaft_r2c.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

r2c/test_mpi_r2c_roundtrip: r2c/test_mpi_r2c_roundtrip.cpp ../parafaft_r2c.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Unit Tests (Backend)
unit/test_fftw_backend: unit/test_fftw_backend.cpp ../backend/fft_backend.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Serial Reference Tests
reference/reference_serial_3d: reference/reference_serial_3d.cpp
	$(SERIAL_CXX) -std=c++11 -I../include -I/opt/homebrew/include $< -o $@ -L.. -L/opt/homebrew/lib -L/usr/lib/aarch64-linux-gnu -lfftw3

reference/reference_serial_r2c_3d: reference/reference_serial_r2c_3d.cpp
	$(SERIAL_CXX) -std=c++11 -I../include -I/opt/homebrew/include $< -o $@ -L.. -L/opt/homebrew/lib -L/usr/lib/aarch64-linux-gnu -lfftw3

# Clean
clean:
	rm -f $(ALL_TESTS)
	rm -f c2c/*.txt r2c/*.txt reference/*.txt
	find . -name "*.dSYM" -type d -exec rm -rf {} + 2>/dev/null || true

# Run targets
run-c2c: $(C2C_TESTS)
	@echo "=== Running C2C Tests ==="
	mpirun -np 4 c2c/test_mpi_constant_32
	mpirun -np 4 c2c/test_mpi_gaussian_roundtrip
	mpirun -np 4 c2c/test_8cubed
	mpirun -np 4 c2c/test_4d_roundtrip

run-r2c: $(R2C_TESTS)
	@echo "=== Running R2C Tests ==="
	mpirun -np 4 r2c/test_mpi_r2c_gaussian
	mpirun -np 4 r2c/test_mpi_r2c_roundtrip

run-unit: $(UNIT_TESTS)
	@echo "=== Running Unit Tests ==="
	./unit/test_fftw_backend

run-reference: $(REFERENCE_TESTS)
	@echo "=== Running Reference Tests ==="
	./reference/reference_serial_3d
	./reference/reference_serial_r2c_3d

run-all: run-unit run-reference run-c2c run-r2c

# Compare MPI output with serial reference
compare: reference/reference_serial_3d c2c/test_mpi_gaussian_roundtrip
	@echo "=== Comparing MPI vs Serial Output ==="
	./reference/reference_serial_3d > reference/serial_output.txt
	mpirun -np 4 c2c/test_mpi_gaussian_roundtrip > c2c/mpi_output.txt
	python3 scripts/compare_outputs.py reference/serial_output.txt c2c/mpi_output.txt

help:
	@echo "Parallel FFT Test Suite"
	@echo ""
	@echo "Build targets:"
	@echo "  make all          - Build all tests"
	@echo "  make clean        - Remove all built files"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-unit     - Run backend unit tests"
	@echo "  make run-reference - Run serial reference tests"
	@echo "  make run-c2c      - Run C2C MPI tests"
	@echo "  make run-r2c      - Run R2C MPI tests"
	@echo "  make run-all      - Run all tests"
	@echo "  make compare      - Compare MPI vs serial output"
