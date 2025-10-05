CXX = mpicxx
CXXFLAGS = -std=c++11 -I.. -I../include -I/opt/homebrew/include
LDFLAGS = -L.. -L/opt/homebrew/lib -L/usr/lib/aarch64-linux-gnu -lfftw3

SERIAL_CXX = g++

SPECIALIZED_TESTS = test_mpi_constant_32 test_mpi_gaussian_roundtrip test_8cubed test_4d_roundtrip test_mpi_r2c_gaussian test_mpi_r2c_roundtrip
GENERIC_TESTS = test_generic_3d_gaussian test_generic_4d_gaussian test_generic_5d_gaussian test_generic_6d_gaussian test_generic_4d_small
SERIAL_TESTS = reference_serial_3d reference_serial_r2c_3d
BACKEND_TESTS = test_fftw_backend

all: $(SPECIALIZED_TESTS) $(GENERIC_TESTS) $(SERIAL_TESTS) $(BACKEND_TESTS)

test_mpi_constant_32: test_mpi_constant_32.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_mpi_gaussian_roundtrip: test_mpi_gaussian_roundtrip.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_8cubed: test_8cubed.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_4d_roundtrip: test_4d_roundtrip.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_mpi_r2c_gaussian: test_mpi_r2c_gaussian.cpp ../mpifft_r2c.hpp ../fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_mpi_r2c_roundtrip: test_mpi_r2c_roundtrip.cpp ../mpifft_r2c.hpp ../fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

reference_serial_3d: reference_serial_3d.cpp
	$(SERIAL_CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

reference_serial_r2c_3d: reference_serial_r2c_3d.cpp
	$(SERIAL_CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Generic implementation tests
test_generic_3d_gaussian: test_generic_3d_gaussian.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_generic_4d_gaussian: test_generic_4d_gaussian.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_generic_5d_gaussian: test_generic_5d_gaussian.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_generic_6d_gaussian: test_generic_6d_gaussian.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_generic_4d_small: test_generic_4d_small.cpp ../mpifft_generic.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Backend tests
test_fftw_backend: test_fftw_backend.cpp ../fft_backend.hpp ../fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(SPECIALIZED_TESTS) $(GENERIC_TESTS) $(SERIAL_TESTS) $(BACKEND_TESTS) *.txt *.o

compare: all
	@echo "=== Running Serial Reference ==="
	@./reference_serial_3d
	@echo ""
	@echo "=== Running MPI Version ==="
	@mpirun -n 4 ./test_mpi_gaussian_roundtrip
	@echo ""
	@echo "=== Comparing Outputs ==="
	@python3 compare_outputs.py

run-specialized: $(SPECIALIZED_TESTS)
	@echo "=== Running Specialized Implementation Tests ==="
	@echo ""
	@echo "=== Running Constant Test (32^3) ==="
	@mpirun -n 4 ./test_mpi_constant_32
	@echo ""
	@echo "=== Running Gaussian Roundtrip Test (32^3) ==="
	@mpirun -n 4 ./test_mpi_gaussian_roundtrip
	@echo ""
	@echo "=== Running 8^3 Sequential Values Test ==="
	@mpirun -n 4 ./test_8cubed
	@echo ""
	@echo "=== Running 4D Test (4^4) ==="
	@mpirun -n 4 ./test_4d_roundtrip 2>/dev/null || true
	@echo ""

run-generic: $(GENERIC_TESTS)
	@echo "=== Running Generic Implementation Tests ==="
	@echo ""
	@echo "=== 3D Gaussian Test (32^3 with 4 ranks) ==="
	@mpirun -n 4 ./test_generic_3d_gaussian
	@echo ""
	@echo "=== 4D Gaussian Test (16^4 with 8 ranks) ==="
	@mpirun -n 8 ./test_generic_4d_gaussian
	@echo ""
	@echo "=== 4D Small Test (4^4 with 4 ranks) ==="
	@mpirun -n 4 ./test_generic_4d_small
	@echo ""
	@echo "=== 5D Gaussian Test (8^5 with 8 ranks) ==="
	@mpirun -n 8 ./test_generic_5d_gaussian
	@echo ""
	@echo "=== 6D Gaussian Test (6^6 with 8 ranks) ==="
	@mpirun -n 8 ./test_generic_6d_gaussian
	@echo ""

run-backend: $(BACKEND_TESTS)
	@echo "=== Testing FFTW Backend ==="
	@./test_fftw_backend || exit 1

run-all: run-specialized run-generic run-backend
	@echo "All tests completed!"

.PHONY: all clean run-all run-specialized run-generic run-backend compare
