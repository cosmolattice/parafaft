CXX = mpicxx
SERIAL_CXX = g++
CXXFLAGS = -std=c++11 -O3 -I.. -I../include -I/opt/homebrew/include
LDFLAGS = -L.. -L/opt/homebrew/lib -L/usr/lib/aarch64-linux-gnu -lfftw3

all: benchmark_serial benchmark_mpi

benchmark_serial: benchmark_serial.cpp
	$(SERIAL_CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

benchmark_mpi: benchmark_mpi.cpp ../parafaft_generic.hpp ../backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f benchmark_serial benchmark_mpi

run-serial: benchmark_serial
	@echo "=== Running Serial Benchmark ==="
	@./benchmark_serial

run-mpi: benchmark_mpi
	@echo "=== Running MPI Benchmark (4 ranks) ==="
	@mpirun -n 4 ./benchmark_mpi

run-all: all
	@echo "Running serial benchmark..."
	@echo ""
	@./benchmark_serial
	@echo ""
	@echo ""
	@echo "Running MPI benchmark (4 ranks)..."
	@echo ""
	@mpirun -n 4 ./benchmark_mpi

.PHONY: all clean run-serial run-mpi run-all
