# Makefile for MPI FFT Library
#
# Status:
#  - example_3d_pencil: 3D FFT example using template library
#  - example_4d_pencil: 4D FFT example using template library

CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -I. -I./include -I/opt/homebrew/include
LDFLAGS = -L. -L/opt/homebrew/lib -L/usr/lib/aarch64-linux-gnu -lfftw3 -lm

TARGETS = example_3d_pencil example_4d_pencil

.PHONY: all clean test

all: $(TARGETS)

# Template library examples
example_3d_pencil: example_3d_pencil.cpp mpifft_generic.hpp backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

example_4d_pencil: example_4d_pencil.cpp mpifft_generic.hpp backend/fftw3/fft_backend_fftw.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Run 3D example
test: example_3d_pencil
	mpirun -np 4 ./example_3d_pencil

clean:
	rm -f $(TARGETS)
	rm -rf *.dSYM

.PHONY: help
help:
	@echo "MPI FFT Library"
	@echo ""
	@echo "Targets:"
	@echo "  all                  - Build all examples"
	@echo "  example_3d_pencil    - 3D FFT example"
	@echo "  example_4d_pencil    - 4D FFT example"
	@echo "  test                 - Run 3D example"
	@echo "  clean                - Remove built files"
	@echo ""
	@echo "Usage:"
	@echo "  mpirun -np 4 ./example_3d_pencil"
	@echo "  mpirun -np 8 ./example_4d_pencil"
