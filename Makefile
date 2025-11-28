# Makefile for MPI FFT Library

CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -I. -I./include -I/opt/homebrew/include
LDFLAGS = -L. -L/opt/homebrew/lib -L/usr/lib/aarch64-linux-gnu -lfftw3 -lm

.PHONY: all examples test clean help

all: examples

examples:
	$(MAKE) -C examples

test: examples
	mpirun -np 4 ./examples/example_3d_pencil

clean:
	$(MAKE) -C examples clean
	rm -rf *.dSYM

help:
	@echo "MPI FFT Library"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build examples"
	@echo "  examples - Build examples (in examples/ directory)"
	@echo "  test     - Run 3D example"
	@echo "  clean    - Remove built files"
	@echo ""
	@echo "Usage:"
	@echo "  cd examples && make        # Build examples"
	@echo "  mpirun -np 4 ./examples/example_3d_pencil"
	@echo "  mpirun -np 8 ./examples/example_4d_pencil"
