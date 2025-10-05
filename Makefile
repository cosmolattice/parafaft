# Makefile for MPI FFT Library
#
# Status:
#  - minimal_3d_fft: ✓ Fully working (forward + backward)
#  - example_3d_pencil: ⚠️ Forward only (backward has bugs)
#  - example_4d_pencil: ⚠️ Forward only (backward has bugs)

CXX = mpicxx
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -lfftw3 -lm

TARGETS = minimal_3d_fft example_3d_pencil example_4d_pencil

.PHONY: all clean test

all: $(TARGETS)

# Working standalone implementation
minimal_3d_fft: minimal_3d_fft.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Template library examples (forward only)
example_3d_pencil: example_3d_pencil.cpp mpifft_pencil.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

example_4d_pencil: example_4d_pencil.cpp mpifft_pencil.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Run working standalone version
test: minimal_3d_fft
	mpirun -np 4 ./minimal_3d_fft

clean:
	rm -f $(TARGETS)
	rm -rf *.dSYM

.PHONY: help
help:
	@echo "MPI FFT Library"
	@echo ""
	@echo "Targets:"
	@echo "  all                  - Build all examples"
	@echo "  minimal_3d_fft       - ✓ Working standalone 3D FFT"
	@echo "  example_3d_pencil    - ⚠️ Template library (forward only)"
	@echo "  example_4d_pencil    - ⚠️ Template library (forward only)"
	@echo "  test                 - Run working standalone version"
	@echo "  clean                - Remove built files"
	@echo ""
	@echo "Usage:"
	@echo "  mpirun -np 4 ./minimal_3d_fft       # Working version"
	@echo "  mpirun -np 4 ./example_3d_pencil    # Forward only"
	@echo "  mpirun -np 8 ./example_4d_pencil    # Forward only"
