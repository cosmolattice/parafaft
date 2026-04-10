// ============================================================================
// Generalized N-dimensional C2C Gaussian roundtrip test for parafaft (FFTW).
//
// Usage:  mpirun -np <P> ./test_fftw_mpi_c2c_roundtrip_nd [N] [D] [S]
//
//   N  – grid side length (default 32, or 16 when D >= 4)
//   D  – number of dimensions (default 3, range 2..6)
//   S  – shape: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial (default 0)
//
// The test generates a D-dimensional Gaussian on an N^D hypercubic grid,
// performs forward + backward C2C transforms with parafaft (MPI-distributed,
// FFTW backend), normalises, and checks that the result matches the original.
// ============================================================================

#include "../../parafaft_c2c.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

// ============================================================================
// Templated roundtrip test – works for any D.
// ============================================================================
template <int D> int roundtrip_test(const int N, int rank, int shape_id) {
  using namespace parafaft_test;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing C2C roundtrip for " << D << "D transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << " [shape=" << shape_name(shape_id) << "]" << std::endl;
  }

  // ---- ParaFaFT setup -------------------------------------------------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT_C2C<D, parafaft::FFTWBackend> fft(global_shape.data());

  int local_size = fft.get_local_size();
  int buffer_size = fft.get_required_output_size();
  int local_shape[D];
  int global_start[D];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);

  // ---- Generate local portion of data
  // ----------------------------------------
  std::vector<std::complex<double>> data(buffer_size);
  std::vector<std::complex<double>> original(local_size);

  iterate_nd<D>(local_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = global_start[d] + lidx[d];
    double value = point_value<D>(gidx, N, shape_id);
    int flat = nd_index<D>(lidx.data(), local_shape);
    data[flat] = std::complex<double>(value, 0.0);
    original[flat] = data[flat];
  });

  // ---- Forward FFT ----------------------------------------------------------
  fft.forward(data.data());

  // ---- Backward FFT ---------------------------------------------------------
  fft.backward(data.data());

  // ---- Normalise ------------------------------------------------------------
  long long total_size = 1;
  for (int d = 0; d < D; ++d)
    total_size *= N;
  double scale = 1.0 / total_size;
  for (int i = 0; i < local_size; ++i)
    data[i] *= scale;

  // ---- Compute error --------------------------------------------------------
  double local_max_error = 0.0;
  for (int i = 0; i < local_size; ++i) {
    double err = std::abs(data[i] - original[i]);
    if (err > local_max_error)
      local_max_error = err;
  }

  double global_max_error = 0.0;
  MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  const double tolerance = 1e-10;
  int local_pass = (local_max_error < tolerance) ? 1 : 0;
  int global_pass = 0;
  MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Maximum roundtrip error: " << global_max_error << std::endl;
    if (global_pass) {
      std::cout << "Test passed." << std::endl;
    } else {
      std::cout << "Test FAILED." << std::endl;
    }
  }

  return global_pass ? 0 : 1;
}

// ============================================================================
// Dispatch
// ============================================================================
int dispatch(int D, int N, int rank, int shape_id) {
  switch (D) {
  case 1:
    return roundtrip_test<1>(N, rank, shape_id);
  case 2:
    return roundtrip_test<2>(N, rank, shape_id);
  case 3:
    return roundtrip_test<3>(N, rank, shape_id);
  case 4:
    return roundtrip_test<4>(N, rank, shape_id);
  case 5:
    return roundtrip_test<5>(N, rank, shape_id);
  case 6:
    return roundtrip_test<6>(N, rank, shape_id);
  default:
    if (rank == 0)
      std::cerr << "Unsupported dimensionality D=" << D << " (supported: 2..6)"
                << std::endl;
    return 1;
  }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int D = 3;
  int N = -1;
  int S = 0;

  if (argc > 1)
    N = std::atoi(argv[1]);
  if (argc > 2)
    D = std::atoi(argv[2]);
  if (argc > 3)
    S = std::atoi(argv[3]);

  if (D == 1 && size > 1) {
    if (rank == 0)
      std::cerr << "Error: D=1 test cannot be run with multiple processes."
                << std::endl;
    MPI_Finalize();
    return 1;
  }

  if (N <= 0)
    N = (D >= 4) ? 16 : 32;

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: c2c/fftw_mpi_c2c_roundtrip_nd" << std::endl;
    std::cout << "# D = " << D << ",  N = " << N
              << ",  S = " << parafaft_test::shape_name(S) << std::endl;
    std::cout << "########################################" << std::endl;
  }

  int total_failures = dispatch(D, N, rank, S);

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "RESULTS SUMMARY" << std::endl;
    std::cout << "Total failures: " << total_failures << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return total_failures;
}
