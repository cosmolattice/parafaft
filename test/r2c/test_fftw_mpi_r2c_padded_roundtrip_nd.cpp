// ============================================================================
// Generalized N-dimensional padded R2C Gaussian roundtrip test for parafaft
// (FFTW backend). In-place variant: forward_in_place + backward_in_place.
//
// Usage:  mpirun -np <P> ./test_fftw_mpi_r2c_padded_roundtrip_nd [N] [D]
//
//   N  – grid side length (default 32, or 16 when D >= 4)
//   D  – number of dimensions (default 3, range 2..6)
//
// The test generates a D-dimensional Gaussian on an N^D hypercubic grid using
// padded layout (last dim stride = N+2), performs forward_in_place R2C +
// backward_in_place C2R transforms with parafaft (MPI-distributed, FFTW
// backend), normalises, and checks that the result matches the original data.
// ============================================================================

#include "../../parafaft_r2c.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

// ============================================================================
// Templated roundtrip test – works for any D.
// ============================================================================
template <int D> int roundtrip_test(const int N, int rank)
{
  using namespace parafaft_test;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing R2C padded roundtrip (in-place) for " << D << "D transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;
  }

  // ---- ParaFaFT setup -------------------------------------------------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT_R2C<D, parafaft::FFTWBackend> fft(global_shape.data());

  int local_real_shape[D], real_start[D];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);

  int local_padded_size = fft.get_required_output_size();

  // Padded stride: last dimension gets +2
  const int padded_last = local_real_shape[D - 1] + 2;

  // ---- Generate local portion of Gaussian in padded layout ------------------
  const double sigma = 4.0;
  const double center = N / 2.0;

  std::vector<double> padded_buffer(local_padded_size, 0.0);
  // Keep an unpadded copy of the original real values
  int local_real_size = fft.get_local_real_size();
  std::vector<double> original(local_real_size);

  int orig_idx = 0;
  iterate_nd<D>(local_real_shape, [&](const std::array<int, D> &lidx) {
    double r2 = 0.0;
    for (int d = 0; d < D; ++d) {
      double x = (real_start[d] + lidx[d]) - center;
      r2 += x * x;
    }
    double value = std::exp(-r2 / (2.0 * sigma * sigma));

    // Padded flat index: dims 0..D-2 use local_real_shape, last dim uses padded_last
    int padded_flat = lidx[0];
    for (int d = 1; d < D - 1; ++d)
      padded_flat = padded_flat * local_real_shape[d] + lidx[d];
    padded_flat = padded_flat * padded_last + lidx[D - 1];

    padded_buffer[padded_flat] = value;
    original[orig_idx++] = value;
  });

  // ---- Forward R2C FFT (in-place) -------------------------------------------
  fft.forward_in_place(padded_buffer.data());

  // ---- Backward C2R FFT (in-place) ------------------------------------------
  fft.backward_in_place(padded_buffer.data());

  // ---- Normalise ------------------------------------------------------------
  long long total_real_size_global = 1;
  for (int d = 0; d < D; ++d)
    total_real_size_global *= N;
  double scale = 1.0 / total_real_size_global;
  for (int i = 0; i < local_padded_size; ++i)
    padded_buffer[i] *= scale;

  // ---- Compute error (only on non-padded elements) --------------------------
  double local_max_error = 0.0;
  orig_idx = 0;

  iterate_nd<D>(local_real_shape, [&](const std::array<int, D> &lidx) {
    int padded_flat = lidx[0];
    for (int d = 1; d < D - 1; ++d)
      padded_flat = padded_flat * local_real_shape[d] + lidx[d];
    padded_flat = padded_flat * padded_last + lidx[D - 1];

    double err = std::abs(padded_buffer[padded_flat] - original[orig_idx++]);
    if (err > local_max_error) local_max_error = err;
  });

  double global_max_error = 0.0;
  MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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
int dispatch(int D, int N, int rank)
{
  switch (D) {
  case 2:
    return roundtrip_test<2>(N, rank);
  case 3:
    return roundtrip_test<3>(N, rank);
  case 4:
    return roundtrip_test<4>(N, rank);
  case 5:
    return roundtrip_test<5>(N, rank);
  case 6:
    return roundtrip_test<6>(N, rank);
  default:
    if (rank == 0) std::cerr << "Unsupported dimensionality D=" << D << " (supported: 2..6)" << std::endl;
    return 1;
  }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int D = 3;
  int N = -1;

  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) D = std::atoi(argv[2]);

  if (N <= 0) N = (D >= 4) ? 16 : 32;

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: r2c/fftw_mpi_r2c_padded_roundtrip_nd" << std::endl;
    std::cout << "# D = " << D << ",  N = " << N << std::endl;
    std::cout << "########################################" << std::endl;
  }

  int total_failures = dispatch(D, N, rank);

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "RESULTS SUMMARY" << std::endl;
    std::cout << "Total failures: " << total_failures << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return total_failures;
}
