// ============================================================================
// Single-precision N-dimensional C2C smoke test for parafaft (FFTW backend).
//
// Usage:  mpirun -np <P> ./test_fftw_c2c_nd_float [N] [D] [S]
//
// Mirrors test_fftw_c2c_nd.cpp but instantiates
//   parafaft::ParaFaFT_C2C<D, parafaft::FFTWBackend<float>, float>
// to verify that the single-precision FFTW path is wired end-to-end.
//
// The reference FFT is computed in double precision; inputs and local
// results are cast to/from float with a float-appropriate tolerance.
//
// This test only builds/runs when libfftw3f was found at configure time
// (PARAFAFT_FFTW3F_AVAILABLE).
// ============================================================================

#include "../../backend/fftw3/fft_backend_fftw.hpp"
#include "../../parafaft_c2c.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

template <int D> int compare_fftwBackend_float(const int N, int rank, int shape_id) {
  using namespace parafaft_test;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft FFTWBackend<float> for " << D
              << "D C2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << " [shape=" << shape_name(shape_id) << "]" << std::endl;
  }

  // Generate reference data in double precision.
  const auto global_original_data_d = generate_c2c_data_nd<D>(N, shape_id);
  auto global_fftw_reference_d = global_original_data_d;
  fftw_c2c_reference_nd<D>(global_fftw_reference_d.data(), N);

  // ParaFaFT setup at float precision.
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT_C2C<D, parafaft::FFTWBackend<float>, float> fft(
      global_shape.data());

  int buffer_size = fft.get_required_output_size();
  int local_shape[D];
  int global_start[D];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);

  std::array<int, D> full_shape;
  full_shape.fill(N);

  // Scatter global (double) data into local float partition.
  std::vector<std::complex<float>> local_data(buffer_size);
  iterate_nd<D>(local_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = global_start[d] + lidx[d];

    int global_flat = nd_index<D>(gidx, full_shape);
    int local_flat = nd_index<D>(lidx.data(), local_shape);
    const auto &c = global_original_data_d[global_flat];
    local_data[local_flat] =
        std::complex<float>(static_cast<float>(c.real()), static_cast<float>(c.imag()));
  });

  fft.forward(local_data.data());

  // Compare against double reference using final (redistributed) layout.
  int final_shape[D];
  int final_start[D];
  fft.get_final_shape(final_shape);
  fft.get_final_start(final_start);

  double max_error = 0.0;
  double ref_magnitude = 0.0;
  iterate_nd<D>(final_shape, [&](const std::array<int, D> &fidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = final_start[d] + fidx[d];

    int global_flat = nd_index<D>(gidx, full_shape);
    int local_flat = nd_index<D>(fidx.data(), final_shape);
    std::complex<double> got(local_data[local_flat].real(),
                             local_data[local_flat].imag());
    double err = std::abs(got - global_fftw_reference_d[global_flat]);
    if (err > max_error)
      max_error = err;
    double mag = std::abs(global_fftw_reference_d[global_flat]);
    if (mag > ref_magnitude)
      ref_magnitude = mag;
  });

  // Float-appropriate relative tolerance. Pick a few ulps-of-FFT-output:
  // FFT accumulates ~log2(N^D) rounding errors in float (~7 decimal digits),
  // so 1e-4 * peak_magnitude is a safe smoke-test threshold.
  MPI_Allreduce(MPI_IN_PLACE, &ref_magnitude, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  const double tolerance = 1e-4 * std::max(ref_magnitude, 1e-6);
  const bool local_success = (max_error < tolerance);
  int global_success = local_success ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);

  if (global_success == 1 && rank == 0) {
    std::cout << "Test passed: FFTWBackend<float> produces correct results."
              << std::endl;
    std::cout << "Maximum error: " << max_error << " (tol " << tolerance << ")"
              << std::endl;
  } else if (!global_success) {
    std::cout << "Rank " << rank
              << ": Test failed: FFTWBackend<float> produces incorrect results."
              << std::endl;
    std::cout << "Maximum error: " << max_error << " (tol " << tolerance << ")"
              << std::endl;
    return 1;
  }
  return 0;
}

int dispatch(int D, int N, int rank, int shape_id) {
  switch (D) {
  case 2: return compare_fftwBackend_float<2>(N, rank, shape_id);
  case 3: return compare_fftwBackend_float<3>(N, rank, shape_id);
  case 4: return compare_fftwBackend_float<4>(N, rank, shape_id);
  case 5: return compare_fftwBackend_float<5>(N, rank, shape_id);
  default:
    if (rank == 0)
      std::cerr << "Unsupported dimensionality D=" << D << " (supported: 2..5)"
                << std::endl;
    return 1;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  (void)size;

  int D = 3;
  int N = -1;
  int S = 0;

  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) D = std::atoi(argv[2]);
  if (argc > 3) S = std::atoi(argv[3]);

  if (N <= 0) N = (D >= 4) ? 16 : 32;

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: c2c/fftw_c2c_nd_float" << std::endl;
    std::cout << "# D = " << D << ",  N = " << N
              << ",  S = " << parafaft_test::shape_name(S) << std::endl;
    std::cout << "########################################" << std::endl;
  }

  int total_failures = dispatch(D, N, rank, S);

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Total failures: " << total_failures << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return total_failures;
}
