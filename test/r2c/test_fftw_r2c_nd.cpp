// ============================================================================
// Generalized N-dimensional R2C Gaussian test for parafaft (FFTW backend).
//
// Usage:  mpirun -np <P> ./test_fftw_mpi_r2c_gaussian_nd [N] [D] [S]
//
//   N  – grid side length (default 32, or 16 when D >= 4)
//   D  – number of dimensions (default 3, range 2..6)
//   S  – shape: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial (default 0)
//
// The test generates a D-dimensional Gaussian on an N^D hypercubic grid,
// computes the forward R2C transform with parafaft (MPI-distributed, FFTW
// backend, out-of-place), and compares against a serial FFTW reference.
// ============================================================================

#include "../../parafaft_r2c.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

// ============================================================================
// Templated comparison function – works for any D.
// ============================================================================
template <int D> int compare_fftwBackend(const int N, int rank, int shape_id)
{
  using namespace parafaft_test;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft FFTWBackend for " << D << "D R2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << " [shape=" << shape_name(shape_id) << "]" << std::endl;
  }

  // ---- Generate global data (every rank keeps a full copy for reference) ----
  const auto global_original_data = generate_r2c_data_nd<D>(N, shape_id);

  // ---- Serial FFTW R2C reference --------------------------------------------
  auto global_fftw_reference = global_original_data;
  fftw_r2c_reference_nd<D>(global_fftw_reference.data(), N);

  // ---- ParaFaFT setup -------------------------------------------------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT_R2C<D, parafaft::FFTWBackend<>> fft(global_shape.data());

  int local_real_shape[D], real_start[D];
  int local_complex_shape[D], complex_start[D];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);
  fft.get_local_complex_shape(local_complex_shape);
  fft.get_complex_global_start(complex_start);

  int local_real_size = fft.get_required_output_size();
  int local_complex_size = fft.get_local_complex_size();

  // Print local real shape
  std::cout << "Local real shape on rank " << rank << ": ";
  print_shape<D>(std::cout, local_real_shape);
  std::cout << std::endl;

  // Validate local complex size
  {
    int expected = 1;
    for (int d = 0; d < D; ++d)
      expected *= local_complex_shape[d];
    if (local_complex_size != expected) {
      std::cout << "Rank " << rank << ": Test failed: Local complex size mismatch." << std::endl;
      return 1;
    }
  }

  // ---- Scatter global real data into local partition ------------------------
  // Local real data uses stride N in last dimension (non-padded, out-of-place).
  std::vector<double> local_data(local_real_size);

  iterate_nd<D>(local_real_shape, [&](const std::array<int, D> &lidx) {
    // Build global multi-index
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = real_start[d] + lidx[d];

    // Global flat index (all dims stride N)
    std::array<int, D> full_shape;
    full_shape.fill(N);
    int global_flat = nd_index<D>(gidx, full_shape);

    // Local flat index: dimensions 0..D-2 use local_real_shape, last dim stride N
    std::array<int, D> local_strides_shape;
    for (int d = 0; d < D - 1; ++d)
      local_strides_shape[d] = local_real_shape[d];
    local_strides_shape[D - 1] = N; // last dim stride = N for non-padded layout
    int local_flat = nd_index_real<D>(lidx, local_strides_shape.data());

    local_data[local_flat] = global_original_data[global_flat];
  });

  // ---- Allocate output and perform forward FFT (out-of-place) ---------------
  int local_result_buffer_size = local_real_size / 2;
  std::vector<std::complex<double>> local_result(local_result_buffer_size);
  fft.forward(local_data.data(), local_result.data());

  // ---- Compare against reference using final complex layout -----------------
  std::complex<double> *const global_result = reinterpret_cast<std::complex<double> *>(global_fftw_reference.data());

  // Global complex shape: all dims N except last = N/2+1
  auto global_complex_shape = r2c_complex_shape<D>(N);

  double max_error = 0.0;

  iterate_nd<D>(local_complex_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = complex_start[d] + lidx[d];

    int global_flat = nd_index_complex<D>(gidx, global_complex_shape.data());
    int local_flat = nd_index_complex<D>(lidx, local_complex_shape);

    double error = std::abs(local_result[local_flat] - global_result[global_flat]);
    if (error > max_error) max_error = error;
  });

  // ---- Gather results -------------------------------------------------------
  const double tolerance = 1e-10;
  const bool local_success = (max_error < tolerance);
  int global_success = local_success ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (global_success == 1 && rank == 0) {
    std::cout << "Test passed: parafaft FFTWBackend produces correct results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
  } else if (!global_success) {
    std::cout << "Rank " << rank << ": Test failed: parafaft FFTWBackend produces incorrect results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 1;
  }

  return 0;
}

// ============================================================================
// Dispatch
// ============================================================================
int dispatch(int D, int N, int rank, int shape_id)
{
  switch (D) {
  case 1:
    return compare_fftwBackend<1>(N, rank, shape_id);
  case 2:
    return compare_fftwBackend<2>(N, rank, shape_id);
  case 3:
    return compare_fftwBackend<3>(N, rank, shape_id);
  case 4:
    return compare_fftwBackend<4>(N, rank, shape_id);
  case 5:
    return compare_fftwBackend<5>(N, rank, shape_id);
  case 6:
    return compare_fftwBackend<6>(N, rank, shape_id);
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
  int S = 0;

  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) D = std::atoi(argv[2]);
  if (argc > 3) S = std::atoi(argv[3]);

  if (D == 1 && size > 1) {
    if (rank == 0) std::cerr << "Error: D=1 test cannot be run with multiple processes." << std::endl;
    MPI_Finalize();
    return 1;
  }

  if (N <= 0) N = (D >= 4) ? 16 : 32;

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: r2c/fftw_mpi_r2c_gaussian_nd" << std::endl;
    std::cout << "# D = " << D << ",  N = " << N << ",  S = " << parafaft_test::shape_name(S) << std::endl;
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
