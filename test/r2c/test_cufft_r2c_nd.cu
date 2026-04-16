// ============================================================================
// Generalized N-dimensional R2C Gaussian test for parafaft (cuFFT backend).
//
// Usage:  mpirun -np <P> ./test_cufft_mpi_r2c_gaussian_nd [N] [D] [S]
//
//   N  – grid side length (default 32, or 16 when D >= 4)
//   D  – number of dimensions (default 3, range 2..6)
//   S  – shape: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial (default 0)
//
// The test generates a D-dimensional Gaussian on an N^D hypercubic grid,
// computes the forward R2C transform with parafaft (MPI-distributed, cuFFT
// backend, out-of-place), and compares against a serial FFTW reference.
//
// For D = 2 or 3 an additional sanity check compares sequential cuFFT against
// sequential FFTW.  For D >= 4, this is skipped.
// ============================================================================

#include "../../parafaft_r2c.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

// ============================================================================
// Templated comparison: parafaft(cuFFT) vs serial FFTW reference.
// ============================================================================
template <int D> int compare_cuFFTBackend(const int N, int rank, int shape_id)
{
  using namespace parafaft_test;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft CuFFTBackend for " << D << "D R2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << " [shape=" << shape_name(shape_id) << "]" << std::endl;
  }

  // ---- Generate global data -------------------------------------------------
  const auto global_original_data = generate_r2c_data_nd<D>(N, shape_id);

  // ---- Serial FFTW R2C reference --------------------------------------------
  auto global_fftw_reference = global_original_data;
  fftw_r2c_reference_nd<D>(global_fftw_reference.data(), N);

  // ---- ParaFaFT setup -------------------------------------------------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT_R2C<D, parafaft::CuFFTBackend<>> fft(global_shape.data());

  int local_real_shape[D], real_start[D];
  int local_complex_shape[D], complex_start[D];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);
  fft.get_local_complex_shape(local_complex_shape);
  fft.get_complex_global_start(complex_start);

  int local_real_size = fft.get_required_output_size();
  int local_complex_size = fft.get_local_complex_size();

  std::cout << "Local real shape on rank " << rank << ": ";
  print_shape<D>(std::cout, local_real_shape);
  std::cout << std::endl;

  // Validate
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
  std::vector<double> local_data(local_real_size);

  iterate_nd<D>(local_real_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = real_start[d] + lidx[d];

    std::array<int, D> full_shape;
    full_shape.fill(N);
    int global_flat = nd_index<D>(gidx, full_shape);

    // Local: dims 0..D-2 use local_real_shape, last dim stride N
    std::array<int, D> local_strides_shape;
    for (int d = 0; d < D - 1; ++d)
      local_strides_shape[d] = local_real_shape[d];
    local_strides_shape[D - 1] = N;
    int local_flat = nd_index_real<D>(lidx, local_strides_shape.data());

    local_data[local_flat] = global_original_data[global_flat];
  });

  // ---- Copy to device -------------------------------------------------------
  double *d_data = nullptr;
  cudaMalloc((void **)&d_data, local_real_size * sizeof(double));
  cudaMemcpy(d_data, local_data.data(), local_real_size * sizeof(double), cudaMemcpyHostToDevice);

  // Allocate device memory for complex output
  parafaft::CuFFTBackend<>::Complex *d_result = nullptr;
  cudaMalloc((void **)&d_result, (local_real_size / 2) * sizeof(parafaft::CuFFTBackend<>::Complex));

  // ---- Forward FFT ----------------------------------------------------------
  fft.forward(d_data, d_result);

  // ---- Copy back to host ----------------------------------------------------
  std::vector<std::complex<double>> local_result(local_complex_size);
  cudaMemcpy(local_result.data(), d_result, local_complex_size * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  cudaFree(d_result);

  // ---- Compare against reference --------------------------------------------
  std::complex<double> *const global_result = reinterpret_cast<std::complex<double> *>(global_fftw_reference.data());

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

  const double tolerance = 1e-10;
  const bool local_success = (max_error < tolerance);
  int global_success = local_success ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (global_success == 1 && rank == 0) {
    std::cout << "Test passed: parafaft CuFFTBackend produces correct results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
  } else if (!global_success) {
    std::cout << "Rank " << rank << ": Test failed: parafaft CuFFTBackend produces incorrect results." << std::endl;
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
  using namespace parafaft_test;
  int failures = 0;

  // cuFFT-vs-FFTW R2C sanity check (only for D = 2, 3)
  if (D == 2)
    failures += compare_cufft_vs_fftw_r2c<2>(N, rank);
  else if (D == 3)
    failures += compare_cufft_vs_fftw_r2c<3>(N, rank);

  // parafaft comparison
  switch (D) {
  case 1:
    failures += compare_cuFFTBackend<1>(N, rank, shape_id);
    break;
  case 2:
    failures += compare_cuFFTBackend<2>(N, rank, shape_id);
    break;
  case 3:
    failures += compare_cuFFTBackend<3>(N, rank, shape_id);
    break;
  case 4:
    failures += compare_cuFFTBackend<4>(N, rank, shape_id);
    break;
  case 5:
    failures += compare_cuFFTBackend<5>(N, rank, shape_id);
    break;
  case 6:
    failures += compare_cuFFTBackend<6>(N, rank, shape_id);
    break;
  default:
    if (rank == 0) std::cerr << "Unsupported dimensionality D=" << D << " (supported: 2..6)" << std::endl;
    return 1;
  }

  return failures;
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
    std::cout << "# TEST: r2c/cufft_mpi_r2c_gaussian_nd" << std::endl;
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
