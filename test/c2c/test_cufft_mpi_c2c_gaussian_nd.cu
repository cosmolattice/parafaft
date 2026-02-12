// ============================================================================
// Generalized N-dimensional C2C Gaussian test for parafaft (cuFFT backend).
//
// Usage:  mpirun -np <P> ./test_cufft_mpi_c2c_gaussian_nd [N] [D]
//
//   N  – grid side length (default 32, or 16 when D >= 4)
//   D  – number of dimensions (default 3, range 2..6)
//
// The test generates a D-dimensional Gaussian on an N^D hypercubic grid,
// computes the forward C2C transform with parafaft (MPI-distributed, cuFFT
// backend), and compares against a serial FFTW reference.
//
// For D = 2 or 3 an additional sanity check compares sequential cuFFT against
// sequential FFTW.  For D >= 4, cuFFT has no cufftPlanNd equivalent, so this
// sanity check is skipped.
// ============================================================================

#include "../../parafaft_generic.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

// ============================================================================
// Templated comparison: parafaft(cuFFT) vs serial FFTW reference.
// ============================================================================
template <int D> int compare_cuFFTBackend(const int N, int rank)
{
  using namespace parafaft_test;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft CuFFTBackend for " << D << "D C2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;
  }

  // ---- Generate global data on every rank -----------------------------------
  const auto global_original_data = generate_gaussian_nd<D>(N, 4.0);

  // ---- Serial FFTW reference ------------------------------------------------
  auto global_fftw_reference = global_original_data;
  fftw_c2c_reference_nd<D>(global_fftw_reference.data(), N);

  // ---- ParaFaFT setup -------------------------------------------------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT<D, parafaft::CuFFTBackend> fft(global_shape.data());

  int local_size = fft.get_local_size();
  int local_shape[D];
  int global_start[D];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);

  // ---- Scatter global data into local partition -----------------------------
  std::vector<std::complex<double>> local_data(local_size);
  std::array<int, D> full_shape;
  full_shape.fill(N);

  iterate_nd<D>(local_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = global_start[d] + lidx[d];

    int global_flat = nd_index<D>(gidx, full_shape);
    int local_flat = nd_index<D>(lidx.data(), local_shape);
    local_data[local_flat] = global_original_data[global_flat];
  });

  std::cout << "Local shape on rank " << rank << ": ";
  print_shape<D>(std::cout, local_shape);
  std::cout << std::endl;

  // ---- Copy to device -------------------------------------------------------
  parafaft::CuFFTBackend::Complex *d_data = nullptr;
  cudaMalloc((void **)&d_data, local_size * sizeof(parafaft::CuFFTBackend::Complex));
  cudaMemcpy(d_data, local_data.data(), local_size * sizeof(parafaft::CuFFTBackend::Complex), cudaMemcpyHostToDevice);

  // ---- Forward FFT ----------------------------------------------------------
  fft.forward(d_data);

  // ---- Copy back to host ----------------------------------------------------
  int final_shape[D];
  int final_start[D];
  fft.get_final_shape(final_shape);
  fft.get_final_start(final_start);

  cudaMemcpy(local_data.data(), d_data, local_size * sizeof(parafaft::CuFFTBackend::Complex), cudaMemcpyDeviceToHost);
  cudaFree(d_data);

  // ---- Compare against reference --------------------------------------------
  double max_error = 0.0;

  iterate_nd<D>(final_shape, [&](const std::array<int, D> &fidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = final_start[d] + fidx[d];

    int global_flat = nd_index<D>(gidx, full_shape);
    int local_flat = nd_index<D>(fidx.data(), final_shape);
    double error = std::abs(local_data[local_flat] - global_fftw_reference[global_flat]);
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
int dispatch(int D, int N, int rank)
{
  using namespace parafaft_test;
  int failures = 0;

  // cuFFT-vs-FFTW sanity check (only for D = 2, 3)
  if (D == 2)
    failures += compare_cufft_vs_fftw<2>(N, rank);
  else if (D == 3)
    failures += compare_cufft_vs_fftw<3>(N, rank);

  // parafaft comparison
  switch (D) {
  case 2:
    failures += compare_cuFFTBackend<2>(N, rank);
    break;
  case 3:
    failures += compare_cuFFTBackend<3>(N, rank);
    break;
  case 4:
    failures += compare_cuFFTBackend<4>(N, rank);
    break;
  case 5:
    failures += compare_cuFFTBackend<5>(N, rank);
    break;
  case 6:
    failures += compare_cuFFTBackend<6>(N, rank);
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

  if (argc > 1) N = std::atoi(argv[1]);
  if (argc > 2) D = std::atoi(argv[2]);

  if (N <= 0) N = (D >= 4) ? 16 : 32;

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: c2c/cufft_mpi_c2c_gaussian_nd" << std::endl;
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
