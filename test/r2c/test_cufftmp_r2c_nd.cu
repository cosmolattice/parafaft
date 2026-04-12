// ============================================================================
// N-dimensional R2C test for parafaft (cuFFTMp distributed backend).
//
// Usage:  mpirun -np <P> ./test_cufftmp_r2c_nd [N] [D] [S]
//
//   N  – grid side length (default 32)
//   D  – number of dimensions (2 or 3 only — cuFFTMp limitation)
//   S  – shape: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial (default 0)
//
// Uses the in-place API (forward_in_place / backward_in_place) with the
// backend-managed NVSHMEM buffer. Compares against a serial FFTW reference.
// ============================================================================

#include "../../parafaft_r2c.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

// ============================================================================
// GPU device assignment (required before cuFFTMp plan creation)
// ============================================================================
static void assign_gpu() {
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  cudaSetDevice(local_rank % num_devices);
  MPI_Comm_free(&local_comm);
}

// ============================================================================
// Templated comparison: parafaft(cuFFTMp) vs serial FFTW reference.
// ============================================================================
template <int D>
int compare_cuFFTMpBackend(const int N, int rank, int shape_id) {
  using namespace parafaft_test;
  using Backend = parafaft::CuFFTMpBackend<D>;
  using Complex = typename Backend::Complex;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft CuFFTMpBackend for " << D
              << "D R2C transform of size ";
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

  // ---- ParaFaFT setup (cuFFTMp handles decomposition internally) ------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT_R2C<D, Backend> fft(global_shape.data());

  int local_real_shape[D], real_start[D];
  int local_complex_shape[D], complex_start[D];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);
  fft.get_local_complex_shape(local_complex_shape);
  fft.get_complex_global_start(complex_start);

  int local_padded_size = fft.get_required_output_size();
  int local_complex_size = fft.get_local_complex_size();
  const int padded_last = local_real_shape[D - 1] + 2;

  if (rank == 0) {
    std::cout << "Local real shape on rank " << rank << ": ";
    print_shape<D>(std::cout, local_real_shape);
    std::cout << std::endl;
  }

  // ---- Scatter global real data into backend-managed buffer -----------------
  std::vector<double> local_padded(local_padded_size, 0.0);

  iterate_nd<D>(local_real_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = real_start[d] + lidx[d];

    std::array<int, D> full_shape;
    full_shape.fill(N);
    int global_flat = nd_index<D>(gidx, full_shape);

    // Padded flat index: last dim uses padded_last stride
    int padded_flat = lidx[0];
    for (int d = 1; d < D - 1; ++d)
      padded_flat = padded_flat * local_real_shape[d] + lidx[d];
    padded_flat = padded_flat * padded_last + lidx[D - 1];

    local_padded[padded_flat] = global_original_data[global_flat];
  });

  // Copy into the NVSHMEM-backed buffer
  double *buffer = fft.get_real_buffer();
  cudaMemcpy(buffer, local_padded.data(), local_padded_size * sizeof(double),
             cudaMemcpyHostToDevice);

  // ---- Forward R2C FFT ------------------------------------------------------
  fft.forward_in_place(buffer);
  cudaDeviceSynchronize();

  // ---- Copy complex result back to host -------------------------------------
  std::vector<std::complex<double>> local_result(local_complex_size);
  Complex *complex_buffer = fft.get_buffer();
  cudaMemcpy(local_result.data(), complex_buffer,
             local_complex_size * sizeof(std::complex<double>),
             cudaMemcpyDeviceToHost);

  // ---- Compare against reference --------------------------------------------
  std::complex<double> *const global_result =
      reinterpret_cast<std::complex<double> *>(global_fftw_reference.data());
  auto global_complex_shape = r2c_complex_shape<D>(N);

  double max_error = 0.0;

  iterate_nd<D>(local_complex_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = complex_start[d] + lidx[d];

    int global_flat = nd_index_complex<D>(gidx, global_complex_shape.data());
    int local_flat = nd_index_complex<D>(lidx, local_complex_shape);

    double error =
        std::abs(local_result[local_flat] - global_result[global_flat]);
    if (error > max_error)
      max_error = error;
  });

  const double tolerance = 1e-10;
  const bool local_success = (max_error < tolerance);
  int global_success = local_success ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);

  if (global_success == 1 && rank == 0) {
    std::cout << "Test passed: parafaft CuFFTMpBackend R2C produces correct results."
              << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
  } else if (!global_success) {
    std::cout << "Rank " << rank
              << ": Test failed: parafaft CuFFTMpBackend R2C produces incorrect results."
              << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 1;
  }

  return 0;
}

// ============================================================================
// Dispatch (cuFFTMp supports only D=2 and D=3)
// ============================================================================
int dispatch(int D, int N, int rank, int shape_id) {
  int failures = 0;

  switch (D) {
  case 2:
    failures += compare_cuFFTMpBackend<2>(N, rank, shape_id);
    break;
  case 3:
    failures += compare_cuFFTMpBackend<3>(N, rank, shape_id);
    break;
  default:
    if (rank == 0)
      std::cerr << "Unsupported dimensionality D=" << D
                << " for cuFFTMp (supported: 2, 3)" << std::endl;
    return 1;
  }

  return failures;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  assign_gpu();

  int D = 3;
  int N = 32;
  int S = 0;

  if (argc > 1)
    N = std::atoi(argv[1]);
  if (argc > 2)
    D = std::atoi(argv[2]);
  if (argc > 3)
    S = std::atoi(argv[3]);

  if (D < 2 || D > 3) {
    if (rank == 0)
      std::cerr << "Error: cuFFTMp only supports D=2 or D=3." << std::endl;
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: r2c/cufftmp_r2c_nd" << std::endl;
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
