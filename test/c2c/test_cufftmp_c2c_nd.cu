// ============================================================================
// N-dimensional C2C test for parafaft (cuFFTMp distributed backend).
//
// Usage:  mpirun -np <P> ./test_cufftmp_c2c_nd [N] [D] [S]
//
//   N  – grid side length (default 32)
//   D  – number of dimensions (2 or 3 only — cuFFTMp limitation)
//   S  – shape: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial (default 0)
//
// Each MPI rank is assigned a GPU. cuFFTMp handles decomposition and
// communication internally via NVSHMEM. The test compares against a serial
// FFTW reference.
// ============================================================================

#include "../../parafaft_c2c.hpp"
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
              << "D C2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << " [shape=" << shape_name(shape_id) << "]" << std::endl;
  }

  // ---- Generate global data on every rank -----------------------------------
  const auto global_original_data = generate_c2c_data_nd<D>(N, shape_id);

  // ---- Serial FFTW reference ------------------------------------------------
  auto global_fftw_reference = global_original_data;
  fftw_c2c_reference_nd<D>(global_fftw_reference.data(), N);

  // ---- ParaFaFT setup (cuFFTMp handles decomposition internally) ------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT_C2C<D, Backend> fft(global_shape.data());

  int local_size = fft.get_local_size();
  int local_shape[D];
  int global_start[D];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);

  if (rank == 0) {
    std::cout << "Local shape on rank " << rank << ": ";
    print_shape<D>(std::cout, local_shape);
    std::cout << std::endl;
  }

  // ---- Scatter global data into backend-managed buffer ----------------------
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

  // Copy into the NVSHMEM-backed buffer
  Complex *buffer = fft.get_buffer();
  cudaMemcpy(buffer, local_data.data(), local_size * sizeof(Complex),
             cudaMemcpyHostToDevice);

  // ---- Forward FFT (cuFFTMp handles everything internally) ------------------
  fft.forward(buffer);
  cudaDeviceSynchronize();

  // ---- Copy back to host ----------------------------------------------------
  int final_shape[D];
  int final_start[D];
  fft.get_final_shape(final_shape);
  fft.get_final_start(final_start);

  int final_size = 1;
  for (int d = 0; d < D; ++d) final_size *= final_shape[d];

  std::vector<std::complex<double>> result_data(final_size);
  cudaMemcpy(result_data.data(), buffer, final_size * sizeof(Complex),
             cudaMemcpyDeviceToHost);

  // ---- Compare against reference --------------------------------------------
  double max_error = 0.0;

  iterate_nd<D>(final_shape, [&](const std::array<int, D> &fidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = final_start[d] + fidx[d];

    int global_flat = nd_index<D>(gidx, full_shape);
    int local_flat = nd_index<D>(fidx.data(), final_shape);
    double error =
        std::abs(result_data[local_flat] - global_fftw_reference[global_flat]);
    if (error > max_error)
      max_error = error;
  });

  const double tolerance = 1e-10;
  const bool local_success = (max_error < tolerance);
  int global_success = local_success ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);

  if (global_success == 1 && rank == 0) {
    std::cout << "Test passed: parafaft CuFFTMpBackend produces correct results."
              << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
  } else if (!global_success) {
    std::cout << "Rank " << rank
              << ": Test failed: parafaft CuFFTMpBackend produces incorrect results."
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
    std::cout << "# TEST: c2c/cufftmp_c2c_nd" << std::endl;
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
