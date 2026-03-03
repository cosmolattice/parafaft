// ============================================================================
// Generalized N-dimensional C2C test for parafaft (FFTW backend).
//
// Usage:  mpirun -np <P> ./test_fftw_mpi_c2c_gaussian_nd [N] [D] [S]
//
//   N  – grid side length (default 32, or 16 when D >= 4)
//   D  – number of dimensions (default 3, range 2..6)
//   S  – shape: 0=Gaussian, 1=StepFunction, 2=RandomPolynomial (default 0)
//
// The test generates a D-dimensional Gaussian on an N^D hypercubic grid,
// computes the forward C2C transform with parafaft (MPI-distributed), and
// compares against a serial FFTW reference.
// ============================================================================

#include "../../backend/fftw3/fft_backend_fftw.hpp"
#include "../../parafaft_c2c.hpp"
#include "../test_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

// ============================================================================
// Templated comparison function – works for any D.
// ============================================================================
template <int D> int compare_fftwBackend(const int N, int rank, int shape_id) {
  using namespace parafaft_test;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft FFTWBackend for " << D
              << "D C2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << " [shape=" << shape_name(shape_id) << "]" << std::endl;
  }

  // ---- Generate global data on every rank (reference needs the full grid) ---
  const auto global_original_data = generate_c2c_data_nd<D>(N, shape_id);

  // ---- Serial FFTW reference ------------------------------------------------
  auto global_fftw_reference = global_original_data;
  fftw_c2c_reference_nd<D>(global_fftw_reference.data(), N);

  // ---- ParaFaFT setup -------------------------------------------------------
  std::array<int, D> global_shape;
  global_shape.fill(N);
  parafaft::ParaFaFT<D, parafaft::FFTWBackend> fft(global_shape.data());

  int local_size = fft.get_local_size();
  int buffer_size = fft.get_required_output_size();
  int local_shape[D];
  int global_start[D];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);

  // ---- Scatter global data into local partition -----------------------------
  std::vector<std::complex<double>> local_data(buffer_size);

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

  std::cout << "First 32 local values on rank " << rank << ": ";
  for (int i = 0; i < std::min(32, local_size); ++i) {
    std::cout << local_data[i] << " ";
  }
  std::cout << std::endl;

  // Print local shape
  std::cout << "Local shape on rank " << rank << ": ";
  print_shape<D>(std::cout, local_shape);
  std::cout << std::endl;

  // ---- Forward FFT ----------------------------------------------------------
  fft.forward(local_data.data());

  std::cout << "After transform: First 32 local values on rank " << rank
            << ": ";
  for (int i = 0; i < std::min(32, local_size); ++i) {
    std::cout << local_data[i] << " ";
  }
  std::cout << std::endl;

  // ---- Compare against reference using final (redistributed) layout ---------
  int final_shape[D];
  int final_start[D];
  fft.get_final_shape(final_shape);
  fft.get_final_start(final_start);

  double max_error = 0.0;

  iterate_nd<D>(final_shape, [&](const std::array<int, D> &fidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = final_start[d] + fidx[d];

    int global_flat = nd_index<D>(gidx, full_shape);
    int local_flat = nd_index<D>(fidx.data(), final_shape);
    double error =
        std::abs(local_data[local_flat] - global_fftw_reference[global_flat]);
    if (error > max_error)
      max_error = error;
  });

  // ---- Gather results -------------------------------------------------------
  const double tolerance = 1e-10;
  const bool local_success = (max_error < tolerance);
  int global_success = local_success ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &global_success, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);

  if (global_success == 1 && rank == 0) {
    std::cout << "Test passed: parafaft FFTWBackend produces correct results."
              << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
  } else if (!global_success) {
    std::cout
        << "Rank " << rank
        << ": Test failed: parafaft FFTWBackend produces incorrect results."
        << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 1;
  }

  return 0;
}

// ============================================================================
// Dispatch: select the right template instantiation at runtime.
//
// We explicitly instantiate D = 2..6.  Higher dimensions can be added by
// extending the switch below.
// ============================================================================
int dispatch(int D, int N, int rank, int shape_id) {
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

  // Parse optional arguments: [N] [D] [S]
  int D = 3;  // default dimensionality
  int N = -1; // sentinel — pick a sensible default based on D
  int S = 0;  // default shape: Gaussian

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

  // Pick a reasonable default N if not provided
  if (N <= 0)
    N = (D >= 4) ? 16 : 32;

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: c2c/fftw_mpi_c2c_gaussian_nd" << std::endl;
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
