// cuFFT version of test_mpi_r2c_padded_gaussian.cpp
#include "../../parafaft_r2c.hpp"
#include "../../backend/cufft/fft_backend_cufft.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cstdlib>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Test parameters
  int N = 32; // Default
  if (argc > 1) {
    N = std::atoi(argv[1]);
  }

  const int N0 = N, N1 = N, N2 = N;
  const int complex_N2 = N2 / 2 + 1;
  int global_shape[3] = {N0, N1, N2};

  const double center0 = N0 / 2.0;
  const double center1 = N1 / 2.0;
  const double center2 = N2 / 2.0;
  const double sigma = 4.0;

  // Create R2C FFT object with cuFFT backend
  parafaft::ParaFaFT_R2C<3, parafaft::CuFFTBackend> fft(global_shape);

  // Get local dimensions
  int local_real_shape[3], real_start[3];
  int local_complex_shape[3], complex_start[3];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);
  fft.get_local_complex_shape(local_complex_shape);
  fft.get_complex_global_start(complex_start);

  int local_padded_size = fft.get_local_in_place_buffer_size();
  int local_complex_size = fft.get_local_complex_size();

  // Allocate PADDED local array (used for in-place transform)
  std::vector<double> padded_real_data(local_padded_size);

  // Initialize local portion of Gaussian (only first N2 elements per row)
  int padded_stride = 2 * complex_N2; // Padded row length
  for (int i0 = 0; i0 < local_real_shape[0]; ++i0) {
    for (int i1 = 0; i1 < local_real_shape[1]; ++i1) {
      for (int i2 = 0; i2 < N2; ++i2) {
        double g0 = real_start[0] + i0;
        double g1 = real_start[1] + i1;
        double g2 = real_start[2] + i2;
        double x = g0 - center0;
        double y = g1 - center1;
        double z = g2 - center2;
        double r2 = x * x + y * y + z * z;

        int padded_idx = (i0 * local_real_shape[1] + i1) * padded_stride + i2;
        padded_real_data[padded_idx] = std::exp(-r2 / (2.0 * sigma * sigma));
      }
    }
  }

  if (rank == 0) {
    std::cout << "R2C Padded FFT Test (cuFFT backend): " << N0 << "x" << N1 << "x" << N2 << " with " << size
              << " processes\n";
    std::cout << "Complex output shape: " << N0 << "x" << N1 << "x" << complex_N2 << "\n";
  }

  // Perform in-place parallel R2C FFT (in-place: real → complex)
  fft.forward_in_place(padded_real_data.data());

  // Access complex result via reinterpret_cast
  std::complex<double> *complex_data = reinterpret_cast<std::complex<double> *>(padded_real_data.data());

  // Gather shape and start info from all ranks for reconstruction
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  std::vector<int> all_shapes(nranks * 3);
  std::vector<int> all_starts(nranks * 3);
  int shape_data[3] = {local_complex_shape[0], local_complex_shape[1], local_complex_shape[2]};
  int start_data[3] = {complex_start[0], complex_start[1], complex_start[2]};

  MPI_Gather(shape_data, 3, MPI_INT, all_shapes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(start_data, 3, MPI_INT, all_starts.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

  // Gather data sizes
  std::vector<int> sizes(nranks);
  std::vector<int> displs(nranks);
  MPI_Gather(&local_complex_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < nranks; ++i) {
      displs[i] = displs[i - 1] + sizes[i - 1];
    }
  }

  // Gather all complex data to rank 0
  std::vector<std::complex<double>> gathered_data;
  if (rank == 0) {
    gathered_data.resize(displs[nranks - 1] + sizes[nranks - 1]);
  }

  MPI_Gatherv(complex_data, local_complex_size, MPI_C_DOUBLE_COMPLEX, gathered_data.data(), sizes.data(), displs.data(),
              MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

  // On rank 0: reconstruct global array and verify
  int test_passed = 1;
  if (rank == 0) {
    int complex_size = N0 * N1 * complex_N2;
    std::vector<std::complex<double>> global_complex(complex_size);

    int offset = 0;
    for (int r = 0; r < nranks; ++r) {
      int lshape0 = all_shapes[r * 3 + 0];
      int lshape1 = all_shapes[r * 3 + 1];
      int lshape2 = all_shapes[r * 3 + 2];
      int lstart0 = all_starts[r * 3 + 0];
      int lstart1 = all_starts[r * 3 + 1];
      int lstart2 = all_starts[r * 3 + 2];

      for (int i0 = 0; i0 < lshape0; ++i0) {
        for (int i1 = 0; i1 < lshape1; ++i1) {
          for (int i2 = 0; i2 < lshape2; ++i2) {
            int local_idx = (i0 * lshape1 + i1) * lshape2 + i2;
            int global_i0 = lstart0 + i0;
            int global_i1 = lstart1 + i1;
            int global_i2 = lstart2 + i2;
            int global_idx = (global_i0 * N1 + global_i1) * complex_N2 + global_i2;
            global_complex[global_idx] = gathered_data[offset + local_idx];
          }
        }
      }
      offset += sizes[r];
    }

    std::cout << "DC component: " << global_complex[0] << "\n";

    // Basic sanity check: DC component should be positive real for a Gaussian
    if (global_complex[0].real() > 0 && std::abs(global_complex[0].imag()) < 1e-10) {
      std::cout << "PASSED: DC component is positive real\n";
      test_passed = 1;
    } else {
      std::cout << "FAILED: DC component check failed\n";
      test_passed = 0;
    }
  }

  // Broadcast result
  MPI_Bcast(&test_passed, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Finalize();
  return test_passed ? 0 : 1;
}
