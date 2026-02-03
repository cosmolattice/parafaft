// cuFFT version of test_8cubed.cpp
#include "../../parafaft_generic.hpp"
#include "../../backend/cufft/fft_backend_cufft.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int global_shape[3] = {8, 8, 8};
  parafaft::ParaFaFT<3, parafaft::CuFFTBackend> fft(global_shape);

  int local_size = fft.get_local_size();

  // Initialize with sequential values
  std::vector<std::complex<double>> data(local_size);
  for (int i = 0; i < local_size; ++i) {
    data[i] = std::complex<double>(i + 1.0, 0.0);
  }

  std::vector<std::complex<double>> original = data;

  if (rank == 0) {
    std::cout << "Testing 8x8x8 with sequential values (cuFFT backend)\n";
    std::cout << "Local size: " << local_size << "\n";
  }

  fft.forward(data.data());
  fft.backward(data.data());

  // Normalize
  double scale = 1.0 / 512.0;
  for (int i = 0; i < local_size; ++i) {
    data[i] *= scale;
  }

  double max_error = 0.0;
  for (int i = 0; i < local_size; ++i) {
    double err = std::abs(data[i] - original[i]);
    max_error = std::max(max_error, err);
  }

  double global_max;
  MPI_Reduce(&max_error, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Max error: " << global_max << "\n";
    if (global_max < 1e-10) {
      std::cout << "SUCCESS!\n";
    } else {
      std::cout << "FAILED\n";
      std::cout << "data[0] = " << data[0] << " (expected " << original[0] << ")\n";
    }
  }

  MPI_Finalize();
  return 0;
}
