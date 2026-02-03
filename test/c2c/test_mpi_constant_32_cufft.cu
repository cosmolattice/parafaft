// cuFFT version of test_mpi_constant_32.cpp
#include "../../parafaft_generic.hpp"
#include "../../backend/cufft/fft_backend_cufft.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int global_shape[3] = {32, 32, 32};
  parafaft::ParaFaFT<3, parafaft::CuFFTBackend> fft(global_shape);

  int local_size = fft.get_local_size();
  std::vector<std::complex<double>> data(local_size, {1.0, 0.0});

  if (rank == 0) {
    std::cout << "Testing 32x32x32 constant array (cuFFT backend)\n";
    std::cout << "Initial data[0] = " << data[0] << "\n";
  }

  fft.forward(data.data());

  if (rank == 0) {
    std::cout << "After forward: data[0] = " << data[0] << "\n";
    std::cout << "Expected: (32768, 0)\n";
  }

  fft.backward(data.data());

  if (rank == 0) {
    std::cout << "After backward (before norm): data[0] = " << data[0] << "\n";
    std::cout << "Expected: (32768, 0)\n";
  }

  // Normalize
  double scale = 1.0 / (global_shape[0] * global_shape[1] * global_shape[2]);
  for (int i = 0; i < local_size; ++i) {
    data[i] *= scale;
  }

  if (rank == 0) {
    std::cout << "After normalization: data[0] = " << data[0] << "\n";
    std::cout << "Expected: (1, 0)\n";

    bool all_correct = true;
    for (int i = 0; i < local_size; ++i) {
      if (std::abs(data[i].real() - 1.0) > 1e-10 || std::abs(data[i].imag()) > 1e-10) {
        all_correct = false;
        std::cout << "ERROR at index " << i << ": " << data[i] << "\n";
        if (i > 5) break;
      }
    }
    if (all_correct) {
      std::cout << "SUCCESS: All elements are (1,0)!\n";
    }
  }

  MPI_Finalize();
  return 0;
}
