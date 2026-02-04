#include "../../parafaft_generic.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char **argv)
{
  std::cout << "########################################" << std::endl;
  std::cout << "# TEST: c2c/fftw_mpi_4d_roundtrip" << std::endl;
  std::cout << "########################################" << std::endl;

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int global_shape[4] = {4, 4, 4, 4};
  parafaft::ParaFaFT<4> fft(global_shape);

  int local_size = fft.get_local_size();
  std::vector<std::complex<double>> data(local_size, {1.0, 0.0});

  if (rank == 0) {
    std::cout << "Testing 4D roundtrip with constant (1,0) input\n";
    std::cout << "Global shape: 4x4x4x4 = 256 elements\n";
  }

  fft.forward(data.data());

  if (rank == 0) {
    std::cout << "After forward: data[0]=" << data[0] << " (expected 256)\n";
  }

  fft.backward(data.data());

  // Normalize
  double scale = 1.0 / (global_shape[0] * global_shape[1] * global_shape[2] * global_shape[3]);
  for (int i = 0; i < local_size; ++i) {
    data[i] *= scale;
  }

  if (rank == 0) {
    std::cout << "After roundtrip (normalized): data[0]=" << data[0] << " (expected 1)\n";

    bool all_correct = true;
    for (int i = 0; i < local_size; ++i) {
      if (std::abs(data[i].real() - 1.0) > 1e-10 || std::abs(data[i].imag()) > 1e-10) {
        all_correct = false;
        std::cout << "ERROR at index " << i << ": " << data[i] << "\n";
        if (i > 5) break; // Don't spam
      }
    }
    if (all_correct) {
      std::cout << "SUCCESS: All elements are (1,0)!\n";
    }
  }

  MPI_Finalize();
  return 0;
}
