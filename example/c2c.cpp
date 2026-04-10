#include "parafaft_c2c.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  // Works for any dimension!
  const int global_shape[3] = {32, 32, 32};
  parafaft::ParaFaFT_C2C<3> fft(global_shape);

  // Allocate local data (use get_required_output_size for sufficient space)
  const int local_size = fft.get_local_size();
  const int buffer_size = fft.get_required_output_size();
  std::vector<std::complex<double>> data(buffer_size);

  // Initialize data
  int local_shape[3], global_start[3];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);
  // ...

  // Forward FFT (in-place)
  fft.forward(data.data());

  // Backward FFT (in-place)
  fft.backward(data.data());

  // Normalize (FFTW convention)
  const double scale =
      1.0 / (global_shape[0] * global_shape[1] * global_shape[2]);
  for (auto &val : data)
    val *= scale;

  MPI_Finalize();
  return 0;
}
