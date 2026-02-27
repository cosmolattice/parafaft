#include "parafaft_r2c.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  const int global_shape[3] = {32, 32, 32};
  parafaft::ParaFaFT_R2C<3> fft(global_shape);

  // Get local size, required for R2C transforms. This may be larger than
  // N₀*N₁*(N₂+2) due to additional memory requirements for intermediate stages.
  const int local_padded_size = fft.get_required_output_size();

  // Allocate arrays
  std::vector<double> real_data(local_padded_size);

  // Initialize real data...
  int local_real_shape[3], real_start[3];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);

  // Fill the first N₀*N₁*(N₂+2) elements with your data ...

  // Forward R2C FFT: real -> complex (in-place)
  // Note: the data passed must be padded to the required size for the R2C
  // transform, i.e. the last (smallest) dimension must have N + 2 (real)
  // elements, where the last 2 are padding for the complex output (see also the
  // fftw documentation for R2C transforms)
  fft.forward_in_place(real_data.data());

  // ... process in frequency domain ...

  // Backward C2R FFT: complex -> real
  fft.backward_in_place(real_data.data());

  // Normalize (FFTW convention)
  const double scale =
      1.0 / (global_shape[0] * global_shape[1] * global_shape[2]);
  for (auto &val : real_data)
    val *= scale;

  MPI_Finalize();
  return 0;
}
