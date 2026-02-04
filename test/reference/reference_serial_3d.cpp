#include <fftw3.h>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

int main()
{
  // Array dimensions
  const int N0 = 32;
  const int N1 = 32;
  const int N2 = 32;
  const int total_size = N0 * N1 * N2;

  // Allocate array
  std::vector<std::complex<double>> data(total_size);

  // Gaussian parameters
  const double center0 = N0 / 2.0;
  const double center1 = N1 / 2.0;
  const double center2 = N2 / 2.0;
  const double sigma = 4.0; // Width of Gaussian

  std::cout << "Creating 3D Gaussian distribution...\n";
  std::cout << "Array size: " << N0 << " x " << N1 << " x " << N2 << "\n";
  std::cout << "Center: (" << center0 << ", " << center1 << ", " << center2 << ")\n";
  std::cout << "Sigma: " << sigma << "\n\n";

  // Initialize with 3D Gaussian
  for (int i0 = 0; i0 < N0; ++i0) {
    for (int i1 = 0; i1 < N1; ++i1) {
      for (int i2 = 0; i2 < N2; ++i2) {
        double x = i0 - center0;
        double y = i1 - center1;
        double z = i2 - center2;
        double r2 = x * x + y * y + z * z;
        double value = std::exp(-r2 / (2.0 * sigma * sigma));

        int idx = (i0 * N1 + i1) * N2 + i2;
        data[idx] = std::complex<double>(value, 0.0);
      }
    }
  }

  // Save initial distribution
  {
    std::ofstream file("initial_gaussian.txt");
    file << std::scientific << std::setprecision(15);
    file << "# 3D Gaussian distribution\n";
    file << "# Format: i0 i1 i2 real imag\n";
    file << "# Array size: " << N0 << " x " << N1 << " x " << N2 << "\n";

    for (int i0 = 0; i0 < N0; ++i0) {
      for (int i1 = 0; i1 < N1; ++i1) {
        for (int i2 = 0; i2 < N2; ++i2) {
          int idx = (i0 * N1 + i1) * N2 + i2;
          file << i0 << " " << i1 << " " << i2 << " " << data[idx].real() << " " << data[idx].imag() << "\n";
        }
      }
    }
    file.close();
    std::cout << "Saved initial distribution to: initial_gaussian.txt\n";
  }

  // Save copy of original for comparison
  std::vector<std::complex<double>> original = data;

  // Create FFTW plan for 3D transform
  std::cout << "Creating FFTW plan for 3D FFT...\n";
  fftw_plan plan_forward = fftw_plan_dft_3d(N0, N1, N2, reinterpret_cast<fftw_complex *>(data.data()),
                                            reinterpret_cast<fftw_complex *>(data.data()), FFTW_FORWARD, FFTW_ESTIMATE);

  // Forward FFT
  std::cout << "Executing forward FFT...\n";
  fftw_execute(plan_forward);

  // Save transformed distribution
  {
    std::ofstream file("transformed_gaussian.txt");
    file << std::scientific << std::setprecision(15);
    file << "# 3D FFT of Gaussian distribution\n";
    file << "# Format: i0 i1 i2 real imag amplitude\n";
    file << "# Array size: " << N0 << " x " << N1 << " x " << N2 << "\n";

    for (int i0 = 0; i0 < N0; ++i0) {
      for (int i1 = 0; i1 < N1; ++i1) {
        for (int i2 = 0; i2 < N2; ++i2) {
          int idx = (i0 * N1 + i1) * N2 + i2;
          double amp = std::abs(data[idx]);
          file << i0 << " " << i1 << " " << i2 << " " << data[idx].real() << " " << data[idx].imag() << " " << amp
               << "\n";
        }
      }
    }
    file.close();
    std::cout << "Saved transformed distribution to: transformed_gaussian.txt\n";
  }

  // Create backward plan
  fftw_plan plan_backward =
      fftw_plan_dft_3d(N0, N1, N2, reinterpret_cast<fftw_complex *>(data.data()),
                       reinterpret_cast<fftw_complex *>(data.data()), FFTW_BACKWARD, FFTW_ESTIMATE);

  // Backward FFT
  std::cout << "Executing backward FFT...\n";
  fftw_execute(plan_backward);

  // Normalize (FFTW doesn't normalize)
  for (int i = 0; i < total_size; ++i) {
    data[i] /= total_size;
  }

  // Compute error
  double max_error = 0.0;
  double avg_error = 0.0;
  for (int i = 0; i < total_size; ++i) {
    double err = std::abs(data[i] - original[i]);
    max_error = std::max(max_error, err);
    avg_error += err;
  }
  avg_error /= total_size;

  std::cout << "\nRoundtrip verification:\n";
  std::cout << "  Maximum error: " << std::scientific << max_error << "\n";
  std::cout << "  Average error: " << std::scientific << avg_error << "\n";

  if (max_error < 1e-10) {
    std::cout << "  ✓ PASSED - Roundtrip accurate\n";
  } else {
    std::cout << "  ✗ FAILED - Roundtrip has errors\n";
  }

  // Save roundtrip distribution
  {
    std::ofstream file("roundtrip_gaussian.txt");
    file << std::scientific << std::setprecision(15);
    file << "# 3D Gaussian after forward+backward FFT\n";
    file << "# Format: i0 i1 i2 real imag original_real error\n";
    file << "# Array size: " << N0 << " x " << N1 << " x " << N2 << "\n";
    file << "# Maximum error: " << max_error << "\n";
    file << "# Average error: " << avg_error << "\n";

    for (int i0 = 0; i0 < N0; ++i0) {
      for (int i1 = 0; i1 < N1; ++i1) {
        for (int i2 = 0; i2 < N2; ++i2) {
          int idx = (i0 * N1 + i1) * N2 + i2;
          double err = std::abs(data[idx] - original[idx]);
          file << i0 << " " << i1 << " " << i2 << " " << data[idx].real() << " " << data[idx].imag() << " "
               << original[idx].real() << " " << err << "\n";
        }
      }
    }
    file.close();
    std::cout << "Saved roundtrip distribution to: roundtrip_gaussian.txt\n";
  }

  // Cleanup
  fftw_destroy_plan(plan_forward);
  fftw_destroy_plan(plan_backward);

  std::cout << "\nAll files saved successfully!\n";

  return 0;
}
