// hipFFT version of test_mpi_gaussian.cpp (mirrors test_cufft_mpi_c2c_gaussian.cu)
#include "../../parafaft_generic.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <fftw3.h>

// Test: Compare C2C transform of a 3D Gaussian against raw FFTW library
// This test verifies that parafaft produces identical results even when using MPI, in comparison
// to direct FFTW library calls for a 3D Gaussian distribution.

std::vector<std::complex<double>> generate_gaussian(int N0, int N1, int N2, double center0, double center1,
                                                    double center2, double sigma)
{
  const int total_size = N0 * N1 * N2;
  std::vector<std::complex<double>> data(total_size);

  // initialize with 3d gaussian
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

  return data;
}

// generate serial fftw c2c reference result
void generate_fftw_c2c_reference(parafaft::FFTWBackend::Complex *host_data, const int N0, const int N1, const int N2)
{
  // create fftw plan for 3d forward transform
  fftw_plan plan_forward = fftw_plan_dft_3d(N0, N1, N2, reinterpret_cast<fftw_complex *>(host_data),
                                            reinterpret_cast<fftw_complex *>(host_data), FFTW_FORWARD, FFTW_ESTIMATE);

  // execute forward fft
  fftw_execute(plan_forward);
  fftw_destroy_plan(plan_forward);
}

// Generate serial hipFFT C2C reference result
void generate_hipFFT_c2c_reference(parafaft::HipFFTBackend::Complex *host_data, const int N0, const int N1, const int N2)
{
  const int total_size = N0 * N1 * N2;

  // Create a device copy of the data
  parafaft::HipFFTBackend::Complex *d_data = nullptr;
  hipMalloc((void **)&d_data, total_size * sizeof(parafaft::HipFFTBackend::Complex));
  hipMemcpy(d_data, host_data, total_size * sizeof(parafaft::HipFFTBackend::Complex), hipMemcpyHostToDevice);

  // Create hipFFT plan for 3D forward transform
  hipfftHandle plan;
  hipfftPlan3d(&plan, N0, N1, N2, HIPFFT_Z2Z);
  hipfftExecZ2Z(plan, reinterpret_cast<hipfftDoubleComplex *>(d_data), reinterpret_cast<hipfftDoubleComplex *>(d_data),
                HIPFFT_FORWARD);
  hipfftDestroy(plan);

  // Copy back to host
  hipMemcpy(host_data, d_data, total_size * sizeof(parafaft::HipFFTBackend::Complex), hipMemcpyDeviceToHost);
  hipFree(d_data);
}

int compare_libraries(int N, int rank)
{
  if (rank != 0) return 0;

  std::cout << "\n==========================================" << std::endl;
  std::cout << "Sanity check: Comparing (sequential) FFTW and hipFFT for 3D C2C transform of size " << N << "x" << N
            << "x" << N << std::endl;

  // Simply check that both libraries produce the same result for the same input
  const int N0 = N, N1 = N, N2 = N;
  const double center0 = N0 / 2.0;
  const double center1 = N1 / 2.0;
  const double center2 = N2 / 2.0;
  const double sigma = 4.0;

  auto fftw_reference = generate_gaussian(N0, N1, N2, center0, center1, center2, sigma);
  auto hipfft_reference = generate_gaussian(N0, N1, N2, center0, center1, center2, sigma);

  // Generate reference results
  generate_fftw_c2c_reference(fftw_reference.data(), N0, N1, N2);
  generate_hipFFT_c2c_reference(reinterpret_cast<parafaft::HipFFTBackend::Complex *>(hipfft_reference.data()), N0, N1, N2);

  // Compare results
  double max_error = 0.0;
  for (size_t i = 0; i < fftw_reference.size(); ++i) {
    double error = std::abs(fftw_reference[i] - hipfft_reference[i]);
    if (error > max_error) {
      max_error = error;
    }
  }
  const double tolerance = 1e-10;

  const int success = (max_error < tolerance);

  if (success == 1) {
    std::cout << "Test passed: FFTW and hipFFT produce identical results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 0;
  } else {
    std::cout << "Test failed: FFTW and hipFFT produce different results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 1;
  }

  return success ? 0 : 1;
}

int compare_hipFFTBackend(const int N, int rank)
{
  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft HipFFTBackend for 3D C2C transform of size " << N << "x" << N << "x" << N
              << std::endl;
  }

  // Create global base data
  const auto global_original_data = generate_gaussian(N, N, N, N / 2.0, N / 2.0, N / 2.0, 4.0);

  // Create a global reference using the FFTW library
  auto global_fftw_reference = global_original_data;
  generate_fftw_c2c_reference(global_fftw_reference.data(), N, N, N);

  // Create ParaFaFT object with hipFFT backend
  const int global_shape[3] = {N, N, N};
  parafaft::ParaFaFT<3, parafaft::HipFFTBackend> fft(global_shape);

  // Get local size
  int local_size = fft.get_local_size();
  int local_shape[3];
  int global_start[3];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);
  std::vector<std::complex<double>> local_data(local_size);
  // Initialize local data from global original data
  for (int i0 = 0; i0 < local_shape[0]; ++i0) {
    for (int i1 = 0; i1 < local_shape[1]; ++i1) {
      for (int i2 = 0; i2 < local_shape[2]; ++i2) {
        int global_idx = ((global_start[0] + i0) * N * N) + ((global_start[1] + i1) * N) + (global_start[2] + i2);
        int local_idx = ((i0 * local_shape[1]) + i1) * local_shape[2] + i2;
        local_data[local_idx] = global_original_data[global_idx];
      }
    }
  }
  std::cout << "Local shape on rank " << rank << ": " << local_shape[0] << " x " << local_shape[1] << " x "
            << local_shape[2] << std::endl;

  // Copy local data to device
  parafaft::HipFFTBackend::Complex *d_data = nullptr;
  hipMalloc((void **)&d_data, local_size * sizeof(parafaft::HipFFTBackend::Complex));
  hipMemcpy(d_data, local_data.data(), local_size * sizeof(parafaft::HipFFTBackend::Complex), hipMemcpyHostToDevice);

  // Perform forward FFT
  fft.forward(d_data);

  // Get final shape and start (data is redistributed after forward FFT)
  int final_shape[3];
  int final_start[3];
  fft.get_final_shape(final_shape);
  fft.get_final_start(final_start);

  // Copy back to host
  hipMemcpy(local_data.data(), d_data, local_size * sizeof(parafaft::HipFFTBackend::Complex), hipMemcpyDeviceToHost);
  hipFree(d_data);

  // Compare local results to global FFTW reference using FINAL shape/start
  double max_error = 0.0;
  for (int i0 = 0; i0 < final_shape[0]; ++i0) {
    for (int i1 = 0; i1 < final_shape[1]; ++i1) {
      for (int i2 = 0; i2 < final_shape[2]; ++i2) {
        int global_idx = ((final_start[0] + i0) * N * N) + ((final_start[1] + i1) * N) + (final_start[2] + i2);
        int local_idx = ((i0 * final_shape[1]) + i1) * final_shape[2] + i2;
        double error = std::abs(local_data[local_idx] - global_fftw_reference[global_idx]);
        if (error > max_error) {
          max_error = error;
        }
      }
    }
  }
  const double tolerance = 1e-10;
  const bool local_success = (max_error < tolerance);
  // Gather success flags from all processes
  int global_success = local_success ? 1 : 0;
  MPI_Allreduce(&global_success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (global_success == 1 && rank == 0) {
    std::cout << "Test passed: parafaft HipFFTBackend produces correct results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
  } else if (!global_success) {
    std::cout << "Rank " << rank << ": Test failed: parafaft HipFFTBackend produces incorrect results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 1;
  }

  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "########################################" << std::endl;
    std::cout << "# TEST: c2c/hipfft_mpi_c2c_gaussian" << std::endl;
    std::cout << "########################################" << std::endl;
  }

  int N = 32; // Default
  if (argc > 1) {
    N = std::atoi(argv[1]);
  }

  int total_failures = 0;

  total_failures += compare_libraries(N, rank);
  total_failures += compare_hipFFTBackend(N, rank);

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    if (total_failures == 0) {
      std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
      std::cout << "SOME TESTS FAILED" << std::endl;
    }
  }

  MPI_Finalize();
  return total_failures;
}
