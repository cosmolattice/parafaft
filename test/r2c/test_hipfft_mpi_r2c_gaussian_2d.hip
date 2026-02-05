// hipFFT version of test_fftw_mpi_r2c_gaussian.cpp (mirrors test_cufft_mpi_r2c_gaussian_2d.cu)
#include "../../parafaft_r2c.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <fftw3.h>

// Test: Compare R2C transform of a 2D Gaussian against raw FFTW library
// This test verifies that parafaft produces identical results even when using MPI, in comparison
// to direct FFTW library calls for a 2D Gaussian distribution.

std::vector<double> generate_gaussian(int N0, int N1, double center0, double center1, double sigma)
{
  // allocate more space for later transform
  const int total_size = N0 * (N1 / 2 + 1) * 2;
  std::vector<double> data(total_size);

  // initialize with 2d gaussian
  for (int i0 = 0; i0 < N0; ++i0) {
    for (int i1 = 0; i1 < N1; ++i1) {
      double x = i0 - center0;
      double y = i1 - center1;
      double r2 = x * x + y * y;
      double value = std::exp(-r2 / (2.0 * sigma * sigma));

      int idx = i0 * N1 + i1;
      data[idx] = value;
    }
  }

  return data;
}

// generate serial fftw r2c reference result
void generate_fftw_r2c_reference(double *host_data, const int N0, const int N1)
{
  // copy data into an out-of-place array
  std::vector<double> temp_data(N0 * N1);
  std::copy(host_data, host_data + N0 * N1, temp_data.data());

  // wipe original data to store result
  std::fill(host_data, host_data + N0 * (N1 + 2), 0.0);

  // create fftw plan for 2d forward transform
  fftw_plan plan_forward =
      fftw_plan_dft_r2c_2d(N0, N1, temp_data.data(), reinterpret_cast<fftw_complex *>(host_data), FFTW_ESTIMATE);

  // execute forward fft
  fftw_execute(plan_forward);
  fftw_destroy_plan(plan_forward);
}

// Generate serial hipFFT r2c reference result
void generate_hipFFT_r2c_reference(double *host_data, const int N0, const int N1)
{
  const int real_size = N0 * N1;
  const int complex_size = N0 * (N1 / 2 + 1);

  // Create device arrays for input and output
  hipfftDoubleReal *d_real = nullptr;
  hipfftDoubleComplex *d_complex = nullptr;
  hipMalloc((void **)&d_real, real_size * sizeof(hipfftDoubleReal));
  hipMalloc((void **)&d_complex, complex_size * sizeof(hipfftDoubleComplex));
  hipMemcpy(d_real, host_data, real_size * sizeof(hipfftDoubleReal), hipMemcpyHostToDevice);

  // Create hipFFT plan for 2D forward transform
  hipfftHandle plan;
  hipfftPlan2d(&plan, N0, N1, HIPFFT_D2Z);
  // execute forward fft
  hipfftExecD2Z(plan, d_real, d_complex);
  hipfftDestroy(plan);

  // Copy back to host
  hipMemcpy(host_data, d_complex, complex_size * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
  hipFree(d_real);
  hipFree(d_complex);
}

int compare_libraries(int N, int rank)
{
  if (rank != 0) return 0;

  std::cout << "\n==========================================" << std::endl;
  std::cout << "Sanity check: Comparing (sequential) FFTW and hipFFT for 2D R2C transform of size " << N << "x" << N
            << std::endl;

  // Simply check that both libraries produce the same result for the same input
  const int N0 = N, N1 = N;
  const double center0 = N0 / 2.0;
  const double center1 = N1 / 2.0;
  const double sigma = 4.0;

  auto fftw_reference = generate_gaussian(N0, N1, center0, center1, sigma);
  auto hipfft_reference = generate_gaussian(N0, N1, center0, center1, sigma);

  // Generate reference results
  generate_fftw_r2c_reference(fftw_reference.data(), N0, N1);
  generate_hipFFT_r2c_reference(hipfft_reference.data(), N0, N1);

  // Compare results (complex output size)
  const int complex_size = N0 * (N1 / 2 + 1);
  double max_error = 0.0;
  std::complex<double> *fftw_complex = reinterpret_cast<std::complex<double> *>(fftw_reference.data());
  std::complex<double> *hipfft_complex = reinterpret_cast<std::complex<double> *>(hipfft_reference.data());
  for (int i = 0; i < complex_size; ++i) {
    double error = std::abs(fftw_complex[i] - hipfft_complex[i]);
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
    std::cout << "Testing parafaft HipFFTBackend for 2D R2C transform of size " << N << "x" << N << std::endl;
  }

  // Create global base data
  const auto global_original_data = generate_gaussian(N, N, N / 2.0, N / 2.0, 4.0);

  // Create a global reference using the FFTW library
  auto global_fftw_reference = global_original_data;
  generate_fftw_r2c_reference(global_fftw_reference.data(), N, N);

  // Create ParaFaFT object with hipFFT backend
  const int global_shape[2] = {N, N};
  parafaft::ParaFaFT_R2C<2, parafaft::HipFFTBackend> fft(global_shape);

  // Get local size
  int local_real_shape[2], real_start[2];
  int local_complex_shape[2], complex_start[2];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);
  fft.get_local_complex_shape(local_complex_shape);
  fft.get_complex_global_start(complex_start);

  int local_real_size = fft.get_required_output_size();
  int local_complex_size = fft.get_local_complex_size();

  std::cout << "Local real shape on rank " << rank << ": " << local_real_shape[0] << " x " << local_real_shape[1]
            << std::endl;

  if (!(local_complex_size == (local_complex_shape[0] * local_complex_shape[1]))) {
    std::cout << "Rank " << rank << ": Test failed: Local complex size mismatch." << std::endl;
    return 1;
  }

  std::vector<double> local_data(local_real_size);
  // Initialize local data from global original data
  for (int i0 = 0; i0 < local_real_shape[0]; ++i0) {
    for (int i1 = 0; i1 < local_real_shape[1]; ++i1) {
      int global_idx = (real_start[0] + i0) * N + (real_start[1] + i1);
      int local_idx = i0 * N + i1;
      local_data[local_idx] = global_original_data[global_idx];
    }
  }

  // Copy local data to device
  double *d_data = nullptr;
  hipMalloc((void **)&d_data, local_real_size * sizeof(double));
  hipMemcpy(d_data, local_data.data(), local_real_size * sizeof(double), hipMemcpyHostToDevice);

  // Allocate device memory for complex output
  parafaft::HipFFTBackend::Complex *d_result = nullptr;
  hipMalloc((void **)&d_result, local_real_size / 2 * sizeof(parafaft::HipFFTBackend::Complex));

  // Perform forward FFT
  fft.forward(d_data, d_result);

  // Copy back to host
  std::vector<std::complex<double>> local_result(local_complex_size);
  hipMemcpy(local_result.data(), d_result, local_complex_size * sizeof(std::complex<double>), hipMemcpyDeviceToHost);
  hipFree(d_data);
  hipFree(d_result);

  // Access complex result via reinterpret_cast
  std::complex<double> *const global_result = reinterpret_cast<std::complex<double> *>(global_fftw_reference.data());

  // Compare local results to global FFTW reference using FINAL shape/start
  double max_error = 0.0;
  for (int i0 = 0; i0 < local_complex_shape[0]; ++i0) {
    for (int i1 = 0; i1 < local_complex_shape[1]; ++i1) {
      const int global_idx = (complex_start[0] + i0) * (N / 2 + 1) + (complex_start[1] + i1);
      const int local_idx = i0 * local_complex_shape[1] + i1;
      const double error = std::abs(local_result[local_idx] - global_result[global_idx]);
      if (error > max_error) {
        max_error = error;
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
    std::cout << "# TEST: r2c/hipfft_mpi_r2c_gaussian_2d" << std::endl;
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
