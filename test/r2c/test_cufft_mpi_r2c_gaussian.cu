// cuFFT version of test_fftw_mpi_r2c_gaussian.cpp
#include "../../parafaft_r2c.hpp"
#include "../../backend/cufft/fft_backend_cufft.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <fftw3.h>

// Test: Compare R2C transform of a 3D Gaussian against raw FFTW library
// This test verifies that parafaft produces identical results even when using MPI, in comparison
// to direct FFTW library calls for a 3D Gaussian distribution.

std::vector<double> generate_gaussian(int N0, int N1, int N2, double center0, double center1, double center2,
                                      double sigma)
{
  // allocate more space for later transform
  const int total_size = N0 * N1 * (N2 / 2 + 1) * 2;
  std::vector<double> data(total_size);

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
        data[idx] = value;
      }
    }
  }

  return data;
}

// generate serial fftw r2c reference result
void generate_fftw_r2c_reference(double *host_data, const int N0, const int N1, const int N2)
{
  // copy data into an out-of-place array
  std::vector<double> temp_data(N0 * N1 * N2);
  std::copy(host_data, host_data + N0 * N1 * N2, temp_data.data());

  // wipe original data to store result
  std::fill(host_data, host_data + N0 * N1 * (N2 + 2), 0.0);

  // create fftw plan for 3d forward transform
  fftw_plan plan_forward =
      fftw_plan_dft_r2c_3d(N0, N1, N2, temp_data.data(), reinterpret_cast<fftw_complex *>(host_data), FFTW_ESTIMATE);

  // execute forward fft
  fftw_execute(plan_forward);
  fftw_destroy_plan(plan_forward);
}

// Generate serial cuFFT r2c reference result
void generate_cuFFT_r2c_reference(double *host_data, const int N0, const int N1, const int N2)
{
  const int real_size = N0 * N1 * N2;
  const int complex_size = N0 * N1 * (N2 / 2 + 1);

  // Create device arrays for input and output
  cufftDoubleReal *d_real = nullptr;
  cufftDoubleComplex *d_complex = nullptr;
  cudaMalloc((void **)&d_real, real_size * sizeof(cufftDoubleReal));
  cudaMalloc((void **)&d_complex, complex_size * sizeof(cufftDoubleComplex));
  cudaMemcpy(d_real, host_data, real_size * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

  // Create cuFFT plan for 3D forward transform
  cufftHandle plan;
  cufftPlan3d(&plan, N0, N1, N2, CUFFT_D2Z);
  // execute forward fft
  cufftExecD2Z(plan, d_real, d_complex);
  cufftDestroy(plan);

  // Copy back to host
  cudaMemcpy(host_data, d_complex, complex_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
  cudaFree(d_real);
  cudaFree(d_complex);
}

int compare_libraries(int N, int rank)
{
  if (rank != 0) return 0;

  std::cout << "\n==========================================" << std::endl;
  std::cout << "Sanity check: Comparing (sequential) FFTW and cuFFT for 3D R2C transform of size " << N << "x" << N
            << "x" << N << std::endl;

  // Simply check that both libraries produce the same result for the same input
  const int N0 = N, N1 = N, N2 = N;
  const double center0 = N0 / 2.0;
  const double center1 = N1 / 2.0;
  const double center2 = N2 / 2.0;
  const double sigma = 4.0;

  auto fftw_reference = generate_gaussian(N0, N1, N2, center0, center1, center2, sigma);
  auto cufft_reference = generate_gaussian(N0, N1, N2, center0, center1, center2, sigma);

  // Generate reference results
  generate_fftw_r2c_reference(fftw_reference.data(), N0, N1, N2);
  generate_cuFFT_r2c_reference(cufft_reference.data(), N0, N1, N2);

  // Compare results (complex output size)
  const int complex_size = N0 * N1 * (N2 / 2 + 1);
  double max_error = 0.0;
  std::complex<double> *fftw_complex = reinterpret_cast<std::complex<double> *>(fftw_reference.data());
  std::complex<double> *cufft_complex = reinterpret_cast<std::complex<double> *>(cufft_reference.data());
  for (int i = 0; i < complex_size; ++i) {
    double error = std::abs(fftw_complex[i] - cufft_complex[i]);
    if (error > max_error) {
      max_error = error;
    }
  }
  const double tolerance = 1e-10;

  const int success = (max_error < tolerance);

  if (success == 1) {
    std::cout << "Test passed: FFTW and cuFFT produce identical results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 0;
  } else {
    std::cout << "Test failed: FFTW and cuFFT produce different results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
    return 1;
  }

  return success ? 0 : 1;
}

int compare_cuFFTBackend(const int N, int rank)
{
  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Testing parafaft CuFFTBackend for 3D R2C transform of size " << N << "x" << N << "x" << N
              << std::endl;
  }

  // Create global base data
  const auto global_original_data = generate_gaussian(N, N, N, N / 2.0, N / 2.0, N / 2.0, 4.0);

  // Create a global reference using the FFTW library
  auto global_fftw_reference = global_original_data;
  generate_fftw_r2c_reference(global_fftw_reference.data(), N, N, N);

  // Create ParaFaFT object with cuFFT backend
  const int global_shape[3] = {N, N, N};
  parafaft::ParaFaFT_R2C<3, parafaft::CuFFTBackend> fft(global_shape);

  // Get local size
  int local_real_shape[3], real_start[3];
  int local_complex_shape[3], complex_start[3];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);
  fft.get_local_complex_shape(local_complex_shape);
  fft.get_complex_global_start(complex_start);

  int local_real_size = fft.get_required_output_size();
  int local_complex_size = fft.get_local_complex_size();

  std::cout << "Local real shape on rank " << rank << ": " << local_real_shape[0] << " x " << local_real_shape[1]
            << " x " << local_real_shape[2] << std::endl;

  if (!(local_complex_size == (local_complex_shape[0] * local_complex_shape[1] * local_complex_shape[2]))) {
    std::cout << "Rank " << rank << ": Test failed: Local complex size mismatch." << std::endl;
    return 1;
  }

  std::vector<double> local_data(local_real_size);
  // Initialize local data from global original data
  for (int i0 = 0; i0 < local_real_shape[0]; ++i0) {
    for (int i1 = 0; i1 < local_real_shape[1]; ++i1) {
      for (int i2 = 0; i2 < local_real_shape[2]; ++i2) {
        int global_idx = ((real_start[0] + i0) * N * N) + ((real_start[1] + i1) * N) + (real_start[2] + i2);
        int local_idx = ((i0 * local_real_shape[1]) + i1) * N + i2;
        local_data[local_idx] = global_original_data[global_idx];
      }
    }
  }

  // Copy local data to device
  double *d_data = nullptr;
  cudaMalloc((void **)&d_data, local_real_size * sizeof(double));
  cudaMemcpy(d_data, local_data.data(), local_real_size * sizeof(double), cudaMemcpyHostToDevice);

  // Allocate device memory for complex output
  parafaft::CuFFTBackend::Complex *d_result = nullptr;
  cudaMalloc((void **)&d_result, local_complex_size * sizeof(parafaft::CuFFTBackend::Complex));

  // Perform forward FFT
  fft.forward(d_data, d_result);

  // Copy back to host
  std::vector<std::complex<double>> local_result(local_complex_size);
  cudaMemcpy(local_result.data(), d_result, local_complex_size * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  cudaFree(d_result);

  // Access complex result via reinterpret_cast
  std::complex<double> *const global_result = reinterpret_cast<std::complex<double> *>(global_fftw_reference.data());

  // Compare local results to global FFTW reference using FINAL shape/start
  double max_error = 0.0;
  for (int i0 = 0; i0 < local_complex_shape[0]; ++i0) {
    for (int i1 = 0; i1 < local_complex_shape[1]; ++i1) {
      for (int i2 = 0; i2 < local_complex_shape[2]; ++i2) {
        const int global_idx = ((complex_start[0] + i0) * N * (N / 2 + 1)) + ((complex_start[1] + i1) * (N / 2 + 1)) +
                               (complex_start[2] + i2);
        const int local_idx = ((i0 * local_complex_shape[1]) + i1) * local_complex_shape[2] + i2;
        const double error = std::abs(local_result[local_idx] - global_result[global_idx]);
        if (error > max_error) {
          max_error = error;
          std::cout << "Rank " << rank << ": New max error " << std::scientific << max_error << " at local index ("
                    << i0 << "," << i1 << "," << i2 << ")"
                    << " global index (" << (complex_start[0] + i0) << "," << (complex_start[1] + i1) << ","
                    << (complex_start[2] + i2) << ")"
                    << " parallel=" << local_result[local_idx] << " reference=" << global_result[global_idx]
                    << std::endl;
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
    std::cout << "Test passed: parafaft CuFFTBackend produces correct results." << std::endl;
    std::cout << "Maximum error: " << max_error << std::endl;
  } else if (!global_success) {
    std::cout << "Rank " << rank << ": Test failed: parafaft CuFFTBackend produces incorrect results." << std::endl;
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
    std::cout << "# TEST: r2c/cufft_mpi_r2c_gaussian" << std::endl;
    std::cout << "########################################" << std::endl;
  }

  int N = 32; // Default
  if (argc > 1) {
    N = std::atoi(argv[1]);
  }

  int total_failures = 0;

  total_failures += compare_libraries(N, rank);
  total_failures += compare_cuFFTBackend(N, rank);

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "SUMMARY: r2c/cufft_mpi_r2c_gaussian" << std::endl;
    std::cout << "Total failures: " << total_failures << std::endl;
    std::cout << "\n==========================================" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return total_failures;
}
