// Unit tests for hipFFT backend (mirrors test_cufft_backend.cu and test_fftw_backend.cpp)
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <hip/hip_runtime.h>
#include "../../backend/hipfft/fft_backend_hipfft.hpp"

// Helper to check HIP errors in tests
#define HIP_CHECK(call)                                                                                                \
  do {                                                                                                                 \
    hipError_t err = call;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(err) << std::endl;      \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (0)

// Test 1: Contiguous data layout (stride=1, dist=N)
void test_contiguous_layout()
{
  const int N = 256;
  const int batch = 3;

  // Allocate host data
  std::vector<std::complex<double>> host_data(N * batch);

  // Initialize with Gaussian
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < N; ++i) {
      double x = (i - N / 2) / 16.0;
      host_data[b * N + i] = std::exp(-x * x / 2.0);
    }
  }
  std::vector<std::complex<double>> original = host_data;

  // Allocate device memory
  using Complex = hipDoubleComplex;
  parafaft::hipvector<Complex> device_data(N * batch);

  // Copy to device
  HIP_CHECK(hipMemcpy(device_data.data(), host_data.data(), N * batch * sizeof(Complex), hipMemcpyHostToDevice));

  try {
    // Create backend and plan
    parafaft::HipFFTBackend<> backend(1);
    backend.create_stage_plan(0, N, batch, device_data.data(), 1, N);

    // Roundtrip
    backend.execute_stage(0, parafaft::FFTDirection::Forward, device_data.data());
    backend.execute_stage(0, parafaft::FFTDirection::Backward, device_data.data());

    // Copy back to host
    HIP_CHECK(hipMemcpy(host_data.data(), device_data.data(), N * batch * sizeof(Complex), hipMemcpyDeviceToHost));

    // Normalize
    for (auto &val : host_data) {
      val /= N;
    }

    // Check error
    double max_error = 0.0;
    for (size_t i = 0; i < host_data.size(); ++i) {
      max_error = std::max(max_error, std::abs(host_data[i] - original[i]));
    }

    std::cout << "Test contiguous: max_error = " << max_error;
    if (max_error < 1e-10) {
      std::cout << " [PASS]" << std::endl;
    } else {
      std::cout << " [FAIL]" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "Test contiguous: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 2: Strided data layout (stride=batch, dist=1)
void test_strided_layout()
{
  const int N = 128;
  const int batch = 4;

  // Allocate host data
  std::vector<std::complex<double>> host_data(N * batch);

  // Initialize interleaved
  for (int i = 0; i < N; ++i) {
    for (int b = 0; b < batch; ++b) {
      double phase = 2.0 * M_PI * b / batch;
      host_data[i * batch + b] = std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
  std::vector<std::complex<double>> original = host_data;

  // Allocate device memory
  using Complex = hipDoubleComplex;
  parafaft::hipvector<Complex> device_data(N * batch);

  // Copy to device
  HIP_CHECK(hipMemcpy(device_data.data(), host_data.data(), N * batch * sizeof(Complex), hipMemcpyHostToDevice));

  try {
    // Create backend and plan
    parafaft::HipFFTBackend<> backend(1);
    backend.create_stage_plan(0, N, batch, device_data.data(), batch, 1);

    // Roundtrip
    backend.execute_stage(0, parafaft::FFTDirection::Forward, device_data.data());
    backend.execute_stage(0, parafaft::FFTDirection::Backward, device_data.data());

    // Copy back to host
    HIP_CHECK(hipMemcpy(host_data.data(), device_data.data(), N * batch * sizeof(Complex), hipMemcpyDeviceToHost));

    // Normalize
    for (auto &val : host_data) {
      val /= N;
    }

    // Check error
    double max_error = 0.0;
    for (size_t i = 0; i < host_data.size(); ++i) {
      max_error = std::max(max_error, std::abs(host_data[i] - original[i]));
    }

    std::cout << "Test strided: max_error = " << max_error;
    if (max_error < 1e-10) {
      std::cout << " [PASS]" << std::endl;
    } else {
      std::cout << " [FAIL]" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "Test strided: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 3: Multiple stages with plan reuse
void test_multiple_stages()
{
  const int N = 64;
  const int num_stages = 3;

  using Complex = hipDoubleComplex;

  // Host and device data for each stage
  std::vector<std::vector<std::complex<double>>> host_data(num_stages);
  std::vector<std::vector<std::complex<double>>> original_data(num_stages);
  std::vector<parafaft::hipvector<Complex>> device_data(num_stages);

  for (int stage = 0; stage < num_stages; ++stage) {
    host_data[stage].resize(N);
    for (int i = 0; i < N; ++i) {
      host_data[stage][i] = std::complex<double>(i + stage * 100, 0.0);
    }
    original_data[stage] = host_data[stage];

    // Allocate and copy to device
    device_data[stage].resize(N);
    hipError_t err =
        hipMemcpy(device_data[stage].data(), host_data[stage].data(), N * sizeof(Complex), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      std::cout << "Test multiple stages: [FAIL] hipMemcpy failed" << std::endl;
      return;
    }
  }

  try {
    // Create backend with 3 stages
    parafaft::HipFFTBackend<> backend(num_stages);
    for (int stage = 0; stage < num_stages; ++stage) {
      backend.create_stage_plan(stage, N, 1, device_data[stage].data(), 1, N);
    }

    // Execute multiple times (test plan reuse)
    double max_error = 0.0;
    for (int repeat = 0; repeat < 3; ++repeat) {
      for (int stage = 0; stage < num_stages; ++stage) {
        backend.execute_stage(stage, parafaft::FFTDirection::Forward, device_data[stage].data());
        backend.execute_stage(stage, parafaft::FFTDirection::Backward, device_data[stage].data());

        // Copy back to host for normalization and error check
        HIP_CHECK(
            hipMemcpy(host_data[stage].data(), device_data[stage].data(), N * sizeof(Complex), hipMemcpyDeviceToHost));

        for (auto &val : host_data[stage]) {
          val /= N;
        }

        // Check error after each roundtrip
        for (size_t i = 0; i < host_data[stage].size(); ++i) {
          max_error = std::max(max_error, std::abs(host_data[stage][i] - original_data[stage][i]));
        }

        // Copy normalized data back to device for next iteration
        HIP_CHECK(
            hipMemcpy(device_data[stage].data(), host_data[stage].data(), N * sizeof(Complex), hipMemcpyHostToDevice));
      }
    }

    std::cout << "Test multiple stages with reuse: max_error = " << max_error;
    if (max_error < 1e-10) {
      std::cout << " [PASS]" << std::endl;
    } else {
      std::cout << " [FAIL]" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "Test multiple stages: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 4: Same plan applied to different data arrays
void test_different_pointers()
{
  const int N = 64;
  const int batch = 2;

  using Complex = hipDoubleComplex;

  // Create three different host data arrays
  std::vector<std::complex<double>> host_data1(N * batch);
  std::vector<std::complex<double>> host_data2(N * batch);
  std::vector<std::complex<double>> host_data3(N * batch);

  // Initialize with different patterns
  for (int i = 0; i < N * batch; ++i) {
    host_data1[i] = std::complex<double>(i % 10, 0.0);
    host_data2[i] = std::complex<double>(0.0, i % 5);
    host_data3[i] = std::exp(-static_cast<double>((i - N * batch / 2) * (i - N * batch / 2)) / (2.0 * 100.0));
  }
  auto original1 = host_data1;
  auto original2 = host_data2;
  auto original3 = host_data3;

  // Allocate device memory
  parafaft::hipvector<Complex> device_data1(N * batch);
  parafaft::hipvector<Complex> device_data2(N * batch);
  parafaft::hipvector<Complex> device_data3(N * batch);

  // Copy to device
  HIP_CHECK(hipMemcpy(device_data1.data(), host_data1.data(), N * batch * sizeof(Complex), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_data2.data(), host_data2.data(), N * batch * sizeof(Complex), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_data3.data(), host_data3.data(), N * batch * sizeof(Complex), hipMemcpyHostToDevice));

  try {
    // Create single backend and plan (using data1 for plan creation)
    parafaft::HipFFTBackend<> backend(1);
    backend.create_stage_plan(0, N, batch, device_data1.data(), 1, N);

    // Apply the same plan to all three different arrays
    backend.execute_stage(0, parafaft::FFTDirection::Forward, device_data1.data());
    backend.execute_stage(0, parafaft::FFTDirection::Backward, device_data1.data());
    HIP_CHECK(hipMemcpy(host_data1.data(), device_data1.data(), N * batch * sizeof(Complex), hipMemcpyDeviceToHost));
    for (auto &val : host_data1)
      val /= N;

    backend.execute_stage(0, parafaft::FFTDirection::Forward, device_data2.data());
    backend.execute_stage(0, parafaft::FFTDirection::Backward, device_data2.data());
    HIP_CHECK(hipMemcpy(host_data2.data(), device_data2.data(), N * batch * sizeof(Complex), hipMemcpyDeviceToHost));
    for (auto &val : host_data2)
      val /= N;

    backend.execute_stage(0, parafaft::FFTDirection::Forward, device_data3.data());
    backend.execute_stage(0, parafaft::FFTDirection::Backward, device_data3.data());
    HIP_CHECK(hipMemcpy(host_data3.data(), device_data3.data(), N * batch * sizeof(Complex), hipMemcpyDeviceToHost));
    for (auto &val : host_data3)
      val /= N;

    // Check all arrays recovered correctly
    double max_error = 0.0;
    for (size_t i = 0; i < host_data1.size(); ++i) {
      max_error = std::max(max_error, std::abs(host_data1[i] - original1[i]));
      max_error = std::max(max_error, std::abs(host_data2[i] - original2[i]));
      max_error = std::max(max_error, std::abs(host_data3[i] - original3[i]));
    }

    std::cout << "Test different pointers: max_error = " << max_error;
    if (max_error < 1e-10) {
      std::cout << " [PASS]" << std::endl;
    } else {
      std::cout << " [FAIL]" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "Test different pointers: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 5: R2C/C2R in-place roundtrip
void test_r2c_c2r_inplace()
{
  const int N = 128; // Real-space length
  const int batch = 2;
  const int padded_dist = 2 * (N / 2 + 1); // Padded distance for in-place R2C

  // Allocate host data with padding
  std::vector<double> host_data(batch * padded_dist);

  // Initialize with Gaussian pattern (only the first N elements of each batch)
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < N; ++i) {
      double x = (i - N / 2) / 16.0;
      host_data[b * padded_dist + i] = std::exp(-x * x / 2.0);
    }
    // Zero the padding
    for (int i = N; i < padded_dist; ++i) {
      host_data[b * padded_dist + i] = 0.0;
    }
  }

  // Save original (only real data, not padding)
  std::vector<double> original(batch * N);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < N; ++i) {
      original[b * N + i] = host_data[b * padded_dist + i];
    }
  }

  // Allocate device memory
  parafaft::hipvector<double> device_data(batch * padded_dist);

  // Copy to device
  HIP_CHECK(hipMemcpy(device_data.data(), host_data.data(), batch * padded_dist * sizeof(double), hipMemcpyHostToDevice));

  try {
    // Create backend with R2C and C2R plans
    parafaft::HipFFTBackend<> backend(0); // No C2C stages needed
    backend.create_r2c_inplace_plan(N, batch, device_data.data(), 1, padded_dist);
    backend.create_c2r_inplace_plan(N, batch, device_data.data(), 1, padded_dist);

    // R2C forward
    backend.execute_r2c_inplace(device_data.data());

    // C2R backward
    backend.execute_c2r_inplace(device_data.data());

    // Copy back to host
    HIP_CHECK(hipMemcpy(host_data.data(), device_data.data(), batch * padded_dist * sizeof(double), hipMemcpyDeviceToHost));

    // Normalize (hipFFT doesn't normalize, need to divide by N)
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < N; ++i) {
        host_data[b * padded_dist + i] /= N;
      }
    }

    // Check error
    double max_error = 0.0;
    for (int b = 0; b < batch; ++b) {
      for (int i = 0; i < N; ++i) {
        max_error = std::max(max_error, std::abs(host_data[b * padded_dist + i] - original[b * N + i]));
      }
    }

    std::cout << "Test R2C/C2R in-place: max_error = " << max_error;
    if (max_error < 1e-10) {
      std::cout << " [PASS]" << std::endl;
    } else {
      std::cout << " [FAIL]" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "Test R2C/C2R in-place: [FAIL] Exception: " << e.what() << std::endl;
  }
}

int main(int argc, char **argv) {
  std::cout << "########################################" << std::endl;
  std::cout << "# TEST: unit/hipfft_backend" << std::endl;
  std::cout << "########################################" << std::endl;

  MPI_Init(&argc, &argv);

  // Check if HIP device is available
  int deviceCount = 0;
  hipError_t err = hipGetDeviceCount(&deviceCount);
  if (err != hipSuccess || deviceCount == 0) {
    std::cout << "No HIP devices available. Skipping hipFFT backend tests." << std::endl;
    return 0;
  }

  std::cout << "Running hipFFT backend tests..." << std::endl;
  std::cout << "================================" << std::endl;

  test_contiguous_layout();
  test_strided_layout();
  test_multiple_stages();
  test_different_pointers();
  test_r2c_c2r_inplace();

  std::cout << "================================" << std::endl;
  std::cout << "hipFFT backend tests complete." << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
