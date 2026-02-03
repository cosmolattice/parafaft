// cuFFT Backend C2C Unit Tests
// Equivalent to test_fftw_backend.cpp but using CuFFTBackend
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include "../../backend/cufft/fft_backend_cufft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test 1: Contiguous data layout (stride=1, dist=N)
void test_contiguous_layout()
{
  const int N = 256;
  const int batch = 3;

  // Create data on host, for convenience
  std::vector<std::complex<double>> host_data(N * batch);

  // Initialize with Gaussian
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < N; ++i) {
      double x = (i - N / 2) / 16.0;
      host_data[b * N + i] = std::exp(-x * x / 2.0);
    }
  }
  std::vector<std::complex<double>> host_original = host_data;

  // Copy to device
  parafaft::CuFFTBackend::ComplexBuffer data(N * batch);
  parafaft::CuFFTBackend::memcpy(data.data(), host_data.data(), N * batch * sizeof(std::complex<double>));

  // Create backend and plan
  parafaft::CuFFTBackend backend(1);
  backend.create_stage_plan(0, N, batch, data.data(), 1, N);

  // Roundtrip
  backend.execute_stage(0, parafaft::FFTDirection::Forward, data.data());
  backend.execute_stage(0, parafaft::FFTDirection::Backward, data.data());

  // Copy back to host
  parafaft::CuFFTBackend::memcpy(host_data.data(), data.data(), N * batch * sizeof(std::complex<double>));

  // Normalize
  for (auto &val : host_data) {
    val /= N;
  }

  // Check error
  double max_error = 0.0;
  for (size_t i = 0; i < host_data.size(); ++i) {
    max_error = std::max(max_error, std::abs(host_data[i] - host_original[i]));
  }

  std::cout << "Test contiguous: max_error = " << max_error;
  if (max_error < 1e-10) {
    std::cout << " [PASS]" << std::endl;
  } else {
    std::cout << " [FAIL]" << std::endl;
  }
}

// Test 2: Strided data layout (stride=batch, dist=1)
void test_strided_layout()
{
  const int N = 128;
  const int batch = 4;

  std::vector<std::complex<double>> host_data(N * batch);

  // Initialize interleaved
  for (int i = 0; i < N; ++i) {
    for (int b = 0; b < batch; ++b) {
      double phase = 2.0 * M_PI * b / batch;
      host_data[i * batch + b] = std::complex<double>(std::cos(phase), std::sin(phase));
    }
  }
  std::vector<std::complex<double>> host_original = host_data;

  // Copy to device
  parafaft::CuFFTBackend::ComplexBuffer data(N * batch);
  parafaft::CuFFTBackend::memcpy(data.data(), host_data.data(), N * batch * sizeof(std::complex<double>));

  // Create backend and plan
  parafaft::CuFFTBackend backend(1);
  backend.create_stage_plan(0, N, batch, data.data(), batch, 1);

  // Roundtrip
  backend.execute_stage(0, parafaft::FFTDirection::Forward, data.data());
  backend.execute_stage(0, parafaft::FFTDirection::Backward, data.data());

  // Copy back to host
  parafaft::CuFFTBackend::memcpy(host_data.data(), data.data(), N * batch * sizeof(std::complex<double>));

  // Normalize
  for (auto &val : host_data) {
    val /= N;
  }

  // Check error
  double max_error = 0.0;
  for (size_t i = 0; i < host_data.size(); ++i) {
    max_error = std::max(max_error, std::abs(host_data[i] - host_original[i]));
  }

  std::cout << "Test strided: max_error = " << max_error;
  if (max_error < 1e-10) {
    std::cout << " [PASS]" << std::endl;
  } else {
    std::cout << " [FAIL]" << std::endl;
  }
}

// Test 3: Multiple stages with plan reuse
void test_multiple_stages()
{
  const int N = 64;
  const int num_stages = 3;

  std::vector<std::vector<std::complex<double>>> host_stage_data(num_stages);
  std::vector<std::vector<std::complex<double>>> host_original_data(num_stages);

  for (int stage = 0; stage < num_stages; ++stage) {
    host_stage_data[stage].resize(N);
    for (int i = 0; i < N; ++i) {
      host_stage_data[stage][i] = std::complex<double>(i + stage * 100, 0.0);
    }
    host_original_data[stage] = host_stage_data[stage];
  }

  // Copy to device
  std::vector<parafaft::CuFFTBackend::ComplexBuffer> stage_data(num_stages);
  for (int stage = 0; stage < num_stages; ++stage) {
    stage_data[stage].resize(N);
    parafaft::CuFFTBackend::memcpy(stage_data[stage].data(), host_stage_data[stage].data(),
                                   N * sizeof(std::complex<double>));
  }

  // Create backend with 3 stages
  parafaft::CuFFTBackend backend(num_stages);
  for (int stage = 0; stage < num_stages; ++stage) {
    backend.create_stage_plan(stage, N, 1, stage_data[stage].data(), 1, N);
  }

  // Execute multiple times (test plan reuse)
  double max_error = 0.0;
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (int stage = 0; stage < num_stages; ++stage) {
      backend.execute_stage(stage, parafaft::FFTDirection::Forward, stage_data[stage].data());
      backend.execute_stage(stage, parafaft::FFTDirection::Backward, stage_data[stage].data());

      // Copy back to host for error checking
      parafaft::CuFFTBackend::memcpy(host_stage_data[stage].data(), stage_data[stage].data(),
                                     N * sizeof(std::complex<double>));

      // Normalize
      for (auto &val : host_stage_data[stage]) {
        val /= N;
      }

      // Check error after each roundtrip
      for (size_t i = 0; i < stage_data[stage].size(); ++i) {
        max_error = std::max(max_error, std::abs(host_stage_data[stage][i] - host_original_data[stage][i]));
      }
    }
  }

  std::cout << "Test multiple stages with reuse: max_error = " << max_error;
  if (max_error < 1e-10) {
    std::cout << " [PASS]" << std::endl;
  } else {
    std::cout << " [FAIL]" << std::endl;
  }
}

// Test 4: Same plan applied to different data arrays
void test_different_pointers()
{
  const int N = 64;
  const int batch = 2;

  // Create three different data arrays
  std::vector<std::complex<double>> host_data1(N * batch);
  std::vector<std::complex<double>> host_data2(N * batch);
  std::vector<std::complex<double>> host_data3(N * batch);

  // Initialize with different patterns
  for (int i = 0; i < N * batch; ++i) {
    host_data1[i] = std::complex<double>(i % 10, 0.0);
    host_data2[i] = std::complex<double>(0.0, i % 5);
    host_data3[i] = std::exp(-static_cast<double>((i - N * batch / 2) * (i - N * batch / 2)) / (2.0 * 100.0));
  }
  auto host_original1 = host_data1;
  auto host_original2 = host_data2;
  auto host_original3 = host_data3;

  // Copy to device
  parafaft::CuFFTBackend::ComplexBuffer data1(N * batch);
  parafaft::CuFFTBackend::ComplexBuffer data2(N * batch);
  parafaft::CuFFTBackend::ComplexBuffer data3(N * batch);
  parafaft::CuFFTBackend::memcpy(data1.data(), host_data1.data(), N * batch * sizeof(std::complex<double>));
  parafaft::CuFFTBackend::memcpy(data2.data(), host_data2.data(), N * batch * sizeof(std::complex<double>));
  parafaft::CuFFTBackend::memcpy(data3.data(), host_data3.data(), N * batch * sizeof(std::complex<double>));

  // Create single backend and plan (using data1 for plan creation)
  parafaft::CuFFTBackend backend(1);
  backend.create_stage_plan(0, N, batch, data1.data(), 1, N);

  // Apply the same plan to all three different arrays
  backend.execute_stage(0, parafaft::FFTDirection::Forward, data1.data());
  backend.execute_stage(0, parafaft::FFTDirection::Backward, data1.data());
  // Copy back to host for normalization
  parafaft::CuFFTBackend::memcpy(host_data1.data(), data1.data(), N * batch * sizeof(std::complex<double>));
  for (auto &val : host_data1)
    val /= N;

  backend.execute_stage(0, parafaft::FFTDirection::Forward, data2.data());
  backend.execute_stage(0, parafaft::FFTDirection::Backward, data2.data());
  // Copy back to host for normalization
  parafaft::CuFFTBackend::memcpy(host_data2.data(), data2.data(), N * batch * sizeof(std::complex<double>));
  for (auto &val : host_data2)
    val /= N;

  backend.execute_stage(0, parafaft::FFTDirection::Forward, data3.data());
  backend.execute_stage(0, parafaft::FFTDirection::Backward, data3.data());
  // Copy back to host for normalization
  parafaft::CuFFTBackend::memcpy(host_data3.data(), data3.data(), N * batch * sizeof(std::complex<double>));
  for (auto &val : host_data3)
    val /= N;

  // Check all arrays recovered correctly
  double max_error = 0.0;
  for (size_t i = 0; i < data1.size(); ++i) {
    max_error = std::max(max_error, std::abs(host_data1[i] - host_original1[i]));
    max_error = std::max(max_error, std::abs(host_data2[i] - host_original2[i]));
    max_error = std::max(max_error, std::abs(host_data3[i] - host_original3[i]));
  }

  std::cout << "Test different pointers: max_error = " << max_error;
  if (max_error < 1e-10) {
    std::cout << " [PASS]" << std::endl;
  } else {
    std::cout << " [FAIL]" << std::endl;
  }
}

// Test 5: Move constructor
void test_move_constructor()
{
  const int N = 32;
  const int batch = 1;

  std::vector<std::complex<double>> host_data(N);
  for (int i = 0; i < N; ++i) {
    host_data[i] = std::complex<double>(std::cos(2.0 * M_PI * i / N), 0.0);
  }
  std::vector<std::complex<double>> host_original = host_data;

  // Copy to device
  parafaft::CuFFTBackend::ComplexBuffer data(N);
  parafaft::CuFFTBackend::memcpy(data.data(), host_data.data(), N * sizeof(std::complex<double>));

  parafaft::CuFFTBackend backend1(1);
  backend1.create_stage_plan(0, N, batch, data.data(), 1, N);

  // Move to new backend
  parafaft::CuFFTBackend backend2(std::move(backend1));

  // Execute on moved backend
  backend2.execute_stage(0, parafaft::FFTDirection::Forward, data.data());
  backend2.execute_stage(0, parafaft::FFTDirection::Backward, data.data());

  // Copy back to host
  parafaft::CuFFTBackend::memcpy(host_data.data(), data.data(), N * sizeof(std::complex<double>));

  for (auto &val : host_data) {
    val /= N;
  }

  double max_error = 0.0;
  for (size_t i = 0; i < host_data.size(); ++i) {
    max_error = std::max(max_error, std::abs(host_data[i] - host_original[i]));
  }

  std::cout << "Test move constructor: max_error = " << max_error;
  if (max_error < 1e-10) {
    std::cout << " [PASS]" << std::endl;
  } else {
    std::cout << " [FAIL]" << std::endl;
  }
}

void test_reinterpret()
{
  // Test reinterpret_cast of complex to cufftDoubleComplex
  std::complex<double> host_val(1.0, -1.0);
  cufftDoubleComplex cufft_val = *reinterpret_cast<cufftDoubleComplex *>(&host_val);
  // Change values
  cufft_val.x = 2.0;
  cufft_val.y = -2.0;
  // Reinterpret back
  std::complex<double> host_val2 = *reinterpret_cast<std::complex<double> *>(&cufft_val);
  if (host_val2.real() == 2.0 && host_val2.imag() == -2.0) {
    std::cout << "Test reinterpret_cast: [PASS]" << std::endl;
  } else {
    std::cout << "Test reinterpret_cast: [FAIL]" << std::endl;
  }
}

int main()
{
  std::cout << "=== CuFFT Backend C2C Unit Tests ===" << std::endl;

  test_contiguous_layout();
  test_strided_layout();
  test_multiple_stages();
  test_different_pointers();
  test_move_constructor();

  return 0;
}
