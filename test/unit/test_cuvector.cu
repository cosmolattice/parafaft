// Unit tests for cuvector CUDA device memory wrapper
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <memory>
#include <cuda_runtime.h>
#include "../../backend/cufft/fft_backend_cufft.hpp"

// Test 1: Basic allocation and size
void test_basic_allocation()
{
  const size_t size = 1024;

  try {
    parafaft::cuvector<double> vec(size);

    bool passed = true;

    // Check size
    if (vec.size() != size) {
      std::cout << "Size mismatch: expected " << size << ", got " << vec.size() << std::endl;
      passed = false;
    }

    // Check data pointer is not null
    if (vec.data() == nullptr) {
      std::cout << "Data pointer is null" << std::endl;
      passed = false;
    }

    std::cout << "Test basic allocation: " << (passed ? "[PASS]" : "[FAIL]") << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test basic allocation: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 2: Data transfer (host to device and back)
void test_data_transfer_double()
{
  const size_t size = 256;

  try {
    parafaft::cuvector<double> vec(size);

    // Create host data
    std::vector<double> host_data(size);
    for (size_t i = 0; i < size; ++i) {
      host_data[i] = static_cast<double>(i) * 1.5;
    }

    // Copy to device
    cudaError_t err = cudaMemcpy(vec.data(), host_data.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cout << "Test data transfer (double): [FAIL] cudaMemcpy H2D failed: " << cudaGetErrorString(err)
                << std::endl;
      return;
    }

    // Copy back to host
    std::vector<double> result(size);
    err = cudaMemcpy(result.data(), vec.data(), size * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "Test data transfer (double): [FAIL] cudaMemcpy D2H failed: " << cudaGetErrorString(err)
                << std::endl;
      return;
    }

    // Verify data
    double max_error = 0.0;
    for (size_t i = 0; i < size; ++i) {
      max_error = std::max(max_error, std::abs(result[i] - host_data[i]));
    }

    bool passed = (max_error < 1e-15);
    std::cout << "Test data transfer (double): max_error = " << max_error << (passed ? " [PASS]" : " [FAIL]")
              << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test data transfer (double): [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 3: Data transfer with complex type
void test_data_transfer_complex()
{
  const size_t size = 128;

  using Complex = cuda::std::complex<double>;

  try {
    parafaft::cuvector<Complex> vec(size);

    // Create host data (using std::complex for convenience)
    std::vector<std::complex<double>> host_data(size);
    for (size_t i = 0; i < size; ++i) {
      host_data[i] = std::complex<double>(static_cast<double>(i), static_cast<double>(size - i));
    }

    // Copy to device (std::complex and cuda::std::complex have compatible memory layout)
    cudaError_t err = cudaMemcpy(vec.data(), host_data.data(), size * sizeof(Complex), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cout << "Test data transfer (complex): [FAIL] cudaMemcpy H2D failed: " << cudaGetErrorString(err)
                << std::endl;
      return;
    }

    // Copy back to host
    std::vector<std::complex<double>> result(size);
    err = cudaMemcpy(result.data(), vec.data(), size * sizeof(Complex), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "Test data transfer (complex): [FAIL] cudaMemcpy D2H failed: " << cudaGetErrorString(err)
                << std::endl;
      return;
    }

    // Verify data
    double max_error = 0.0;
    for (size_t i = 0; i < size; ++i) {
      max_error = std::max(max_error, std::abs(result[i] - host_data[i]));
    }

    bool passed = (max_error < 1e-15);
    std::cout << "Test data transfer (complex): max_error = " << max_error << (passed ? " [PASS]" : " [FAIL]")
              << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test data transfer (complex): [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 4: Const correctness
void test_const_correctness()
{
  const size_t size = 64;

  try {
    parafaft::cuvector<double> vec(size);
    const parafaft::cuvector<double> &const_vec = vec;

    bool passed = true;

    // Check that const data() returns const pointer
    const double *const_ptr = const_vec.data();
    if (const_ptr == nullptr) {
      std::cout << "Const data pointer is null" << std::endl;
      passed = false;
    }

    // Check size on const reference
    if (const_vec.size() != size) {
      std::cout << "Const size mismatch" << std::endl;
      passed = false;
    }

    // Verify both pointers point to same memory
    if (vec.data() != const_vec.data()) {
      std::cout << "Pointer mismatch between const and non-const" << std::endl;
      passed = false;
    }

    std::cout << "Test const correctness: " << (passed ? "[PASS]" : "[FAIL]") << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test const correctness: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 5: Large allocation
void test_large_allocation()
{
  // 64 MB of doubles (8 million elements)
  const size_t size = 8 * 1024 * 1024;

  try {
    parafaft::cuvector<double> vec(size);

    bool passed = true;

    if (vec.size() != size) {
      std::cout << "Size mismatch for large allocation" << std::endl;
      passed = false;
    }

    if (vec.data() == nullptr) {
      std::cout << "Data pointer is null for large allocation" << std::endl;
      passed = false;
    }

    // Test writing to first and last elements
    std::vector<double> test_data = {42.0, 99.0};
    cudaError_t err = cudaMemcpy(vec.data(), &test_data[0], sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cout << "Failed to write first element" << std::endl;
      passed = false;
    }

    err = cudaMemcpy(vec.data() + size - 1, &test_data[1], sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cout << "Failed to write last element" << std::endl;
      passed = false;
    }

    // Read back and verify
    double first = 0.0, last = 0.0;
    cudaMemcpy(&first, vec.data(), sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last, vec.data() + size - 1, sizeof(double), cudaMemcpyDeviceToHost);

    if (std::abs(first - 42.0) > 1e-15 || std::abs(last - 99.0) > 1e-15) {
      std::cout << "Data verification failed: first=" << first << ", last=" << last << std::endl;
      passed = false;
    }

    std::cout << "Test large allocation (64MB): " << (passed ? "[PASS]" : "[FAIL]") << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test large allocation: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 6: Multiple allocations (stress test)
void test_multiple_allocations()
{
  const size_t size = 1024;
  const int num_vectors = 100;

  try {
    std::vector<std::unique_ptr<parafaft::cuvector<double>>> vectors;
    vectors.reserve(num_vectors);

    for (int i = 0; i < num_vectors; ++i) {
      vectors.push_back(std::make_unique<parafaft::cuvector<double>>(size));
    }

    bool passed = true;

    // Verify all allocations
    for (int i = 0; i < num_vectors; ++i) {
      if (vectors[i]->size() != size || vectors[i]->data() == nullptr) {
        passed = false;
        break;
      }
    }

    // Clear and verify cleanup happens without errors
    vectors.clear();

    // Synchronize to ensure all frees completed
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cout << "Device sync after cleanup failed" << std::endl;
      passed = false;
    }

    std::cout << "Test multiple allocations (" << num_vectors << " vectors): " << (passed ? "[PASS]" : "[FAIL]")
              << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test multiple allocations: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 7: Integer type
void test_integer_type()
{
  const size_t size = 512;

  try {
    parafaft::cuvector<int> vec(size);

    // Create host data
    std::vector<int> host_data(size);
    for (size_t i = 0; i < size; ++i) {
      host_data[i] = static_cast<int>(i * 7 - 100);
    }

    // Copy to device and back
    cudaMemcpy(vec.data(), host_data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> result(size);
    cudaMemcpy(result.data(), vec.data(), size * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify
    bool passed = true;
    for (size_t i = 0; i < size; ++i) {
      if (result[i] != host_data[i]) {
        passed = false;
        break;
      }
    }

    std::cout << "Test integer type: " << (passed ? "[PASS]" : "[FAIL]") << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test integer type: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 8: Float type
void test_float_type()
{
  const size_t size = 512;

  try {
    parafaft::cuvector<float> vec(size);

    // Create host data
    std::vector<float> host_data(size);
    for (size_t i = 0; i < size; ++i) {
      host_data[i] = static_cast<float>(i) * 0.1f;
    }

    // Copy to device and back
    cudaMemcpy(vec.data(), host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> result(size);
    cudaMemcpy(result.data(), vec.data(), size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    float max_error = 0.0f;
    for (size_t i = 0; i < size; ++i) {
      max_error = std::max(max_error, std::abs(result[i] - host_data[i]));
    }

    bool passed = (max_error < 1e-6f);
    std::cout << "Test float type: max_error = " << max_error << (passed ? " [PASS]" : " [FAIL]") << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test float type: [FAIL] Exception: " << e.what() << std::endl;
  }
}

// Test 9: Resize functionality
void test_resize()
{
  const size_t initial_size = 8;
  const size_t new_size = 512;

  try {
    parafaft::cuvector<double> vec(initial_size);

    bool passed = true;

    // Resize to larger size
    vec.resize(new_size);
    if (vec.size() != new_size) {
      std::cout << "Resize to larger size failed" << std::endl;
      passed = false;
    }

    // Copy some arbitrary data to new size, just to see that it does not segfault
    std::vector<double> host_data(new_size);
    for (size_t i = 0; i < new_size; ++i) {
      host_data[i] = static_cast<double>(i) * 2.0;
    }
    cudaMemcpy(vec.data(), host_data.data(), new_size * sizeof(double), cudaMemcpyHostToDevice);

    // Resize back to smaller size
    vec.resize(initial_size);
    if (vec.size() != initial_size) {
      std::cout << "Resize back to smaller size failed" << std::endl;
      passed = false;
    }

    std::cout << "Test resize functionality: " << (passed ? "[PASS]" : "[FAIL]") << std::endl;
  } catch (const std::exception &e) {
    std::cout << "Test resize functionality: [FAIL] Exception: " << e.what() << std::endl;
  }
}

int main()
{
  std::cout << "========================================" << std::endl;
  std::cout << "cuvector Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  // Check for CUDA device
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::cout << "No CUDA devices available. Tests cannot run." << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using CUDA device: " << prop.name << std::endl;
  std::cout << "========================================" << std::endl;

  test_basic_allocation();
  test_data_transfer_double();
  test_data_transfer_complex();
  test_const_correctness();
  test_large_allocation();
  test_multiple_allocations();
  test_integer_type();
  test_float_type();
  test_resize();

  std::cout << "========================================" << std::endl;
  std::cout << "All tests completed." << std::endl;

  return 0;
}
