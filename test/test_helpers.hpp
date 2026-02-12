#ifndef PARAFAFT_TEST_HELPERS_HPP
#define PARAFAFT_TEST_HELPERS_HPP

// ============================================================================
// Reusable test helpers for arbitrary-dimension parafaft tests.
//
// Provides:
//   - nd_index<D>           : convert D multi-indices + shape to flat index
//   - generate_gaussian_nd<D> : fill a hypercubic grid with a Gaussian
//   - fftw_c2c_reference_nd : serial FFTW forward C2C on D-dimensional data
//   - print_shape<D>        : pretty-print a shape array
//   - iterate_nd<D>         : generic nested-loop replacement (runtime dims)
// ============================================================================

#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <vector>
#include <fftw3.h>

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVCC__)
#include <cufft.h>
#include <cuda_runtime.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIPCC__)
#include <hipfft/hipfft.h>
#include <hip/hip_runtime.h>
#endif

namespace parafaft_test
{

  // --------------------------------------------------------------------------
  // Flat-index computation for a D-dimensional array in row-major order.
  // idx[d] is the coordinate in dimension d, shape[d] the extent.
  // --------------------------------------------------------------------------
  template <int D> inline int nd_index(const int idx[D], const int shape[D])
  {
    int flat = idx[0];
    for (int d = 1; d < D; ++d)
      flat = flat * shape[d] + idx[d];
    return flat;
  }

  // Overload accepting std::array
  template <int D> inline int nd_index(const std::array<int, D> &idx, const std::array<int, D> &shape)
  {
    return nd_index<D>(idx.data(), shape.data());
  }

  // --------------------------------------------------------------------------
  // Iterate over all points in a D-dimensional grid of given shape and call
  // a callback with the multi-index.  Replaces the D nested for-loops.
  // --------------------------------------------------------------------------
  template <int D> void iterate_nd(const int shape[D], const std::function<void(const std::array<int, D> &)> &callback)
  {
    std::array<int, D> idx{};
    // Total number of points
    int total = 1;
    for (int d = 0; d < D; ++d)
      total *= shape[d];

    for (int i = 0; i < total; ++i) {
      callback(idx);
      // Increment multi-index (last index varies fastest, row-major order)
      for (int d = D - 1; d >= 0; --d) {
        if (++idx[d] < shape[d]) break;
        idx[d] = 0;
      }
    }
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional Gaussian on a hypercubic grid of side N.
  // center[d] = N/2, sigma as given.  Returns row-major complex data.
  // --------------------------------------------------------------------------
  template <int D> std::vector<std::complex<double>> generate_gaussian_nd(int N, double sigma)
  {
    int total = 1;
    for (int d = 0; d < D; ++d)
      total *= N;

    std::vector<std::complex<double>> data(total);
    std::array<int, D> shape;
    shape.fill(N);

    const double center = N / 2.0;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      double value = std::exp(-r2 / (2.0 * sigma * sigma));
      int flat = nd_index<D>(idx, shape);
      data[flat] = std::complex<double>(value, 0.0);
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Serial FFTW forward C2C transform on D-dimensional data (in-place).
  // Uses the generic fftw_plan_dft which accepts arbitrary rank.
  // --------------------------------------------------------------------------
  template <int D> void fftw_c2c_reference_nd(std::complex<double> *data, int N)
  {
    std::array<int, D> n;
    n.fill(N);

    fftw_plan plan = fftw_plan_dft(D, n.data(), reinterpret_cast<fftw_complex *>(data),
                                   reinterpret_cast<fftw_complex *>(data), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  // --------------------------------------------------------------------------
  // Pretty-print a shape array: "N0 x N1 x ... x N_{D-1}"
  // --------------------------------------------------------------------------
  template <int D> void print_shape(std::ostream &os, const int shape[D])
  {
    for (int d = 0; d < D; ++d) {
      if (d > 0) os << " x ";
      os << shape[d];
    }
  }

  // --------------------------------------------------------------------------
  // cuFFT serial C2C reference (D = 2 or 3 only, guarded by preprocessor).
  // For D >= 4 cuFFT has no cufftPlanNd for C2C, so we use FFTW as reference.
  // --------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVCC__)

  inline void cufft_c2c_reference_2d(std::complex<double> *host_data, int N0, int N1)
  {
    const int total = N0 * N1;
    cufftDoubleComplex *d_data = nullptr;
    cudaMalloc((void **)&d_data, total * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_data, host_data, total * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan2d(&plan, N0, N1, CUFFT_Z2Z);
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    cufftDestroy(plan);

    cudaMemcpy(host_data, d_data, total * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
  }

  inline void cufft_c2c_reference_3d(std::complex<double> *host_data, int N0, int N1, int N2)
  {
    const int total = N0 * N1 * N2;
    cufftDoubleComplex *d_data = nullptr;
    cudaMalloc((void **)&d_data, total * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_data, host_data, total * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan3d(&plan, N0, N1, N2, CUFFT_Z2Z);
    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    cufftDestroy(plan);

    cudaMemcpy(host_data, d_data, total * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
  }

  /// Run the cuFFT-vs-FFTW sanity check.  Returns 0 on success, 1 on failure.
  /// Only callable for D = 2 or 3; for D >= 4 this is skipped.
  template <int D> int compare_cufft_vs_fftw(int N, int rank)
  {
    static_assert(D == 2 || D == 3, "cuFFT library reference only available for D=2,3");
    if (rank != 0) return 0;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Sanity check: Comparing (sequential) FFTW and cuFFT for " << D << "D C2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;

    auto fftw_ref = generate_gaussian_nd<D>(N, 4.0);
    auto cufft_ref = fftw_ref;

    fftw_c2c_reference_nd<D>(fftw_ref.data(), N);

    if (D == 2)
      cufft_c2c_reference_2d(cufft_ref.data(), N, N);
    else
      cufft_c2c_reference_3d(cufft_ref.data(), N, N, N);

    double max_error = 0.0;
    for (size_t i = 0; i < fftw_ref.size(); ++i) {
      double err = std::abs(fftw_ref[i] - cufft_ref[i]);
      if (err > max_error) max_error = err;
    }

    if (max_error < 1e-10) {
      std::cout << "Test passed: FFTW and cuFFT produce identical results." << std::endl;
      std::cout << "Maximum error: " << max_error << std::endl;
      return 0;
    } else {
      std::cout << "Test failed: FFTW and cuFFT produce different results." << std::endl;
      std::cout << "Maximum error: " << max_error << std::endl;
      return 1;
    }
  }

#endif // cuFFT helpers

  // --------------------------------------------------------------------------
  // hipFFT serial C2C reference (D = 2 or 3 only, guarded by preprocessor).
  // --------------------------------------------------------------------------
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIPCC__)

  inline void hipfft_c2c_reference_2d(std::complex<double> *host_data, int N0, int N1)
  {
    const int total = N0 * N1;
    hipfftDoubleComplex *d_data = nullptr;
    hipMalloc((void **)&d_data, total * sizeof(hipfftDoubleComplex));
    hipMemcpy(d_data, host_data, total * sizeof(hipfftDoubleComplex), hipMemcpyHostToDevice);

    hipfftHandle plan;
    hipfftPlan2d(&plan, N0, N1, HIPFFT_Z2Z);
    hipfftExecZ2Z(plan, d_data, d_data, HIPFFT_FORWARD);
    hipfftDestroy(plan);

    hipMemcpy(host_data, d_data, total * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
    hipFree(d_data);
  }

  inline void hipfft_c2c_reference_3d(std::complex<double> *host_data, int N0, int N1, int N2)
  {
    const int total = N0 * N1 * N2;
    hipfftDoubleComplex *d_data = nullptr;
    hipMalloc((void **)&d_data, total * sizeof(hipfftDoubleComplex));
    hipMemcpy(d_data, host_data, total * sizeof(hipfftDoubleComplex), hipMemcpyHostToDevice);

    hipfftHandle plan;
    hipfftPlan3d(&plan, N0, N1, N2, HIPFFT_Z2Z);
    hipfftExecZ2Z(plan, d_data, d_data, HIPFFT_FORWARD);
    hipfftDestroy(plan);

    hipMemcpy(host_data, d_data, total * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
    hipFree(d_data);
  }

  /// Run the hipFFT-vs-FFTW sanity check.  Returns 0 on success, 1 on failure.
  /// Only callable for D = 2 or 3.
  template <int D> int compare_hipfft_vs_fftw(int N, int rank)
  {
    static_assert(D == 2 || D == 3, "hipFFT library reference only available for D=2,3");
    if (rank != 0) return 0;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Sanity check: Comparing (sequential) FFTW and hipFFT for " << D << "D C2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;

    auto fftw_ref = generate_gaussian_nd<D>(N, 4.0);
    auto hipfft_ref = fftw_ref;

    fftw_c2c_reference_nd<D>(fftw_ref.data(), N);

    if (D == 2)
      hipfft_c2c_reference_2d(hipfft_ref.data(), N, N);
    else
      hipfft_c2c_reference_3d(hipfft_ref.data(), N, N, N);

    double max_error = 0.0;
    for (size_t i = 0; i < fftw_ref.size(); ++i) {
      double err = std::abs(fftw_ref[i] - hipfft_ref[i]);
      if (err > max_error) max_error = err;
    }

    if (max_error < 1e-10) {
      std::cout << "Test passed: FFTW and hipFFT produce identical results." << std::endl;
      std::cout << "Maximum error: " << max_error << std::endl;
      return 0;
    } else {
      std::cout << "Test failed: FFTW and hipFFT produce different results." << std::endl;
      std::cout << "Maximum error: " << max_error << std::endl;
      return 1;
    }
  }

#endif // hipFFT helpers

} // namespace parafaft_test

#endif // PARAFAFT_TEST_HELPERS_HPP
