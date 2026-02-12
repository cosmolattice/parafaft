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

#include <algorithm>

namespace parafaft_test
{

  // --------------------------------------------------------------------------
  // Shape enumeration for test data generation.
  //   0 = Gaussian
  //   1 = Step function (sphere)
  //   2 = Random polynomial
  // --------------------------------------------------------------------------
  enum TestShape { SHAPE_GAUSSIAN = 0, SHAPE_STEP = 1, SHAPE_RANDOM_POLY = 2 };

  inline const char *shape_name(int shape)
  {
    switch (shape) {
    case SHAPE_GAUSSIAN:
      return "Gaussian";
    case SHAPE_STEP:
      return "StepFunction";
    case SHAPE_RANDOM_POLY:
      return "RandomPolynomial";
    default:
      return "Unknown";
    }
  }

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
  // Generate a D-dimensional step function (sphere) on a hypercubic grid of
  // side N.  Value is 1 inside a sphere of radius N/4 centered at N/2, else 0.
  // --------------------------------------------------------------------------
  template <int D> std::vector<std::complex<double>> generate_step_nd(int N)
  {
    int total = 1;
    for (int d = 0; d < D; ++d)
      total *= N;

    std::vector<std::complex<double>> data(total);
    std::array<int, D> shape;
    shape.fill(N);

    const double center = N / 2.0;
    const double radius = N / 4.0;
    const double r2_max = radius * radius;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      double value = (r2 <= r2_max) ? 1.0 : 0.0;
      int flat = nd_index<D>(idx, shape);
      data[flat] = std::complex<double>(value, 0.0);
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional random polynomial on a hypercubic grid of side N.
  // Uses a deterministic seed so all MPI ranks produce the same data.
  // The polynomial is a sum of random monomials of degree <= 3.
  // --------------------------------------------------------------------------
  template <int D> std::vector<std::complex<double>> generate_random_poly_nd(int N)
  {
    int total = 1;
    for (int d = 0; d < D; ++d)
      total *= N;

    std::vector<std::complex<double>> data(total);
    std::array<int, D> shape;
    shape.fill(N);

    // Deterministic coefficients (seeded pseudo-random via simple LCG)
    // We generate D+1 coefficients for a polynomial:
    //   f(x) = c0 + sum_d c1[d]*x_d + sum_d c2[d]*x_d^2 + sum_d c3[d]*x_d^3
    const int num_coeffs = 1 + 3 * D;
    std::vector<double> coeffs(num_coeffs);
    unsigned int seed = 42u + D * 7u + N * 13u;
    for (int i = 0; i < num_coeffs; ++i) {
      seed = seed * 1103515245u + 12345u;
      coeffs[i] = ((seed >> 16) & 0x7FFF) / 32768.0 - 0.5; // in [-0.5, 0.5)
    }

    const double inv_N = 1.0 / N;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double value = coeffs[0]; // constant term
      for (int d = 0; d < D; ++d) {
        double x = idx[d] * inv_N; // normalised to [0,1)
        value += coeffs[1 + d] * x;
        value += coeffs[1 + D + d] * x * x;
        value += coeffs[1 + 2 * D + d] * x * x * x;
      }
      int flat = nd_index<D>(idx, shape);
      data[flat] = std::complex<double>(value, 0.0);
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generic C2C data generator: dispatches on shape id.
  // --------------------------------------------------------------------------
  template <int D> std::vector<std::complex<double>> generate_c2c_data_nd(int N, int shape_id)
  {
    switch (shape_id) {
    case SHAPE_GAUSSIAN:
      return generate_gaussian_nd<D>(N, 4.0);
    case SHAPE_STEP:
      return generate_step_nd<D>(N);
    case SHAPE_RANDOM_POLY:
      return generate_random_poly_nd<D>(N);
    default:
      return generate_gaussian_nd<D>(N, 4.0);
    }
  }

  // --------------------------------------------------------------------------
  // Point-wise value generator for local data (used by roundtrip tests).
  // Returns the real value at a given global multi-index for the chosen shape.
  // --------------------------------------------------------------------------
  template <int D> double point_value(const std::array<int, D> &gidx, int N, int shape_id)
  {
    const double center = N / 2.0;
    switch (shape_id) {
    case SHAPE_GAUSSIAN: {
      const double sigma = 4.0;
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = gidx[d] - center;
        r2 += x * x;
      }
      return std::exp(-r2 / (2.0 * sigma * sigma));
    }
    case SHAPE_STEP: {
      const double radius = N / 4.0;
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = gidx[d] - center;
        r2 += x * x;
      }
      return (r2 <= radius * radius) ? 1.0 : 0.0;
    }
    case SHAPE_RANDOM_POLY: {
      const int num_coeffs = 1 + 3 * D;
      std::vector<double> coeffs(num_coeffs);
      unsigned int seed = 42u + D * 7u + N * 13u;
      for (int i = 0; i < num_coeffs; ++i) {
        seed = seed * 1103515245u + 12345u;
        coeffs[i] = ((seed >> 16) & 0x7FFF) / 32768.0 - 0.5;
      }
      const double inv_N = 1.0 / N;
      double value = coeffs[0];
      for (int d = 0; d < D; ++d) {
        double x = gidx[d] * inv_N;
        value += coeffs[1 + d] * x;
        value += coeffs[1 + D + d] * x * x;
        value += coeffs[1 + 2 * D + d] * x * x * x;
      }
      return value;
    }
    default:
      return 0.0;
    }
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

  // ==========================================================================
  // R2C helpers
  // ==========================================================================

  // --------------------------------------------------------------------------
  // Flat-index for R2C real data in row-major order (NON-PADDED layout).
  // All dimensions use their natural shape; last dim stride is shape[D-1].
  // --------------------------------------------------------------------------
  template <int D> inline int nd_index_real(const std::array<int, D> &idx, const int shape[D])
  {
    int flat = idx[0];
    for (int d = 1; d < D; ++d)
      flat = flat * shape[d] + idx[d];
    return flat;
  }

  // --------------------------------------------------------------------------
  // Flat-index for R2C real data in row-major order (PADDED layout).
  // The last dimension has stride (shape[D-1] + 2) to accommodate in-place R2C.
  // --------------------------------------------------------------------------
  template <int D> inline int nd_index_real_padded(const std::array<int, D> &idx, const int shape[D])
  {
    const int padded_last = shape[D - 1] + 2;
    int flat = idx[0];
    for (int d = 1; d < D - 1; ++d)
      flat = flat * shape[d] + idx[d];
    flat = flat * padded_last + idx[D - 1];
    return flat;
  }

  // --------------------------------------------------------------------------
  // Flat-index for R2C complex output in row-major order.
  // shape_complex[d] = shape_real[d] for d < D-1, and N_{D-1}/2+1 for last dim.
  // --------------------------------------------------------------------------
  template <int D> inline int nd_index_complex(const std::array<int, D> &idx, const int complex_shape[D])
  {
    int flat = idx[0];
    for (int d = 1; d < D; ++d)
      flat = flat * complex_shape[d] + idx[d];
    return flat;
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional Gaussian for R2C (NON-PADDED, out-of-place).
  // Allocates enough for the complex output: product(N, ..., N, (N/2+1)) * 2.
  // Fills only the real part (N^D entries).
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_gaussian_r2c_nd(int N, double sigma)
  {
    // Total real elements
    int real_total = 1;
    for (int d = 0; d < D; ++d)
      real_total *= N;

    // Complex output total (doubles)
    int complex_doubles = 1;
    for (int d = 0; d < D - 1; ++d)
      complex_doubles *= N;
    complex_doubles *= (N / 2 + 1) * 2;

    // Allocate max of both (complex output >= real for even N)
    const int total_size = std::max(real_total, complex_doubles);
    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);
    const double center = N / 2.0;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      int flat = nd_index_real<D>(idx, shape.data());
      data[flat] = std::exp(-r2 / (2.0 * sigma * sigma));
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional Gaussian for R2C (PADDED, in-place).
  // The last dimension has stride (N+2).
  // Total allocation: product(N, ..., N_{D-2}, N+2).
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_gaussian_r2c_padded_nd(int N, double sigma)
  {
    int total_size = 1;
    for (int d = 0; d < D - 1; ++d)
      total_size *= N;
    total_size *= (N + 2);

    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);
    const double center = N / 2.0;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      int flat = nd_index_real_padded<D>(idx, shape.data());
      data[flat] = std::exp(-r2 / (2.0 * sigma * sigma));
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional step function for R2C (NON-PADDED, out-of-place).
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_step_r2c_nd(int N)
  {
    int real_total = 1;
    for (int d = 0; d < D; ++d)
      real_total *= N;

    int complex_doubles = 1;
    for (int d = 0; d < D - 1; ++d)
      complex_doubles *= N;
    complex_doubles *= (N / 2 + 1) * 2;

    const int total_size = std::max(real_total, complex_doubles);
    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);
    const double center = N / 2.0;
    const double radius = N / 4.0;
    const double r2_max = radius * radius;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      int flat = nd_index_real<D>(idx, shape.data());
      data[flat] = (r2 <= r2_max) ? 1.0 : 0.0;
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional random polynomial for R2C (NON-PADDED).
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_random_poly_r2c_nd(int N)
  {
    int real_total = 1;
    for (int d = 0; d < D; ++d)
      real_total *= N;

    int complex_doubles = 1;
    for (int d = 0; d < D - 1; ++d)
      complex_doubles *= N;
    complex_doubles *= (N / 2 + 1) * 2;

    const int total_size = std::max(real_total, complex_doubles);
    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);

    const int num_coeffs = 1 + 3 * D;
    std::vector<double> coeffs(num_coeffs);
    unsigned int seed = 42u + D * 7u + N * 13u;
    for (int i = 0; i < num_coeffs; ++i) {
      seed = seed * 1103515245u + 12345u;
      coeffs[i] = ((seed >> 16) & 0x7FFF) / 32768.0 - 0.5;
    }
    const double inv_N = 1.0 / N;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double value = coeffs[0];
      for (int d = 0; d < D; ++d) {
        double x = idx[d] * inv_N;
        value += coeffs[1 + d] * x;
        value += coeffs[1 + D + d] * x * x;
        value += coeffs[1 + 2 * D + d] * x * x * x;
      }
      int flat = nd_index_real<D>(idx, shape.data());
      data[flat] = value;
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional step function for R2C (PADDED, in-place).
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_step_r2c_padded_nd(int N)
  {
    int total_size = 1;
    for (int d = 0; d < D - 1; ++d)
      total_size *= N;
    total_size *= (N + 2);

    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);
    const double center = N / 2.0;
    const double radius = N / 4.0;
    const double r2_max = radius * radius;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      int flat = nd_index_real_padded<D>(idx, shape.data());
      data[flat] = (r2 <= r2_max) ? 1.0 : 0.0;
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generate a D-dimensional random polynomial for R2C (PADDED, in-place).
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_random_poly_r2c_padded_nd(int N)
  {
    int total_size = 1;
    for (int d = 0; d < D - 1; ++d)
      total_size *= N;
    total_size *= (N + 2);

    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);

    const int num_coeffs = 1 + 3 * D;
    std::vector<double> coeffs(num_coeffs);
    unsigned int seed = 42u + D * 7u + N * 13u;
    for (int i = 0; i < num_coeffs; ++i) {
      seed = seed * 1103515245u + 12345u;
      coeffs[i] = ((seed >> 16) & 0x7FFF) / 32768.0 - 0.5;
    }
    const double inv_N = 1.0 / N;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double value = coeffs[0];
      for (int d = 0; d < D; ++d) {
        double x = idx[d] * inv_N;
        value += coeffs[1 + d] * x;
        value += coeffs[1 + D + d] * x * x;
        value += coeffs[1 + 2 * D + d] * x * x * x;
      }
      int flat = nd_index_real_padded<D>(idx, shape.data());
      data[flat] = value;
    });

    return data;
  }

  // --------------------------------------------------------------------------
  // Generic R2C data generator (NON-PADDED): dispatches on shape id.
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_r2c_data_nd(int N, int shape_id)
  {
    switch (shape_id) {
    case SHAPE_GAUSSIAN:
      return generate_gaussian_r2c_nd<D>(N, 4.0);
    case SHAPE_STEP:
      return generate_step_r2c_nd<D>(N);
    case SHAPE_RANDOM_POLY:
      return generate_random_poly_r2c_nd<D>(N);
    default:
      return generate_gaussian_r2c_nd<D>(N, 4.0);
    }
  }

  // --------------------------------------------------------------------------
  // Generic R2C data generator (PADDED): dispatches on shape id.
  // --------------------------------------------------------------------------
  template <int D> std::vector<double> generate_r2c_padded_data_nd(int N, int shape_id)
  {
    switch (shape_id) {
    case SHAPE_GAUSSIAN:
      return generate_gaussian_r2c_padded_nd<D>(N, 4.0);
    case SHAPE_STEP:
      return generate_step_r2c_padded_nd<D>(N);
    case SHAPE_RANDOM_POLY:
      return generate_random_poly_r2c_padded_nd<D>(N);
    default:
      return generate_gaussian_r2c_padded_nd<D>(N, 4.0);
    }
  }

  // --------------------------------------------------------------------------
  // Serial FFTW forward R2C transform on D-dimensional data (out-of-place).
  // Input: real data in host_data (N^D entries, contiguous row-major).
  // Output: complex data written back into host_data (reinterpreted).
  // Caller must ensure host_data has enough room for complex output.
  // --------------------------------------------------------------------------
  template <int D> void fftw_r2c_reference_nd(double *host_data, int N)
  {
    // Copy real data out
    int real_total = 1;
    for (int d = 0; d < D; ++d)
      real_total *= N;

    std::vector<double> temp(real_total);
    std::copy(host_data, host_data + real_total, temp.data());

    // Compute complex output size (in doubles)
    int complex_doubles = 1;
    for (int d = 0; d < D - 1; ++d)
      complex_doubles *= N;
    complex_doubles *= (N + 2); // (N/2+1) complex entries = (N+2) doubles in last dim

    std::fill(host_data, host_data + complex_doubles, 0.0);

    std::array<int, D> n;
    n.fill(N);

    fftw_plan plan =
        fftw_plan_dft_r2c(D, n.data(), temp.data(), reinterpret_cast<fftw_complex *>(host_data), FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  // --------------------------------------------------------------------------
  // Serial FFTW forward R2C transform on D-dimensional PADDED data (in-place).
  // The last dimension already has stride (N+2).
  // --------------------------------------------------------------------------
  template <int D> void fftw_r2c_reference_padded_nd(double *host_data, int N)
  {
    std::array<int, D> n;
    n.fill(N);

    fftw_plan plan =
        fftw_plan_dft_r2c(D, n.data(), host_data, reinterpret_cast<fftw_complex *>(host_data), FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  // --------------------------------------------------------------------------
  // Compute the global complex shape for an R2C transform of a hypercubic grid.
  // All dimensions stay N except the last, which becomes N/2+1.
  // --------------------------------------------------------------------------
  template <int D> std::array<int, D> r2c_complex_shape(int N)
  {
    std::array<int, D> s;
    s.fill(N);
    s[D - 1] = N / 2 + 1;
    return s;
  }

  // --------------------------------------------------------------------------
  // cuFFT serial R2C reference helpers (D = 2, 3 only).
  // --------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVCC__)

  // Out-of-place cuFFT R2C reference
  inline void cufft_r2c_reference_2d(double *host_data, int N0, int N1)
  {
    const int real_size = N0 * N1;
    const int complex_size = N0 * (N1 / 2 + 1);

    cufftDoubleReal *d_real = nullptr;
    cufftDoubleComplex *d_complex = nullptr;
    cudaMalloc((void **)&d_real, real_size * sizeof(cufftDoubleReal));
    cudaMalloc((void **)&d_complex, complex_size * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_real, host_data, real_size * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan2d(&plan, N0, N1, CUFFT_D2Z);
    cufftExecD2Z(plan, d_real, d_complex);
    cufftDestroy(plan);

    cudaMemcpy(host_data, d_complex, complex_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_real);
    cudaFree(d_complex);
  }

  inline void cufft_r2c_reference_3d(double *host_data, int N0, int N1, int N2)
  {
    const int real_size = N0 * N1 * N2;
    const int complex_size = N0 * N1 * (N2 / 2 + 1);

    cufftDoubleReal *d_real = nullptr;
    cufftDoubleComplex *d_complex = nullptr;
    cudaMalloc((void **)&d_real, real_size * sizeof(cufftDoubleReal));
    cudaMalloc((void **)&d_complex, complex_size * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_real, host_data, real_size * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan3d(&plan, N0, N1, N2, CUFFT_D2Z);
    cufftExecD2Z(plan, d_real, d_complex);
    cufftDestroy(plan);

    cudaMemcpy(host_data, d_complex, complex_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_real);
    cudaFree(d_complex);
  }

  // In-place cuFFT R2C reference (padded layout)
  inline void cufft_r2c_reference_padded_2d(double *host_data, int N0, int N1)
  {
    const int total_size = N0 * (N1 / 2 + 1);
    cufftDoubleComplex *d_data = nullptr;
    cudaMalloc((void **)&d_data, total_size * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_data, host_data, total_size * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan2d(&plan, N0, N1, CUFFT_D2Z);
    cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal *>(d_data), reinterpret_cast<cufftDoubleComplex *>(d_data));
    cufftDestroy(plan);

    cudaMemcpy(host_data, d_data, total_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
  }

  inline void cufft_r2c_reference_padded_3d(double *host_data, int N0, int N1, int N2)
  {
    const int total_size = N0 * N1 * (N2 / 2 + 1);
    cufftDoubleComplex *d_data = nullptr;
    cudaMalloc((void **)&d_data, total_size * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_data, host_data, total_size * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan3d(&plan, N0, N1, N2, CUFFT_D2Z);
    cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal *>(d_data), reinterpret_cast<cufftDoubleComplex *>(d_data));
    cufftDestroy(plan);

    cudaMemcpy(host_data, d_data, total_size * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
  }

  /// cuFFT-vs-FFTW R2C sanity check (out-of-place). D = 2 or 3 only.
  template <int D> int compare_cufft_vs_fftw_r2c(int N, int rank)
  {
    static_assert(D == 2 || D == 3, "cuFFT R2C library reference only available for D=2,3");
    if (rank != 0) return 0;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Sanity check: Comparing (sequential) FFTW and cuFFT for " << D << "D R2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;

    auto fftw_ref = generate_gaussian_r2c_nd<D>(N, 4.0);
    auto cufft_ref = fftw_ref;

    fftw_r2c_reference_nd<D>(fftw_ref.data(), N);

    if (D == 2)
      cufft_r2c_reference_2d(cufft_ref.data(), N, N);
    else
      cufft_r2c_reference_3d(cufft_ref.data(), N, N, N);

    auto gcs = r2c_complex_shape<D>(N);
    int complex_total = 1;
    for (int d = 0; d < D; ++d)
      complex_total *= gcs[d];

    double max_error = 0.0;
    std::complex<double> *fc = reinterpret_cast<std::complex<double> *>(fftw_ref.data());
    std::complex<double> *cc = reinterpret_cast<std::complex<double> *>(cufft_ref.data());
    for (int i = 0; i < complex_total; ++i) {
      double err = std::abs(fc[i] - cc[i]);
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

  /// cuFFT-vs-FFTW R2C sanity check (padded/in-place). D = 2 or 3 only.
  template <int D> int compare_cufft_vs_fftw_r2c_padded(int N, int rank)
  {
    static_assert(D == 2 || D == 3, "cuFFT R2C padded library reference only available for D=2,3");
    if (rank != 0) return 0;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Sanity check: Comparing (sequential) FFTW and cuFFT for " << D << "D padded R2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;

    auto fftw_ref = generate_gaussian_r2c_padded_nd<D>(N, 4.0);
    auto cufft_ref = fftw_ref;

    fftw_r2c_reference_padded_nd<D>(fftw_ref.data(), N);

    if (D == 2)
      cufft_r2c_reference_padded_2d(cufft_ref.data(), N, N);
    else
      cufft_r2c_reference_padded_3d(cufft_ref.data(), N, N, N);

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

#endif // cuFFT R2C helpers

  // --------------------------------------------------------------------------
  // hipFFT serial R2C reference helpers (D = 2, 3 only).
  // --------------------------------------------------------------------------
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIPCC__)

  // Out-of-place hipFFT R2C reference
  inline void hipfft_r2c_reference_2d(double *host_data, int N0, int N1)
  {
    const int real_size = N0 * N1;
    const int complex_size = N0 * (N1 / 2 + 1);

    hipfftDoubleReal *d_real = nullptr;
    hipfftDoubleComplex *d_complex = nullptr;
    hipMalloc((void **)&d_real, real_size * sizeof(hipfftDoubleReal));
    hipMalloc((void **)&d_complex, complex_size * sizeof(hipfftDoubleComplex));
    hipMemcpy(d_real, host_data, real_size * sizeof(hipfftDoubleReal), hipMemcpyHostToDevice);

    hipfftHandle plan;
    hipfftPlan2d(&plan, N0, N1, HIPFFT_D2Z);
    hipfftExecD2Z(plan, d_real, d_complex);
    hipfftDestroy(plan);

    hipMemcpy(host_data, d_complex, complex_size * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
    hipFree(d_real);
    hipFree(d_complex);
  }

  inline void hipfft_r2c_reference_3d(double *host_data, int N0, int N1, int N2)
  {
    const int real_size = N0 * N1 * N2;
    const int complex_size = N0 * N1 * (N2 / 2 + 1);

    hipfftDoubleReal *d_real = nullptr;
    hipfftDoubleComplex *d_complex = nullptr;
    hipMalloc((void **)&d_real, real_size * sizeof(hipfftDoubleReal));
    hipMalloc((void **)&d_complex, complex_size * sizeof(hipfftDoubleComplex));
    hipMemcpy(d_real, host_data, real_size * sizeof(hipfftDoubleReal), hipMemcpyHostToDevice);

    hipfftHandle plan;
    hipfftPlan3d(&plan, N0, N1, N2, HIPFFT_D2Z);
    hipfftExecD2Z(plan, d_real, d_complex);
    hipfftDestroy(plan);

    hipMemcpy(host_data, d_complex, complex_size * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
    hipFree(d_real);
    hipFree(d_complex);
  }

  // In-place hipFFT R2C reference (padded layout)
  inline void hipfft_r2c_reference_padded_2d(double *host_data, int N0, int N1)
  {
    const int total_size = N0 * (N1 / 2 + 1);
    hipfftDoubleComplex *d_data = nullptr;
    hipMalloc((void **)&d_data, total_size * sizeof(hipfftDoubleComplex));
    hipMemcpy(d_data, host_data, total_size * sizeof(hipfftDoubleComplex), hipMemcpyHostToDevice);

    hipfftHandle plan;
    hipfftPlan2d(&plan, N0, N1, HIPFFT_D2Z);
    hipfftExecD2Z(plan, reinterpret_cast<hipfftDoubleReal *>(d_data), reinterpret_cast<hipfftDoubleComplex *>(d_data));
    hipfftDestroy(plan);

    hipMemcpy(host_data, d_data, total_size * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
    hipFree(d_data);
  }

  inline void hipfft_r2c_reference_padded_3d(double *host_data, int N0, int N1, int N2)
  {
    const int total_size = N0 * N1 * (N2 / 2 + 1);
    hipfftDoubleComplex *d_data = nullptr;
    hipMalloc((void **)&d_data, total_size * sizeof(hipfftDoubleComplex));
    hipMemcpy(d_data, host_data, total_size * sizeof(hipfftDoubleComplex), hipMemcpyHostToDevice);

    hipfftHandle plan;
    hipfftPlan3d(&plan, N0, N1, N2, HIPFFT_D2Z);
    hipfftExecD2Z(plan, reinterpret_cast<hipfftDoubleReal *>(d_data), reinterpret_cast<hipfftDoubleComplex *>(d_data));
    hipfftDestroy(plan);

    hipMemcpy(host_data, d_data, total_size * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);
    hipFree(d_data);
  }

  /// hipFFT-vs-FFTW R2C sanity check (out-of-place). D = 2 or 3 only.
  template <int D> int compare_hipfft_vs_fftw_r2c(int N, int rank)
  {
    static_assert(D == 2 || D == 3, "hipFFT R2C library reference only available for D=2,3");
    if (rank != 0) return 0;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Sanity check: Comparing (sequential) FFTW and hipFFT for " << D << "D R2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;

    auto fftw_ref = generate_gaussian_r2c_nd<D>(N, 4.0);
    auto hipfft_ref = fftw_ref;

    fftw_r2c_reference_nd<D>(fftw_ref.data(), N);

    if (D == 2)
      hipfft_r2c_reference_2d(hipfft_ref.data(), N, N);
    else
      hipfft_r2c_reference_3d(hipfft_ref.data(), N, N, N);

    auto gcs = r2c_complex_shape<D>(N);
    int complex_total = 1;
    for (int d = 0; d < D; ++d)
      complex_total *= gcs[d];

    double max_error = 0.0;
    std::complex<double> *fc = reinterpret_cast<std::complex<double> *>(fftw_ref.data());
    std::complex<double> *hc = reinterpret_cast<std::complex<double> *>(hipfft_ref.data());
    for (int i = 0; i < complex_total; ++i) {
      double err = std::abs(fc[i] - hc[i]);
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

  /// hipFFT-vs-FFTW R2C sanity check (padded/in-place). D = 2 or 3 only.
  template <int D> int compare_hipfft_vs_fftw_r2c_padded(int N, int rank)
  {
    static_assert(D == 2 || D == 3, "hipFFT R2C padded library reference only available for D=2,3");
    if (rank != 0) return 0;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Sanity check: Comparing (sequential) FFTW and hipFFT for " << D << "D padded R2C transform of size ";
    std::array<int, D> s;
    s.fill(N);
    print_shape<D>(std::cout, s.data());
    std::cout << std::endl;

    auto fftw_ref = generate_gaussian_r2c_padded_nd<D>(N, 4.0);
    auto hipfft_ref = fftw_ref;

    fftw_r2c_reference_padded_nd<D>(fftw_ref.data(), N);

    if (D == 2)
      hipfft_r2c_reference_padded_2d(hipfft_ref.data(), N, N);
    else
      hipfft_r2c_reference_padded_3d(hipfft_ref.data(), N, N, N);

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

#endif // hipFFT R2C helpers

} // namespace parafaft_test

#endif // PARAFAFT_TEST_HELPERS_HPP
