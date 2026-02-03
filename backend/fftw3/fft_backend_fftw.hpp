#ifndef PARAFAFT_BACKEND_FFTW_HPP
#define PARAFAFT_BACKEND_FFTW_HPP

#include <fftw3.h>
#include <complex>
#include <vector>
#include <cstring>
#include "../fft_backend.hpp"

namespace parafaft
{
  class FFTWBackend
  {
  public:
    using Complex = std::complex<double>;
    using Buffer = std::vector<double>;
    using ComplexBuffer = std::vector<Complex>;

    // Constructor: Initialize storage for plans
    explicit FFTWBackend(int num_stages) : forward_plans_(num_stages, nullptr), backward_plans_(num_stages, nullptr) {}

    // Create and store plans for a specific stage
    void create_stage_plan(int stage, int length, int batch, Complex *data, int stride, int dist)
    {
      int n[] = {length};
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(data);

      // Create forward plan (bound to data pointer)
      forward_plans_[stage] = fftw_plan_many_dft(1, n, batch, fftw_data, NULL, stride, dist, fftw_data, NULL, stride,
                                                 dist, FFTW_FORWARD, FFTW_ESTIMATE);

      // Create backward plan (bound to data pointer)
      backward_plans_[stage] = fftw_plan_many_dft(1, n, batch, fftw_data, NULL, stride, dist, fftw_data, NULL, stride,
                                                  dist, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    // Create in-place R2C plan for padded memory optimization
    // Input: real values in padded buffer
    // Output: complex values in same padded buffer (reinterpreted)
    void create_r2c_inplace_plan(int length,          // Real-space input length N
                                 int batch,           // Number of transforms
                                 double *padded_real, // Padded real buffer
                                 int stride,          // Stride
                                 int dist             // Distance between batches (padded: 2*(N/2+1) doubles)
    )
    {
      int n[] = {length};
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(padded_real);

      // In-place R2C: real → complex in same buffer
      r2c_inplace_plan_ =
          fftw_plan_many_dft_r2c(1, n, batch, padded_real, NULL, stride, dist, // real input (dist in doubles)
                                 fftw_data, NULL, stride, dist / 2,            // complex output (dist/2 in complex)
                                 FFTW_ESTIMATE);
    }

    // Execute in-place R2C plan
    void execute_r2c_inplace(double *padded_real)
    {
      fftw_execute_dft_r2c(r2c_inplace_plan_, padded_real, reinterpret_cast<fftw_complex *>(padded_real));
    }

    // Create in-place C2R plan for padded memory optimization
    // Input: complex values in padded real buffer (reinterpreted)
    // Output: real values in same padded buffer
    void create_c2r_inplace_plan(int length,          // Real-space output length N
                                 int batch,           // Number of transforms
                                 double *padded_real, // Padded real buffer (also used as complex input)
                                 int stride,          // Stride
                                 int dist             // Distance between batches (padded: 2*(N/2+1))
    )
    {
      int n[] = {length};
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(padded_real);

      // In-place C2R: complex → real in same buffer
      c2r_inplace_plan_ =
          fftw_plan_many_dft_c2r(1, n, batch, fftw_data, NULL, stride, dist / 2, // dist/2 for complex stride
                                 padded_real, NULL, stride, dist, FFTW_ESTIMATE);
    }

    // Execute in-place C2R plan
    void execute_c2r_inplace(double *padded_real)
    {
      fftw_execute_dft_c2r(c2r_inplace_plan_, reinterpret_cast<fftw_complex *>(padded_real), padded_real);
    }

    // Execute pre-created plan for specified stage on given data
    void execute_stage(int stage, FFTDirection direction, Complex *data)
    {
      fftw_plan plan = (direction == FFTDirection::Forward) ? forward_plans_[stage] : backward_plans_[stage];
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(data);
      fftw_execute_dft(plan, fftw_data, fftw_data);
    }

    // Memory copy method
    static void memcpy(void *dest, const void *src, size_t bytes) { std::memcpy(dest, src, bytes); }

    // Destructor: Clean up all plans
    ~FFTWBackend()
    {
      for (auto plan : forward_plans_) {
        if (plan) fftw_destroy_plan(plan);
      }
      for (auto plan : backward_plans_) {
        if (plan) fftw_destroy_plan(plan);
      }
      if (r2c_inplace_plan_) fftw_destroy_plan(r2c_inplace_plan_);
      if (c2r_inplace_plan_) fftw_destroy_plan(c2r_inplace_plan_);
    }

    // Disable copying (plans can't be safely copied)
    FFTWBackend(const FFTWBackend &) = delete;
    FFTWBackend &operator=(const FFTWBackend &) = delete;

    // Enable moving
    FFTWBackend(FFTWBackend &&other) noexcept
        : forward_plans_(std::move(other.forward_plans_)), backward_plans_(std::move(other.backward_plans_)),
          r2c_inplace_plan_(other.r2c_inplace_plan_), c2r_inplace_plan_(other.c2r_inplace_plan_)
    {
      std::fill(other.forward_plans_.begin(), other.forward_plans_.end(), nullptr);
      std::fill(other.backward_plans_.begin(), other.backward_plans_.end(), nullptr);
      other.r2c_inplace_plan_ = nullptr;
      other.c2r_inplace_plan_ = nullptr;
    }

  private:
    std::vector<fftw_plan> forward_plans_;
    std::vector<fftw_plan> backward_plans_;
    fftw_plan r2c_inplace_plan_ = nullptr;
    fftw_plan c2r_inplace_plan_ = nullptr;
  };

} // namespace parafaft

#endif // PARAFAFT_BACKEND_FFTW_HPP
