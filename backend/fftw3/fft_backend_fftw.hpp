#ifndef MPIFFT_BACKEND_FFTW_HPP
#define MPIFFT_BACKEND_FFTW_HPP

#include <fftw3.h>
#include <complex>
#include <vector>
#include "../fft_backend.hpp"

namespace mpifft {

class FFTWBackend {
public:
    // Constructor: Initialize storage for plans
    explicit FFTWBackend(int num_stages)
        : forward_plans_(num_stages, nullptr),
          backward_plans_(num_stages, nullptr) {
    }

    // Create and store plans for a specific stage
    void create_stage_plan(
        int stage,
        int length,
        int batch,
        std::complex<double>* data,
        int stride,
        int dist
    ) {
        int n[] = {length};
        fftw_complex* fftw_data = reinterpret_cast<fftw_complex*>(data);

        // Create forward plan (bound to data pointer)
        forward_plans_[stage] = fftw_plan_many_dft(
            1, n, batch,
            fftw_data, NULL, stride, dist,
            fftw_data, NULL, stride, dist,
            FFTW_FORWARD, FFTW_ESTIMATE
        );

        // Create backward plan (bound to data pointer)
        backward_plans_[stage] = fftw_plan_many_dft(
            1, n, batch,
            fftw_data, NULL, stride, dist,
            fftw_data, NULL, stride, dist,
            FFTW_BACKWARD, FFTW_ESTIMATE
        );
    }

    // Create R2C plan for first stage (real input → complex output)
    void create_r2c_plan(
        int length,      // Real-space length N
        int batch,       // Number of transforms
        double* real_in,           // Real input array
        std::complex<double>* complex_out,  // Complex output array
        int istride,     // Input stride
        int idist,       // Input distance between batches
        int ostride,     // Output stride
        int odist        // Output distance between batches
    ) {
        int n[] = {length};
        fftw_complex* fftw_out = reinterpret_cast<fftw_complex*>(complex_out);

        // R2C forward plan: real → complex (N → N/2+1)
        r2c_plan_ = fftw_plan_many_dft_r2c(
            1, n, batch,
            real_in, NULL, istride, idist,
            fftw_out, NULL, ostride, odist,
            FFTW_ESTIMATE
        );
    }

    // Execute R2C plan
    void execute_r2c(double* real_in, std::complex<double>* complex_out) {
        fftw_execute_dft_r2c(r2c_plan_, real_in,
                             reinterpret_cast<fftw_complex*>(complex_out));
    }

    // Create C2R plan for last stage (complex input → real output)
    void create_c2r_plan(
        int length,      // Real-space output length N
        int batch,       // Number of transforms
        std::complex<double>* complex_in,   // Complex input (size N/2+1)
        double* real_out,                   // Real output (size N)
        int istride,     // Input stride
        int idist,       // Input distance between batches
        int ostride,     // Output stride
        int odist        // Output distance between batches
    ) {
        int n[] = {length};
        fftw_complex* fftw_in = reinterpret_cast<fftw_complex*>(complex_in);

        // C2R backward plan: complex → real (N/2+1 → N)
        c2r_plan_ = fftw_plan_many_dft_c2r(
            1, n, batch,
            fftw_in, NULL, istride, idist,
            real_out, NULL, ostride, odist,
            FFTW_ESTIMATE
        );
    }

    // Execute C2R plan
    void execute_c2r(std::complex<double>* complex_in, double* real_out) {
        fftw_execute_dft_c2r(c2r_plan_,
                             reinterpret_cast<fftw_complex*>(complex_in),
                             real_out);
    }

    // Execute pre-created plan for specified stage on given data
    void execute_stage(int stage, FFTDirection direction, std::complex<double>* data) {
        fftw_plan plan = (direction == FFTDirection::Forward)
                         ? forward_plans_[stage]
                         : backward_plans_[stage];
        fftw_complex* fftw_data = reinterpret_cast<fftw_complex*>(data);
        fftw_execute_dft(plan, fftw_data, fftw_data);
    }

    // Destructor: Clean up all plans
    ~FFTWBackend() {
        for (auto plan : forward_plans_) {
            if (plan) fftw_destroy_plan(plan);
        }
        for (auto plan : backward_plans_) {
            if (plan) fftw_destroy_plan(plan);
        }
        if (r2c_plan_) fftw_destroy_plan(r2c_plan_);
        if (c2r_plan_) fftw_destroy_plan(c2r_plan_);
    }

    // Disable copying (plans can't be safely copied)
    FFTWBackend(const FFTWBackend&) = delete;
    FFTWBackend& operator=(const FFTWBackend&) = delete;

    // Enable moving
    FFTWBackend(FFTWBackend&& other) noexcept
        : forward_plans_(std::move(other.forward_plans_)),
          backward_plans_(std::move(other.backward_plans_)),
          r2c_plan_(other.r2c_plan_),
          c2r_plan_(other.c2r_plan_) {
        std::fill(other.forward_plans_.begin(), other.forward_plans_.end(), nullptr);
        std::fill(other.backward_plans_.begin(), other.backward_plans_.end(), nullptr);
        other.r2c_plan_ = nullptr;
        other.c2r_plan_ = nullptr;
    }

private:
    std::vector<fftw_plan> forward_plans_;
    std::vector<fftw_plan> backward_plans_;
    fftw_plan r2c_plan_ = nullptr;
    fftw_plan c2r_plan_ = nullptr;
};

} // namespace mpifft

#endif // MPIFFT_BACKEND_FFTW_HPP
