#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include "../../backend/cufft/fft_backend_cufft.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test 1: Basic R2C transform produces correct spectrum
void test_r2c_spectrum() {
    const int N = 64;
    const int batch = 1;
    const int padded_dist = 2 * (N / 2 + 1);

    std::vector<double> padded_real(padded_dist, 0.0);

    // Initialize with single frequency sine wave
    for (int i = 0; i < N; ++i) {
        padded_real[i] = std::sin(2.0 * M_PI * 4 * i / N);  // 4 cycles
    }

    parafaft::CuFFTBackend backend(1);
    backend.create_r2c_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);

    backend.execute_r2c_inplace(padded_real.data());

    // Check spectrum: should have peak at frequency bin 4
    auto* complex_data = reinterpret_cast<std::complex<double>*>(padded_real.data());

    double peak_magnitude = std::abs(complex_data[4]);
    double other_max = 0.0;
    for (int i = 0; i <= N / 2; ++i) {
        if (i != 4) {
            other_max = std::max(other_max, std::abs(complex_data[i]));
        }
    }

    std::cout << "Test R2C spectrum: peak[4]=" << peak_magnitude
              << ", other_max=" << other_max;
    if (peak_magnitude > 10.0 && other_max < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

// Test 2: R2C -> C2R roundtrip
void test_r2c_c2r_roundtrip() {
    const int N = 64;
    const int batch = 1;
    const int padded_dist = 2 * (N / 2 + 1);

    std::vector<double> padded_real(padded_dist, 0.0);

    // Initialize with Gaussian
    for (int i = 0; i < N; ++i) {
        double x = (i - N / 2.0) / 8.0;
        padded_real[i] = std::exp(-x * x / 2.0);
    }

    std::vector<double> original(padded_real.begin(), padded_real.begin() + N);

    parafaft::CuFFTBackend backend(1);
    backend.create_r2c_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);
    backend.create_c2r_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);

    // Forward R2C
    backend.execute_r2c_inplace(padded_real.data());

    // Backward C2R
    backend.execute_c2r_inplace(padded_real.data());

    // Normalize
    for (int i = 0; i < N; ++i) {
        padded_real[i] /= N;
    }

    // Check error
    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        max_error = std::max(max_error, std::abs(padded_real[i] - original[i]));
    }

    std::cout << "Test R2C/C2R roundtrip: max_error = " << max_error;
    if (max_error < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

// Test 3: Batched R2C roundtrip
void test_batched_r2c_roundtrip() {
    const int N = 32;
    const int batch = 4;
    const int padded_dist = 2 * (N / 2 + 1);

    std::vector<double> padded_real(batch * padded_dist, 0.0);

    // Initialize each batch with different data
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < N; ++i) {
            padded_real[b * padded_dist + i] =
                std::sin(2.0 * M_PI * (b + 1) * i / N);
        }
    }

    std::vector<double> original = padded_real;

    parafaft::CuFFTBackend backend(1);
    backend.create_r2c_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);
    backend.create_c2r_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);

    backend.execute_r2c_inplace(padded_real.data());
    backend.execute_c2r_inplace(padded_real.data());

    // Normalize
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < N; ++i) {
            padded_real[b * padded_dist + i] /= N;
        }
    }

    // Check error (only in unpadded regions)
    double max_error = 0.0;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error,
                std::abs(padded_real[b * padded_dist + i] -
                         original[b * padded_dist + i]));
        }
    }

    std::cout << "Test batched R2C roundtrip: max_error = " << max_error;
    if (max_error < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

// Test 4: Large transform size
void test_large_r2c() {
    const int N = 1024;
    const int batch = 1;
    const int padded_dist = 2 * (N / 2 + 1);

    std::vector<double> padded_real(padded_dist, 0.0);

    for (int i = 0; i < N; ++i) {
        padded_real[i] = static_cast<double>(i) / N;
    }
    std::vector<double> original(padded_real.begin(), padded_real.begin() + N);

    parafaft::CuFFTBackend backend(1);
    backend.create_r2c_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);
    backend.create_c2r_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);

    backend.execute_r2c_inplace(padded_real.data());
    backend.execute_c2r_inplace(padded_real.data());

    for (int i = 0; i < N; ++i) {
        padded_real[i] /= N;
    }

    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        max_error = std::max(max_error, std::abs(padded_real[i] - original[i]));
    }

    std::cout << "Test large R2C (N=1024): max_error = " << max_error;
    if (max_error < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

// Test 5: Compare with FFTW results
void test_fftw_comparison() {
    std::cout << "Test FFTW comparison: [SKIPPED - requires dual linking]" << std::endl;
}

// Test 6: Move constructor
void test_move_constructor() {
    const int N = 32;
    const int batch = 1;
    const int padded_dist = 2 * (N / 2 + 1);

    std::vector<double> padded_real(padded_dist, 0.0);
    for (int i = 0; i < N; ++i) {
        padded_real[i] = std::cos(2.0 * M_PI * i / N);
    }
    std::vector<double> original(padded_real.begin(), padded_real.begin() + N);

    parafaft::CuFFTBackend backend1(1);
    backend1.create_r2c_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);
    backend1.create_c2r_inplace_plan(N, batch, padded_real.data(), 1, padded_dist);

    // Move to new backend
    parafaft::CuFFTBackend backend2(std::move(backend1));

    // Execute on moved backend
    backend2.execute_r2c_inplace(padded_real.data());
    backend2.execute_c2r_inplace(padded_real.data());

    for (int i = 0; i < N; ++i) {
        padded_real[i] /= N;
    }

    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        max_error = std::max(max_error, std::abs(padded_real[i] - original[i]));
    }

    std::cout << "Test move constructor: max_error = " << max_error;
    if (max_error < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

int main() {
    std::cout << "=== CuFFT Backend R2C/C2R Unit Tests ===" << std::endl;

    test_r2c_spectrum();
    test_r2c_c2r_roundtrip();
    test_batched_r2c_roundtrip();
    test_large_r2c();
    test_fftw_comparison();
    test_move_constructor();

    return 0;
}
