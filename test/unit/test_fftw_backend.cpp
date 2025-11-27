#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include "../../backend/fftw3/fft_backend_fftw.hpp"

// Test 1: Contiguous data layout (stride=1, dist=N)
void test_contiguous_layout() {
    const int N = 256;
    const int batch = 3;

    std::vector<std::complex<double>> data(N * batch);

    // Initialize with Gaussian
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < N; ++i) {
            double x = (i - N/2) / 16.0;
            data[b * N + i] = std::exp(-x * x / 2.0);
        }
    }
    std::vector<std::complex<double>> original = data;

    // Create backend and plan
    mpifft::FFTWBackend backend(1);
    backend.create_stage_plan(0, N, batch, data.data(), 1, N);

    // Roundtrip
    backend.execute_stage(0, mpifft::FFTDirection::Forward, data.data());
    backend.execute_stage(0, mpifft::FFTDirection::Backward, data.data());

    // Normalize
    for (auto& val : data) {
        val /= N;
    }

    // Check error
    double max_error = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        max_error = std::max(max_error, std::abs(data[i] - original[i]));
    }

    std::cout << "Test contiguous: max_error = " << max_error;
    if (max_error < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

// Test 2: Strided data layout (stride=batch, dist=1)
void test_strided_layout() {
    const int N = 128;
    const int batch = 4;

    std::vector<std::complex<double>> data(N * batch);

    // Initialize interleaved
    for (int i = 0; i < N; ++i) {
        for (int b = 0; b < batch; ++b) {
            double phase = 2.0 * M_PI * b / batch;
            data[i * batch + b] = std::complex<double>(std::cos(phase), std::sin(phase));
        }
    }
    std::vector<std::complex<double>> original = data;

    // Create backend and plan
    mpifft::FFTWBackend backend(1);
    backend.create_stage_plan(0, N, batch, data.data(), batch, 1);

    // Roundtrip
    backend.execute_stage(0, mpifft::FFTDirection::Forward, data.data());
    backend.execute_stage(0, mpifft::FFTDirection::Backward, data.data());

    // Normalize
    for (auto& val : data) {
        val /= N;
    }

    // Check error
    double max_error = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        max_error = std::max(max_error, std::abs(data[i] - original[i]));
    }

    std::cout << "Test strided: max_error = " << max_error;
    if (max_error < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

// Test 3: Multiple stages with plan reuse
void test_multiple_stages() {
    const int N = 64;
    const int num_stages = 3;

    std::vector<std::vector<std::complex<double>>> stage_data(num_stages);
    std::vector<std::vector<std::complex<double>>> original_data(num_stages);

    for (int stage = 0; stage < num_stages; ++stage) {
        stage_data[stage].resize(N);
        for (int i = 0; i < N; ++i) {
            stage_data[stage][i] = std::complex<double>(i + stage * 100, 0.0);
        }
        original_data[stage] = stage_data[stage];
    }

    // Create backend with 3 stages
    mpifft::FFTWBackend backend(num_stages);
    for (int stage = 0; stage < num_stages; ++stage) {
        backend.create_stage_plan(stage, N, 1, stage_data[stage].data(), 1, N);
    }

    // Execute multiple times (test plan reuse)
    double max_error = 0.0;
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (int stage = 0; stage < num_stages; ++stage) {
            backend.execute_stage(stage, mpifft::FFTDirection::Forward, stage_data[stage].data());
            backend.execute_stage(stage, mpifft::FFTDirection::Backward, stage_data[stage].data());
            for (auto& val : stage_data[stage]) {
                val /= N;
            }

            // Check error after each roundtrip
            for (size_t i = 0; i < stage_data[stage].size(); ++i) {
                max_error = std::max(max_error, std::abs(stage_data[stage][i] - original_data[stage][i]));
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
void test_different_pointers() {
    const int N = 64;
    const int batch = 2;

    // Create three different data arrays
    std::vector<std::complex<double>> data1(N * batch);
    std::vector<std::complex<double>> data2(N * batch);
    std::vector<std::complex<double>> data3(N * batch);

    // Initialize with different patterns
    for (int i = 0; i < N * batch; ++i) {
        data1[i] = std::complex<double>(i % 10, 0.0);
        data2[i] = std::complex<double>(0.0, i % 5);
        data3[i] = std::exp(-static_cast<double>((i - N*batch/2) * (i - N*batch/2)) / (2.0 * 100.0));
    }
    auto original1 = data1;
    auto original2 = data2;
    auto original3 = data3;

    // Create single backend and plan (using data1 for plan creation)
    mpifft::FFTWBackend backend(1);
    backend.create_stage_plan(0, N, batch, data1.data(), 1, N);

    // Apply the same plan to all three different arrays
    backend.execute_stage(0, mpifft::FFTDirection::Forward, data1.data());
    backend.execute_stage(0, mpifft::FFTDirection::Backward, data1.data());
    for (auto& val : data1) val /= N;

    backend.execute_stage(0, mpifft::FFTDirection::Forward, data2.data());
    backend.execute_stage(0, mpifft::FFTDirection::Backward, data2.data());
    for (auto& val : data2) val /= N;

    backend.execute_stage(0, mpifft::FFTDirection::Forward, data3.data());
    backend.execute_stage(0, mpifft::FFTDirection::Backward, data3.data());
    for (auto& val : data3) val /= N;

    // Check all arrays recovered correctly
    double max_error = 0.0;
    for (size_t i = 0; i < data1.size(); ++i) {
        max_error = std::max(max_error, std::abs(data1[i] - original1[i]));
        max_error = std::max(max_error, std::abs(data2[i] - original2[i]));
        max_error = std::max(max_error, std::abs(data3[i] - original3[i]));
    }

    std::cout << "Test different pointers: max_error = " << max_error;
    if (max_error < 1e-10) {
        std::cout << " [PASS]" << std::endl;
    } else {
        std::cout << " [FAIL]" << std::endl;
    }
}

int main() {
    test_contiguous_layout();
    test_strided_layout();
    test_multiple_stages();
    test_different_pointers();
    return 0;
}
