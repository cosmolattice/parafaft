#include <fftw3.h>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    // Array dimensions - 256^3
    const int N0 = 256;
    const int N1 = 256;
    const int N2 = 256;
    const int total_size = N0 * N1 * N2;

    std::cout << "=== Serial FFTW Benchmark ===\n";
    std::cout << "Array size: " << N0 << " x " << N1 << " x " << N2 << "\n";
    std::cout << "Total elements: " << total_size << "\n";
    std::cout << "Memory: " << (total_size * sizeof(std::complex<double>) / (1024.0*1024.0)) << " MB\n\n";

    // Allocate array
    std::vector<std::complex<double>> data(total_size);

    // Gaussian parameters
    const double center0 = N0 / 2.0;
    const double center1 = N1 / 2.0;
    const double center2 = N2 / 2.0;
    const double sigma = 16.0;  // Scaled for larger array

    std::cout << "Initializing with 3D Gaussian (sigma=" << sigma << ")...\n";
    auto t_init_start = std::chrono::high_resolution_clock::now();

    // Initialize with 3D Gaussian
    for (int i0 = 0; i0 < N0; ++i0) {
        for (int i1 = 0; i1 < N1; ++i1) {
            for (int i2 = 0; i2 < N2; ++i2) {
                double x = i0 - center0;
                double y = i1 - center1;
                double z = i2 - center2;
                double r2 = x*x + y*y + z*z;
                double value = std::exp(-r2 / (2.0 * sigma * sigma));

                int idx = (i0 * N1 + i1) * N2 + i2;
                data[idx] = std::complex<double>(value, 0.0);
            }
        }
    }

    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_time = std::chrono::duration<double>(t_init_end - t_init_start).count();
    std::cout << "Initialization time: " << init_time << " seconds\n\n";

    // Save copy of original for comparison
    std::vector<std::complex<double>> original = data;

    // Create FFTW plan for 3D transform
    std::cout << "Creating FFTW plans...\n";
    auto t_plan_start = std::chrono::high_resolution_clock::now();

    fftw_plan plan_forward = fftw_plan_dft_3d(
        N0, N1, N2,
        reinterpret_cast<fftw_complex*>(data.data()),
        reinterpret_cast<fftw_complex*>(data.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_plan plan_backward = fftw_plan_dft_3d(
        N0, N1, N2,
        reinterpret_cast<fftw_complex*>(data.data()),
        reinterpret_cast<fftw_complex*>(data.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);

    auto t_plan_end = std::chrono::high_resolution_clock::now();
    double plan_time = std::chrono::duration<double>(t_plan_end - t_plan_start).count();
    std::cout << "Plan creation time: " << plan_time << " seconds\n\n";

    // Forward FFT
    std::cout << "Executing forward FFT...\n";
    auto t_forward_start = std::chrono::high_resolution_clock::now();
    fftw_execute(plan_forward);
    auto t_forward_end = std::chrono::high_resolution_clock::now();
    double forward_time = std::chrono::duration<double>(t_forward_end - t_forward_start).count();

    std::cout << "Forward FFT time: " << forward_time << " seconds\n";
    std::cout << "DC component: " << data[0] << "\n\n";

    // Backward FFT
    std::cout << "Executing backward FFT...\n";
    auto t_backward_start = std::chrono::high_resolution_clock::now();
    fftw_execute(plan_backward);
    auto t_backward_end = std::chrono::high_resolution_clock::now();
    double backward_time = std::chrono::duration<double>(t_backward_end - t_backward_start).count();

    std::cout << "Backward FFT time: " << backward_time << " seconds\n\n";

    // Normalize (FFTW doesn't normalize)
    std::cout << "Normalizing...\n";
    for (int i = 0; i < total_size; ++i) {
        data[i] /= total_size;
    }

    // Compute error
    std::cout << "Computing roundtrip error...\n";
    double max_error = 0.0;
    double avg_error = 0.0;
    for (int i = 0; i < total_size; ++i) {
        double err = std::abs(data[i] - original[i]);
        max_error = std::max(max_error, err);
        avg_error += err;
    }
    avg_error /= total_size;

    // Summary
    std::cout << "\n=== RESULTS ===\n";
    std::cout << "Plan creation:  " << std::setw(10) << std::fixed << std::setprecision(6) << plan_time << " seconds\n";
    std::cout << "Forward FFT:    " << std::setw(10) << std::fixed << std::setprecision(6) << forward_time << " seconds\n";
    std::cout << "Backward FFT:   " << std::setw(10) << std::fixed << std::setprecision(6) << backward_time << " seconds\n";
    std::cout << "Total FFT time: " << std::setw(10) << std::fixed << std::setprecision(6) << (forward_time + backward_time) << " seconds\n\n";

    std::cout << "Roundtrip accuracy:\n";
    std::cout << "  Maximum error: " << std::scientific << max_error << "\n";
    std::cout << "  Average error: " << std::scientific << avg_error << "\n";

    if (max_error < 1e-10) {
        std::cout << "  ✓ PASSED - Roundtrip accurate\n";
    } else {
        std::cout << "  ✗ FAILED - Roundtrip has errors\n";
    }

    // Cleanup
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);

    return 0;
}
