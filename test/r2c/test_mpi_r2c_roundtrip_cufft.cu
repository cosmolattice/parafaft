// MPI Integration Test for cuFFT Backend with R2C Transforms
//
// This test validates that the CuFFTBackend produces correct results when
// used with the ParaFaFT_R2C distributed FFT library.

#include "../../parafaft_r2c.hpp"
#include "../../backend/cufft/fft_backend_cufft.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Test parameters
    int N = 32;  // Default
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    const int N0 = N, N1 = N, N2 = N;
    const int complex_N2 = N2 / 2 + 1;
    int global_shape[3] = {N0, N1, N2};
    const long long total_real_size = (long long)N0 * N1 * N2;

    const double center0 = N0 / 2.0;
    const double center1 = N1 / 2.0;
    const double center2 = N2 / 2.0;
    const double sigma = 4.0;

    // Create R2C FFT object with CuFFT backend
    parafaft::ParaFaFT_R2C<3, parafaft::CuFFTBackend> fft(global_shape);

    // Get local dimensions
    int local_real_shape[3], real_start[3];
    int local_complex_shape[3], complex_start[3];
    fft.get_local_real_shape(local_real_shape);
    fft.get_real_global_start(real_start);
    fft.get_local_complex_shape(local_complex_shape);
    fft.get_complex_global_start(complex_start);

    int local_real_size = fft.get_local_real_size();
    int local_padded_size = fft.get_local_in_place_buffer_size();
    int local_complex_size = fft.get_local_complex_size();

    // Verify size relationship
    if (rank == 0) {
        std::cout << "=== cuFFT R2C MPI Roundtrip Test ===" << std::endl;
        std::cout << "Grid size: " << N0 << "x" << N1 << "x" << N2 << std::endl;
        std::cout << "MPI ranks: " << size << std::endl;
        std::cout << "local_real_size = " << local_real_size << "\n";
        std::cout << "local_padded_size = " << local_padded_size << " (doubles)\n";
        std::cout << "local_complex_size = " << local_complex_size << " (complex)\n";
    }

    // Allocate single padded buffer for in-place operations
    std::vector<double> padded_buffer(local_padded_size);

    // Also keep unpadded copy for comparison
    std::vector<double> original_data(local_real_size);

    // Initialize local portion of Gaussian (only first N2 elements per row)
    int padded_stride = 2 * complex_N2;  // Padded row length
    int idx = 0;
    for (int i0 = 0; i0 < local_real_shape[0]; ++i0) {
        for (int i1 = 0; i1 < local_real_shape[1]; ++i1) {
            for (int i2 = 0; i2 < N2; ++i2) {
                double g0 = real_start[0] + i0;
                double g1 = real_start[1] + i1;
                double g2 = real_start[2] + i2;
                double x = g0 - center0;
                double y = g1 - center1;
                double z = g2 - center2;
                double r2 = x*x + y*y + z*z;
                double value = std::exp(-r2 / (2.0 * sigma * sigma));

                // Store in padded buffer
                int padded_idx = (i0 * local_real_shape[1] + i1) * padded_stride + i2;
                padded_buffer[padded_idx] = value;

                // Store in unpadded copy
                original_data[idx++] = value;
            }
        }
    }

    // Forward R2C FFT (in-place: real -> complex)
    fft.forward_in_place(padded_buffer.data());

    // Backward C2R FFT (in-place: complex -> real)
    fft.backward_in_place(padded_buffer.data());

    // Normalize (FFTW/cuFFT convention: forward*backward = N * original)
    double scale = 1.0 / total_real_size;
    for (int i = 0; i < local_padded_size; ++i) {
        padded_buffer[i] *= scale;
    }

    // Compute local error (only on non-padded elements)
    double local_max_error = 0.0;
    idx = 0;
    for (int i0 = 0; i0 < local_real_shape[0]; ++i0) {
        for (int i1 = 0; i1 < local_real_shape[1]; ++i1) {
            for (int i2 = 0; i2 < N2; ++i2) {
                int padded_idx = (i0 * local_real_shape[1] + i1) * padded_stride + i2;
                double err = std::abs(padded_buffer[padded_idx] - original_data[idx]);
                local_max_error = std::max(local_max_error, err);
                idx++;
            }
        }
    }

    // Gather global max error
    double global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int test_passed = 1;
    if (rank == 0) {
        std::cout << "Maximum roundtrip error: " << std::scientific << global_max_error << "\n";

        if (global_max_error < 1e-10) {
            std::cout << "PASSED\n";
            test_passed = 1;
        } else {
            std::cout << "FAILED\n";
            test_passed = 0;
        }
    }

    // Broadcast result
    MPI_Bcast(&test_passed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return test_passed ? 0 : 1;
}
