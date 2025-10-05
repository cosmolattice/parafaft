#include "../mpifft_r2c.hpp"
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

    // Create R2C FFT object
    mpifft::PencilFFT_R2C<3> fft(global_shape);

    // Get local dimensions
    int local_real_shape[3], real_start[3];
    int local_complex_shape[3], complex_start[3];
    fft.get_local_real_shape(local_real_shape);
    fft.get_real_global_start(real_start);
    fft.get_local_complex_shape(local_complex_shape);
    fft.get_complex_global_start(complex_start);

    int local_real_size = fft.get_local_real_size();
    int local_complex_size = fft.get_local_complex_size();

    // Allocate local arrays
    std::vector<double> real_data(local_real_size);
    std::vector<double> real_result(local_real_size);
    std::vector<std::complex<double>> complex_data(local_complex_size);

    // Initialize local portion of Gaussian
    int idx = 0;
    for (int i0 = 0; i0 < local_real_shape[0]; ++i0) {
        for (int i1 = 0; i1 < local_real_shape[1]; ++i1) {
            for (int i2 = 0; i2 < local_real_shape[2]; ++i2) {
                double g0 = real_start[0] + i0;
                double g1 = real_start[1] + i1;
                double g2 = real_start[2] + i2;
                double x = g0 - center0;
                double y = g1 - center1;
                double z = g2 - center2;
                double r2 = x*x + y*y + z*z;
                real_data[idx++] = std::exp(-r2 / (2.0 * sigma * sigma));
            }
        }
    }

    if (rank == 0) {
        std::cout << "R2C Roundtrip Test: " << N0 << "x" << N1 << "x" << N2
                  << " with " << size << " processes\n";
    }

    // Forward R2C FFT
    fft.forward(real_data.data(), complex_data.data());

    // Backward C2R FFT
    fft.backward(complex_data.data(), real_result.data());

    // Normalize (FFTW convention: forward*backward = N * original)
    double scale = 1.0 / total_real_size;
    for (int i = 0; i < local_real_size; ++i) {
        real_result[i] *= scale;
    }

    // Compute local error
    double local_max_error = 0.0;
    for (int i = 0; i < local_real_size; ++i) {
        double err = std::abs(real_result[i] - real_data[i]);
        local_max_error = std::max(local_max_error, err);
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
