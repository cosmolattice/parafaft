#include "../mpifft_r2c.hpp"
#include <fftw3.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdlib>

// Generate serial FFTW R2C reference result
std::vector<std::complex<double>> generate_fftw_r2c_reference(
    int N0, int N1, int N2,
    double center0, double center1, double center2, double sigma)
{
    int complex_N2 = N2 / 2 + 1;
    int real_size = N0 * N1 * N2;
    int complex_size = N0 * N1 * complex_N2;

    std::vector<double> real_data(real_size);
    std::vector<std::complex<double>> complex_data(complex_size);

    // Initialize Gaussian
    for (int i0 = 0; i0 < N0; ++i0) {
        for (int i1 = 0; i1 < N1; ++i1) {
            for (int i2 = 0; i2 < N2; ++i2) {
                double x = i0 - center0;
                double y = i1 - center1;
                double z = i2 - center2;
                double r2 = x*x + y*y + z*z;
                int idx = (i0 * N1 + i1) * N2 + i2;
                real_data[idx] = std::exp(-r2 / (2.0 * sigma * sigma));
            }
        }
    }

    // Execute FFTW R2C
    fftw_plan plan = fftw_plan_dft_r2c_3d(
        N0, N1, N2,
        real_data.data(),
        reinterpret_cast<fftw_complex*>(complex_data.data()),
        FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    return complex_data;
}

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
        std::cout << "R2C FFT Test: " << N0 << "x" << N1 << "x" << N2
                  << " with " << size << " processes\n";
        std::cout << "Complex output shape: " << N0 << "x" << N1 << "x" << complex_N2 << "\n";
    }

    // Perform parallel R2C FFT
    fft.forward(real_data.data(), complex_data.data());

    // Gather shape and start info from all ranks for reconstruction
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    std::vector<int> all_shapes(nranks * 3);
    std::vector<int> all_starts(nranks * 3);
    int shape_data[3] = {local_complex_shape[0], local_complex_shape[1], local_complex_shape[2]};
    int start_data[3] = {complex_start[0], complex_start[1], complex_start[2]};

    MPI_Gather(shape_data, 3, MPI_INT, all_shapes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(start_data, 3, MPI_INT, all_starts.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather data sizes
    std::vector<int> sizes(nranks);
    std::vector<int> displs(nranks);
    MPI_Gather(&local_complex_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nranks; ++i) {
            displs[i] = displs[i-1] + sizes[i-1];
        }
    }

    // Gather all complex data to rank 0
    std::vector<std::complex<double>> gathered_data;
    if (rank == 0) {
        gathered_data.resize(displs[nranks-1] + sizes[nranks-1]);
    }

    MPI_Gatherv(complex_data.data(), local_complex_size, MPI_C_DOUBLE_COMPLEX,
                gathered_data.data(), sizes.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

    // On rank 0: reconstruct global array and compare with reference
    int test_passed = 1;
    if (rank == 0) {
        // Reconstruct global complex array
        int total_complex_size = N0 * N1 * complex_N2;
        std::vector<std::complex<double>> global_array(total_complex_size);

        int offset = 0;
        for (int r = 0; r < nranks; ++r) {
            int shape0 = all_shapes[r*3 + 0];
            int shape1 = all_shapes[r*3 + 1];
            int shape2 = all_shapes[r*3 + 2];
            int start0 = all_starts[r*3 + 0];
            int start1 = all_starts[r*3 + 1];
            int start2 = all_starts[r*3 + 2];

            for (int i0 = 0; i0 < shape0; ++i0) {
                for (int i1 = 0; i1 < shape1; ++i1) {
                    for (int i2 = 0; i2 < shape2; ++i2) {
                        int local_idx = (i0 * shape1 + i1) * shape2 + i2;
                        int global_i0 = start0 + i0;
                        int global_i1 = start1 + i1;
                        int global_i2 = start2 + i2;
                        int global_idx = (global_i0 * N1 + global_i1) * complex_N2 + global_i2;
                        global_array[global_idx] = gathered_data[offset + local_idx];
                    }
                }
            }
            offset += sizes[r];
        }

        // Generate reference
        auto fftw_reference = generate_fftw_r2c_reference(
            N0, N1, N2, center0, center1, center2, sigma);

        // Compare element-wise
        double max_error = 0.0;
        int error_count = 0;
        for (int i = 0; i < total_complex_size; ++i) {
            double err = std::abs(global_array[i] - fftw_reference[i]);
            max_error = std::max(max_error, err);
            if (err > 1e-10) {
                if (error_count < 5) {
                    std::cout << "Error at index " << i << ": parallel="
                              << global_array[i] << " reference=" << fftw_reference[i]
                              << " diff=" << err << "\n";
                }
                error_count++;
            }
        }

        std::cout << "\nDC component (parallel): " << global_array[0] << "\n";
        std::cout << "DC component (reference): " << fftw_reference[0] << "\n";
        std::cout << "Maximum error: " << std::scientific << max_error << "\n";

        if (max_error < 1e-10) {
            std::cout << "PASSED\n";
            test_passed = 1;
        } else {
            std::cout << "FAILED (error count: " << error_count << ")\n";
            test_passed = 0;
        }
    }

    // Broadcast result to all ranks
    MPI_Bcast(&test_passed, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return test_passed ? 0 : 1;
}
