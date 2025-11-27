#include "../mpifft_generic.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Array dimensions - 256^3
    const int N0 = 256;
    const int N1 = 256;
    const int N2 = 256;
    int global_shape[3] = {N0, N1, N2};
    const int total_size = N0 * N1 * N2;

    if (rank == 0) {
        std::cout << "=== MPI Parallel FFT Benchmark ===\n";
        std::cout << "Array size: " << N0 << " x " << N1 << " x " << N2 << "\n";
        std::cout << "Total elements: " << total_size << "\n";
        std::cout << "Memory per rank: ~" << (total_size * sizeof(std::complex<double>) / (nranks * 1024.0*1024.0)) << " MB\n";
        std::cout << "MPI ranks: " << nranks << "\n\n";
    }

    // Gaussian parameters
    const double center0 = N0 / 2.0;
    const double center1 = N1 / 2.0;
    const double center2 = N2 / 2.0;
    const double sigma = 16.0;  // Scaled for larger array

    // Create FFT object
    if (rank == 0) {
        std::cout << "Creating FFT object and setting up MPI topology...\n";
    }
    auto t_setup_start = std::chrono::high_resolution_clock::now();

    mpifft::PencilFFT<3> fft(global_shape);

    auto t_setup_end = std::chrono::high_resolution_clock::now();
    double setup_time = std::chrono::duration<double>(t_setup_end - t_setup_start).count();

    // Get local dimensions
    int local_size = fft.get_local_size();
    int local_shape[3];
    int global_start[3];
    fft.get_local_shape(local_shape);
    fft.get_global_start(global_start);

    if (rank == 0) {
        std::cout << "Setup time: " << setup_time << " seconds\n";
        std::cout << "Local shape on rank 0: [" << local_shape[0] << ", "
                  << local_shape[1] << ", " << local_shape[2] << "]\n\n";
    }

    // Allocate local data
    std::vector<std::complex<double>> data(local_size);

    // Initialize with 3D Gaussian using global coordinates
    if (rank == 0) {
        std::cout << "Initializing with 3D Gaussian (sigma=" << sigma << ")...\n";
    }
    auto t_init_start = std::chrono::high_resolution_clock::now();

    for (int i0 = 0; i0 < local_shape[0]; ++i0) {
        for (int i1 = 0; i1 < local_shape[1]; ++i1) {
            for (int i2 = 0; i2 < local_shape[2]; ++i2) {
                double g0 = global_start[0] + i0;
                double g1 = global_start[1] + i1;
                double g2 = global_start[2] + i2;

                double x = g0 - center0;
                double y = g1 - center1;
                double z = g2 - center2;
                double r2 = x*x + y*y + z*z;
                double value = std::exp(-r2 / (2.0 * sigma * sigma));

                int idx = (i0 * local_shape[1] + i1) * local_shape[2] + i2;
                data[idx] = std::complex<double>(value, 0.0);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_time = std::chrono::duration<double>(t_init_end - t_init_start).count();

    if (rank == 0) {
        std::cout << "Initialization time: " << init_time << " seconds\n\n";
    }

    // Save original
    std::vector<std::complex<double>> original = data;

    // Forward FFT
    if (rank == 0) {
        std::cout << "Executing forward FFT...\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_forward_start = std::chrono::high_resolution_clock::now();

    fft.forward(data.data());

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_forward_end = std::chrono::high_resolution_clock::now();
    double forward_time = std::chrono::duration<double>(t_forward_end - t_forward_start).count();

    if (rank == 0) {
        std::cout << "Forward FFT time: " << forward_time << " seconds\n";
        std::cout << "DC component: " << data[0] << "\n\n";
    }

    // Backward FFT
    if (rank == 0) {
        std::cout << "Executing backward FFT...\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_backward_start = std::chrono::high_resolution_clock::now();

    fft.backward(data.data());

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_backward_end = std::chrono::high_resolution_clock::now();
    double backward_time = std::chrono::duration<double>(t_backward_end - t_backward_start).count();

    if (rank == 0) {
        std::cout << "Backward FFT time: " << backward_time << " seconds\n\n";
    }

    // Normalize
    if (rank == 0) {
        std::cout << "Normalizing...\n";
    }
    double scale = 1.0 / total_size;
    for (int i = 0; i < local_size; ++i) {
        data[i] *= scale;
    }

    // Compute local error
    if (rank == 0) {
        std::cout << "Computing roundtrip error...\n";
    }
    double local_max_error = 0.0;
    double local_sum_error = 0.0;
    for (int i = 0; i < local_size; ++i) {
        double err = std::abs(data[i] - original[i]);
        local_max_error = std::max(local_max_error, err);
        local_sum_error += err;
    }

    // Reduce to get global errors
    double global_max_error, global_sum_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_error, &global_sum_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Summary on rank 0
    if (rank == 0) {
        double avg_error = global_sum_error / total_size;

        std::cout << "\n=== RESULTS ===\n";
        std::cout << "Setup time:     " << std::setw(10) << std::fixed << std::setprecision(6) << setup_time << " seconds\n";
        std::cout << "Forward FFT:    " << std::setw(10) << std::fixed << std::setprecision(6) << forward_time << " seconds\n";
        std::cout << "Backward FFT:   " << std::setw(10) << std::fixed << std::setprecision(6) << backward_time << " seconds\n";
        std::cout << "Total FFT time: " << std::setw(10) << std::fixed << std::setprecision(6) << (forward_time + backward_time) << " seconds\n\n";

        std::cout << "Roundtrip accuracy:\n";
        std::cout << "  Maximum error: " << std::scientific << global_max_error << "\n";
        std::cout << "  Average error: " << std::scientific << avg_error << "\n";

        if (global_max_error < 1e-10) {
            std::cout << "  ✓ PASSED - Roundtrip accurate\n";
        } else {
            std::cout << "  ✗ FAILED - Roundtrip has errors\n";
        }
    }

    MPI_Finalize();
    return 0;
}
