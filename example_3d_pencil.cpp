#include "mpifft_generic.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define global 3D array size
    int global_shape[3] = {32, 32, 32};

    // Create 3D FFT object (use a scope to ensure cleanup before MPI_Finalize)
    mpifft::PencilFFT<3> fft(global_shape);

    // Get local array size and shape
    int local_size = fft.get_local_size();
    int local_shape[3];
    fft.get_local_shape(local_shape);

    // Note: generic implementation doesn't expose get_proc_grid()
    // int proc_grid[2];
    // fft.get_proc_grid(proc_grid);

    if (rank == 0) {
        std::cout << "\n=== 3D Pencil FFT Example ===\n";
        std::cout << "Global shape: " << global_shape[0] << " × "
                  << global_shape[1] << " × " << global_shape[2] << "\n";
        std::cout << "MPI processes: " << size << "\n";
        // std::cout << "Processor grid: " << proc_grid[0] << " × " << proc_grid[1] << "\n";
        std::cout << "Local shape (on rank 0): " << local_shape[0] << " × "
                  << local_shape[1] << " × " << local_shape[2] << "\n\n";
    }

    // Allocate local data
    std::vector<std::complex<double>> data(local_size);

    // Initialize with a simple pattern (constant array)
    for (int i = 0; i < local_size; ++i) {
        data[i] = std::complex<double>(1.0, 0.0);
    }

    // Save original data for comparison
    std::vector<std::complex<double>> original = data;

    if (rank == 0) {
        std::cout << "Performing forward FFT...\n";
    }

    // Forward FFT
    double t_start = MPI_Wtime();
    fft.forward(data.data());
    double t_forward = MPI_Wtime() - t_start;

    if (rank == 0) {
        std::cout << "Forward FFT completed in " << std::fixed << std::setprecision(4)
                  << t_forward << " seconds\n";
        std::cout << "Performing backward FFT...\n";
    }

    // Backward FFT
    t_start = MPI_Wtime();
    fft.backward(data.data());
    double t_backward = MPI_Wtime() - t_start;

    if (rank == 0) {
        std::cout << "Backward FFT completed in " << std::fixed << std::setprecision(4)
                  << t_backward << " seconds\n\n";
    }

    // Normalize (FFTW doesn't normalize, so divide by N)
    int total_size = global_shape[0] * global_shape[1] * global_shape[2];
    for (int i = 0; i < local_size; ++i) {
        data[i] /= total_size;
    }

    // Compute error (should recover original data after normalization)
    double max_error = 0.0;
    for (int i = 0; i < local_size; ++i) {
        double err_real = std::abs(data[i].real() - original[i].real());
        double err_imag = std::abs(data[i].imag() - original[i].imag());
        max_error = std::max(max_error, std::max(err_real, err_imag));
    }

    // Global reduction to find maximum error across all processes
    double global_max_error;
    MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "=== Verification ===\n";
        std::cout << "Maximum error: " << std::scientific << std::setprecision(2)
                  << global_max_error << "\n";

        if (global_max_error < 1e-10) {
            std::cout << "✓ Test PASSED - Forward + Backward recovers original data\n";
        } else {
            std::cout << "✗ Test FAILED - Error too large\n";
        }
    }

    // Note: generic implementation uses RAII, no manual cleanup needed
    // fft.cleanup();

    MPI_Finalize();
    return 0;
}
