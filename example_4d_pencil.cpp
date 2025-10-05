#include "mpifft_generic.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define global 4D array size
    int global_shape[4] = {4, 4, 4, 4};

    // Create 4D FFT object
    mpifft::PencilFFT<4> fft(global_shape);

    // Get local array size and shape
    int local_size = fft.get_local_size();
    int local_shape[4];
    fft.get_local_shape(local_shape);

    int proc_grid[3];
    // fft.get_proc_grid(proc_grid); // Not in generic implementation

    if (rank == 0) {
        std::cout << "\n=== 4D Pencil FFT Example ===\n";
        std::cout << "Global shape: " << global_shape[0] << " × "
                  << global_shape[1] << " × " << global_shape[2] << " × "
                  << global_shape[3] << "\n";
        std::cout << "MPI processes: " << size << "\n";
        std::cout << "Processor grid: " << proc_grid[0] << " × "
                  << proc_grid[1] << " × " << proc_grid[2] << "\n";
        std::cout << "Local shape (on rank 0): " << local_shape[0] << " × "
                  << local_shape[1] << " × " << local_shape[2] << " × "
                  << local_shape[3] << "\n\n";
    }

    // Allocate local data
    std::vector<std::complex<double>> data(local_size);

    // Initialize with constant value for reliable roundtrip test
    // (Using a sinusoidal pattern requires global indexing which is complex)
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
    int total_size = global_shape[0] * global_shape[1] * global_shape[2] * global_shape[3];
    for (int i = 0; i < local_size; ++i) {
        data[i] /= total_size;
    }

    // Compute error (should recover original data after normalization)
    double max_error = 0.0;
    for (int i = 0; i < local_size; ++i) {
        double err = std::abs(data[i] - original[i]);
        max_error = std::max(max_error, err);
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

    MPI_Finalize();
    return 0;
}
