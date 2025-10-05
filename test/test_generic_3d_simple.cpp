#include "../mpifft_generic.hpp"
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

    // Create 3D FFT object using generic implementation
    mpifft::PencilFFT<3> fft(global_shape);

    // Get local array size
    int local_size = fft.get_local_size();

    if (rank == 0) {
        std::cout << "\n=== Generic 3D FFT Test ===\n";
        std::cout << "Global shape: " << global_shape[0] << " × "
                  << global_shape[1] << " × " << global_shape[2] << "\n";
        std::cout << "MPI processes: " << size << "\n\n";
    }

    // Allocate local data
    std::vector<std::complex<double>> data(local_size);

    // Initialize with Gaussian
    int local_shape[3];
    fft.get_local_shape(local_shape);
    int global_start[3];
    fft.get_global_start(global_start);

    for (int i = 0; i < local_shape[0]; ++i) {
        for (int j = 0; j < local_shape[1]; ++j) {
            for (int k = 0; k < local_shape[2]; ++k) {
                int gi = global_start[0] + i;
                int gj = global_start[1] + j;
                int gk = global_start[2] + k;

                double x = (gi - global_shape[0]/2.0) / 8.0;
                double y = (gj - global_shape[1]/2.0) / 8.0;
                double z = (gk - global_shape[2]/2.0) / 8.0;

                int idx = i * local_shape[1] * local_shape[2] + j * local_shape[2] + k;
                data[idx] = std::exp(-(x*x + y*y + z*z) / 2.0);
            }
        }
    }

    // Save original data
    std::vector<std::complex<double>> original = data;

    if (rank == 0) {
        std::cout << "Performing forward FFT...\n";
    }

    // Forward FFT
    fft.forward(data.data());

    if (rank == 0) {
        std::cout << "Performing backward FFT...\n";
    }

    // Backward FFT
    fft.backward(data.data());

    // Normalize
    int total_size = global_shape[0] * global_shape[1] * global_shape[2];
    for (int i = 0; i < local_size; ++i) {
        data[i] /= total_size;
    }

    // Compute error
    double max_error = 0.0;
    for (int i = 0; i < local_size; ++i) {
        double err = std::abs(data[i] - original[i]);
        max_error = std::max(max_error, err);
    }

    // Global reduction
    double global_max_error;
    MPI_Reduce(&max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Verification ===\n";
        std::cout << "Maximum error: " << std::scientific << std::setprecision(2)
                  << global_max_error << "\n";

        if (global_max_error < 1e-10) {
            std::cout << "✓ Test PASSED\n";
        } else {
            std::cout << "✗ Test FAILED - Error too large\n";
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
