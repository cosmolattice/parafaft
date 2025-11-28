#include "../../parafaft_generic.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Same parameters as serial reference
    const int N0 = 32;
    const int N1 = 32;
    const int N2 = 32;
    int global_shape[3] = {N0, N1, N2};
    const int total_size = N0 * N1 * N2;

    // Gaussian parameters
    const double center0 = N0 / 2.0;
    const double center1 = N1 / 2.0;
    const double center2 = N2 / 2.0;
    const double sigma = 4.0;

    // Create FFT object
    parafaft::ParaFaFT<3> fft(global_shape);

    // Get local dimensions
    int local_size = fft.get_local_size();
    int local_shape[3];
    int global_start[3];
    fft.get_local_shape(local_shape);
    fft.get_global_start(global_start);

    // Allocate local data
    std::vector<std::complex<double>> data(local_size);

    // Initialize with 3D Gaussian using global coordinates
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

    // Save original
    std::vector<std::complex<double>> original = data;

    if (rank == 0) {
        std::cout << "Initial data[0] = " << data[0] << "\n";
        std::cout << "Performing forward FFT...\n";
    }

    // Forward FFT
    fft.forward(data.data());

    if (rank == 0) {
        std::cout << "After forward: data[0] = " << data[0] << "\n";
    }

    // Save transformed data - need to gather from all ranks
    // After forward FFT, data is in stage C distribution: [N0, n0, n1]
    int final_shape[3];
    int final_start[3];
    fft.get_final_shape(final_shape);
    fft.get_final_start(final_start);
    int final_size = final_shape[0] * final_shape[1] * final_shape[2];

    // Gather shape and start info from all ranks
    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    std::vector<int> all_shapes(nranks * 3);
    std::vector<int> all_starts(nranks * 3);
    int shape_data[3] = {final_shape[0], final_shape[1], final_shape[2]};
    int start_data[3] = {final_start[0], final_start[1], final_start[2]};

    MPI_Gather(shape_data, 3, MPI_INT, all_shapes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(start_data, 3, MPI_INT, all_starts.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather data from all ranks
    std::vector<int> sizes(nranks);
    std::vector<int> displs(nranks);
    MPI_Gather(&final_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nranks; ++i) {
            displs[i] = displs[i-1] + sizes[i-1];
        }
    }

    std::vector<std::complex<double>> gathered_data;
    if (rank == 0) {
        gathered_data.resize(displs[nranks-1] + sizes[nranks-1]);
    }

    MPI_Gatherv(data.data(), final_size, MPI_C_DOUBLE_COMPLEX,
                gathered_data.data(), sizes.data(), displs.data(), MPI_C_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);

    // Reconstruct global array in correct order on rank 0
    if (rank == 0) {
        std::vector<std::complex<double>> global_array(total_size);

        int offset = 0;
        for (int r = 0; r < nranks; ++r) {
            int shape0 = all_shapes[r*3 + 0];
            int shape1 = all_shapes[r*3 + 1];
            int shape2 = all_shapes[r*3 + 2];
            int start0 = all_starts[r*3 + 0];
            int start1 = all_starts[r*3 + 1];
            int start2 = all_starts[r*3 + 2];

            // Stage C: [N0, n0, n1] where axis 1,2 are distributed
            for (int i0 = 0; i0 < shape0; ++i0) {
                for (int i1 = 0; i1 < shape1; ++i1) {
                    for (int i2 = 0; i2 < shape2; ++i2) {
                        int local_idx = (i0 * shape1 + i1) * shape2 + i2;
                        int global_i0 = start0 + i0;  // = i0 (axis 0 not distributed in stage C)
                        int global_i1 = start1 + i1;
                        int global_i2 = start2 + i2;
                        int global_idx = (global_i0 * N1 + global_i1) * N2 + global_i2;
                        global_array[global_idx] = gathered_data[offset + local_idx];
                    }
                }
            }
            offset += sizes[r];
        }

        // Save to file
        std::ofstream outfile("gaussian_transformed_mpi.txt");
        outfile << std::scientific << std::setprecision(15);

        for (int i = 0; i < total_size; ++i) {
            outfile << global_array[i].real() << " " << global_array[i].imag() << "\n";
        }
        outfile.close();
        std::cout << "Saved transformed data to gaussian_transformed_mpi.txt\n";
        std::cout << "Performing backward FFT...\n";
    }

    // Backward FFT
    fft.backward(data.data());

    if (rank == 0) {
        std::cout << "After backward (before norm): data[0] = " << data[0] << "\n";
        std::cout << "Original: " << original[0] << "\n";
        std::cout << "Ratio: " << data[0].real() / original[0].real() << "\n";
    }

    // Normalize
    double scale = 1.0 / total_size;
    for (int i = 0; i < local_size; ++i) {
        data[i] *= scale;
    }

    if (rank == 0) {
        std::cout << "\nAfter normalization:\n";
        std::cout << "  data[0] = " << data[0] << "\n";
        std::cout << "  original[0] = " << original[0] << "\n";
        std::cout << "  error = " << std::abs(data[0] - original[0]) << "\n";
    }

    // Compute errors
    double local_max_error = 0.0;
    for (int i = 0; i < local_size; ++i) {
        double err = std::abs(data[i] - original[i]);
        local_max_error = std::max(local_max_error, err);
    }

    double global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\nMaximum error: " << std::scientific << global_max_error << "\n";
        if (global_max_error < 1e-10) {
            std::cout << "✓ PASSED\n";
        } else {
            std::cout << "✗ FAILED\n";
        }
    }

    MPI_Finalize();
    return 0;
}
