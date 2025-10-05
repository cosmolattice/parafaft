#include <fftw3.h>
#include <complex>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    int N0 = 32;
    int N1 = 32;
    int N2 = 32;

    // Allow command-line override
    if (argc > 1) {
        N0 = N1 = N2 = std::atoi(argv[1]);
    }

    const int complex_N2 = N2 / 2 + 1;

    const int real_size = N0 * N1 * N2;
    const int complex_size = N0 * N1 * complex_N2;

    // Allocate arrays
    std::vector<double> real_data(real_size);
    std::vector<std::complex<double>> complex_data(complex_size);

    // Gaussian parameters
    const double center0 = N0 / 2.0;
    const double center1 = N1 / 2.0;
    const double center2 = N2 / 2.0;
    const double sigma = 4.0;

    std::cout << "Creating 3D Gaussian distribution (real-valued)...\n";
    std::cout << "Real array size: " << N0 << " x " << N1 << " x " << N2 << "\n";
    std::cout << "Complex array size: " << N0 << " x " << N1 << " x " << complex_N2 << "\n";

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
                real_data[idx] = value;
            }
        }
    }

    // Save initial real distribution
    {
        std::ofstream file("r2c_initial_gaussian.txt");
        file << std::scientific << std::setprecision(15);
        for (int i = 0; i < real_size; ++i) {
            file << real_data[i] << "\n";
        }
        file.close();
        std::cout << "Saved initial distribution to: r2c_initial_gaussian.txt\n";
    }

    // Create FFTW R2C plan
    std::cout << "Creating FFTW plan for 3D R2C FFT...\n";
    fftw_plan plan_r2c = fftw_plan_dft_r2c_3d(
        N0, N1, N2,
        real_data.data(),
        reinterpret_cast<fftw_complex*>(complex_data.data()),
        FFTW_ESTIMATE);

    // Execute R2C FFT
    std::cout << "Executing R2C FFT...\n";
    fftw_execute(plan_r2c);

    // Save transformed distribution (complex, reduced size)
    {
        std::ofstream file("r2c_transformed_gaussian.txt");
        file << std::scientific << std::setprecision(15);

        for (int i0 = 0; i0 < N0; ++i0) {
            for (int i1 = 0; i1 < N1; ++i1) {
                for (int i2 = 0; i2 < complex_N2; ++i2) {
                    int idx = (i0 * N1 + i1) * complex_N2 + i2;
                    file << complex_data[idx].real() << " "
                         << complex_data[idx].imag() << "\n";
                }
            }
        }
        file.close();
        std::cout << "Saved R2C transform to: r2c_transformed_gaussian.txt\n";
    }

    // Print DC component and first few values for verification
    std::cout << "\nFirst few complex values:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  [" << i << "] = " << complex_data[i] << "\n";
    }

    // Verify DC component (should be sum of all real values)
    double sum = 0.0;
    for (int i = 0; i < real_size; ++i) {
        sum += real_data[i];
    }
    std::cout << "\nSum of real values: " << sum << "\n";
    std::cout << "DC component (should match): " << complex_data[0].real() << "\n";
    std::cout << "Difference: " << std::abs(sum - complex_data[0].real()) << "\n";

    fftw_destroy_plan(plan_r2c);
    std::cout << "\nDone!\n";

    return 0;
}
