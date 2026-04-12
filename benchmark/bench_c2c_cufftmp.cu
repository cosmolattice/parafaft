#include "../parafaft_c2c.hpp"
#include "../backend/cufft/fft_backend_cufftmp.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

namespace parafaft_bench {

class Statistics {
public:
  void add(double value) { values_.push_back(value); }

  double mean() const {
    if (values_.empty())
      return 0.0;
    double sum = std::accumulate(values_.begin(), values_.end(), 0.0);
    return sum / values_.size();
  }

  double stddev() const {
    if (values_.size() < 2)
      return 0.0;
    double m = mean();
    double sq_sum = 0.0;
    for (const auto &v : values_) {
      sq_sum += (v - m) * (v - m);
    }
    return std::sqrt(sq_sum / (values_.size() - 1));
  }

private:
  std::vector<double> values_;
};

template <int D>
int nd_index(const int idx[], const int shape[]) {
  int flat = idx[0];
  for (int d = 1; d < D; ++d)
    flat = flat * shape[d] + idx[d];
  return flat;
}

template <int D>
void iterate_nd(
    const int shape[D],
    const std::function<void(const std::array<int, D> &)> &callback) {
  std::array<int, D> idx{};
  int total = 1;
  for (int d = 0; d < D; ++d)
    total *= shape[d];

  for (int i = 0; i < total; ++i) {
    callback(idx);
    for (int d = D - 1; d >= 0; --d) {
      if (++idx[d] < shape[d])
        break;
      idx[d] = 0;
    }
  }
}

inline void write_csv(const std::string &filename, int mpi_procs, int N,
                      double mean, double std, int iterations) {
  std::ifstream check_file(filename);
  bool file_exists = check_file.good();
  check_file.close();

  std::ofstream csv_file(filename, std::ios::app);
  if (!file_exists) {
    csv_file << "mpi_procs,N,parafaft_mean,parafaft_std,iterations\n";
  }
  csv_file << std::fixed << std::setprecision(6);
  csv_file << mpi_procs << "," << N << "," << mean << "," << std << ","
           << iterations << "\n";
  csv_file.close();
}

} // namespace parafaft_bench

template <int D>
void run_benchmark(int N, int rank, int mpi_size, int iterations) {
  using namespace parafaft_bench;
  using Backend = parafaft::CuFFTMpBackend<D>;
  using Complex = typename Backend::Complex;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Benchmark: ParaFaFT C2C (cuFFTMp) " << D << "D, N=" << N
              << ", MPI procs=" << mpi_size << ", iterations=" << iterations
              << std::endl;
  }

  std::array<int, D> global_shape;
  global_shape.fill(N);

  Statistics stats;

  {
    parafaft::ParaFaFT_C2C<D, Backend> fft(global_shape.data());

    int local_shape[D];
    int global_start[D];
    fft.get_local_shape(local_shape);
    fft.get_global_start(global_start);

    int local_size = fft.get_local_size();

    // Fill Gaussian data on host
    std::vector<std::complex<double>> host_buffer(local_size);
    const double center = N / 2.0;
    const double sigma = 4.0;

    iterate_nd<D>(local_shape, [&](const std::array<int, D> &lidx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = (global_start[d] + lidx[d]) - center;
        r2 += x * x;
      }
      int local_flat = nd_index<D>(lidx.data(), local_shape);
      host_buffer[local_flat] =
          std::complex<double>(std::exp(-r2 / (2.0 * sigma * sigma)), 0.0);
    });

    // Copy into backend-managed NVSHMEM buffer
    Complex *buffer = fft.get_buffer();
    cudaMemcpy(buffer, host_buffer.data(), local_size * sizeof(Complex),
               cudaMemcpyHostToDevice);

    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup
    for (int iter = 0; iter < 5; ++iter) {
      fft.forward(buffer);
      fft.backward(buffer);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    for (int iter = 0; iter < iterations; ++iter) {
      // Re-upload original data
      cudaMemcpy(buffer, host_buffer.data(), local_size * sizeof(Complex),
                 cudaMemcpyHostToDevice);

      MPI_Barrier(MPI_COMM_WORLD);
      double start = MPI_Wtime();
      fft.forward(buffer);
      fft.backward(buffer);
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double elapsed = MPI_Wtime() - start;
      stats.add(elapsed);
    }
  }

  double parafaft_mean = stats.mean();
  double parafaft_std = stats.stddev();

  if (rank == 0) {
    std::cout << "ParaFaFT (cuFFTMp): mean=" << parafaft_mean
              << "s, std=" << parafaft_std << "s" << std::endl;

    write_csv("bench_c2c_cufftmp.csv", mpi_size, N, parafaft_mean,
              parafaft_std, iterations);
    std::cout << "CSV written to bench_c2c_cufftmp.csv" << std::endl;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Set GPU device based on node-local rank
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                      MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  cudaSetDevice(local_rank % num_devices);
  MPI_Comm_free(&local_comm);

  int N = 32;
  int D = 3;
  int iterations = 50;

  if (argc > 2) {
    N = std::atoi(argv[1]);
    D = std::atoi(argv[2]);
  } else if (argc > 1) {
    D = std::atoi(argv[1]);
  }
  if (argc > 3)
    iterations = std::atoi(argv[3]);

  if (D < 2 || D > 3) {
    if (rank == 0)
      std::cerr << "Error: cuFFTMp only supports D=2 or D=3." << std::endl;
    MPI_Finalize();
    return 1;
  }

  switch (D) {
  case 2:
    run_benchmark<2>(N, rank, mpi_size, iterations);
    break;
  case 3:
    run_benchmark<3>(N, rank, mpi_size, iterations);
    break;
  }

  MPI_Finalize();
  return 0;
}
