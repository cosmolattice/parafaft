#include "../parafaft_r2c.hpp"
#include "../backend/cufft/fft_backend_cufft.hpp"

#include <array>
#include <cmath>
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

inline void write_csv_cuda(const std::string &filename, int mpi_procs, int N,
                           double parafaft_mean, double parafaft_std,
                           int iterations) {
  std::ifstream check_file(filename);
  bool file_exists = check_file.good();
  check_file.close();

  std::ofstream csv_file(filename, std::ios::app);
  if (!file_exists) {
    csv_file << "mpi_procs,N,parafaft_mean,parafaft_std,iterations\n";
  }
  csv_file << std::fixed << std::setprecision(6);
  csv_file << mpi_procs << "," << N << "," << parafaft_mean << ","
           << parafaft_std << "," << iterations << "\n";
  csv_file.close();
}

} // namespace parafaft_bench

template <int D>
void run_benchmark(int N, int rank, int mpi_size, int iterations) {
  using namespace parafaft_bench;

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Benchmark: ParaFaFT R2C (cuFFT) " << D << "D, N=" << N
              << ", MPI procs=" << mpi_size << ", iterations=" << iterations
              << std::endl;
  }

  std::array<int, D> global_shape;
  global_shape.fill(N);

  Statistics parafaft_stats;

  {
    parafaft::ParaFaFT_R2C<D, parafaft::CuFFTBackend<>> fft(global_shape.data());

    int local_real_shape[D], real_start[D];
    fft.get_local_real_shape(local_real_shape);
    fft.get_real_global_start(real_start);

    int local_padded_size = fft.get_required_output_size();

    const int padded_last = local_real_shape[D - 1] + 2;

    // Fill Gaussian data on host
    std::vector<double> host_buffer(local_padded_size, 0.0);

    const double center = N / 2.0;
    const double sigma = 4.0;

    iterate_nd<D>(local_real_shape, [&](const std::array<int, D> &lidx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = (real_start[d] + lidx[d]) - center;
        r2 += x * x;
      }
      double value = std::exp(-r2 / (2.0 * sigma * sigma));

      int padded_flat = lidx[0];
      for (int d = 1; d < D - 1; ++d)
        padded_flat = padded_flat * local_real_shape[d] + lidx[d];
      padded_flat = padded_flat * padded_last + lidx[D - 1];

      host_buffer[padded_flat] = value;
    });

    // Allocate device buffer and copy host data
    double *d_data = nullptr;
    cudaMalloc((void **)&d_data, local_padded_size * sizeof(double));
    cudaMemcpy(d_data, host_buffer.data(),
               local_padded_size * sizeof(double), cudaMemcpyHostToDevice);

    MPI_Barrier(MPI_COMM_WORLD);

    // Warmup
    for (int iter = 0; iter < 5; ++iter) {
      fft.forward_in_place(d_data);
      fft.backward_in_place(d_data);
    }

    // Timed iterations
    for (int iter = 0; iter < iterations; ++iter) {
      // Re-upload original data for consistent input each iteration
      cudaMemcpy(d_data, host_buffer.data(),
                 local_padded_size * sizeof(double), cudaMemcpyHostToDevice);

      MPI_Barrier(MPI_COMM_WORLD);
      double start = MPI_Wtime();
      fft.forward_in_place(d_data);
      fft.backward_in_place(d_data);
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double elapsed = MPI_Wtime() - start;
      parafaft_stats.add(elapsed);
    }

    cudaFree(d_data);
  }

  double parafaft_mean = parafaft_stats.mean();
  double parafaft_std = parafaft_stats.stddev();

  if (rank == 0) {
    std::cout << "ParaFaFT (cuFFT): mean=" << parafaft_mean
              << "s, std=" << parafaft_std << "s" << std::endl;

    write_csv_cuda("bench_r2c_cuda.csv", mpi_size, N, parafaft_mean,
                   parafaft_std, iterations);
    std::cout << "CSV written to bench_r2c_cuda.csv" << std::endl;
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
  bool explicit_n = (argc > 2);

  if (argc > 2) {
    N = std::atoi(argv[1]);
    D = std::atoi(argv[2]);
  } else if (argc > 1) {
    D = std::atoi(argv[1]);
  }
  if (argc > 3)
    iterations = std::atoi(argv[3]);

  if (D >= 4 && !explicit_n)
    N = 16;

  switch (D) {
  case 2:
    run_benchmark<2>(N, rank, mpi_size, iterations);
    break;
  case 3:
    run_benchmark<3>(N, rank, mpi_size, iterations);
    break;
  case 4:
    run_benchmark<4>(N, rank, mpi_size, iterations);
    break;
  case 5:
    run_benchmark<5>(N, rank, mpi_size, iterations);
    break;
  case 6:
    run_benchmark<6>(N, rank, mpi_size, iterations);
    break;
  default:
    if (rank == 0) {
      std::cerr << "Unsupported dimensions: " << D << " (supported: 2-6)"
                << std::endl;
    }
  }

  MPI_Finalize();
  return 0;
}
