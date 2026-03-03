#include "../backend/fftw3/fft_backend_fftw.hpp"
#include "../parafaft_c2c.hpp"
#include "bench_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

template <int D>
void run_benchmark(int N, int rank, int mpi_size, int iterations) {
  using namespace parafaft_bench;

  int num_threads = detect_thread_count(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Benchmark: ParaFaFT C2C " << D << "D, N=" << N
              << ", MPI procs=" << mpi_size << ", FFTW threads=" << num_threads
              << ", iterations=" << iterations << std::endl;
  }

  std::array<int, D> global_shape;
  global_shape.fill(N);

  parafaft::ParaFaFT<D, parafaft::FFTWBackend> fft(global_shape.data());

  int local_size = fft.get_local_size();
  int buffer_size = fft.get_required_output_size();
  int local_shape[D];
  int global_start[D];
  fft.get_local_shape(local_shape);
  fft.get_global_start(global_start);

  std::vector<std::complex<double>> local_data(buffer_size);

  const double center = N / 2.0;
  const double sigma = 4.0;

  iterate_nd<D>(local_shape, [&](const std::array<int, D> &lidx) {
    double r2 = 0.0;
    for (int d = 0; d < D; ++d) {
      double x = (global_start[d] + lidx[d]) - center;
      r2 += x * x;
    }
    int local_flat = nd_index<D>(lidx.data(), local_shape);
    local_data[local_flat] =
        std::complex<double>(std::exp(-r2 / (2.0 * sigma * sigma)), 0.0);
  });

  Statistics parafaft_stats;
  Statistics fftw_stats;

  MPI_Barrier(MPI_COMM_WORLD);

  for (int iter = 0; iter < 5; ++iter) {
    auto local_copy = local_data;
    fft.forward(local_copy.data());
    fft.backward(local_copy.data());
  }

  for (int iter = 0; iter < iterations; ++iter) {
    auto local_copy = local_data;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    fft.forward(local_copy.data());
    fft.backward(local_copy.data());
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    parafaft_stats.add(elapsed);
  }

  // fftw_init_threads() must be called before fftw_mpi_init()
  // (see FFTW docs: "Combining MPI and Threads")
  parafaft_bench::init_fftw_threads(MPI_COMM_WORLD);
  parafaft_bench::init_fftw_mpi();

  FFTWMPIReferenceCtoC<D> fftw_ref(N, MPI_COMM_WORLD);

  for (int iter = 0; iter < 5; ++iter)
    fftw_ref.execute();

  double fftw_mean = 0.0, fftw_std = 0.0;
  for (int iter = 0; iter < iterations; ++iter) {
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    fftw_ref.execute();
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    fftw_stats.add(elapsed);
  }

  double parafaft_mean = parafaft_stats.mean();
  double parafaft_std = parafaft_stats.stddev();
  fftw_mean = fftw_stats.mean();
  fftw_std = fftw_stats.stddev();

  if (rank == 0) {

    std::cout << "ParaFaFT: mean=" << parafaft_mean << "s, std=" << parafaft_std
              << "s" << std::endl;
    std::cout << "FFTW:     mean=" << fftw_mean << "s, std=" << fftw_std << "s"
              << std::endl;

    write_csv("bench_c2c.csv", mpi_size, num_threads, N, parafaft_mean,
              parafaft_std, fftw_mean, fftw_std, iterations);
    std::cout << "CSV written to bench_c2c.csv" << std::endl;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
