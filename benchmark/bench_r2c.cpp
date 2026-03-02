#include "../parafaft_r2c.hpp"
#include "../backend/fftw3/fft_backend_fftw.hpp"
#include "bench_helpers.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

template <int D> void run_benchmark(int N, int rank, int mpi_size, int iterations)
{
  using namespace parafaft_bench;

  int num_threads = detect_thread_count(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "Benchmark: ParaFaFT R2C " << D << "D, N=" << N << ", MPI procs=" << mpi_size
              << ", FFTW threads=" << num_threads << ", iterations=" << iterations << std::endl;
  }

  std::array<int, D> global_shape;
  global_shape.fill(N);

  parafaft::ParaFaFT_R2C<D, parafaft::FFTWBackend> fft(global_shape.data());

  int local_real_shape[D], real_start[D];
  fft.get_local_real_shape(local_real_shape);
  fft.get_real_global_start(real_start);

  int local_padded_size = fft.get_required_output_size();

  const int padded_last = local_real_shape[D - 1] + 2;

  std::vector<double> padded_buffer(local_padded_size, 0.0);

  int local_real_size = fft.get_local_real_size();
  std::vector<double> original(local_real_size);

  std::array<int, D> full_shape;
  full_shape.fill(N);

  auto global_original = generate_gaussian_r2c_nd<D>(N, 4.0);

  int orig_idx = 0;
  iterate_nd<D>(local_real_shape, [&](const std::array<int, D> &lidx) {
    std::array<int, D> gidx;
    for (int d = 0; d < D; ++d)
      gidx[d] = real_start[d] + lidx[d];

    int global_flat = nd_index<D>(gidx, full_shape);
    double value = global_original[global_flat];

    int padded_flat = lidx[0];
    for (int d = 1; d < D - 1; ++d)
      padded_flat = padded_flat * local_real_shape[d] + lidx[d];
    padded_flat = padded_flat * padded_last + lidx[D - 1];

    padded_buffer[padded_flat] = value;
    original[orig_idx++] = value;
  });

  Statistics parafaft_stats;
  Statistics fftw_stats;

  MPI_Barrier(MPI_COMM_WORLD);

  for (int iter = 0; iter < 5; ++iter) {
    auto padded_copy = padded_buffer;
    fft.forward_in_place(padded_copy.data());
    fft.backward_in_place(padded_copy.data());
  }

  for (int iter = 0; iter < iterations; ++iter) {
    auto padded_copy = padded_buffer;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    fft.forward_in_place(padded_copy.data());
    fft.backward_in_place(padded_copy.data());
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    parafaft_stats.add(elapsed);
  }

  if (rank == 0) {
    auto fftw_data = generate_gaussian_r2c_padded_nd<D>(N, 4.0);

    Timer fftw_timer;

    for (int iter = 0; iter < 5; ++iter) {
      auto copy = fftw_data;
      fftw_r2c_roundtrip_reference_nd<D>(copy.data(), N);
    }

    for (int iter = 0; iter < iterations; ++iter) {
      auto copy = fftw_data;
      fftw_timer.start();
      fftw_r2c_roundtrip_reference_nd<D>(copy.data(), N);
      fftw_timer.stop();
      fftw_stats.add(fftw_timer.elapsed());
    }

    double parafaft_mean = parafaft_stats.mean();
    double parafaft_std = parafaft_stats.stddev();
    double fftw_mean = fftw_stats.mean();
    double fftw_std = fftw_stats.stddev();

    std::cout << "ParaFaFT: mean=" << parafaft_mean << "s, std=" << parafaft_std << "s" << std::endl;
    std::cout << "FFTW:     mean=" << fftw_mean << "s, std=" << fftw_std << "s" << std::endl;

    write_csv("bench_r2c.csv", mpi_size, num_threads, N, parafaft_mean, parafaft_std, fftw_mean, fftw_std, iterations);
    std::cout << "CSV written to bench_r2c.csv" << std::endl;
  }
}

int main(int argc, char **argv)
{
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
  if (argc > 3) iterations = std::atoi(argv[3]);

  if (D >= 4 && !explicit_n) N = 16;

  parafaft_bench::init_fftw_threads(MPI_COMM_WORLD);

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
      std::cerr << "Unsupported dimensions: " << D << " (supported: 2-6)" << std::endl;
    }
  }

  MPI_Finalize();
  return 0;
}
