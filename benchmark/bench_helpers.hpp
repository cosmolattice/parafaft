#ifndef PARAFAFT_BENCHMARK_HELPERS_HPP
#define PARAFAFT_BENCHMARK_HELPERS_HPP

#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include <mpi.h>
#include <thread>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <string>
#include <sstream>

namespace parafaft_bench
{

  template <int D> inline int nd_index(const int idx[D], const int shape[D])
  {
    int flat = idx[0];
    for (int d = 1; d < D; ++d)
      flat = flat * shape[d] + idx[d];
    return flat;
  }

  template <int D> inline int nd_index(const std::array<int, D> &idx, const std::array<int, D> &shape)
  {
    return nd_index<D>(idx.data(), shape.data());
  }

  template <int D> void iterate_nd(const int shape[D], const std::function<void(const std::array<int, D> &)> &callback)
  {
    std::array<int, D> idx{};
    int total = 1;
    for (int d = 0; d < D; ++d)
      total *= shape[d];

    for (int i = 0; i < total; ++i) {
      callback(idx);
      for (int d = D - 1; d >= 0; --d) {
        if (++idx[d] < shape[d]) break;
        idx[d] = 0;
      }
    }
  }

  template <int D> std::vector<std::complex<double>> generate_gaussian_nd(int N, double sigma)
  {
    int total = 1;
    for (int d = 0; d < D; ++d)
      total *= N;

    std::vector<std::complex<double>> data(total);
    std::array<int, D> shape;
    shape.fill(N);

    const double center = N / 2.0;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      double value = std::exp(-r2 / (2.0 * sigma * sigma));
      int flat = nd_index<D>(idx, shape);
      data[flat] = std::complex<double>(value, 0.0);
    });

    return data;
  }

  template <int D> std::vector<double> generate_gaussian_r2c_nd(int N, double sigma)
  {
    int real_total = 1;
    for (int d = 0; d < D; ++d)
      real_total *= N;

    int complex_doubles = 1;
    for (int d = 0; d < D - 1; ++d)
      complex_doubles *= N;
    complex_doubles *= (N / 2 + 1) * 2;

    const int total_size = std::max(real_total, complex_doubles);
    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);
    const double center = N / 2.0;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      int flat = nd_index<D>(idx, shape);
      data[flat] = std::exp(-r2 / (2.0 * sigma * sigma));
    });

    return data;
  }

  template <int D> std::vector<double> generate_gaussian_r2c_padded_nd(int N, double sigma)
  {
    int total_size = 1;
    for (int d = 0; d < D - 1; ++d)
      total_size *= N;
    total_size *= (N + 2);

    std::vector<double> data(total_size, 0.0);

    std::array<int, D> shape;
    shape.fill(N);
    const double center = N / 2.0;

    iterate_nd<D>(shape.data(), [&](const std::array<int, D> &idx) {
      double r2 = 0.0;
      for (int d = 0; d < D; ++d) {
        double x = idx[d] - center;
        r2 += x * x;
      }
      const int padded_last = shape[D - 1] + 2;
      int flat = idx[0];
      for (int d = 1; d < D - 1; ++d)
        flat = flat * shape[d] + idx[d];
      flat = flat * padded_last + idx[D - 1];
      data[flat] = std::exp(-r2 / (2.0 * sigma * sigma));
    });

    return data;
  }

  class Timer
  {
  public:
    void start() { start_time_ = MPI_Wtime(); }
    void stop() { end_time_ = MPI_Wtime(); }
    double elapsed() const { return end_time_ - start_time_; }

  private:
    double start_time_;
    double end_time_;
  };

  class Statistics
  {
  public:
    void add(double value) { values_.push_back(value); }

    double mean() const
    {
      if (values_.empty()) return 0.0;
      double sum = std::accumulate(values_.begin(), values_.end(), 0.0);
      return sum / values_.size();
    }

    double stddev() const
    {
      if (values_.size() < 2) return 0.0;
      double m = mean();
      double sq_sum = 0.0;
      for (const auto &v : values_) {
        sq_sum += (v - m) * (v - m);
      }
      return std::sqrt(sq_sum / (values_.size() - 1));
    }

    const std::vector<double> &values() const { return values_; }

  private:
    std::vector<double> values_;
  };

  inline int detect_thread_count(MPI_Comm comm)
  {
    int threads = 1;

#if defined(PARAFAFT_FFTW_OMP)
    const char *omp_threads = std::getenv("OMP_NUM_THREADS");
    if (omp_threads != nullptr) {
      threads = std::atoi(omp_threads);
      if (threads > 0) {
        return threads;
      }
    }
#endif

    const char *kokkos_threads = std::getenv("KOKKOS_NUM_THREADS");
    if (kokkos_threads != nullptr) {
      threads = std::atoi(kokkos_threads);
      if (threads > 0) {
        return threads;
      }
    }

    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads > 0) {
      int mpi_size = 1;
      MPI_Comm_size(comm, &mpi_size);
      threads = static_cast<int>(hw_threads) / mpi_size;
      if (threads < 1) {
        threads = 1;
      }
    }

    return threads;
  }

  inline void init_fftw_threads(MPI_Comm comm)
  {
#if defined(PARAFAFT_FFTW_THREADS) || defined(PARAFAFT_FFTW_OMP)
    fftw_init_threads();
    int num_threads = detect_thread_count(comm);
    fftw_plan_with_nthreads(num_threads);
#endif
  }

  template <int D> void fftw_c2c_reference_nd(std::complex<double> *data, int N)
  {
    std::array<int, D> n;
    n.fill(N);

    fftw_plan plan = fftw_plan_dft(D, n.data(), reinterpret_cast<fftw_complex *>(data),
                                   reinterpret_cast<fftw_complex *>(data), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  template <int D> void fftw_c2c_roundtrip_reference_nd(std::complex<double> *data, int N)
  {
    std::array<int, D> n;
    n.fill(N);

    fftw_plan forward_plan = fftw_plan_dft(D, n.data(), reinterpret_cast<fftw_complex *>(data),
                                           reinterpret_cast<fftw_complex *>(data), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(forward_plan);
    fftw_destroy_plan(forward_plan);

    fftw_plan backward_plan = fftw_plan_dft(D, n.data(), reinterpret_cast<fftw_complex *>(data),
                                            reinterpret_cast<fftw_complex *>(data), FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(backward_plan);
    fftw_destroy_plan(backward_plan);

    double scale = 1.0;
    for (int d = 0; d < D; ++d) scale *= N;
    int total = 1;
    for (int d = 0; d < D; ++d) total *= N;
    for (int i = 0; i < total; ++i) data[i] /= scale;
  }

  template <int D>
  void fftw_c2c_mpi_reference_nd(std::complex<double> *local_data, const int local_shape[D],
                                  const int global_start[D], int N, MPI_Comm comm)
  {
    int total_local = 1;
    for (int d = 0; d < D; ++d) total_local *= local_shape[d];

    for (int axis = 0; axis < D; ++axis) {
      int fft_length = local_shape[axis];
      int stride = 1;
      for (int d = axis + 1; d < D; ++d) {
        stride *= local_shape[d];
      }
      int batch = total_local / (fft_length * stride);
      int dist = stride * fft_length;

      fftw_plan forward_plan = fftw_plan_many_dft(1, &fft_length, batch,
                                                    reinterpret_cast<fftw_complex *>(local_data),
                                                    NULL, stride, dist,
                                                    reinterpret_cast<fftw_complex *>(local_data),
                                                    NULL, stride, dist,
                                                    FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(forward_plan);
      fftw_destroy_plan(forward_plan);
    }

    for (int axis = 0; axis < D; ++axis) {
      int fft_length = local_shape[axis];
      int stride = 1;
      for (int d = axis + 1; d < D; ++d) {
        stride *= local_shape[d];
      }
      int batch = total_local / (fft_length * stride);
      int dist = stride * fft_length;

      fftw_plan backward_plan = fftw_plan_many_dft(1, &fft_length, batch,
                                                     reinterpret_cast<fftw_complex *>(local_data),
                                                     NULL, stride, dist,
                                                     reinterpret_cast<fftw_complex *>(local_data),
                                                     NULL, stride, dist,
                                                     FFTW_BACKWARD, FFTW_ESTIMATE);
      fftw_execute(backward_plan);
      fftw_destroy_plan(backward_plan);
    }

    double scale = 1.0;
    for (int d = 0; d < D; ++d) scale *= N;
    for (int i = 0; i < total_local; ++i) local_data[i] /= scale;
  }

  template <int D> void fftw_r2c_reference_padded_nd(double *host_data, int N)
  {
    std::array<int, D> n;
    n.fill(N);

    fftw_plan plan =
        fftw_plan_dft_r2c(D, n.data(), host_data, reinterpret_cast<fftw_complex *>(host_data), FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  template <int D> void fftw_c2r_reference_padded_nd(double *host_data, int N)
  {
    std::array<int, D> n;
    n.fill(N);

    fftw_plan plan =
        fftw_plan_dft_c2r(D, n.data(), reinterpret_cast<fftw_complex *>(host_data), host_data, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
  }

  template <int D> void fftw_r2c_roundtrip_reference_nd(double *data, int N)
  {
    std::array<int, D> n;
    n.fill(N);

    fftw_plan forward_plan =
        fftw_plan_dft_r2c(D, n.data(), data, reinterpret_cast<fftw_complex *>(data), FFTW_ESTIMATE);
    fftw_execute(forward_plan);
    fftw_destroy_plan(forward_plan);

    fftw_plan backward_plan =
        fftw_plan_dft_c2r(D, n.data(), reinterpret_cast<fftw_complex *>(data), data, FFTW_ESTIMATE);
    fftw_execute(backward_plan);
    fftw_destroy_plan(backward_plan);

    double scale = 1.0;
    for (int d = 0; d < D; ++d) scale *= N;
    int total = 1;
    for (int d = 0; d < D - 1; ++d) total *= N;
    total *= (N + 2);
    for (int i = 0; i < total; ++i) {
      data[i] /= scale;
    }
  }

  template <int D>
  void fftw_r2c_mpi_reference_nd(double *padded_buffer, const int local_shape[D], int N, MPI_Comm comm)
  {
    int padded_last = local_shape[D - 1] + 2;

    int total_local = 1;
    for (int d = 0; d < D - 1; ++d) total_local *= local_shape[d];
    total_local *= padded_last;

    for (int axis = D - 2; axis >= 0; --axis) {
      int fft_length = local_shape[axis];
      int batch = total_local / fft_length / padded_last;

      int stride = padded_last;
      for (int d = axis + 1; d < D - 1; ++d) {
        stride *= local_shape[d];
      }

      fftw_plan forward_plan = fftw_plan_many_dft(1, &fft_length, batch,
                                                  reinterpret_cast<fftw_complex *>(padded_buffer),
                                                  NULL, stride, padded_last,
                                                  reinterpret_cast<fftw_complex *>(padded_buffer),
                                                  NULL, stride, padded_last,
                                                  FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(forward_plan);
      fftw_destroy_plan(forward_plan);
    }

    {
      int axis = D - 1;
      int fft_length = local_shape[axis];
      int batch = 1;
      for (int d = 0; d < D - 1; ++d) {
        batch *= local_shape[d];
      }

      fftw_plan forward_plan = fftw_plan_many_dft(1, &fft_length, batch,
                                                  reinterpret_cast<fftw_complex *>(padded_buffer),
                                                  NULL, 1, padded_last,
                                                  reinterpret_cast<fftw_complex *>(padded_buffer),
                                                  NULL, 1, padded_last,
                                                  FFTW_FORWARD, FFTW_ESTIMATE);
      fftw_execute(forward_plan);
      fftw_destroy_plan(forward_plan);
    }

    for (int axis = D - 2; axis >= 0; --axis) {
      int fft_length = local_shape[axis];
      int batch = total_local / fft_length / padded_last;

      int stride = padded_last;
      for (int d = axis + 1; d < D - 1; ++d) {
        stride *= local_shape[d];
      }

      fftw_plan backward_plan = fftw_plan_many_dft(1, &fft_length, batch,
                                                   reinterpret_cast<fftw_complex *>(padded_buffer),
                                                   NULL, stride, padded_last,
                                                   reinterpret_cast<fftw_complex *>(padded_buffer),
                                                   NULL, stride, padded_last,
                                                   FFTW_BACKWARD, FFTW_ESTIMATE);
      fftw_execute(backward_plan);
      fftw_destroy_plan(backward_plan);
    }

    {
      int axis = D - 1;
      int fft_length = local_shape[axis];
      int batch = 1;
      for (int d = 0; d < D - 1; ++d) {
        batch *= local_shape[d];
      }

      fftw_plan backward_plan = fftw_plan_many_dft(1, &fft_length, batch,
                                                   reinterpret_cast<fftw_complex *>(padded_buffer),
                                                   NULL, 1, padded_last,
                                                   reinterpret_cast<fftw_complex *>(padded_buffer),
                                                   NULL, 1, padded_last,
                                                   FFTW_BACKWARD, FFTW_ESTIMATE);
      fftw_execute(backward_plan);
      fftw_destroy_plan(backward_plan);
    }

    double scale = 1.0;
    for (int d = 0; d < D; ++d) scale *= N;
    for (int i = 0; i < total_local; ++i) {
      padded_buffer[i] /= scale;
    }
  }

  inline std::string format_csv(const std::string &benchmark, const std::string &type, const std::string &test_name,
                                  int N, int D, int mpi_procs, int fftw_threads, double parafaft_mean,
                                  double parafaft_std, double fftw_mean, double fftw_std, int iterations)
  {
    std::ostringstream oss;
    oss << benchmark << "," << type << "," << test_name << "," << N << "," << D << "," << mpi_procs << ","
        << fftw_threads << "," << parafaft_mean << "," << parafaft_std << "," << fftw_mean << "," << fftw_std
        << "," << iterations;
    return oss.str();
  }

  inline void write_csv(const std::string &filename, int mpi_procs, int threads, int N,
                        double parafaft_mean, double parafaft_std, double fftw_mean, double fftw_std,
                        int iterations)
  {
    std::ifstream check_file(filename);
    bool file_exists = check_file.good();
    check_file.close();

    std::ofstream csv_file(filename, std::ios::app);
    if (!file_exists) {
      csv_file << "mpi_procs,threads,N,parafaft_mean,parafaft_std,fftw_mean,fftw_std,iterations\n";
    }
    csv_file << std::fixed << std::setprecision(6);
    csv_file << mpi_procs << "," << threads << "," << N << ","
             << parafaft_mean << "," << parafaft_std << "," << fftw_mean << "," << fftw_std << "," << iterations << "\n";
    csv_file.close();
  }

} // namespace parafaft_bench

#endif // PARAFAFT_BENCHMARK_HELPERS_HPP
