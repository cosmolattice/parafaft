#ifndef PARAFAFT_BENCHMARK_HELPERS_HPP
#define PARAFAFT_BENCHMARK_HELPERS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <fftw3-mpi.h>
#include <fftw3.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace parafaft_bench {

template <int D>
inline std::size_t nd_index(const int idx[D], const int shape[D]) {
  std::size_t flat = static_cast<std::size_t>(idx[0]);
  for (int d = 1; d < D; ++d)
    flat = flat * static_cast<std::size_t>(shape[d]) + static_cast<std::size_t>(idx[d]);
  return flat;
}

template <int D>
inline std::size_t nd_index(const std::array<int, D> &idx,
                            const std::array<int, D> &shape) {
  return nd_index<D>(idx.data(), shape.data());
}

template <int D>
void iterate_nd(
    const int shape[D],
    const std::function<void(const std::array<int, D> &)> &callback) {
  std::ptrdiff_t total = 1;
  for (int d = 0; d < D; ++d)
    total *= static_cast<std::ptrdiff_t>(shape[d]);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < total; ++i) {
    std::array<int, D> idx;
    std::ptrdiff_t rem = i;
    for (int d = D - 1; d >= 0; --d) {
      idx[d] = static_cast<int>(rem % shape[d]);
      rem /= shape[d];
    }
    callback(idx);
  }
}

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

  const std::vector<double> &values() const { return values_; }

private:
  std::vector<double> values_;
};

inline int detect_thread_count(MPI_Comm comm) {
  int threads = 1;

  // Honour OMP_NUM_THREADS whichever FFTW threading library is linked (fftw3_omp
  // or fftw3_threads): it is the explicit "threads per rank" request.
  const char *omp_threads = std::getenv("OMP_NUM_THREADS");
  if (omp_threads != nullptr) {
    threads = std::atoi(omp_threads);
    if (threads > 0) {
      return threads;
    }
  }

  const char *kokkos_threads = std::getenv("KOKKOS_NUM_THREADS");
  if (kokkos_threads != nullptr) {
    threads = std::atoi(kokkos_threads);
    if (threads > 0) {
      return threads;
    }
  }

  unsigned int hw_threads = std::thread::hardware_concurrency();
  if (hw_threads > 0) {
    // Split the node's cores among the ranks *sharing that node*. Dividing by the
    // size of comm would shrink the per-rank thread count as the job scales out.
    int local_size = 1;
    MPI_Comm shared_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);
    MPI_Comm_size(shared_comm, &local_size);
    MPI_Comm_free(&shared_comm);

    threads = static_cast<int>(hw_threads) / local_size;
    if (threads < 1) {
      threads = 1;
    }
  }

  return threads;
}

inline void init_fftw_threads(MPI_Comm comm) {
#if defined(PARAFAFT_FFTW_THREADS) || defined(PARAFAFT_FFTW_OMP)
  fftw_init_threads();
  int num_threads = detect_thread_count(comm);
  fftw_plan_with_nthreads(num_threads);
#endif
}

inline void init_fftw_mpi() { fftw_mpi_init(); }

/**
 * @brief RAII wrapper for FFTW MPI C2C reference benchmark.
 *
 * Uses fftw_mpi_plan_dft for true MPI-parallel FFTs (slab decomposition).
 * Plans and buffers are created once and reused across iterations.
 */
template <int D> struct FFTWMPIReferenceCtoC {
  fftw_plan forward_plan_;
  fftw_plan backward_plan_;
  fftw_complex *buffer_;
  ptrdiff_t alloc_local_;
  ptrdiff_t local_n0_;
  ptrdiff_t local_0_start_;
  int N_;

  FFTWMPIReferenceCtoC(int N, MPI_Comm comm) : N_(N) {
    ptrdiff_t n[D];
    for (int d = 0; d < D; ++d)
      n[d] = N;

    alloc_local_ = fftw_mpi_local_size(D, n, comm, &local_n0_, &local_0_start_);
    buffer_ = fftw_alloc_complex(alloc_local_);

    forward_plan_ = fftw_mpi_plan_dft(D, n, buffer_, buffer_, comm,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    backward_plan_ = fftw_mpi_plan_dft(D, n, buffer_, buffer_, comm,
                                       FFTW_BACKWARD, FFTW_ESTIMATE);

    // Fill buffer with Gaussian data
    const double center = N / 2.0;
    const double sigma = 4.0;
    ptrdiff_t stride = 1;
    for (int d = 1; d < D; ++d)
      stride *= N;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < local_n0_; ++i) {
      ptrdiff_t gi = local_0_start_ + i;
      for (ptrdiff_t j = 0; j < stride; ++j) {
        // Decompose j into indices for dimensions 1..D-1
        double r2 = 0.0;
        double x0 = gi - center;
        r2 += x0 * x0;

        ptrdiff_t rem = j;
        for (int d = D - 1; d >= 1; --d) {
          ptrdiff_t idx_d = rem % N;
          rem /= N;
          double x = idx_d - center;
          r2 += x * x;
        }

        ptrdiff_t flat = i * stride + j;
        buffer_[flat][0] = std::exp(-r2 / (2.0 * sigma * sigma));
        buffer_[flat][1] = 0.0;
      }
    }
  }

  ~FFTWMPIReferenceCtoC() {
    fftw_destroy_plan(forward_plan_);
    fftw_destroy_plan(backward_plan_);
    fftw_free(buffer_);
  }

  void execute() {
    fftw_execute(forward_plan_);
    fftw_execute(backward_plan_);
  }

  // Non-copyable
  FFTWMPIReferenceCtoC(const FFTWMPIReferenceCtoC &) = delete;
  FFTWMPIReferenceCtoC &operator=(const FFTWMPIReferenceCtoC &) = delete;
};

/**
 * @brief RAII wrapper for FFTW MPI R2C/C2R reference benchmark.
 *
 * Uses fftw_mpi_plan_dft_r2c/c2r for true MPI-parallel FFTs (slab
 * decomposition). Plans and buffers are created once and reused across
 * iterations.
 */
template <int D> struct FFTWMPIReferenceRtoC {
  fftw_plan forward_plan_;
  fftw_plan backward_plan_;
  double *buffer_;
  ptrdiff_t alloc_local_;
  ptrdiff_t local_n0_;
  ptrdiff_t local_0_start_;
  int N_;

  FFTWMPIReferenceRtoC(int N, MPI_Comm comm) : N_(N) {
    ptrdiff_t n_real[D];
    for (int d = 0; d < D; ++d)
      n_real[d] = N;

    // Query distribution using the complex shape (last dim = N/2+1)
    ptrdiff_t n_complex[D];
    for (int d = 0; d < D - 1; ++d)
      n_complex[d] = N;
    n_complex[D - 1] = N / 2 + 1;

    alloc_local_ =
        fftw_mpi_local_size(D, n_complex, comm, &local_n0_, &local_0_start_);
    buffer_ = fftw_alloc_real(2 * alloc_local_);

    forward_plan_ = fftw_mpi_plan_dft_r2c(
        D, n_real, buffer_, reinterpret_cast<fftw_complex *>(buffer_), comm,
        FFTW_ESTIMATE);
    backward_plan_ = fftw_mpi_plan_dft_c2r(
        D, n_real, reinterpret_cast<fftw_complex *>(buffer_), buffer_, comm,
        FFTW_ESTIMATE);

    // Fill buffer with Gaussian data in FFTW's padded layout
    const double center = N / 2.0;
    const double sigma = 4.0;
    const ptrdiff_t padded_last = 2 * (N / 2 + 1);

    // Stride for dimensions 1..D-2 (each is N)
    ptrdiff_t inner_stride = 1;
    for (int d = 1; d < D - 1; ++d)
      inner_stride *= N;

    // Zero the entire buffer (including padding)
    ptrdiff_t total_doubles = local_n0_ * inner_stride * padded_last;
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < total_doubles; ++i)
      buffer_[i] = 0.0;

    // Fill real values
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < local_n0_; ++i) {
      ptrdiff_t gi = local_0_start_ + i;
      for (ptrdiff_t j = 0; j < inner_stride; ++j) {
        for (ptrdiff_t k = 0; k < N; ++k) {
          double r2 = 0.0;
          double x0 = gi - center;
          r2 += x0 * x0;

          ptrdiff_t rem = j;
          for (int d = D - 2; d >= 1; --d) {
            ptrdiff_t idx_d = rem % N;
            rem /= N;
            double x = idx_d - center;
            r2 += x * x;
          }

          double xk = k - center;
          r2 += xk * xk;

          ptrdiff_t flat = (i * inner_stride + j) * padded_last + k;
          buffer_[flat] = std::exp(-r2 / (2.0 * sigma * sigma));
        }
      }
    }
  }

  ~FFTWMPIReferenceRtoC() {
    fftw_destroy_plan(forward_plan_);
    fftw_destroy_plan(backward_plan_);
    fftw_free(buffer_);
  }

  void execute() {
    fftw_execute(forward_plan_);
    fftw_execute(backward_plan_);
  }

  // Non-copyable
  FFTWMPIReferenceRtoC(const FFTWMPIReferenceRtoC &) = delete;
  FFTWMPIReferenceRtoC &operator=(const FFTWMPIReferenceRtoC &) = delete;
};

inline std::string
format_csv(const std::string &benchmark, const std::string &type,
           const std::string &test_name, int N, int D, int mpi_procs,
           int fftw_threads, double parafaft_mean, double parafaft_std,
           double fftw_mean, double fftw_std, int iterations) {
  std::ostringstream oss;
  oss << benchmark << "," << type << "," << test_name << "," << N << "," << D
      << "," << mpi_procs << "," << fftw_threads << "," << parafaft_mean << ","
      << parafaft_std << "," << fftw_mean << "," << fftw_std << ","
      << iterations;
  return oss.str();
}

inline void write_csv(const std::string &filename, int mpi_procs, int threads,
                      int N, double parafaft_mean, double parafaft_std,
                      double fftw_mean, double fftw_std, int iterations) {
  std::ifstream check_file(filename);
  bool file_exists = check_file.good();
  check_file.close();

  std::ofstream csv_file(filename, std::ios::app);
  if (!file_exists) {
    csv_file << "mpi_procs,threads,N,parafaft_mean,parafaft_std,fftw_mean,fftw_"
                "std,iterations\n";
  }
  csv_file << std::fixed << std::setprecision(6);
  csv_file << mpi_procs << "," << threads << "," << N << "," << parafaft_mean
           << "," << parafaft_std << "," << fftw_mean << "," << fftw_std << ","
           << iterations << "\n";
  csv_file.close();
}

} // namespace parafaft_bench

#endif // PARAFAFT_BENCHMARK_HELPERS_HPP
