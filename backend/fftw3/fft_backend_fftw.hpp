/**
 * @file fft_backend_fftw.hpp
 * @brief FFTW3 backend implementation for ParaFaFT.
 *
 * This header provides an FFT backend using the FFTW3 library for CPU-based
 * FFT operations. Supports C2C, R2C, and C2R transforms as required by
 * ParaFaFT and ParaFaFT_R2C.
 *
 * Threading Support:
 * - If compiled with PARAFAFT_FFTW_THREADS, uses libfftw3_threads (POSIX threads)
 * - If compiled with PARAFAFT_FFTW_OMP, uses fftw3_omp (OpenMP)
 * - Otherwise, uses serial FFTW
 */

#ifndef PARAFAFT_BACKEND_FFTW_HPP
#define PARAFAFT_BACKEND_FFTW_HPP

#include <fftw3.h>
#include <complex>
#include <vector>
#include <cstring>
#include <thread>
#include <cstdlib>
#include <mpi.h>
#include "../fft_backend.hpp"

namespace parafaft
{
  /**
   * @enum FFTBackendType
   * @brief Specifies the FFTW backend threading type.
   */
  enum class FFTBackendType {
    Serial,   ///< Serial (single-threaded) FFTW
    Threads,  ///< POSIX threads (libfftw3_threads)
    OpenMP    ///< OpenMP-based (fftw3_omp)
  };

  /**
   * @class FFTWBackend
   * @brief FFTW3 backend for CPU-based FFT operations.
   *
   * Provides an interface compatible with ParaFaFT for executing FFT transforms
   * using FFTW3. Supports C2C, R2C, and C2R transforms with in-place operations.
   *
   * Memory management: Uses std::vector for host memory allocation.
   * All data pointers passed to this backend must be host (CPU) pointers.
   *
   * @note FFTW plans are not copyable; this class is move-only.
   */
  class FFTWBackend
  {
  public:
    using Complex = std::complex<double>;       ///< Complex number type (CPU-compatible)
    using Buffer = std::vector<double>;         ///< Real buffer type (host memory)
    using ComplexBuffer = std::vector<Complex>; ///< Complex buffer type (host memory)

    /**
     * @brief Construct an FFTW backend with storage for the given number of stages.
     *
     * @param num_stages Number of FFT stages (typically D for D-dimensional transform)
     * @param plan_flag Planning strategy (default: Estimate for quick planning)
     */
    explicit FFTWBackend(int num_stages, FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
        : forward_plans_(num_stages, nullptr), backward_plans_(num_stages, nullptr), backend_type_(FFTBackendType::Serial),
          num_threads_(1), plan_flag_(convertPlanFlag(plan_flag))
    {
#if defined(PARAFAFT_FFTW_THREADS)
      backend_type_ = FFTBackendType::Threads;
      fftw_init_threads();
      num_threads_ = detect_thread_count(MPI_COMM_SELF);
      fftw_plan_with_nthreads(num_threads_);
#elif defined(PARAFAFT_FFTW_OMP)
      backend_type_ = FFTBackendType::OpenMP;
      fftw_init_threads();
      num_threads_ = detect_thread_count(MPI_COMM_SELF);
      fftw_plan_with_nthreads(num_threads_);
#endif
    }

    /**
     * @brief Construct an FFTW backend with MPI communicator for thread count calculation.
     *
     * @param num_stages Number of FFT stages (typically D for D-dimensional transform)
     * @param comm MPI communicator (used to determine number of MPI tasks for thread count)
     * @param plan_flag Planning strategy (default: Estimate for quick planning)
     */
    explicit FFTWBackend(int num_stages, MPI_Comm comm, FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
        : forward_plans_(num_stages, nullptr), backward_plans_(num_stages, nullptr), backend_type_(FFTBackendType::Serial),
          num_threads_(1), plan_flag_(convertPlanFlag(plan_flag))
    {
#if defined(PARAFAFT_FFTW_THREADS)
      backend_type_ = FFTBackendType::Threads;
      fftw_init_threads();
      num_threads_ = detect_thread_count(comm);
      fftw_plan_with_nthreads(num_threads_);
#elif defined(PARAFAFT_FFTW_OMP)
      backend_type_ = FFTBackendType::OpenMP;
      fftw_init_threads();
      num_threads_ = detect_thread_count(comm);
      fftw_plan_with_nthreads(num_threads_);
#else
      (void)comm; // Suppress unused parameter warning
#endif
    }

    /**
     * @brief Create and store FFTW plans for a specific stage (C2C transforms).
     *
     * Creates both forward and backward plans for the given stage, using the
     * configured planning flag (FFTW_ESTIMATE by default, configurable via
     * constructor).
     *
     * @param stage Stage index for plan storage (0 to num_stages-1)
     * @param length FFT length (number of complex elements per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param data Host pointer for plan creation (alignment reference)
     * @param stride Stride between consecutive elements in a transform
     * @param dist Distance between first elements of consecutive transforms
     */
    void create_stage_plan(int stage, int length, int batch, Complex *data, int stride, int dist)
    {
      int n[] = {length};
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(data);

      // Create forward plan (bound to data pointer)
      forward_plans_[stage] = fftw_plan_many_dft(1, n, batch, fftw_data, NULL, stride, dist, fftw_data, NULL, stride,
                                                 dist, FFTW_FORWARD, plan_flag_);

      // Create backward plan (bound to data pointer)
      backward_plans_[stage] = fftw_plan_many_dft(1, n, batch, fftw_data, NULL, stride, dist, fftw_data, NULL, stride,
                                                  dist, FFTW_BACKWARD, plan_flag_);
    }

    /**
     * @brief Create in-place R2C plan for padded memory layout.
     *
     * Creates an R2C (real-to-complex) plan for in-place transformation.
     * The input is N real values per transform, output is N/2+1 complex values,
     * stored in the same padded buffer.
     *
     * @param length Real-space input length N (number of real values per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param padded_real Host pointer to padded real buffer (2*(N/2+1) doubles per row)
     * @param stride Element stride (typically 1 for contiguous data)
     * @param dist Distance between batches in doubles (should be 2*(N/2+1) for in-place)
     */
    void create_r2c_inplace_plan(int length,          // Real-space input length N
                                 int batch,           // Number of transforms
                                 double *padded_real, // Padded real buffer
                                 int stride,          // Stride
                                 int dist             // Distance between batches (padded: 2*(N/2+1) doubles)
    )
    {
      int n[] = {length};
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(padded_real);

      // In-place R2C: real → complex in same buffer
      r2c_inplace_plan_ =
          fftw_plan_many_dft_r2c(1, n, batch, padded_real, NULL, stride, dist, // real input (dist in doubles)
                                 fftw_data, NULL, stride, dist / 2,            // complex output (dist/2 in complex)
                                 plan_flag_);
    }

    /**
     * @brief Execute in-place R2C transform.
     *
     * Transforms real data to complex data in-place. The input buffer must be
     * padded (2*(N/2+1) doubles per row) to accommodate the complex output.
     *
     * @param padded_real Host pointer to padded real input buffer (overwritten with complex output)
     */
    void execute_r2c_inplace(double *padded_real)
    {
      fftw_execute_dft_r2c(r2c_inplace_plan_, padded_real, reinterpret_cast<fftw_complex *>(padded_real));
    }

    /**
     * @brief Create in-place C2R plan for padded memory layout.
     *
     * Creates a C2R (complex-to-real) plan for in-place transformation.
     * The input is N/2+1 complex values per transform, output is N real values,
     * stored in the same padded buffer.
     *
     * @param length Real-space output length N (number of real values per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param padded_real Host pointer to padded buffer (used for both complex input and real output)
     * @param stride Element stride (typically 1 for contiguous data)
     * @param dist Distance between batches (padded: 2*(N/2+1) doubles)
     */
    void create_c2r_inplace_plan(int length,          // Real-space output length N
                                 int batch,           // Number of transforms
                                 double *padded_real, // Padded real buffer (also used as complex input)
                                 int stride,          // Stride
                                 int dist             // Distance between batches (padded: 2*(N/2+1))
    )
    {
      int n[] = {length};
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(padded_real);

      // In-place C2R: complex → real in same buffer
      c2r_inplace_plan_ =
          fftw_plan_many_dft_c2r(1, n, batch, fftw_data, NULL, stride, dist / 2, // dist/2 for complex stride
                                 padded_real, NULL, stride, dist, plan_flag_);
    }

    /**
     * @brief Execute in-place C2R transform.
     *
     * Transforms complex data to real data in-place. The buffer layout must
     * match the plan created by create_c2r_inplace_plan().
     *
     * @param padded_real Host pointer to buffer (complex input, real output)
     */
    void execute_c2r_inplace(double *padded_real)
    {
      fftw_execute_dft_c2r(c2r_inplace_plan_, reinterpret_cast<fftw_complex *>(padded_real), padded_real);
    }

    /**
     * @brief Execute pre-created plan for specified stage.
     *
     * Executes the C2C transform for the given stage using the new-array
     * execution interface (fftw_execute_dft), allowing the plan to be
     * applied to data at a different address with the same layout.
     *
     * @param stage Stage index (0 to num_stages-1)
     * @param direction Transform direction (Forward or Backward)
     * @param data Host pointer to complex data buffer
     */
    void execute_stage(int stage, FFTDirection direction, Complex *data)
    {
      fftw_plan plan = (direction == FFTDirection::Forward) ? forward_plans_[stage] : backward_plans_[stage];
      fftw_complex *fftw_data = reinterpret_cast<fftw_complex *>(data);
      fftw_execute_dft(plan, fftw_data, fftw_data);
    }

    /**
     * @brief Copy memory between host buffers.
     *
     * Standard memory copy using std::memcpy. For CPU-based operations.
     *
     * @param dest Destination pointer
     * @param src Source pointer
     * @param bytes Number of bytes to copy
     */
    void memcpy(void *dest, const void *src, size_t bytes) const { std::memcpy(dest, src, bytes); }

    /// Always use MPI_Alltoallw with derived types on CPU (efficient)
    static constexpr bool use_alltoallw = true;

    /**
     * @brief 2D strided memory copy.
     *
     * Copies `height` rows of `width` bytes each, with independent
     * source and destination pitches (strides between rows).
     */
    void memcpy2d(void *dst, size_t dpitch, const void *src,
                  size_t spitch, size_t width, size_t height) const {
      for (size_t row = 0; row < height; ++row) {
        std::memcpy(static_cast<char *>(dst) + row * dpitch,
                    static_cast<const char *>(src) + row * spitch, width);
      }
    }

    /// No-op: CPU operations are synchronous
    void sync() const {}

    /// No P2P on CPU
    static constexpr bool use_p2p = false;

    /**
     * @brief Destructor. Cleans up all FFTW plans.
     */
    ~FFTWBackend()
    {
      for (auto plan : forward_plans_) {
        if (plan) fftw_destroy_plan(plan);
      }
      for (auto plan : backward_plans_) {
        if (plan) fftw_destroy_plan(plan);
      }
      if (r2c_inplace_plan_) fftw_destroy_plan(r2c_inplace_plan_);
      if (c2r_inplace_plan_) fftw_destroy_plan(c2r_inplace_plan_);

#if defined(PARAFAFT_FFTW_THREADS) || defined(PARAFAFT_FFTW_OMP)
      fftw_cleanup_threads();
#endif
    }

    /// @brief Deleted copy constructor (FFTW plans cannot be safely copied)
    FFTWBackend(const FFTWBackend &) = delete;
    /// @brief Deleted copy assignment (FFTW plans cannot be safely copied)
    FFTWBackend &operator=(const FFTWBackend &) = delete;

    /**
     * @brief Move constructor.
     *
     * Transfers ownership of all plans from another backend instance.
     * The moved-from object is left in a valid but empty state.
     *
     * @param other Backend to move from
     */
    FFTWBackend(FFTWBackend &&other) noexcept
        : forward_plans_(std::move(other.forward_plans_)), backward_plans_(std::move(other.backward_plans_)),
          r2c_inplace_plan_(other.r2c_inplace_plan_), c2r_inplace_plan_(other.c2r_inplace_plan_),
          plan_flag_(other.plan_flag_)
    {
      std::fill(other.forward_plans_.begin(), other.forward_plans_.end(), nullptr);
      std::fill(other.backward_plans_.begin(), other.backward_plans_.end(), nullptr);
      other.r2c_inplace_plan_ = nullptr;
      other.c2r_inplace_plan_ = nullptr;
    }

    // ========== FFTW Wisdom Support ==========

    /**
     * @brief Export accumulated FFTW wisdom to a file.
     *
     * Saves the current FFTW planning results (wisdom) so they can be reloaded
     * in future runs, avoiding re-benchmarking when using FFTW_MEASURE or
     * FFTW_PATIENT. Only meaningful when plans were created with Measure or Patient.
     *
     * @param filename Path to the wisdom file to create/overwrite.
     * @return true if wisdom was successfully exported, false otherwise.
     */
    static bool exportWisdom(const char *filename)
    {
      return fftw_export_wisdom_to_filename(filename) != 0;
    }

    /**
     * @brief Import FFTW wisdom from a file.
     *
     * Loads previously saved planning results. Must be called before constructing
     * the backend (i.e., before plan creation) for the wisdom to take effect.
     *
     * @param filename Path to the wisdom file to read.
     * @return true if wisdom was successfully imported, false otherwise.
     */
    static bool importWisdom(const char *filename)
    {
      return fftw_import_wisdom_from_filename(filename) != 0;
    }

    /**
     * @brief Discard all accumulated FFTW wisdom.
     *
     * Clears the internal wisdom database. Subsequent plans will be created
     * from scratch.
     */
    static void forgetWisdom()
    {
      fftw_forget_wisdom();
    }

  private:
    std::vector<fftw_plan> forward_plans_;  ///< Forward C2C plans (one per stage)
    std::vector<fftw_plan> backward_plans_; ///< Backward C2C plans (one per stage)
    fftw_plan r2c_inplace_plan_ = nullptr;  ///< In-place R2C plan
    fftw_plan c2r_inplace_plan_ = nullptr;  ///< In-place C2R plan

    FFTBackendType backend_type_; ///< Threading backend type
    int num_threads_;              ///< Number of threads for FFTW
    unsigned plan_flag_;           ///< FFTW planning flag (FFTW_ESTIMATE, FFTW_MEASURE, etc.)

    /**
     * @brief Convert FFTPlanFlag enum to FFTW's unsigned flag value.
     *
     * @param flag The FFTPlanFlag enum value
     * @return Corresponding FFTW flag constant
     */
    static unsigned convertPlanFlag(FFTPlanFlag flag)
    {
      switch (flag) {
      case FFTPlanFlag::Measure:
        return FFTW_MEASURE;
      case FFTPlanFlag::Patient:
        return FFTW_PATIENT;
      case FFTPlanFlag::Estimate:
      default:
        return FFTW_ESTIMATE;
      }
    }

    /**
     * @brief Detect optimal thread count based on environment and hardware.
     *
     * Priority:
     * 1. OMP_NUM_THREADS environment variable (if using OpenMP backend)
     * 2. KOKKOS_NUM_THREADS environment variable
     * 3. std::thread::hardware_concurrency() / mpi_task_count
     * 4. Default to 1 (serial)
     *
     * @param comm MPI communicator for determining task count
     * @return Optimal number of threads
     */
    int detect_thread_count(MPI_Comm comm)
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
  };

} // namespace parafaft

#endif // PARAFAFT_BACKEND_FFTW_HPP
