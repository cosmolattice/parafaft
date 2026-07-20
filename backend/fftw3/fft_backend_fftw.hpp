/**
 * @file fft_backend_fftw.hpp
 * @brief FFTW3 backend implementation for ParaFaFT.
 *
 * This header provides an FFT backend using the FFTW3 library for CPU-based
 * FFT operations. Supports C2C, R2C, and C2R transforms as required by
 * ParaFaFT and ParaFaFT_R2C.
 *
 * Precision: FFTWBackend is a class template parameterised on FloatType.
 *   - FFTWBackend<double> (default) uses the double-precision fftw_* API
 *   - FFTWBackend<float>  uses the single-precision fftwf_* API, only
 *     available when PARAFAFT_FFTW3F_AVAILABLE is defined (set by CMake
 *     when libfftw3f is found).
 *
 * Threading Support:
 * Two strategies, chosen at configure time.
 *
 * 1. Batch-parallel (PARAFAFT_BATCH_PARALLEL, the default). Every stage is a
 *    batch of independent 1D transforms, so the batch is split across threads
 *    and each chunk runs a *serial* FFTW plan. FFTW's own threading is pinned
 *    to one thread to avoid nesting. The executor is OpenMP or a std::thread
 *    pool (see ../parallel_for.hpp). Buffers are first-touched with the same
 *    block partition so pages land in the touching thread's NUMA domain.
 *
 * 2. FFTW-internal threading (PARAFAFT_BATCH_PARALLEL=OFF). One plan per stage
 *    with fftw_plan_with_nthreads(N), parallelising *within* each transform:
 *    - PARAFAFT_FFTW_THREADS uses libfftw3_threads (POSIX threads)
 *    - PARAFAFT_FFTW_OMP uses fftw3_omp (OpenMP)
 *
 * Both are expressed as the same chunked-plan structure — strategy 2 is simply
 * the single-chunk case — so there is one execution path to maintain.
 */

#ifndef PARAFAFT_BACKEND_FFTW_HPP
#define PARAFAFT_BACKEND_FFTW_HPP

#include <fftw3.h>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <vector>
#include <thread>
#include <cstdlib>
#include <mpi.h>
#include "../fft_backend.hpp"
#include "../parallel_for.hpp"

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
   * @brief Precision-specific FFTW symbol mapping.
   *
   * Routes FFTWBackend<FloatType> to the right FFTW symbol family
   * (fftw_* for double, fftwf_* for float). The float specialization
   * is only defined when libfftw3f was found at configure time
   * (PARAFAFT_FFTW3F_AVAILABLE). Instantiating FFTWBackend<float>
   * without that macro yields a clean "incomplete type" error.
   */
  template <typename T> struct fftw_traits;

  template <> struct fftw_traits<double> {
    using real_t = double;
    using complex_t = fftw_complex;
    using plan_t = fftw_plan;
    using iodim64_t = fftw_iodim64;

    static real_t *alloc_real(size_t n) { return fftw_alloc_real(n); }
    static complex_t *alloc_complex(size_t n) { return fftw_alloc_complex(n); }
    static void free_mem(void *p) { fftw_free(p); }

    static int init_threads() { return fftw_init_threads(); }
    static void plan_with_nthreads(int n) { fftw_plan_with_nthreads(n); }
    static void cleanup_threads() { fftw_cleanup_threads(); }

    static plan_t plan_guru64_dft(int rank, const iodim64_t *dims, int howmany_rank,
                                  const iodim64_t *howmany_dims, complex_t *in, complex_t *out,
                                  int sign, unsigned flags) {
      return fftw_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
    }
    static plan_t plan_guru64_dft_r2c(int rank, const iodim64_t *dims, int howmany_rank,
                                      const iodim64_t *howmany_dims, real_t *in, complex_t *out,
                                      unsigned flags) {
      return fftw_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }
    static plan_t plan_guru64_dft_c2r(int rank, const iodim64_t *dims, int howmany_rank,
                                      const iodim64_t *howmany_dims, complex_t *in, real_t *out,
                                      unsigned flags) {
      return fftw_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }
    static void execute_dft(plan_t p, complex_t *in, complex_t *out) { fftw_execute_dft(p, in, out); }
    static void execute_dft_r2c(plan_t p, real_t *in, complex_t *out) { fftw_execute_dft_r2c(p, in, out); }
    static void execute_dft_c2r(plan_t p, complex_t *in, real_t *out) { fftw_execute_dft_c2r(p, in, out); }
    static void destroy_plan(plan_t p) { fftw_destroy_plan(p); }

    static int export_wisdom(const char *fn) { return fftw_export_wisdom_to_filename(fn); }
    static int import_wisdom(const char *fn) { return fftw_import_wisdom_from_filename(fn); }
    static void forget_wisdom() { fftw_forget_wisdom(); }
  };

#ifdef PARAFAFT_FFTW3F_AVAILABLE
  template <> struct fftw_traits<float> {
    using real_t = float;
    using complex_t = fftwf_complex;
    using plan_t = fftwf_plan;
    using iodim64_t = fftwf_iodim64;

    static real_t *alloc_real(size_t n) { return fftwf_alloc_real(n); }
    static complex_t *alloc_complex(size_t n) { return fftwf_alloc_complex(n); }
    static void free_mem(void *p) { fftwf_free(p); }

    static int init_threads() { return fftwf_init_threads(); }
    static void plan_with_nthreads(int n) { fftwf_plan_with_nthreads(n); }
    static void cleanup_threads() { fftwf_cleanup_threads(); }

    static plan_t plan_guru64_dft(int rank, const iodim64_t *dims, int howmany_rank,
                                  const iodim64_t *howmany_dims, complex_t *in, complex_t *out,
                                  int sign, unsigned flags) {
      return fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
    }
    static plan_t plan_guru64_dft_r2c(int rank, const iodim64_t *dims, int howmany_rank,
                                      const iodim64_t *howmany_dims, real_t *in, complex_t *out,
                                      unsigned flags) {
      return fftwf_plan_guru64_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }
    static plan_t plan_guru64_dft_c2r(int rank, const iodim64_t *dims, int howmany_rank,
                                      const iodim64_t *howmany_dims, complex_t *in, real_t *out,
                                      unsigned flags) {
      return fftwf_plan_guru64_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }
    static void execute_dft(plan_t p, complex_t *in, complex_t *out) { fftwf_execute_dft(p, in, out); }
    static void execute_dft_r2c(plan_t p, real_t *in, complex_t *out) { fftwf_execute_dft_r2c(p, in, out); }
    static void execute_dft_c2r(plan_t p, complex_t *in, real_t *out) { fftwf_execute_dft_c2r(p, in, out); }
    static void destroy_plan(plan_t p) { fftwf_destroy_plan(p); }

    static int export_wisdom(const char *fn) { return fftwf_export_wisdom_to_filename(fn); }
    static int import_wisdom(const char *fn) { return fftwf_import_wisdom_from_filename(fn); }
    static void forget_wisdom() { fftwf_forget_wisdom(); }
  };
#endif // PARAFAFT_FFTW3F_AVAILABLE

  /**
   * @class FFTWBackend
   * @brief FFTW3 backend for CPU-based FFT operations.
   *
   * Provides an interface compatible with ParaFaFT for executing FFT transforms
   * using FFTW3. Supports C2C, R2C, and C2R transforms with in-place operations.
   *
   * @tparam FloatType Precision of the transform: `double` (always available)
   *         or `float` (requires libfftw3f, detected via PARAFAFT_FFTW3F_AVAILABLE).
   *
   * Memory management: Uses FFTW's aligned allocators for host memory.
   * All data pointers passed to this backend must be host (CPU) pointers.
   *
   * @note FFTW plans are not copyable; this class is move-only.
   */
  template <typename FloatTypeT = double>
  class FFTWBackend
  {
  public:
    using FloatType = FloatTypeT;                    ///< Scalar floating-point type
    using Complex = std::complex<FloatType>;          ///< Complex number type (CPU-compatible)
  private:
    using traits = fftw_traits<FloatType>;
    using fftw_complex_t = typename traits::complex_t;
    using fftw_plan_t = typename traits::plan_t;

  public:

    /**
     * @brief FFTW-aligned real buffer.
     *
     * Provides 16/32/64-byte alignment required for FFTW's SIMD codepaths
     * (SSE2, AVX, AVX-512) via the precision-appropriate fftw(f)_alloc_real.
     */
    class AlignedBuffer {
      FloatType *data_ = nullptr;
      size_t size_ = 0;
    public:
      AlignedBuffer() = default;
      explicit AlignedBuffer(size_t n) : data_(traits::alloc_real(n)), size_(n) {
        if (n > 0 && !data_) throw std::bad_alloc();
      }
      ~AlignedBuffer() { if (data_) traits::free_mem(data_); }
      AlignedBuffer(const AlignedBuffer &) = delete;
      AlignedBuffer &operator=(const AlignedBuffer &) = delete;
      AlignedBuffer(AlignedBuffer &&other) noexcept
          : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
      }
      AlignedBuffer &operator=(AlignedBuffer &&other) noexcept {
        if (this != &other) {
          if (data_) traits::free_mem(data_);
          data_ = other.data_;
          size_ = other.size_;
          other.data_ = nullptr;
          other.size_ = 0;
        }
        return *this;
      }
      void resize(size_t n) {
        if (data_) traits::free_mem(data_);
        data_ = (n > 0) ? traits::alloc_real(n) : nullptr;
        size_ = n;
        if (n > 0 && !data_) throw std::bad_alloc();
      }
      FloatType *data() { return data_; }
      const FloatType *data() const { return data_; }
      size_t size() const { return size_; }
    };

    /**
     * @brief FFTW-aligned complex buffer.
     *
     * Provides 16/32/64-byte alignment required for FFTW's SIMD codepaths.
     * FFTW's new-array execute API requires the runtime buffer to have the
     * same alignment as the planning buffer — using this class for both
     * ensures FFTW can select optimal SIMD plans.
     */
    class AlignedComplexBuffer {
      fftw_complex_t *data_ = nullptr;
      size_t size_ = 0;
    public:
      AlignedComplexBuffer() = default;
      explicit AlignedComplexBuffer(size_t n) : data_(traits::alloc_complex(n)), size_(n) {
        if (n > 0 && !data_) throw std::bad_alloc();
      }
      ~AlignedComplexBuffer() { if (data_) traits::free_mem(data_); }
      AlignedComplexBuffer(const AlignedComplexBuffer &) = delete;
      AlignedComplexBuffer &operator=(const AlignedComplexBuffer &) = delete;
      AlignedComplexBuffer(AlignedComplexBuffer &&other) noexcept
          : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
      }
      AlignedComplexBuffer &operator=(AlignedComplexBuffer &&other) noexcept {
        if (this != &other) {
          if (data_) traits::free_mem(data_);
          data_ = other.data_;
          size_ = other.size_;
          other.data_ = nullptr;
          other.size_ = 0;
        }
        return *this;
      }
      void resize(size_t n) {
        if (data_) traits::free_mem(data_);
        data_ = (n > 0) ? traits::alloc_complex(n) : nullptr;
        size_ = n;
        if (n > 0 && !data_) throw std::bad_alloc();
      }
      Complex *data() { return reinterpret_cast<Complex *>(data_); }
      const Complex *data() const { return reinterpret_cast<const Complex *>(data_); }
      size_t size() const { return size_; }
    };

    using Buffer = AlignedBuffer;               ///< Real buffer type (FFTW-aligned)
    using ComplexBuffer = AlignedComplexBuffer;  ///< Complex buffer type (FFTW-aligned)

    /// Which executor the batch-parallel path was compiled against.
#if defined(PARAFAFT_PARALLEL_OMP)
    static constexpr FFTBackendType kExecutorBackendType = FFTBackendType::OpenMP;
#elif defined(PARAFAFT_PARALLEL_THREADS)
    static constexpr FFTBackendType kExecutorBackendType = FFTBackendType::Threads;
#else
    static constexpr FFTBackendType kExecutorBackendType = FFTBackendType::Serial;
#endif

    /**
     * @brief Construct an FFTW backend with storage for the given number of stages.
     *
     * @param num_stages Number of FFT stages (typically D for D-dimensional transform)
     * @param plan_flag Planning strategy (default: Estimate for quick planning)
     */
    explicit FFTWBackend(int num_stages, FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
        : FFTWBackend(num_stages, MPI_COMM_SELF, plan_flag)
    {
    }

    /**
     * @brief Construct an FFTW backend with MPI communicator for thread count calculation.
     *
     * @param num_stages Number of FFT stages (typically D for D-dimensional transform)
     * @param comm MPI communicator (used to determine number of MPI tasks for thread count)
     * @param plan_flag Planning strategy (default: Estimate for quick planning)
     */
    explicit FFTWBackend(int num_stages, MPI_Comm comm, FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
        : forward_plans_(num_stages), backward_plans_(num_stages), backend_type_(FFTBackendType::Serial),
          num_threads_(detect_thread_count(comm)), plan_flag_(convertPlanFlag(plan_flag))
    {
#if defined(PARAFAFT_BATCH_PARALLEL)
      // We parallelise across the batch ourselves, so FFTW must stay serial —
      // otherwise each of our threads would spawn its own FFTW thread team.
      backend_type_ = kExecutorBackendType;
#if defined(PARAFAFT_FFTW_THREADS) || defined(PARAFAFT_FFTW_OMP)
      traits::init_threads();
      traits::plan_with_nthreads(1);
#endif
      executor_.reset(new detail::ParallelExecutor(num_threads_));
#else
      // FFTW does the threading internally; a single chunk per stage.
#if defined(PARAFAFT_FFTW_THREADS)
      backend_type_ = FFTBackendType::Threads;
      traits::init_threads();
      traits::plan_with_nthreads(num_threads_);
#elif defined(PARAFAFT_FFTW_OMP)
      backend_type_ = FFTBackendType::OpenMP;
      traits::init_threads();
      traits::plan_with_nthreads(num_threads_);
#else
      num_threads_ = 1;
#endif
      executor_.reset(new detail::ParallelExecutor(1));
#endif
    }

    /// @brief Threads this backend dispatches FFT chunks to (1 when FFTW threads internally).
    int num_chunks() const { return executor_ ? executor_->size() : 1; }

    /// @brief Threads requested per rank, from OMP_NUM_THREADS or the node topology.
    int num_threads() const { return num_threads_; }

    /// @brief Which threading strategy was compiled in.
    FFTBackendType backend_type() const { return backend_type_; }

    /**
     * @brief Pre-fault a buffer from every worker thread (NUMA first touch).
     *
     * Linux binds a page to the NUMA domain of the thread that first writes it.
     * Allocation alone does not fault pages in, so without this the whole
     * working set is owned by whichever thread initialised it and every worker
     * pulls from that one memory controller.
     *
     * The buffer is split into page-aligned blocks using the same block
     * partition as the FFT chunks. This matches the access pattern for stages
     * whose transforms are contiguous (dist-major); for the strided first-axis
     * stage a thread's chunk is interleaved across the buffer, so this is a
     * distribution heuristic rather than an exact match.
     *
     * @param data  Buffer start. Ignored when null.
     * @param bytes Buffer length in bytes.
     */
    void first_touch(void *data, std::size_t bytes)
    {
      if (data == nullptr || bytes == 0) return;

      constexpr std::size_t kPageSize = 4096;
      char *base = static_cast<char *>(data);
      const std::size_t num_pages = (bytes + kPageSize - 1) / kPageSize;

      executor_->run([base, bytes, num_pages](int tid, int nthreads) {
        std::size_t first_page = 0, last_page = 0;
        detail::chunk_range(num_pages, tid, nthreads, first_page, last_page);
        const std::size_t begin = first_page * kPageSize;
        const std::size_t end = std::min(last_page * kPageSize, bytes);
        if (begin < end) std::memset(base + begin, 0, end - begin);
      });
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
    void create_stage_plan(int stage, int length, std::size_t batch, Complex *data,
                           std::ptrdiff_t stride, std::ptrdiff_t dist)
    {
      // Batched 1D transform: dims describe the transform axis (length N,
      // in-stride == out-stride == stride); howmany_dims describe the batch
      // (howmany = chunk size, dist between batches for in/out).
      // Using guru64 so strides/batch are ptrdiff_t — avoids the int-overflow
      // trap in plan_many_dft when batch*dist exceeds 2^31.
      //
      // One plan per chunk, each planned at the exact offset its thread will
      // execute on. Planning at the offset (rather than at `data` with a
      // shifted execute) is what keeps the new-array execute legal: FFTW
      // records the planning pointer's alignment and requires the runtime
      // pointer to match, and equal offsets into equally-aligned fftw_alloc'd
      // buffers do match.
      forward_plans_[stage] = ChunkedPlan(num_chunks());
      backward_plans_[stage] = ChunkedPlan(num_chunks());

      for (int chunk = 0; chunk < num_chunks(); ++chunk) {
        std::size_t begin = 0, end = 0;
        detail::chunk_range(batch, chunk, num_chunks(), begin, end);
        if (begin == end) continue; // more threads than transforms

        const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(begin) * dist;
        fftw_complex_t *chunk_data = reinterpret_cast<fftw_complex_t *>(data + offset);

        typename traits::iodim64_t dims[] = {{static_cast<std::ptrdiff_t>(length), stride, stride}};
        typename traits::iodim64_t howmany_dims[] = {
            {static_cast<std::ptrdiff_t>(end - begin), dist, dist}};

        forward_plans_[stage].offsets[chunk] = offset;
        backward_plans_[stage].offsets[chunk] = offset;
        forward_plans_[stage].plans[chunk] = traits::plan_guru64_dft(
            1, dims, 1, howmany_dims, chunk_data, chunk_data, FFTW_FORWARD, plan_flag_);
        backward_plans_[stage].plans[chunk] = traits::plan_guru64_dft(
            1, dims, 1, howmany_dims, chunk_data, chunk_data, FFTW_BACKWARD, plan_flag_);
      }
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
     * @param padded_real Host pointer to padded real buffer (2*(N/2+1) FloatType per row)
     * @param stride Element stride (typically 1 for contiguous data)
     * @param dist Distance between batches in FloatType elements (should be 2*(N/2+1) for in-place)
     */
    void create_r2c_inplace_plan(int length,               // Real-space input length N
                                 std::size_t batch,         // Number of transforms
                                 FloatType *padded_real,    // Padded real buffer
                                 std::ptrdiff_t stride,     // Stride
                                 std::ptrdiff_t dist        // Distance between batches (padded: 2*(N/2+1) scalars)
    )
    {
      // For R2C: output dist is dist/2 (complex elements vs scalar elements).
      // Using guru64 so batch*dist isn't truncated to int inside FFTW.
      // Offsets are in FloatType units; the complex view starts at the same
      // byte address, so one offset serves both.
      r2c_inplace_plan_ = ChunkedPlan(num_chunks());

      for (int chunk = 0; chunk < num_chunks(); ++chunk) {
        std::size_t begin = 0, end = 0;
        detail::chunk_range(batch, chunk, num_chunks(), begin, end);
        if (begin == end) continue;

        const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(begin) * dist;
        FloatType *chunk_real = padded_real + offset;

        typename traits::iodim64_t dims[] = {{static_cast<std::ptrdiff_t>(length), stride, stride}};
        typename traits::iodim64_t howmany_dims[] = {
            {static_cast<std::ptrdiff_t>(end - begin), dist, dist / 2}};

        r2c_inplace_plan_.offsets[chunk] = offset;
        r2c_inplace_plan_.plans[chunk] = traits::plan_guru64_dft_r2c(
            1, dims, 1, howmany_dims, chunk_real,
            reinterpret_cast<fftw_complex_t *>(chunk_real), plan_flag_);
      }
    }

    /**
     * @brief Execute in-place R2C transform.
     *
     * Transforms real data to complex data in-place. The input buffer must be
     * padded (2*(N/2+1) FloatType values per row) to accommodate the complex output.
     *
     * @param padded_real Host pointer to padded real input buffer (overwritten with complex output)
     */
    void execute_r2c_inplace(FloatType *padded_real)
    {
      const ChunkedPlan &plan = r2c_inplace_plan_;
      executor_->run([&plan, padded_real](int chunk, int) {
        if (plan.plans[chunk] == nullptr) return;
        FloatType *chunk_real = padded_real + plan.offsets[chunk];
        traits::execute_dft_r2c(plan.plans[chunk], chunk_real,
                                reinterpret_cast<fftw_complex_t *>(chunk_real));
      });
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
     * @param dist Distance between batches (padded: 2*(N/2+1) scalars)
     */
    void create_c2r_inplace_plan(int length,               // Real-space output length N
                                 std::size_t batch,         // Number of transforms
                                 FloatType *padded_real,    // Padded real buffer (also used as complex input)
                                 std::ptrdiff_t stride,     // Stride
                                 std::ptrdiff_t dist        // Distance between batches (padded: 2*(N/2+1))
    )
    {
      // For C2R: input dist is dist/2 (complex elements), output dist is dist (scalars).
      c2r_inplace_plan_ = ChunkedPlan(num_chunks());

      for (int chunk = 0; chunk < num_chunks(); ++chunk) {
        std::size_t begin = 0, end = 0;
        detail::chunk_range(batch, chunk, num_chunks(), begin, end);
        if (begin == end) continue;

        const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(begin) * dist;
        FloatType *chunk_real = padded_real + offset;

        typename traits::iodim64_t dims[] = {{static_cast<std::ptrdiff_t>(length), stride, stride}};
        typename traits::iodim64_t howmany_dims[] = {
            {static_cast<std::ptrdiff_t>(end - begin), dist / 2, dist}};

        c2r_inplace_plan_.offsets[chunk] = offset;
        c2r_inplace_plan_.plans[chunk] = traits::plan_guru64_dft_c2r(
            1, dims, 1, howmany_dims, reinterpret_cast<fftw_complex_t *>(chunk_real),
            chunk_real, plan_flag_);
      }
    }

    /**
     * @brief Execute in-place C2R transform.
     *
     * Transforms complex data to real data in-place. The buffer layout must
     * match the plan created by create_c2r_inplace_plan().
     *
     * @param padded_real Host pointer to buffer (complex input, real output)
     */
    void execute_c2r_inplace(FloatType *padded_real)
    {
      const ChunkedPlan &plan = c2r_inplace_plan_;
      executor_->run([&plan, padded_real](int chunk, int) {
        if (plan.plans[chunk] == nullptr) return;
        FloatType *chunk_real = padded_real + plan.offsets[chunk];
        traits::execute_dft_c2r(plan.plans[chunk],
                                reinterpret_cast<fftw_complex_t *>(chunk_real), chunk_real);
      });
    }

    /**
     * @brief Execute pre-created plan for specified stage.
     *
     * Executes the C2C transform for the given stage using the new-array
     * execution interface, allowing the plan to be applied to data at a
     * different address with the same layout.
     *
     * @param stage Stage index (0 to num_stages-1)
     * @param direction Transform direction (Forward or Backward)
     * @param data Host pointer to complex data buffer
     */
    void execute_stage(int stage, FFTDirection direction, Complex *data)
    {
      const ChunkedPlan &plan =
          (direction == FFTDirection::Forward) ? forward_plans_[stage] : backward_plans_[stage];
      executor_->run([&plan, data](int chunk, int) {
        if (plan.plans[chunk] == nullptr) return;
        fftw_complex_t *chunk_data =
            reinterpret_cast<fftw_complex_t *>(data + plan.offsets[chunk]);
        traits::execute_dft(plan.plans[chunk], chunk_data, chunk_data);
      });
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
    /// FFTW does not handle distributed MPI communication
    static constexpr bool handles_distributed = false;

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
      for (ChunkedPlan &stage : forward_plans_) stage.destroy();
      for (ChunkedPlan &stage : backward_plans_) stage.destroy();
      r2c_inplace_plan_.destroy();
      c2r_inplace_plan_.destroy();

#if defined(PARAFAFT_FFTW_THREADS) || defined(PARAFAFT_FFTW_OMP)
      traits::cleanup_threads();
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
        : forward_plans_(std::move(other.forward_plans_)),
          backward_plans_(std::move(other.backward_plans_)),
          r2c_inplace_plan_(std::move(other.r2c_inplace_plan_)),
          c2r_inplace_plan_(std::move(other.c2r_inplace_plan_)),
          executor_(std::move(other.executor_)), backend_type_(other.backend_type_),
          num_threads_(other.num_threads_), plan_flag_(other.plan_flag_)
    {
      // Moved-from vectors are already empty; clear the in-place plans so the
      // source destructor cannot double-destroy them.
      other.forward_plans_.clear();
      other.backward_plans_.clear();
      other.r2c_inplace_plan_ = ChunkedPlan();
      other.c2r_inplace_plan_ = ChunkedPlan();
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
      return traits::export_wisdom(filename) != 0;
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
      return traits::import_wisdom(filename) != 0;
    }

    /**
     * @brief Discard all accumulated FFTW wisdom.
     *
     * Clears the internal wisdom database. Subsequent plans will be created
     * from scratch.
     */
    static void forgetWisdom()
    {
      traits::forget_wisdom();
    }

  private:
    /**
     * @brief One stage's plans, split across the executor's threads.
     *
     * `plans[i]` transforms the sub-batch starting at `offsets[i]` elements
     * into the data buffer (FloatType units for the R2C/C2R plans, Complex
     * units for C2C). A null entry means that thread has no work, which
     * happens when there are more threads than transforms in the batch.
     *
     * With batch-parallelism disabled there is exactly one chunk at offset 0,
     * reproducing the single-plan behaviour.
     */
    struct ChunkedPlan {
      std::vector<fftw_plan_t> plans;
      std::vector<std::ptrdiff_t> offsets;

      ChunkedPlan() = default;
      explicit ChunkedPlan(int num_chunks) : plans(num_chunks, nullptr), offsets(num_chunks, 0) {}

      void destroy()
      {
        for (fftw_plan_t plan : plans) {
          if (plan) traits::destroy_plan(plan);
        }
        plans.clear();
        offsets.clear();
      }
    };

    std::vector<ChunkedPlan> forward_plans_;  ///< Forward C2C plans (one entry per stage)
    std::vector<ChunkedPlan> backward_plans_; ///< Backward C2C plans (one entry per stage)
    ChunkedPlan r2c_inplace_plan_;            ///< In-place R2C plans
    ChunkedPlan c2r_inplace_plan_;            ///< In-place C2R plans

    /// Dispatches FFT chunks; held by pointer so the backend stays movable.
    std::unique_ptr<detail::ParallelExecutor> executor_;

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
     * 1. OMP_NUM_THREADS environment variable
     * 2. KOKKOS_NUM_THREADS environment variable
     * 3. std::thread::hardware_concurrency() / ranks sharing this node
     * 4. Default to 1 (serial)
     *
     * @param comm MPI communicator for determining task count
     * @return Optimal number of threads
     */
    static int detect_thread_count(MPI_Comm comm)
    {
      int threads = 1;

      // Honour OMP_NUM_THREADS whichever threading library is linked: it is the
      // user's explicit "threads per rank" request, not an OpenMP-only setting.
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
        // Divide the node's cores among the ranks *sharing that node*. Dividing by
        // the size of comm would shrink the thread count as the job scales out
        // across nodes, silently leaving most of each node idle.
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
  };

} // namespace parafaft

#endif // PARAFAFT_BACKEND_FFTW_HPP
