/**
 * @file fft_backend_cufftmp.hpp
 * @brief cuFFTMp distributed multi-GPU FFT backend for ParaFaFT.
 *
 * Uses NVIDIA cuFFTMp (a superset of cuFFT) with NVSHMEM for GPU-to-GPU
 * communication. The backend handles MPI decomposition, all-to-all exchanges,
 * and FFT execution internally — ParaFaFT delegates the full transform.
 *
 * Supports 2D (slab decomposition) and 3D (pencil decomposition) transforms.
 * Requires: NVIDIA HPC SDK with cuFFTMp + NVSHMEM.
 *
 * Build: cmake -DPARAFAFT_CUDA=ON -DPARAFAFT_CUFFTMP=ON
 */

#ifndef PARAFAFT_BACKEND_CUFFTMP_HPP
#define PARAFAFT_BACKEND_CUFFTMP_HPP

#ifdef PARAFAFT_CUFFTMP_ENABLED

#include "../fft_backend.hpp"
#include "./fft_backend_cufft.hpp" // reuse cuvector<T>

#include <cufftMp.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <array>
#include <stdexcept>
#include <string>

namespace parafaft {

namespace detail {

inline void check_cufftmp(cufftResult result, const char *msg) {
  if (result != CUFFT_SUCCESS) {
    throw std::runtime_error(
        std::string("cuFFTMp error: ") + msg +
        " (code " + std::to_string(static_cast<int>(result)) + ")");
  }
}

} // namespace detail

/**
 * @brief Distributed multi-GPU FFT backend using cuFFTMp.
 *
 * @tparam D Number of dimensions (must be 2 or 3).
 *
 * When used as the backend for ParaFaFT<D, CuFFTMpBackend<D>>, the entire
 * distributed transform (decomposition, communication, FFT) is handled by
 * cuFFTMp internally via NVSHMEM.
 *
 * Buffer management: the backend allocates an NVSHMEM-backed buffer via
 * cufftXtMalloc. Users access it via get_buffer() / get_real_buffer() and
 * fill data before calling forward/backward. Transforms are zero-copy.
 *
 * Decomposition strategy:
 *   - 2D: slab decomposition (cufftMpMakePlan2d)
 *   - 3D: pencil decomposition (cufftMpMakePlanDecomposition) -- currently
 *          falls back to slab (cufftMpMakePlan3d) until pencil plan creation
 *          is validated on target hardware.
 */
template <int D>
class CuFFTMpBackend {
  static_assert(D >= 2 && D <= 3,
                "CuFFTMpBackend only supports 2D and 3D transforms");

public:
  using Complex = cuda::std::complex<double>;
  using Buffer = cuvector<double>;
  using ComplexBuffer = cuvector<Complex>;

  /// This backend handles all MPI communication internally
  static constexpr bool handles_distributed = true;
  /// Unused when handles_distributed = true, but required by the interface
  static constexpr bool use_alltoallw = false;
  static constexpr bool use_p2p = false;

  struct DistributedInfo {
    int local_shape[D];
    int global_start[D];
    int output_shape[D];
    int output_start[D];
    int required_size;
  };

  /**
   * @brief Construct the backend (lightweight — no plans created yet).
   */
  explicit CuFFTMpBackend(int num_stages, MPI_Comm comm,
                          FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
      : comm_(comm) {
    (void)num_stages;
    (void)plan_flag;
    check_cuda(cudaStreamCreate(&stream_));
  }

  CuFFTMpBackend(const CuFFTMpBackend &) = delete;
  CuFFTMpBackend &operator=(const CuFFTMpBackend &) = delete;

  CuFFTMpBackend(CuFFTMpBackend &&other) noexcept
      : comm_(other.comm_), stream_(other.stream_),
        plan_c2c_fwd_(other.plan_c2c_fwd_),
        plan_c2c_bwd_(other.plan_c2c_bwd_),
        plan_r2c_(other.plan_r2c_), plan_c2r_(other.plan_c2r_),
        desc_c2c_(other.desc_c2c_), desc_r2c_(other.desc_r2c_),
        global_shape_(other.global_shape_),
        info_(other.info_), initialized_(other.initialized_) {
    other.stream_ = nullptr;
    other.plan_c2c_fwd_ = 0;
    other.plan_c2c_bwd_ = 0;
    other.plan_r2c_ = 0;
    other.plan_c2r_ = 0;
    other.desc_c2c_ = nullptr;
    other.desc_r2c_ = nullptr;
    other.initialized_ = false;
  }

  ~CuFFTMpBackend() {
    if (desc_c2c_) cufftXtFree(desc_c2c_);
    if (desc_r2c_) cufftXtFree(desc_r2c_);
    if (plan_c2c_fwd_) cufftDestroy(plan_c2c_fwd_);
    if (plan_c2c_bwd_) cufftDestroy(plan_c2c_bwd_);
    if (plan_r2c_) cufftDestroy(plan_r2c_);
    if (plan_c2r_) cufftDestroy(plan_c2r_);
    if (stream_) cudaStreamDestroy(stream_);
  }

  // =========================================================================
  // Distributed setup — called by ParaFaFT when handles_distributed = true
  // =========================================================================

  /**
   * @brief Set up distributed FFT plans and allocate NVSHMEM-backed buffers.
   *
   * Creates cuFFTMp plans for both C2C and R2C transforms. For 3D, uses
   * slab decomposition via cufftMpMakePlan3d (pencil support planned).
   * For 2D, uses cufftMpMakePlan2d.
   */
  void setup_distributed(const int global_shape[D], MPI_Comm comm) {
    comm_ = comm;
    for (int i = 0; i < D; ++i) global_shape_[i] = global_shape[i];

    int rank, size;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);

    // Create C2C forward plan
    create_plan(plan_c2c_fwd_, CUFFT_Z2Z);
    // Create C2C backward plan (separate handle for potential workspace reuse)
    create_plan(plan_c2c_bwd_, CUFFT_Z2Z);

    // Allocate NVSHMEM-backed buffer for C2C
    detail::check_cufftmp(
        cufftXtMalloc(plan_c2c_fwd_, &desc_c2c_, CUFFT_XT_FORMAT_INPLACE),
        "cufftXtMalloc C2C");

    // Create R2C and C2R plans
    create_plan(plan_r2c_, CUFFT_D2Z);
    create_plan(plan_c2r_, CUFFT_Z2D);

    // Allocate NVSHMEM-backed buffer for R2C
    detail::check_cufftmp(
        cufftXtMalloc(plan_r2c_, &desc_r2c_, CUFFT_XT_FORMAT_INPLACE),
        "cufftXtMalloc R2C");

    // Query local decomposition info from the C2C plan
    query_local_info(rank, size);
    initialized_ = true;
  }

  DistributedInfo get_distributed_info() const {
    return info_;
  }

  // =========================================================================
  // Buffer access — zero-copy transforms operate on these buffers
  // =========================================================================

  Complex* get_buffer() {
    if (!desc_c2c_) return nullptr;
    return reinterpret_cast<Complex *>(desc_c2c_->descriptor->data[0]);
  }

  double* get_real_buffer() {
    if (!desc_r2c_) return nullptr;
    return reinterpret_cast<double *>(desc_r2c_->descriptor->data[0]);
  }

  // =========================================================================
  // Transforms — operate on internal NVSHMEM-backed buffer
  // =========================================================================

  void forward() {
    detail::check_cufftmp(
        cufftXtExecDescriptor(plan_c2c_fwd_, desc_c2c_, desc_c2c_,
                              CUFFT_FORWARD),
        "cufftXtExecDescriptor C2C forward");
  }

  void backward() {
    detail::check_cufftmp(
        cufftXtExecDescriptor(plan_c2c_bwd_, desc_c2c_, desc_c2c_,
                              CUFFT_INVERSE),
        "cufftXtExecDescriptor C2C backward");
  }

  void forward_r2c() {
    detail::check_cufftmp(
        cufftXtExecDescriptor(plan_r2c_, desc_r2c_, desc_r2c_,
                              CUFFT_FORWARD),
        "cufftXtExecDescriptor R2C forward");
  }

  void backward_c2r() {
    detail::check_cufftmp(
        cufftXtExecDescriptor(plan_c2r_, desc_r2c_, desc_r2c_,
                              CUFFT_INVERSE),
        "cufftXtExecDescriptor C2R backward");
  }

  // =========================================================================
  // Memory operations (required by backend interface)
  // =========================================================================

  void memcpy(void *dst, const void *src, size_t bytes) const {
    check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream_));
  }

  void memcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                size_t width, size_t height) const {
    check_cuda(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                 cudaMemcpyDefault, stream_));
  }

  void sync() const {
    check_cuda(cudaStreamSynchronize(stream_));
  }

  // Unused stubs — required by interface but never called when handles_distributed
  void create_stage_plan(int, int, int, Complex *, int, int) {}
  void execute_stage(int, FFTDirection, Complex *) {}
  void create_r2c_inplace_plan(int, int, double *, int, int) {}
  void execute_r2c_inplace(double *) {}
  void create_c2r_inplace_plan(int, int, double *, int, int) {}
  void execute_c2r_inplace(double *) {}

private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  cudaStream_t stream_ = nullptr;

  cufftHandle plan_c2c_fwd_ = 0;
  cufftHandle plan_c2c_bwd_ = 0;
  cufftHandle plan_r2c_ = 0;
  cufftHandle plan_c2r_ = 0;

  cudaLibXtDesc *desc_c2c_ = nullptr;
  cudaLibXtDesc *desc_r2c_ = nullptr;

  std::array<int, D> global_shape_{};
  DistributedInfo info_{};
  bool initialized_ = false;

  void create_plan(cufftHandle &plan, cufftType type) {
    detail::check_cufftmp(cufftCreate(&plan), "cufftCreate");
    detail::check_cufftmp(
        cufftSetStream(plan, stream_), "cufftSetStream");

    size_t work_size = 0;
    if constexpr (D == 2) {
      detail::check_cufftmp(
          cufftMpMakePlan2d(plan, global_shape_[0], global_shape_[1],
                            type, &comm_, CUFFT_COMM_MPI, &work_size),
          "cufftMpMakePlan2d");
    } else {
      // 3D: use slab decomposition for now
      // TODO: pencil decomposition via cufftMpMakePlanDecomposition
      detail::check_cufftmp(
          cufftMpMakePlan3d(plan, global_shape_[0], global_shape_[1],
                            global_shape_[2], type,
                            &comm_, CUFFT_COMM_MPI, &work_size),
          "cufftMpMakePlan3d");
    }
  }

  /**
   * @brief Query local slab/pencil extents from cuFFTMp.
   *
   * After plan creation, cuFFTMp determines the local decomposition.
   * For slab decomposition (CUFFT_XT_FORMAT_INPLACE), the first axis
   * is distributed evenly across ranks.
   */
  void query_local_info(int rank, int size) {
    // cuFFTMp slab: distributes along the first (X) axis
    // Input: slab along X; Output (shuffled): slab along Y
    for (int i = 0; i < D; ++i) {
      info_.local_shape[i] = global_shape_[i];
      info_.output_shape[i] = global_shape_[i];
      info_.global_start[i] = 0;
      info_.output_start[i] = 0;
    }

    // Input distribution: first axis split
    int base = global_shape_[0] / size;
    int remainder = global_shape_[0] % size;
    info_.local_shape[0] = base + (rank < remainder ? 1 : 0);
    info_.global_start[0] = rank * base + std::min(rank, remainder);

    // Output distribution (shuffled): second axis split
    if (D >= 2) {
      int base_y = global_shape_[1] / size;
      int rem_y = global_shape_[1] % size;
      info_.output_shape[1] = base_y + (rank < rem_y ? 1 : 0);
      info_.output_start[1] = rank * base_y + std::min(rank, rem_y);
      info_.output_shape[0] = global_shape_[0]; // fully local after forward
    }

    // Required buffer size (in complex elements for C2C, doubles for R2C)
    info_.required_size = 1;
    for (int i = 0; i < D; ++i) {
      info_.required_size *= info_.local_shape[i];
    }
    // Take max of input and output sizes
    int output_size = 1;
    for (int i = 0; i < D; ++i) {
      output_size *= info_.output_shape[i];
    }
    if (output_size > info_.required_size) {
      info_.required_size = output_size;
    }
  }

  static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error: ") +
                               cudaGetErrorString(err));
    }
  }
};

} // namespace parafaft

#endif // PARAFAFT_CUFFTMP_ENABLED
#endif // PARAFAFT_BACKEND_CUFFTMP_HPP
