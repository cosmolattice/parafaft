/**
 * @file fft_backend_hipfft.hpp
 * @brief hipFFT backend implementation for ParaFaFT (AMD GPU-accelerated FFT).
 *
 * This header provides an FFT backend using AMD's hipFFT library for GPU-based
 * FFT operations. Supports C2C, R2C, and C2R transforms as required by
 * ParaFaFT and ParaFaFT_R2C.
 *
 * Precision: HipFFTBackend is a class template parameterised on FloatType.
 *   - HipFFTBackend<double> (default) uses HIPFFT_Z2Z / HIPFFT_D2Z / HIPFFT_Z2D.
 *   - HipFFTBackend<float>  uses HIPFFT_C2C / HIPFFT_R2C / HIPFFT_C2R.
 *
 * @note Requires ROCm toolkit and hipFFT library.
 */

#ifndef PARAFAFT_BACKEND_HIPFFT_HPP
#define PARAFAFT_BACKEND_HIPFFT_HPP

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) ||          \
    defined(__HIPCC__)

#include "../fft_backend.hpp"
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef PARAFAFT_GPU_ALLTOALLW
#define PARAFAFT_GPU_ALLTOALLW 0
#endif

namespace parafaft {
/**
 * @brief Check HIP result and throw on error.
 *
 * @param result HIP error code to check
 * @param operation Description of the operation for error messages
 * @throws std::runtime_error if result is not hipSuccess
 */
static void check_hip(hipError_t result, const char *operation) {
  if (result != hipSuccess) {
    throw std::runtime_error(std::string("HIP error in ") + operation + ": " +
                             hipGetErrorString(result));
  }
}

/**
 * @brief (Owning) HIP vector wrapper for device memory management
 *
 * @tparam T Data type
 */
template <typename T> class hipvector {
public:
  hipvector() = default;

  /**
   * @brief Create a hipvector and allocate device memory of given size
   *
   * @param size Number of elements to allocate
   */
  hipvector(size_t size) {
    size_ = size;
    check_hip(hipMalloc(&data_, size_ * sizeof(T)), "hipMalloc hipvector");
  }

  /**
   * @brief Destructor: Free device memory
   */
  ~hipvector() {
    if (data_) {
      hipFree(data_);
    }
  }

  /**
   * @brief Resize the hipvector. Warning: old data is discarded.
   *
   * @param new_size New number of elements
   */
  void resize(size_t new_size) {
    if (new_size == size_)
      return;

    T *new_data = nullptr;
    check_hip(hipMalloc(&new_data, new_size * sizeof(T)),
              "hipMalloc hipvector resize");

    if (data_) {
      // Discard old data
      hipFree(data_);
    }

    data_ = new_data;
    size_ = new_size;
  }

  /**
   * @brief Get raw device pointer
   *
   * @return T* Device pointer
   */
  T *data() { return data_; }
  const T *data() const { return data_; }

  /**
   * @brief Get size of the hipvector
   *
   * @return size_t Number of elements
   */
  size_t size() const { return size_; }

private:
  T *data_ = nullptr;
  size_t size_ = 0;
};

/**
 * @brief Precision-specific hipFFT type/plan/exec mapping.
 *
 * Routes HipFFTBackend<FloatType> to the right hipFFT plan constants
 * (HIPFFT_Z2Z / D2Z / Z2D for double; HIPFFT_C2C / R2C / C2R for float)
 * and the right device complex/real types and exec functions.
 */
template <typename T> struct hipfft_traits;

template <> struct hipfft_traits<double> {
  using real_t = hipfftDoubleReal;
  using complex_t = hipfftDoubleComplex;
  static constexpr hipfftType c2c_type = HIPFFT_Z2Z;
  static constexpr hipfftType r2c_type = HIPFFT_D2Z;
  static constexpr hipfftType c2r_type = HIPFFT_Z2D;
  static hipfftResult exec_c2c(hipfftHandle p, complex_t *in, complex_t *out, int dir) {
    return hipfftExecZ2Z(p, in, out, dir);
  }
  static hipfftResult exec_r2c(hipfftHandle p, real_t *in, complex_t *out) {
    return hipfftExecD2Z(p, in, out);
  }
  static hipfftResult exec_c2r(hipfftHandle p, complex_t *in, real_t *out) {
    return hipfftExecZ2D(p, in, out);
  }
};

template <> struct hipfft_traits<float> {
  using real_t = hipfftReal;
  using complex_t = hipfftComplex;
  static constexpr hipfftType c2c_type = HIPFFT_C2C;
  static constexpr hipfftType r2c_type = HIPFFT_R2C;
  static constexpr hipfftType c2r_type = HIPFFT_C2R;
  static hipfftResult exec_c2c(hipfftHandle p, complex_t *in, complex_t *out, int dir) {
    return hipfftExecC2C(p, in, out, dir);
  }
  static hipfftResult exec_r2c(hipfftHandle p, real_t *in, complex_t *out) {
    return hipfftExecR2C(p, in, out);
  }
  static hipfftResult exec_c2r(hipfftHandle p, complex_t *in, real_t *out) {
    return hipfftExecC2R(p, in, out);
  }
};

/**
 * @brief hipFFT backend for AMD GPU-accelerated FFT operations.
 *
 * Provides an interface compatible with ParaFaFT for executing FFT transforms
 * on AMD GPUs using hipFFT. Supports C2C, R2C, and C2R transforms.
 *
 * @tparam FloatType Precision of the transform: `double` (default, uses
 *         HIPFFT_Z2Z / D2Z / Z2D) or `float` (uses HIPFFT_C2C / R2C / C2R).
 *
 * Memory management: Uses hipvector for device memory allocation.
 * All data pointers passed to this backend must be device pointers.
 */
template <typename FloatTypeT = double>
class HipFFTBackend {
public:
  using FloatType = FloatTypeT;                                      ///< Scalar floating-point type
  using Complex = typename hipfft_traits<FloatType>::complex_t;      ///< Complex number type
  using Buffer = hipvector<FloatType>;                               ///< Real buffer type (device memory)
  using ComplexBuffer = hipvector<Complex>;                          ///< Complex buffer type (device memory)
private:
  using traits = hipfft_traits<FloatType>;
  using hipfft_real_t = typename traits::real_t;
  using hipfft_complex_t = typename traits::complex_t;

public:

  /**
   * @brief Construct a hipFFT backend with storage for the given number of
   * stages.
   *
   * @param num_stages Number of FFT stages (typically D for D-dimensional
   * transform)
   * @param plan_flag Planning strategy (ignored for hipFFT, accepted for API
   * compatibility)
   */
  explicit HipFFTBackend(int num_stages,
                         FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
      : num_stages_(num_stages), forward_plans_(num_stages, 0),
        backward_plans_(num_stages, 0) {
    (void)plan_flag; // hipFFT does not support configurable planning
    check_hip(hipStreamCreate(&stream_), "hipStreamCreate");
  }

  /**
   * @brief Construct a hipFFT backend with MPI communicator (for API
   * compatibility).
   *
   * @param num_stages Number of FFT stages
   * @param comm MPI communicator (unused for hipFFT, but provides API
   * compatibility)
   * @param plan_flag Planning strategy (ignored for hipFFT, accepted for API
   * compatibility)
   */
  explicit HipFFTBackend(int num_stages, MPI_Comm comm,
                         FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
      : num_stages_(num_stages), forward_plans_(num_stages, 0),
        backward_plans_(num_stages, 0) {
    (void)comm;      // Suppress unused parameter warning
    (void)plan_flag; // hipFFT does not support configurable planning
    check_hip(hipStreamCreate(&stream_), "hipStreamCreate");
  }

  /**
   * @brief Destructor. Cleans up stream and all hipFFT plans.
   */
  ~HipFFTBackend() {
    if (stream_) {
      hipStreamSynchronize(stream_);
      hipStreamDestroy(stream_);
    }
    for (auto plan : forward_plans_) {
      if (plan)
        hipfftDestroy(plan);
    }
    for (auto plan : backward_plans_) {
      if (plan)
        hipfftDestroy(plan);
    }
    if (r2c_plan_)
      hipfftDestroy(r2c_plan_);
    if (c2r_plan_)
      hipfftDestroy(c2r_plan_);
  }

  /// @brief Deleted copy constructor (hipFFT plans cannot be safely copied)
  HipFFTBackend(const HipFFTBackend &) = delete;
  /// @brief Deleted copy assignment (hipFFT plans cannot be safely copied)
  HipFFTBackend &operator=(const HipFFTBackend &) = delete;

  /**
   * @brief Move constructor.
   *
   * @param other Backend to move from (will be left in empty state)
   */
  HipFFTBackend(HipFFTBackend &&other) noexcept
      : num_stages_(other.num_stages_),
        forward_plans_(std::move(other.forward_plans_)),
        backward_plans_(std::move(other.backward_plans_)),
        r2c_plan_(other.r2c_plan_), c2r_plan_(other.c2r_plan_),
        r2c_length_(other.r2c_length_), r2c_batch_(other.r2c_batch_),
        r2c_dist_(other.r2c_dist_), stream_(other.stream_) {
    // Clear moved-from object
    std::fill(other.forward_plans_.begin(), other.forward_plans_.end(),
              nullptr);
    std::fill(other.backward_plans_.begin(), other.backward_plans_.end(),
              nullptr);
    other.r2c_plan_ = 0;
    other.c2r_plan_ = 0;
    other.stream_ = nullptr;
  }

  // ========== C2C Transform Methods ==========

  /**
   * @brief Create and store hipFFT plans for a specific stage (C2C transforms).
   *
   * Creates both forward and backward hipFFT plans for the given stage.
   *
   * @note Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are
   * NULL. We explicitly set inembed/onembed to match FFTW behavior.
   *
   * @param stage Stage index for plan storage (0 to num_stages-1)
   * @param length FFT length (number of complex elements per transform)
   * @param batch Number of 1D transforms to execute in batch
   * @param data Device pointer for plan creation (alignment reference)
   * @param stride Stride between consecutive elements in a transform
   * @param dist Distance between first elements of consecutive transforms
   */
  void create_stage_plan(int stage, int length, std::size_t batch, Complex *data,
                         std::ptrdiff_t stride, std::ptrdiff_t dist) {
    (void)data;
    int n[] = {length};
    int inembed[] = {length};
    int onembed[] = {length};

    int batch_i = narrow_plan_arg(batch, "batch");
    int stride_i = narrow_plan_arg(stride, "stride");
    int dist_i = narrow_plan_arg(dist, "dist");

    check_hipfft(hipfftPlanMany(&forward_plans_[stage], 1, n, inembed, stride_i,
                                dist_i, onembed, stride_i, dist_i, traits::c2c_type, batch_i),
                 "hipfftPlanMany C2C forward");
    hipfftSetStream(forward_plans_[stage], stream_);

    check_hipfft(hipfftPlanMany(&backward_plans_[stage], 1, n, inembed, stride_i,
                                dist_i, onembed, stride_i, dist_i, traits::c2c_type, batch_i),
                 "hipfftPlanMany C2C backward");
    hipfftSetStream(backward_plans_[stage], stream_);
  }

  /**
   * @brief Execute pre-created plan for specified stage.
   *
   * Executes the C2C transform for the given stage on the provided device data.
   * Synchronizes the device after execution.
   *
   * @param stage Stage index (0 to num_stages-1)
   * @param direction Transform direction (Forward or Backward)
   * @param data Device pointer to complex data buffer
   */
  void execute_stage(int stage, FFTDirection direction, Complex *data) {
    // Execute C2C transform
    hipfftHandle plan = (direction == FFTDirection::Forward)
                            ? forward_plans_[stage]
                            : backward_plans_[stage];
    int hipfft_direction =
        (direction == FFTDirection::Forward) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
    hipfft_complex_t *hipfft_data =
        reinterpret_cast<hipfft_complex_t *>(data);

    check_hipfft(
        traits::exec_c2c(plan, hipfft_data, hipfft_data, hipfft_direction),
        "hipfftExecC2C/Z2Z");
  }

  // ========== R2C In-Place Transform Methods ==========

  /**
   * @brief Create in-place R2C plan for padded memory layout.
   *
   * Creates an R2C (real-to-complex) plan for in-place transformation on GPU.
   * The input is N real values per transform, output is N/2+1 complex values,
   * stored in the same padded device buffer.
   *
   * @note Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are
   * NULL. We explicitly set inembed/onembed to match FFTW behavior.
   *
   * @param length Real-space FFT length N (number of real values per transform)
   * @param batch Number of 1D transforms to execute in batch
   * @param padded_real Device pointer to padded real buffer (for size
   * calculation only)
   * @param stride Element stride (typically 1 for contiguous data)
   * @param dist Distance between batches in FloatType scalars (should be 2*(N/2+1) for
   * in-place)
   */
  void create_r2c_inplace_plan(
      int length,                 // Real-space FFT length N
      std::size_t batch,           // Number of 1D transforms
      FloatType *padded_real,      // Padded real buffer (for size calculation only)
      std::ptrdiff_t stride,       // Stride (typically 1)
      std::ptrdiff_t dist          // Distance between batches (2*(N/2+1) scalars)
  ) {
    (void)padded_real;
    int n[] = {length};
    int inembed[] = {length};
    int onembed[] = {length / 2 + 1};

    int batch_i = narrow_plan_arg(batch, "batch");
    int stride_i = narrow_plan_arg(stride, "stride");
    int dist_i = narrow_plan_arg(dist, "dist");

    check_hipfft(hipfftPlanMany(&r2c_plan_, 1, n, inembed, stride_i,
                                dist_i,
                                onembed, stride_i,
                                dist_i / 2,
                                traits::r2c_type, batch_i),
                 "hipfftPlanMany R2C");
    hipfftSetStream(r2c_plan_, stream_);
  }

  /**
   * @brief Execute in-place R2C transform.
   *
   * Transforms real data to complex data in-place on GPU. The device buffer
   * must be padded (2*(N/2+1) FloatType scalars per row) to accommodate the complex
   * output. Synchronizes the device after execution.
   *
   * @param device_padded_real Device pointer to padded real input buffer
   * (overwritten with complex output)
   */
  void execute_r2c_inplace(FloatType *device_padded_real) {
    hipfft_real_t *input_data =
        reinterpret_cast<hipfft_real_t *>(device_padded_real);
    hipfft_complex_t *output_data =
        reinterpret_cast<hipfft_complex_t *>(device_padded_real);

    // In-place: output overwrites input buffer
    check_hipfft(traits::exec_r2c(r2c_plan_, input_data, output_data),
                 "hipfftExecR2C/D2Z");
  }

  // ========== C2R In-Place Transform Methods ==========

  /**
   * @brief Create in-place C2R plan for padded memory layout.
   *
   * Creates a C2R (complex-to-real) plan for in-place transformation on GPU.
   * The input is N/2+1 complex values per transform, output is N real values,
   * stored in the same padded device buffer.
   *
   * @note Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are
   * NULL. We explicitly set inembed/onembed to match FFTW behavior.
   *
   * @param length Real-space output length N (number of real values per
   * transform)
   * @param batch Number of 1D transforms to execute in batch
   * @param padded_real Device pointer to padded buffer
   * @param stride Element stride (typically 1 for contiguous data)
   * @param dist Distance between batches (padded: 2*(N/2+1) scalars)
   */
  void create_c2r_inplace_plan(int length,                 // Real-space output length N
                               std::size_t batch,           // Number of transforms
                               FloatType *padded_real,      // Padded real buffer
                               std::ptrdiff_t stride,       // Stride (typically 1)
                               std::ptrdiff_t dist          // Distance between batches (2*(N/2+1))
  ) {
    (void)padded_real;
    int n[] = {length};
    int inembed[] = {length / 2 + 1};
    int onembed[] = {length};

    int batch_i = narrow_plan_arg(batch, "batch");
    int stride_i = narrow_plan_arg(stride, "stride");
    int dist_i = narrow_plan_arg(dist, "dist");

    check_hipfft(hipfftPlanMany(&c2r_plan_, 1, n, inembed, stride_i,
                                dist_i / 2,
                                onembed, stride_i, dist_i,
                                traits::c2r_type, batch_i),
                 "hipfftPlanMany C2R");
    hipfftSetStream(c2r_plan_, stream_);
  }

  /**
   * @brief Execute in-place C2R transform.
   *
   * Transforms complex data to real data in-place on GPU. The device buffer
   * layout must match the plan created by create_c2r_inplace_plan().
   * Synchronizes the device after execution.
   *
   * @param device_padded_real Device pointer to buffer (complex input, real
   * output)
   */
  void execute_c2r_inplace(FloatType *device_padded_real) {
    hipfft_complex_t *input_data =
        reinterpret_cast<hipfft_complex_t *>(device_padded_real);
    hipfft_real_t *output_data =
        reinterpret_cast<hipfft_real_t *>(device_padded_real);

    check_hipfft(traits::exec_c2r(c2r_plan_, input_data, output_data),
                 "hipfftExecC2R/Z2D");
  }

  /**
   * @brief Copy memory between buffers (device-to-device, host-to-device,
   * etc.).
   *
   * Uses hipMemcpy with hipMemcpyDefault to automatically determine the
   * appropriate copy direction based on pointer locations.
   *
   * @param dest Destination pointer (device or host)
   * @param src Source pointer (device or host)
   * @param bytes Number of bytes to copy
   * @throws std::runtime_error if the HIP memcpy operation fails
   */
  void memcpy(void *dest, const void *src, size_t bytes) const {
    check_hip(hipMemcpyAsync(dest, src, bytes, hipMemcpyDefault, stream_),
              "hipMemcpyAsync");
  }

  /// Use manual packing by default; opt-in to MPI_Alltoallw via
  /// PARAFAFT_GPU_ALLTOALLW
  static constexpr bool use_alltoallw =
      static_cast<bool>(PARAFAFT_GPU_ALLTOALLW);
  /// hipFFT does not handle distributed MPI communication (see RocFFTMpBackend)
  static constexpr bool handles_distributed = false;

  /**
   * @brief 2D strided async device memory copy on the backend stream.
   */
  void memcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                size_t width, size_t height) const {
    check_hip(hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                               hipMemcpyDefault, stream_),
              "hipMemcpy2DAsync");
  }

  /**
   * @brief Synchronize the backend stream.
   */
  void sync() const {
    check_hip(hipStreamSynchronize(stream_), "hipStreamSynchronize");
  }

  // ========== P2P / IPC Methods ==========

  static constexpr bool use_p2p = true;

  static int device_id() {
    int dev;
    check_hip(hipGetDevice(&dev), "hipGetDevice");
    return dev;
  }

  static bool can_access_peer(int src_dev, int dst_dev) {
    int can = 0;
    hipDeviceCanAccessPeer(&can, src_dev, dst_dev);
    return can != 0;
  }

  static void enable_peer_access(int peer_dev) {
    hipError_t err = hipDeviceEnablePeerAccess(peer_dev, 0);
    if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
      check_hip(err, "hipDeviceEnablePeerAccess");
  }

  static constexpr size_t ipc_handle_size = sizeof(hipIpcMemHandle_t);

  static void ipc_get_handle(void *devptr, void *handle) {
    check_hip(hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t *>(handle),
                                 devptr),
              "hipIpcGetMemHandle");
  }

  static void *ipc_open_handle(const void *handle) {
    void *ptr;
    check_hip(hipIpcOpenMemHandle(
                  &ptr, *reinterpret_cast<const hipIpcMemHandle_t *>(handle),
                  hipIpcMemLazyEnablePeerAccess),
              "hipIpcOpenMemHandle");
    return ptr;
  }

  static void ipc_close_handle(void *ptr) { hipIpcCloseMemHandle(ptr); }

private:
  template <typename T>
  static int narrow_plan_arg(T value, const char *name) {
    using Limit = std::numeric_limits<int>;
    if (value < static_cast<T>(Limit::min()) || value > static_cast<T>(Limit::max())) {
      throw std::runtime_error(std::string("hipFFT plan parameter '") + name +
                               "' exceeds int range; hipfftPlanMany is int-limited — "
                               "use hipfftXtMakePlanMany for 64-bit batches.");
    }
    return static_cast<int>(value);
  }

  int num_stages_; ///< Number of FFT stages

  // C2C plans (for general complex-to-complex transforms)
  std::vector<hipfftHandle>
      forward_plans_; ///< Forward C2C plans (one per stage)
  std::vector<hipfftHandle>
      backward_plans_; ///< Backward C2C plans (one per stage)

  // R2C/C2R plans
  hipfftHandle r2c_plan_ = 0; ///< In-place R2C plan
  hipfftHandle c2r_plan_ = 0; ///< In-place C2R plan

  // R2C plan metadata for execution
  int r2c_length_ = 0; ///< R2C transform length (stored for potential use)
  int r2c_batch_ = 0;  ///< R2C batch size (stored for potential use)
  int r2c_dist_ =
      0; ///< R2C distance between batches (stored for potential use)

  hipStream_t stream_ = nullptr; ///< HIP stream for async FFT and memcpy ops

  /**
   * @brief Check hipFFT result and throw on error.
   *
   * @param result hipFFT result code to check
   * @param operation Description of the operation for error messages
   * @throws std::runtime_error if result is not HIPFFT_SUCCESS
   */
  static void check_hipfft(hipfftResult result, const char *operation) {
    if (result != HIPFFT_SUCCESS) {
      throw std::runtime_error(std::string("hipFFT error in ") + operation +
                               ": code " + std::to_string(result));
    }
  }
};

} // namespace parafaft

#endif // defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) ||
       // defined(__HIPCC__)

#endif // PARAFAFT_BACKEND_HIPFFT_HPP
