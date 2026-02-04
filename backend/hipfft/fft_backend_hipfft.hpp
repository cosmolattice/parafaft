/**
 * @file fft_backend_hipfft.hpp
 * @brief hipFFT backend implementation for ParaFaFT (AMD GPU-accelerated FFT).
 *
 * This header provides an FFT backend using AMD's hipFFT library for GPU-based
 * FFT operations. Supports C2C, R2C, and C2R transforms as required by
 * ParaFaFT and ParaFaFT_R2C.
 *
 * @note Requires ROCm toolkit and hipFFT library.
 */

#ifndef PARAFAFT_BACKEND_HIPFFT_HPP
#define PARAFAFT_BACKEND_HIPFFT_HPP

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIPCC__)

#include <hipfft/hipfft.h>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "../fft_backend.hpp"

namespace parafaft
{
  /**
   * @brief Check HIP result and throw on error.
   *
   * @param result HIP error code to check
   * @param operation Description of the operation for error messages
   * @throws std::runtime_error if result is not hipSuccess
   */
  static void check_hip(hipError_t result, const char *operation)
  {
    if (result != hipSuccess) {
      throw std::runtime_error(std::string("HIP error in ") + operation + ": " + hipGetErrorString(result));
    }
  }

  /**
   * @brief (Owning) HIP vector wrapper for device memory management
   *
   * @tparam T Data type
   */
  template <typename T> class hipvector
  {
  public:
    hipvector() = default;

    /**
     * @brief Create a hipvector and allocate device memory of given size
     *
     * @param size Number of elements to allocate
     */
    hipvector(size_t size)
    {
      size_ = size;
      check_hip(hipMalloc(&data_, size_ * sizeof(T)), "hipMalloc hipvector");
    }

    /**
     * @brief Destructor: Free device memory
     */
    ~hipvector()
    {
      if (data_) {
        hipFree(data_);
      }
    }

    /**
     * @brief Resize the hipvector. Warning: old data is discarded.
     *
     * @param new_size New number of elements
     */
    void resize(size_t new_size)
    {
      if (new_size == size_) return;

      T *new_data = nullptr;
      check_hip(hipMalloc(&new_data, new_size * sizeof(T)), "hipMalloc hipvector resize");

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
   * @brief hipFFT backend for AMD GPU-accelerated FFT operations.
   *
   * Provides an interface compatible with ParaFaFT for executing FFT transforms
   * on AMD GPUs using hipFFT. Supports C2C, R2C, and C2R transforms.
   *
   * Memory management: Uses hipvector for device memory allocation.
   * All data pointers passed to this backend must be device pointers.
   */
  class HipFFTBackend
  {
  public:
    using Complex = hipDoubleComplex;         ///< Complex number type
    using Buffer = hipvector<double>;         ///< Real buffer type (device memory)
    using ComplexBuffer = hipvector<Complex>; ///< Complex buffer type (device memory)

    /**
     * @brief Construct a hipFFT backend with storage for the given number of stages.
     *
     * @param num_stages Number of FFT stages (typically D for D-dimensional transform)
     */
    explicit HipFFTBackend(int num_stages)
        : num_stages_(num_stages), forward_plans_(num_stages, 0), backward_plans_(num_stages, 0)
    {
    }

    /**
     * @brief Destructor. Cleans up all hipFFT plans.
     */
    ~HipFFTBackend()
    {
      for (auto plan : forward_plans_) {
        if (plan) hipfftDestroy(plan);
      }
      for (auto plan : backward_plans_) {
        if (plan) hipfftDestroy(plan);
      }
      if (r2c_plan_) hipfftDestroy(r2c_plan_);
      if (c2r_plan_) hipfftDestroy(c2r_plan_);
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
        : num_stages_(other.num_stages_), forward_plans_(std::move(other.forward_plans_)),
          backward_plans_(std::move(other.backward_plans_)), r2c_plan_(other.r2c_plan_), c2r_plan_(other.c2r_plan_),
          r2c_length_(other.r2c_length_), r2c_batch_(other.r2c_batch_), r2c_dist_(other.r2c_dist_)
    {
      // Clear moved-from object
      std::fill(other.forward_plans_.begin(), other.forward_plans_.end(), nullptr);
      std::fill(other.backward_plans_.begin(), other.backward_plans_.end(), nullptr);
      other.r2c_plan_ = 0;
      other.c2r_plan_ = 0;
    }

    // ========== C2C Transform Methods ==========

    /**
     * @brief Create and store hipFFT plans for a specific stage (C2C transforms).
     *
     * Creates both forward and backward hipFFT plans for the given stage.
     *
     * @note Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are NULL.
     *       We explicitly set inembed/onembed to match FFTW behavior.
     *
     * @param stage Stage index for plan storage (0 to num_stages-1)
     * @param length FFT length (number of complex elements per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param data Device pointer for plan creation (alignment reference)
     * @param stride Stride between consecutive elements in a transform
     * @param dist Distance between first elements of consecutive transforms
     */
    void create_stage_plan(int stage, int length, int batch, Complex *data, int stride, int dist)
    {
      int n[] = {length};
      // IMPORTANT: Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are NULL!
      // We must explicitly set inembed/onembed = n to match FFTW behavior.
      // With embed = n, hipFFT will use the provided stride and dist values.
      int inembed[] = {length};
      int onembed[] = {length};

      // Create forward plan
      check_hipfft(
          hipfftPlanMany(&forward_plans_[stage], 1, n, inembed, stride, dist, onembed, stride, dist, HIPFFT_Z2Z, batch),
          "hipfftPlanMany C2C forward");

      // Create backward plan
      check_hipfft(hipfftPlanMany(&backward_plans_[stage], 1, n, inembed, stride, dist, onembed, stride, dist,
                                  HIPFFT_Z2Z, batch),
                   "hipfftPlanMany C2C backward");
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
    void execute_stage(int stage, FFTDirection direction, Complex *data)
    {
      // Execute C2C transform
      hipfftHandle plan = (direction == FFTDirection::Forward) ? forward_plans_[stage] : backward_plans_[stage];
      int hipfft_direction = (direction == FFTDirection::Forward) ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
      hipfftDoubleComplex *hipfft_data = reinterpret_cast<hipfftDoubleComplex *>(data);

      check_hipfft(hipfftExecZ2Z(plan, hipfft_data, hipfft_data, hipfft_direction), "hipfftExecZ2Z");

      // Synchronize
      check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize C2C");
    }

    // ========== R2C In-Place Transform Methods ==========

    /**
     * @brief Create in-place R2C plan for padded memory layout.
     *
     * Creates an R2C (real-to-complex) plan for in-place transformation on GPU.
     * The input is N real values per transform, output is N/2+1 complex values,
     * stored in the same padded device buffer.
     *
     * @note Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are NULL.
     *       We explicitly set inembed/onembed to match FFTW behavior.
     *
     * @param length Real-space FFT length N (number of real values per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param padded_real Device pointer to padded real buffer (for size calculation only)
     * @param stride Element stride (typically 1 for contiguous data)
     * @param dist Distance between batches in doubles (should be 2*(N/2+1) for in-place)
     */
    void create_r2c_inplace_plan(int length,          // Real-space FFT length N
                                 int batch,           // Number of 1D transforms
                                 double *padded_real, // Padded real buffer (for size calculation only)
                                 int stride,          // Stride (typically 1)
                                 int dist             // Distance between batches (2*(N/2+1) doubles)
    )
    {
      // R2C: N real inputs -> N/2+1 complex outputs
      // IMPORTANT: Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are NULL!
      // We must explicitly set inembed/onembed to match FFTW behavior.
      int n[] = {length};
      int inembed[] = {length};         // Real input embed = N
      int onembed[] = {length / 2 + 1}; // Complex output embed = N/2+1

      // Note: output dist is half of input dist (complex elements vs doubles)
      check_hipfft(hipfftPlanMany(&r2c_plan_, 1, n, inembed, stride, dist, // Real input layout
                                  onembed, stride, dist / 2,               // Complex output layout
                                  HIPFFT_D2Z, batch),
                   "hipfftPlanMany R2C");
    }

    /**
     * @brief Execute in-place R2C transform.
     *
     * Transforms real data to complex data in-place on GPU. The device buffer must be
     * padded (2*(N/2+1) doubles per row) to accommodate the complex output.
     * Synchronizes the device after execution.
     *
     * @param device_padded_real Device pointer to padded real input buffer (overwritten with complex output)
     */
    void execute_r2c_inplace(double *device_padded_real)
    {
      hipfftDoubleReal *input_data = reinterpret_cast<hipfftDoubleReal *>(device_padded_real);
      hipfftDoubleComplex *output_data = reinterpret_cast<hipfftDoubleComplex *>(device_padded_real);

      // Execute R2C (D2Z = double to Z complex)
      // In-place: output overwrites input buffer
      check_hipfft(hipfftExecD2Z(r2c_plan_, input_data, output_data), "hipfftExecD2Z");

      // Synchronize
      check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize R2C");
    }

    // ========== C2R In-Place Transform Methods ==========

    /**
     * @brief Create in-place C2R plan for padded memory layout.
     *
     * Creates a C2R (complex-to-real) plan for in-place transformation on GPU.
     * The input is N/2+1 complex values per transform, output is N real values,
     * stored in the same padded device buffer.
     *
     * @note Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are NULL.
     *       We explicitly set inembed/onembed to match FFTW behavior.
     *
     * @param length Real-space output length N (number of real values per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param padded_real Device pointer to padded buffer
     * @param stride Element stride (typically 1 for contiguous data)
     * @param dist Distance between batches (padded: 2*(N/2+1) doubles)
     */
    void create_c2r_inplace_plan(int length,          // Real-space output length N
                                 int batch,           // Number of transforms
                                 double *padded_real, // Padded real buffer
                                 int stride,          // Stride (typically 1)
                                 int dist             // Distance between batches (2*(N/2+1))
    )
    {
      // C2R: N/2+1 complex inputs -> N real outputs
      // IMPORTANT: Unlike FFTW, hipFFT ignores stride/dist when inembed/onembed are NULL!
      // We must explicitly set inembed/onembed to match FFTW behavior.
      int n[] = {length};
      int inembed[] = {length / 2 + 1}; // Complex input embed = N/2+1
      int onembed[] = {length};         // Real output embed = N

      check_hipfft(hipfftPlanMany(&c2r_plan_, 1, n, inembed, stride, dist / 2, // Complex input layout
                                  onembed, stride, dist,                       // Real output layout
                                  HIPFFT_Z2D, batch),
                   "hipfftPlanMany C2R");
    }

    /**
     * @brief Execute in-place C2R transform.
     *
     * Transforms complex data to real data in-place on GPU. The device buffer layout
     * must match the plan created by create_c2r_inplace_plan().
     * Synchronizes the device after execution.
     *
     * @param device_padded_real Device pointer to buffer (complex input, real output)
     */
    void execute_c2r_inplace(double *device_padded_real)
    {
      hipfftDoubleComplex *input_data = reinterpret_cast<hipfftDoubleComplex *>(device_padded_real);
      hipfftDoubleReal *output_data = reinterpret_cast<hipfftDoubleReal *>(device_padded_real);

      // Execute C2R (Z2D = Z complex to double)
      check_hipfft(hipfftExecZ2D(c2r_plan_, input_data, output_data), "hipfftExecZ2D");

      // Synchronize
      check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize C2R");
    }

    /**
     * @brief Copy memory between buffers (device-to-device, host-to-device, etc.).
     *
     * Uses hipMemcpy with hipMemcpyDefault to automatically determine the
     * appropriate copy direction based on pointer locations.
     *
     * @param dest Destination pointer (device or host)
     * @param src Source pointer (device or host)
     * @param bytes Number of bytes to copy
     * @throws std::runtime_error if the HIP memcpy operation fails
     */
    static void memcpy(void *dest, const void *src, size_t bytes)
    {
      check_hip(hipMemcpy(dest, src, bytes, hipMemcpyDefault), "hipMemcpy general");
    }

  private:
    int num_stages_; ///< Number of FFT stages

    // C2C plans (for general complex-to-complex transforms)
    std::vector<hipfftHandle> forward_plans_;  ///< Forward C2C plans (one per stage)
    std::vector<hipfftHandle> backward_plans_; ///< Backward C2C plans (one per stage)

    // R2C/C2R plans
    hipfftHandle r2c_plan_ = 0; ///< In-place R2C plan
    hipfftHandle c2r_plan_ = 0; ///< In-place C2R plan

    // R2C plan metadata for execution
    int r2c_length_ = 0; ///< R2C transform length (stored for potential use)
    int r2c_batch_ = 0;  ///< R2C batch size (stored for potential use)
    int r2c_dist_ = 0;   ///< R2C distance between batches (stored for potential use)

    /**
     * @brief Check hipFFT result and throw on error.
     *
     * @param result hipFFT result code to check
     * @param operation Description of the operation for error messages
     * @throws std::runtime_error if result is not HIPFFT_SUCCESS
     */
    static void check_hipfft(hipfftResult result, const char *operation)
    {
      if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error(std::string("hipFFT error in ") + operation + ": code " + std::to_string(result));
      }
    }
  };

} // namespace parafaft

#endif // defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__) || defined(__HIPCC__)

#endif // PARAFAFT_BACKEND_HIPFFT_HPP
