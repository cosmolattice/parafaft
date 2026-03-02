/**
 * @file fft_backend_cufft.hpp
 * @brief cuFFT backend implementation for ParaFaFT (GPU-accelerated FFT).
 *
 * This header provides an FFT backend using NVIDIA's cuFFT library for GPU-based
 * FFT operations. Supports C2C, R2C, and C2R transforms as required by
 * ParaFaFT and ParaFaFT_R2C.
 *
 * @note Requires CUDA toolkit and cuFFT library.
 */

#ifndef PARAFAFT_BACKEND_CUFFT_HPP
#define PARAFAFT_BACKEND_CUFFT_HPP

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVCC__)

#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <vector>
#include <stdexcept>
#include <string>
#include "../fft_backend.hpp"

namespace parafaft
{
  /**
   * @brief Check CUDA result and throw on error.
   *
   * @param result CUDA error code to check
   * @param operation Description of the operation for error messages
   * @throws std::runtime_error if result is not cudaSuccess
   */
  static void check_cuda(cudaError_t result, const char *operation)
  {
    if (result != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA error in ") + operation + ": " + cudaGetErrorString(result));
    }
  }

  /**
   * @brief (Owning) CUDA vector wrapper for device memory management
   *
   * @tparam T Data type
   */
  template <typename T> class cuvector
  {
  public:
    cuvector() = default;

    /**
     * @brief Create a cuvector and allocate device memory of given size
     *
     * @param size Number of elements to allocate
     */
    cuvector(size_t size)
    {
      size_ = size;
      check_cuda(cudaMalloc(&data_, size_ * sizeof(T)), "cudaMalloc cuvector");
    }

    /**
     * @brief Destructor: Free device memory
     */
    ~cuvector()
    {
      if (data_) {
        cudaFree(data_);
      }
    }

    /**
     * @brief Resize the cuvector. Warning: old data is discarded.
     *
     * @param new_size New number of elements
     */
    void resize(size_t new_size)
    {
      if (new_size == size_) return;

      T *new_data = nullptr;
      check_cuda(cudaMalloc(&new_data, new_size * sizeof(T)), "cudaMalloc cuvector resize");

      if (data_) {
        // Discard old data
        cudaFree(data_);
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
     * @brief Get size of the cuvector
     *
     * @return size_t Number of elements
     */
    size_t size() const { return size_; }

  private:
    T *data_ = nullptr;
    size_t size_ = 0;
  };

  /**
   * @brief cuFFT backend for GPU-accelerated FFT operations.
   *
   * Provides an interface compatible with ParaFaFT for executing FFT transforms
   * on NVIDIA GPUs using cuFFT. Supports C2C, R2C, and C2R transforms.
   *
   * Memory management: Uses cuvector for device memory allocation.
   * All data pointers passed to this backend must be device pointers.
   */
  class CuFFTBackend
  {
  public:
    using Complex = cuda::std::complex<double>; ///< Complex number type
    using Buffer = cuvector<double>;            ///< Real buffer type (device memory)
    using ComplexBuffer = cuvector<Complex>;    ///< Complex buffer type (device memory)

    /**
      * @brief Construct a cuFFT backend with storage for the given number of stages.
      *
      * @param num_stages Number of FFT stages (typically D for D-dimensional transform)
      */
    explicit CuFFTBackend(int num_stages)
        : num_stages_(num_stages), forward_plans_(num_stages, 0), backward_plans_(num_stages, 0)
    {
    }

    /**
     * @brief Construct a cuFFT backend with MPI communicator (for API compatibility).
     *
     * @param num_stages Number of FFT stages
     * @param comm MPI communicator (unused for cuFFT, but provides API compatibility)
     */
    explicit CuFFTBackend(int num_stages, MPI_Comm comm)
        : num_stages_(num_stages), forward_plans_(num_stages, 0), backward_plans_(num_stages, 0)
    {
      (void)comm; // Suppress unused parameter warning
    }

    /**
     * @brief Destructor. Cleans up all cuFFT plans.
     */
    ~CuFFTBackend()
    {
      for (auto plan : forward_plans_) {
        if (plan) cufftDestroy(plan);
      }
      for (auto plan : backward_plans_) {
        if (plan) cufftDestroy(plan);
      }
      if (r2c_plan_) cufftDestroy(r2c_plan_);
      if (c2r_plan_) cufftDestroy(c2r_plan_);
    }

    /// @brief Deleted copy constructor (cuFFT plans cannot be safely copied)
    CuFFTBackend(const CuFFTBackend &) = delete;
    /// @brief Deleted copy assignment (cuFFT plans cannot be safely copied)
    CuFFTBackend &operator=(const CuFFTBackend &) = delete;

    /**
     * @brief Move constructor.
     *
     * @param other Backend to move from (will be left in empty state)
     */
    CuFFTBackend(CuFFTBackend &&other) noexcept
        : num_stages_(other.num_stages_), forward_plans_(std::move(other.forward_plans_)),
          backward_plans_(std::move(other.backward_plans_)), r2c_plan_(other.r2c_plan_), c2r_plan_(other.c2r_plan_),
          r2c_length_(other.r2c_length_), r2c_batch_(other.r2c_batch_), r2c_dist_(other.r2c_dist_)
    {
      // Clear moved-from object
      std::fill(other.forward_plans_.begin(), other.forward_plans_.end(), 0);
      std::fill(other.backward_plans_.begin(), other.backward_plans_.end(), 0);
      other.r2c_plan_ = 0;
      other.c2r_plan_ = 0;
    }

    // ========== C2C Transform Methods ==========

    /**
     * @brief Create and store cuFFT plans for a specific stage (C2C transforms).
     *
     * Creates both forward and backward cuFFT plans for the given stage.
     *
     * @note Unlike FFTW, cuFFT ignores stride/dist when inembed/onembed are NULL.
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
      // IMPORTANT: Unlike FFTW, cuFFT ignores stride/dist when inembed/onembed are NULL!
      // We must explicitly set inembed/onembed = n to match FFTW behavior.
      // With embed = n, cuFFT will use the provided stride and dist values.
      int inembed[] = {length};
      int onembed[] = {length};

      // Create forward plan
      check_cufft(
          cufftPlanMany(&forward_plans_[stage], 1, n, inembed, stride, dist, onembed, stride, dist, CUFFT_Z2Z, batch),
          "cufftPlanMany C2C forward");

      // Create backward plan
      check_cufft(
          cufftPlanMany(&backward_plans_[stage], 1, n, inembed, stride, dist, onembed, stride, dist, CUFFT_Z2Z, batch),
          "cufftPlanMany C2C backward");
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
    // Execute pre-created plan for specified stage on given data
    void execute_stage(int stage, FFTDirection direction, Complex *data)
    {
      // Execute C2C transform
      cufftHandle plan = (direction == FFTDirection::Forward) ? forward_plans_[stage] : backward_plans_[stage];
      int cufft_direction = (direction == FFTDirection::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
      cufftDoubleComplex *cufft_data = reinterpret_cast<cufftDoubleComplex *>(data);

      check_cufft(cufftExecZ2Z(plan, cufft_data, cufft_data, cufft_direction), "cufftExecZ2Z");

      // Synchronize
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize C2C");
    }

    // ========== R2C In-Place Transform Methods ==========

    /**
     * @brief Create in-place R2C plan for padded memory layout.
     *
     * Creates an R2C (real-to-complex) plan for in-place transformation on GPU.
     * The input is N real values per transform, output is N/2+1 complex values,
     * stored in the same padded device buffer.
     *
     * @note Unlike FFTW, cuFFT ignores stride/dist when inembed/onembed are NULL.
     *       We explicitly set inembed/onembed to match FFTW behavior.
     *
     * @param length Real-space FFT length N (number of real values per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param padded_real Device pointer to padded real buffer (for size calculation only)
     * @param stride Element stride (typically 1 for contiguous data)
     * @param dist Distance between batches in doubles (should be 2*(N/2+1) for in-place)
     */
    // Create in-place R2C plan for padded memory optimization
    void create_r2c_inplace_plan(int length,          // Real-space FFT length N
                                 int batch,           // Number of 1D transforms
                                 double *padded_real, // Padded real buffer (for size calculation only)
                                 int stride,          // Stride (typically 1)
                                 int dist             // Distance between batches (2*(N/2+1) doubles)
    )
    {
      // R2C: N real inputs -> N/2+1 complex outputs
      // IMPORTANT: Unlike FFTW, cuFFT ignores stride/dist when inembed/onembed are NULL!
      // We must explicitly set inembed/onembed to match FFTW behavior.
      int n[] = {length};
      int inembed[] = {length};         // Real input embed = N
      int onembed[] = {length / 2 + 1}; // Complex output embed = N/2+1

      // Note: output dist is half of input dist (complex elements vs doubles)
      check_cufft(cufftPlanMany(&r2c_plan_, 1, n, inembed, stride, dist, // Real input layout
                                onembed, stride, dist / 2,               // Complex output layout
                                CUFFT_D2Z, batch),
                  "cufftPlanMany R2C");
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
    // Execute in-place R2C plan
    void execute_r2c_inplace(double *device_padded_real)
    {
      cufftDoubleReal *input_data = reinterpret_cast<cufftDoubleReal *>(device_padded_real);
      cufftDoubleComplex *output_data = reinterpret_cast<cufftDoubleComplex *>(device_padded_real);

      // Execute R2C (D2Z = double to Z complex)
      // In-place: output overwrites input buffer
      check_cufft(cufftExecD2Z(r2c_plan_, input_data, output_data), "cufftExecD2Z");

      // Synchronize
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize R2C");
    }

    // ========== C2R In-Place Transform Methods ==========

    /**
     * @brief Create in-place C2R plan for padded memory layout.
     *
     * Creates a C2R (complex-to-real) plan for in-place transformation on GPU.
     * The input is N/2+1 complex values per transform, output is N real values,
     * stored in the same padded device buffer.
     *
     * @note Unlike FFTW, cuFFT ignores stride/dist when inembed/onembed are NULL.
     *       We explicitly set inembed/onembed to match FFTW behavior.
     *
     * @param length Real-space output length N (number of real values per transform)
     * @param batch Number of 1D transforms to execute in batch
     * @param padded_real Device pointer to padded buffer
     * @param stride Element stride (typically 1 for contiguous data)
     * @param dist Distance between batches (padded: 2*(N/2+1) doubles)
     */
    // Create in-place C2R plan for padded memory optimization
    void create_c2r_inplace_plan(int length,          // Real-space output length N
                                 int batch,           // Number of transforms
                                 double *padded_real, // Padded real buffer
                                 int stride,          // Stride (typically 1)
                                 int dist             // Distance between batches (2*(N/2+1))
    )
    {
      // C2R: N/2+1 complex inputs -> N real outputs
      // IMPORTANT: Unlike FFTW, cuFFT ignores stride/dist when inembed/onembed are NULL!
      // We must explicitly set inembed/onembed to match FFTW behavior.
      int n[] = {length};
      int inembed[] = {length / 2 + 1}; // Complex input embed = N/2+1
      int onembed[] = {length};         // Real output embed = N

      check_cufft(cufftPlanMany(&c2r_plan_, 1, n, inembed, stride, dist / 2, // Complex input layout
                                onembed, stride, dist,                       // Real output layout
                                CUFFT_Z2D, batch),
                  "cufftPlanMany C2R");
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
    // Execute in-place C2R plan
    void execute_c2r_inplace(double *device_padded_real)
    {
      cufftDoubleComplex *input_data = reinterpret_cast<cufftDoubleComplex *>(device_padded_real);
      cufftDoubleReal *output_data = reinterpret_cast<cufftDoubleReal *>(device_padded_real);

      // Execute C2R (Z2D = Z complex to double)
      check_cufft(cufftExecZ2D(c2r_plan_, input_data, output_data), "cufftExecZ2D");

      // Synchronize
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize C2R");
    }

    /**
     * @brief Copy memory between buffers (device-to-device, host-to-device, etc.).
     *
     * Uses cudaMemcpy with cudaMemcpyDefault to automatically determine the
     * appropriate copy direction based on pointer locations.
     *
     * @param dest Destination pointer (device or host)
     * @param src Source pointer (device or host)
     * @param bytes Number of bytes to copy
     * @throws std::runtime_error if the CUDA memcpy operation fails
     */
    static void memcpy(void *dest, const void *src, size_t bytes)
    {
      check_cuda(cudaMemcpy(dest, src, bytes, cudaMemcpyDefault), "cudaMemcpy general");
    }

  private:
    int num_stages_; ///< Number of FFT stages

    // C2C plans (for general complex-to-complex transforms)
    std::vector<cufftHandle> forward_plans_;  ///< Forward C2C plans (one per stage)
    std::vector<cufftHandle> backward_plans_; ///< Backward C2C plans (one per stage)

    // R2C/C2R plans
    cufftHandle r2c_plan_ = 0; ///< In-place R2C plan
    cufftHandle c2r_plan_ = 0; ///< In-place C2R plan

    // R2C plan metadata for execution
    int r2c_length_ = 0; ///< R2C transform length (stored for potential use)
    int r2c_batch_ = 0;  ///< R2C batch size (stored for potential use)
    int r2c_dist_ = 0;   ///< R2C distance between batches (stored for potential use)

    /**
     * @brief Check cuFFT result and throw on error.
     *
     * @param result cuFFT result code to check
     * @param operation Description of the operation for error messages
     * @throws std::runtime_error if result is not CUFFT_SUCCESS
     */
    // Helper: Check cuFFT result and throw on error
    static void check_cufft(cufftResult result, const char *operation)
    {
      if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT error in ") + operation + ": code " + std::to_string(result));
      }
    }
  };

} // namespace parafaft

#endif // defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVCC__)

#endif // PARAFAFT_BACKEND_CUFFT_HPP
