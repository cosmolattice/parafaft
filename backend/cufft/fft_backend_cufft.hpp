#ifndef PARAFAFT_BACKEND_CUFFT_HPP
#define PARAFAFT_BACKEND_CUFFT_HPP

#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <vector>
#include <stdexcept>
#include <string>
#include "../fft_backend.hpp"

namespace parafaft
{
  // Helper: Check CUDA result and throw on error
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

  class CuFFTBackend
  {
  public:
    using Complex = cuda::std::complex<double>;
    using Buffer = cuvector<double>;
    using ComplexBuffer = cuvector<Complex>;

    // Constructor: Initialize storage for plans
    explicit CuFFTBackend(int num_stages)
        : num_stages_(num_stages), forward_plans_(num_stages, 0), backward_plans_(num_stages, 0)
    {
    }

    // Destructor: Clean up all plans and device memory
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

    // Disable copying (plans can't be safely copied)
    CuFFTBackend(const CuFFTBackend &) = delete;
    CuFFTBackend &operator=(const CuFFTBackend &) = delete;

    // Enable moving
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

    // ========== C2C Transform Methods (for compatibility with ParaFaFT) ==========

    // Create and store plans for a specific stage (C2C transforms)
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

    // Create in-place R2C plan for padded memory optimization
    void create_r2c_inplace_plan(int length,          // Real-space FFT length N
                                 int batch,           // Number of 1D transforms
                                 double *padded_real, // Padded real buffer (for size calculation only)
                                 int stride,          // Stride (typically 1)
                                 int dist             // Distance between batches (2*(N/2+1) doubles)
    )
    {
      // R2C: N real inputs -> N/2+1 complex outputs
      // Use NULL for embed to let cuFFT infer layout from stride/dist (matches FFTW behavior)
      int n[] = {length};

      // Note: output dist is half of input dist (complex elements vs doubles)
      check_cufft(cufftPlanMany(&r2c_plan_, 1, n, NULL, stride, dist, // Real input layout
                                NULL, stride, dist / 2,               // Complex output layout
                                CUFFT_D2Z, batch),
                  "cufftPlanMany R2C");
    }

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

    // Create in-place C2R plan for padded memory optimization
    void create_c2r_inplace_plan(int length,          // Real-space output length N
                                 int batch,           // Number of transforms
                                 double *padded_real, // Padded real buffer
                                 int stride,          // Stride (typically 1)
                                 int dist             // Distance between batches (2*(N/2+1))
    )
    {
      // C2R: N/2+1 complex inputs -> N real outputs
      // Use NULL for embed to let cuFFT infer layout from stride/dist (matches FFTW behavior)
      int n[] = {length};

      check_cufft(cufftPlanMany(&c2r_plan_, 1, n, NULL, stride, dist / 2, // Complex input layout
                                NULL, stride, dist,                       // Real output layout
                                CUFFT_Z2D, batch),
                  "cufftPlanMany C2R");
    }

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

    static void memcpy(void *dest, const void *src, size_t bytes)
    {
      check_cuda(cudaMemcpy(dest, src, bytes, cudaMemcpyDefault), "cudaMemcpy general");
    }

  private:
    int num_stages_;

    // C2C plans (for general complex-to-complex transforms)
    std::vector<cufftHandle> forward_plans_;
    std::vector<cufftHandle> backward_plans_;

    // R2C/C2R plans
    cufftHandle r2c_plan_ = 0;
    cufftHandle c2r_plan_ = 0;

    // R2C plan metadata for execution
    int r2c_length_ = 0;
    int r2c_batch_ = 0;
    int r2c_dist_ = 0;

    // Helper: Check cuFFT result and throw on error
    static void check_cufft(cufftResult result, const char *operation)
    {
      if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT error in ") + operation + ": code " + std::to_string(result));
      }
    }
  };

} // namespace parafaft

#endif // PARAFAFT_BACKEND_CUFFT_HPP
