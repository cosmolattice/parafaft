#ifndef PARAFAFT_BACKEND_CUFFT_HPP
#define PARAFAFT_BACKEND_CUFFT_HPP

#include <cufft.h>
#include <cuda_runtime.h>
#include <complex>
#include <vector>
#include <stdexcept>
#include <string>
#include "../fft_backend.hpp"

namespace parafaft
{

  class CuFFTBackend
  {
  public:
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
      if (device_buffer_) cudaFree(device_buffer_);
      if (c2c_device_buffer_) cudaFree(c2c_device_buffer_);
    }

    // Disable copying (plans can't be safely copied)
    CuFFTBackend(const CuFFTBackend &) = delete;
    CuFFTBackend &operator=(const CuFFTBackend &) = delete;

    // Enable moving
    CuFFTBackend(CuFFTBackend &&other) noexcept
        : num_stages_(other.num_stages_), forward_plans_(std::move(other.forward_plans_)),
          backward_plans_(std::move(other.backward_plans_)), r2c_plan_(other.r2c_plan_), c2r_plan_(other.c2r_plan_),
          device_buffer_(other.device_buffer_), buffer_size_(other.buffer_size_), r2c_length_(other.r2c_length_),
          r2c_batch_(other.r2c_batch_), r2c_dist_(other.r2c_dist_), c2c_device_buffer_(other.c2c_device_buffer_),
          c2c_buffer_size_(other.c2c_buffer_size_)
    {
      // Clear moved-from object
      std::fill(other.forward_plans_.begin(), other.forward_plans_.end(), 0);
      std::fill(other.backward_plans_.begin(), other.backward_plans_.end(), 0);
      other.r2c_plan_ = 0;
      other.c2r_plan_ = 0;
      other.device_buffer_ = nullptr;
      other.buffer_size_ = 0;
      other.c2c_device_buffer_ = nullptr;
      other.c2c_buffer_size_ = 0;
    }

    // ========== C2C Transform Methods (for compatibility with ParaFaFT) ==========

    // Create and store plans for a specific stage (C2C transforms)
    void create_stage_plan(int stage, int length, int batch, std::complex<double> *data, int stride, int dist)
    {
      // Store stage metadata
      if (stage_metadata_.size() <= static_cast<size_t>(stage)) {
        stage_metadata_.resize(stage + 1);
      }
      stage_metadata_[stage] = {length, batch, stride, dist};

      // Buffer size for complex data
      size_t size_complex = static_cast<size_t>((batch - 1) * dist + length);

      // Allocate or reallocate device buffer if needed
      if (c2c_device_buffer_ == nullptr || size_complex > c2c_buffer_size_) {
        if (c2c_device_buffer_) {
          cudaFree(c2c_device_buffer_);
        }
        c2c_buffer_size_ = size_complex;
        check_cuda(cudaMalloc(&c2c_device_buffer_, size_complex * sizeof(cufftDoubleComplex)), "cudaMalloc C2C buffer");
      }

      int n[] = {length};
      // CRITICAL: Provide explicit embed arrays for cuFFT (NULL doesn't work like FFTW)
      int embed[] = {length};

      // Create forward plan
      check_cufft(
          cufftPlanMany(&forward_plans_[stage], 1, n, embed, stride, dist, embed, stride, dist, CUFFT_Z2Z, batch),
          "cufftPlanMany C2C forward");

      // Create backward plan
      check_cufft(
          cufftPlanMany(&backward_plans_[stage], 1, n, embed, stride, dist, embed, stride, dist, CUFFT_Z2Z, batch),
          "cufftPlanMany C2C backward");
    }

    // Execute pre-created plan for specified stage on given data
    void execute_stage(int stage, FFTDirection direction, std::complex<double> *data)
    {
      auto &meta = stage_metadata_[stage];
      size_t size_bytes = c2c_buffer_size_ * sizeof(cufftDoubleComplex);

      // Transfer host -> device
      check_cuda(cudaMemcpy(c2c_device_buffer_, data, size_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D C2C");

      // Execute C2C transform
      cufftHandle plan = (direction == FFTDirection::Forward) ? forward_plans_[stage] : backward_plans_[stage];
      int cufft_direction = (direction == FFTDirection::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;

      check_cufft(cufftExecZ2Z(plan, static_cast<cufftDoubleComplex *>(c2c_device_buffer_),
                               static_cast<cufftDoubleComplex *>(c2c_device_buffer_), cufft_direction),
                  "cufftExecZ2Z");

      // Synchronize and transfer back
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize C2C");
      check_cuda(cudaMemcpy(data, c2c_device_buffer_, size_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H C2C");
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
      // Store metadata for execution
      r2c_length_ = length;
      r2c_batch_ = batch;
      r2c_dist_ = dist;

      // Buffer size for padded real data
      // dist is in doubles, size needed = (batch-1)*dist + 2*(length/2+1)
      size_t size_doubles = static_cast<size_t>((batch - 1) * dist + 2 * (length / 2 + 1));

      // Allocate or reallocate device buffer
      if (device_buffer_ == nullptr || size_doubles > buffer_size_) {
        if (device_buffer_) {
          cudaFree(device_buffer_);
        }
        buffer_size_ = size_doubles;
        check_cuda(cudaMalloc(&device_buffer_, size_doubles * sizeof(double)), "cudaMalloc R2C buffer");
      }

      // R2C: N real inputs -> N/2+1 complex outputs
      // CRITICAL: Asymmetric embed values - input and output sizes differ!
      int n[] = {length};
      int inembed[] = {2 * (length / 2 + 1)}; // Real input: padded size
      int onembed[] = {length / 2 + 1};       // Complex output: N/2+1 elements

      // Note: output dist is half of input dist (complex elements vs doubles)
      check_cufft(cufftPlanMany(&r2c_plan_, 1, n, inembed, stride, dist, // Real input layout
                                onembed, stride, dist / 2,               // Complex output layout
                                CUFFT_D2Z, batch),
                  "cufftPlanMany R2C");
    }

    // Execute in-place R2C plan
    void execute_r2c_inplace(double *host_padded_real)
    {
      size_t size_bytes = buffer_size_ * sizeof(double);

      // Transfer host -> device
      check_cuda(cudaMemcpy(device_buffer_, host_padded_real, size_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D R2C");

      // Execute R2C (D2Z = double to Z complex)
      // In-place: output overwrites input buffer
      check_cufft(cufftExecD2Z(r2c_plan_, static_cast<cufftDoubleReal *>(device_buffer_),
                               static_cast<cufftDoubleComplex *>(device_buffer_)),
                  "cufftExecD2Z");

      // Ensure FFT completes before transfer
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize R2C");

      // Transfer device -> host
      check_cuda(cudaMemcpy(host_padded_real, device_buffer_, size_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H R2C");
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
      // Buffer size for padded data
      size_t size_doubles = static_cast<size_t>((batch - 1) * dist + 2 * (length / 2 + 1));

      // Allocate or reallocate device buffer if needed
      if (device_buffer_ == nullptr || size_doubles > buffer_size_) {
        if (device_buffer_) {
          cudaFree(device_buffer_);
        }
        buffer_size_ = size_doubles;
        check_cuda(cudaMalloc(&device_buffer_, size_doubles * sizeof(double)), "cudaMalloc C2R buffer");
      }

      // C2R: N/2+1 complex inputs -> N real outputs
      // CRITICAL: Asymmetric embed - reversed from R2C
      int n[] = {length};
      int inembed[] = {length / 2 + 1};       // Complex input: N/2+1 elements
      int onembed[] = {2 * (length / 2 + 1)}; // Real output: padded size

      check_cufft(cufftPlanMany(&c2r_plan_, 1, n, inembed, stride, dist / 2, // Complex input layout
                                onembed, stride, dist,                       // Real output layout
                                CUFFT_Z2D, batch),
                  "cufftPlanMany C2R");
    }

    // Execute in-place C2R plan
    void execute_c2r_inplace(double *host_padded_real)
    {
      size_t size_bytes = buffer_size_ * sizeof(double);

      // Transfer host -> device
      check_cuda(cudaMemcpy(device_buffer_, host_padded_real, size_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D C2R");

      // Execute C2R (Z2D = Z complex to double)
      check_cufft(cufftExecZ2D(c2r_plan_, static_cast<cufftDoubleComplex *>(device_buffer_),
                               static_cast<cufftDoubleReal *>(device_buffer_)),
                  "cufftExecZ2D");

      // Ensure FFT completes before transfer
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize C2R");

      // Transfer device -> host
      check_cuda(cudaMemcpy(host_padded_real, device_buffer_, size_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H C2R");
    }

  private:
    int num_stages_;

    // C2C plans (for general complex-to-complex transforms)
    std::vector<cufftHandle> forward_plans_;
    std::vector<cufftHandle> backward_plans_;

    // R2C/C2R plans
    cufftHandle r2c_plan_ = 0;
    cufftHandle c2r_plan_ = 0;

    // Device buffer for R2C/C2R
    void *device_buffer_ = nullptr;
    size_t buffer_size_ = 0;

    // R2C plan metadata for execution
    int r2c_length_ = 0;
    int r2c_batch_ = 0;
    int r2c_dist_ = 0;

    // Device buffer for C2C transforms
    void *c2c_device_buffer_ = nullptr;
    size_t c2c_buffer_size_ = 0;

    // Stage metadata for C2C transforms
    struct StageMetadata {
      int length;
      int batch;
      int stride;
      int dist;
    };
    std::vector<StageMetadata> stage_metadata_;

    // Helper: Check cuFFT result and throw on error
    static void check_cufft(cufftResult result, const char *operation)
    {
      if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT error in ") + operation + ": code " + std::to_string(result));
      }
    }

    // Helper: Check CUDA result and throw on error
    static void check_cuda(cudaError_t result, const char *operation)
    {
      if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + operation + ": " + cudaGetErrorString(result));
      }
    }
  };

} // namespace parafaft

#endif // PARAFAFT_BACKEND_CUFFT_HPP
