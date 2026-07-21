/**
 * @file fft_backend_cufft.hpp
 * @brief cuFFT backend implementation for ParaFaFT (GPU-accelerated FFT).
 *
 * This header provides an FFT backend using NVIDIA's cuFFT library for
 * GPU-based FFT operations. Supports C2C, R2C, and C2R transforms as required
 * by ParaFaFT and ParaFaFT_R2C.
 *
 * Precision: CuFFTBackend is a class template parameterised on FloatType.
 *   - CuFFTBackend<double> (default) uses CUFFT_Z2Z / CUFFT_D2Z / CUFFT_Z2D.
 *   - CuFFTBackend<float>  uses CUFFT_C2C / CUFFT_R2C / CUFFT_C2R.
 *
 * @note Requires CUDA toolkit and cuFFT library.
 */

#ifndef PARAFAFT_BACKEND_CUFFT_HPP
#define PARAFAFT_BACKEND_CUFFT_HPP

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVCC__)

#include "../fft_backend.hpp"
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <dlfcn.h>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef PARAFAFT_GPU_ALLTOALLW
#define PARAFAFT_GPU_ALLTOALLW 0
#endif

namespace parafaft {
/**
 * @brief Check CUDA result and throw on error.
 *
 * @param result CUDA error code to check
 * @param operation Description of the operation for error messages
 * @throws std::runtime_error if result is not cudaSuccess
 */
static void check_cuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + operation + ": " +
                             cudaGetErrorString(result));
  }
}

/**
 * @brief (Owning) CUDA vector wrapper for device memory management
 *
 * @tparam T Data type
 */
template <typename T> class cuvector {
public:
  cuvector() = default;

  /**
   * @brief Create a cuvector and allocate device memory of given size
   *
   * @param size Number of elements to allocate
   */
  cuvector(size_t size) {
    size_ = size;
    check_cuda(cudaMalloc(&data_, size_ * sizeof(T)), "cudaMalloc cuvector");
  }

  /**
   * @brief Destructor: Free device memory
   */
  ~cuvector() {
    if (data_) {
      cudaFree(data_);
    }
  }

  /**
   * @brief Resize the cuvector. Warning: old data is discarded.
   *
   * @param new_size New number of elements
   */
  void resize(size_t new_size) {
    if (new_size == size_)
      return;

    T *new_data = nullptr;
    check_cuda(cudaMalloc(&new_data, new_size * sizeof(T)),
               "cudaMalloc cuvector resize");

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
 * @brief Precision-specific cuFFT type/plan/exec mapping.
 *
 * Routes CuFFTBackend<FloatType> to the right cuFFT plan constants
 * (CUFFT_Z2Z / D2Z / Z2D for double; CUFFT_C2C / R2C / C2R for float)
 * and the right device complex/real types and exec functions.
 */
template <typename T> struct cufft_traits;

template <> struct cufft_traits<double> {
  using real_t = cufftDoubleReal;
  using complex_t = cufftDoubleComplex;
  static constexpr cufftType c2c_type = CUFFT_Z2Z;
  static constexpr cufftType r2c_type = CUFFT_D2Z;
  static constexpr cufftType c2r_type = CUFFT_Z2D;
  static cufftResult exec_c2c(cufftHandle p, complex_t *in, complex_t *out, int dir) {
    return cufftExecZ2Z(p, in, out, dir);
  }
  static cufftResult exec_r2c(cufftHandle p, real_t *in, complex_t *out) {
    return cufftExecD2Z(p, in, out);
  }
  static cufftResult exec_c2r(cufftHandle p, complex_t *in, real_t *out) {
    return cufftExecZ2D(p, in, out);
  }
};

template <> struct cufft_traits<float> {
  using real_t = cufftReal;
  using complex_t = cufftComplex;
  static constexpr cufftType c2c_type = CUFFT_C2C;
  static constexpr cufftType r2c_type = CUFFT_R2C;
  static constexpr cufftType c2r_type = CUFFT_C2R;
  static cufftResult exec_c2c(cufftHandle p, complex_t *in, complex_t *out, int dir) {
    return cufftExecC2C(p, in, out, dir);
  }
  static cufftResult exec_r2c(cufftHandle p, real_t *in, complex_t *out) {
    return cufftExecR2C(p, in, out);
  }
  static cufftResult exec_c2r(cufftHandle p, complex_t *in, real_t *out) {
    return cufftExecC2R(p, in, out);
  }
};

/**
 * @brief cuFFT backend for GPU-accelerated FFT operations.
 *
 * Provides an interface compatible with ParaFaFT for executing FFT transforms
 * on NVIDIA GPUs using cuFFT. Supports C2C, R2C, and C2R transforms.
 *
 * @tparam FloatType Precision of the transform: `double` (default, uses
 *         CUFFT_Z2Z / D2Z / Z2D) or `float` (uses CUFFT_C2C / R2C / C2R).
 *
 * Memory management: Uses cuvector for device memory allocation.
 * All data pointers passed to this backend must be device pointers.
 */
template <typename FloatTypeT = double>
class CuFFTBackend {
public:
  using FloatType = FloatTypeT;                      ///< Scalar floating-point type
  using Complex = cuda::std::complex<FloatType>;     ///< Complex number type
  using Buffer = cuvector<FloatType>;                ///< Real buffer type (device memory)
  using ComplexBuffer = cuvector<Complex>;           ///< Complex buffer type (device memory)
private:
  using traits = cufft_traits<FloatType>;
  using cufft_real_t = typename traits::real_t;
  using cufft_complex_t = typename traits::complex_t;

public:

  /**
   * @brief Construct a cuFFT backend with storage for the given number of
   * stages.
   *
   * @param num_stages Number of FFT stages (typically D for D-dimensional
   * transform)
   * @param plan_flag Planning strategy (ignored for cuFFT, accepted for API
   * compatibility)
   */
  explicit CuFFTBackend(int num_stages,
                        FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
      : num_stages_(num_stages), forward_plans_(num_stages, 0),
        backward_plans_(num_stages, 0) {
    (void)plan_flag; // cuFFT does not support configurable planning
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  }

  /**
   * @brief Construct a cuFFT backend with MPI communicator (for API
   * compatibility).
   *
   * @param num_stages Number of FFT stages
   * @param comm MPI communicator (unused for cuFFT, but provides API
   * compatibility)
   * @param plan_flag Planning strategy (ignored for cuFFT, accepted for API
   * compatibility)
   */
  explicit CuFFTBackend(int num_stages, MPI_Comm comm,
                        FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
      : num_stages_(num_stages), forward_plans_(num_stages, 0),
        backward_plans_(num_stages, 0) {
    (void)comm;      // Suppress unused parameter warning
    (void)plan_flag; // cuFFT does not support configurable planning
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  }

  /**
   * @brief Destructor. Cleans up stream and all cuFFT plans.
   */
  ~CuFFTBackend() {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
    for (auto plan : forward_plans_) {
      if (plan)
        cufftDestroy(plan);
    }
    for (auto plan : backward_plans_) {
      if (plan)
        cufftDestroy(plan);
    }
    if (r2c_plan_)
      cufftDestroy(r2c_plan_);
    if (c2r_plan_)
      cufftDestroy(c2r_plan_);
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
      : num_stages_(other.num_stages_),
        forward_plans_(std::move(other.forward_plans_)),
        backward_plans_(std::move(other.backward_plans_)),
        r2c_plan_(other.r2c_plan_), c2r_plan_(other.c2r_plan_),
        r2c_length_(other.r2c_length_), r2c_batch_(other.r2c_batch_),
        r2c_dist_(other.r2c_dist_), stream_(other.stream_) {
    // Clear moved-from object
    std::fill(other.forward_plans_.begin(), other.forward_plans_.end(), 0);
    std::fill(other.backward_plans_.begin(), other.backward_plans_.end(), 0);
    other.r2c_plan_ = 0;
    other.c2r_plan_ = 0;
    other.stream_ = nullptr;
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
  void create_stage_plan(int stage, int length, std::size_t batch, Complex *data,
                         std::ptrdiff_t stride, std::ptrdiff_t dist) {
    (void)data;
    int n[] = {length};
    int inembed[] = {length};
    int onembed[] = {length};

    int batch_i = narrow_plan_arg(batch, "batch");
    int stride_i = narrow_plan_arg(stride, "stride");
    int dist_i = narrow_plan_arg(dist, "dist");

    check_cufft(cufftPlanMany(&forward_plans_[stage], 1, n, inembed, stride_i,
                              dist_i, onembed, stride_i, dist_i, traits::c2c_type, batch_i),
                "cufftPlanMany C2C forward");
    cufftSetStream(forward_plans_[stage], stream_);

    check_cufft(cufftPlanMany(&backward_plans_[stage], 1, n, inembed, stride_i,
                              dist_i, onembed, stride_i, dist_i, traits::c2c_type, batch_i),
                "cufftPlanMany C2C backward");
    cufftSetStream(backward_plans_[stage], stream_);
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
  void execute_stage(int stage, FFTDirection direction, Complex *data) {
    // Execute C2C transform
    cufftHandle plan = (direction == FFTDirection::Forward)
                           ? forward_plans_[stage]
                           : backward_plans_[stage];
    int cufft_direction =
        (direction == FFTDirection::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
    cufft_complex_t *cufft_data =
        reinterpret_cast<cufft_complex_t *>(data);

    check_cufft(traits::exec_c2c(plan, cufft_data, cufft_data, cufft_direction),
                "cufftExecC2C/Z2Z");
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
   * @param padded_real Device pointer to padded real buffer (for size
   * calculation only)
   * @param stride Element stride (typically 1 for contiguous data)
   * @param dist Distance between batches in FloatType scalars (should be 2*(N/2+1) for
   * in-place)
   */
  // Create in-place R2C plan for padded memory optimization
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

    check_cufft(cufftPlanMany(&r2c_plan_, 1, n, inembed, stride_i,
                              dist_i,
                              onembed, stride_i,
                              dist_i / 2,
                              traits::r2c_type, batch_i),
                "cufftPlanMany R2C");
    cufftSetStream(r2c_plan_, stream_);
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
  // Execute in-place R2C plan
  void execute_r2c_inplace(FloatType *device_padded_real) {
    cufft_real_t *input_data =
        reinterpret_cast<cufft_real_t *>(device_padded_real);
    cufft_complex_t *output_data =
        reinterpret_cast<cufft_complex_t *>(device_padded_real);

    // In-place: output overwrites input buffer
    check_cufft(traits::exec_r2c(r2c_plan_, input_data, output_data),
                "cufftExecR2C/D2Z");
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
   * @param length Real-space output length N (number of real values per
   * transform)
   * @param batch Number of 1D transforms to execute in batch
   * @param padded_real Device pointer to padded buffer
   * @param stride Element stride (typically 1 for contiguous data)
   * @param dist Distance between batches (padded: 2*(N/2+1) scalars)
   */
  // Create in-place C2R plan for padded memory optimization
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

    check_cufft(cufftPlanMany(&c2r_plan_, 1, n, inembed, stride_i,
                              dist_i / 2,
                              onembed, stride_i, dist_i,
                              traits::c2r_type, batch_i),
                "cufftPlanMany C2R");
    cufftSetStream(c2r_plan_, stream_);
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
  // Execute in-place C2R plan
  void execute_c2r_inplace(FloatType *device_padded_real) {
    cufft_complex_t *input_data =
        reinterpret_cast<cufft_complex_t *>(device_padded_real);
    cufft_real_t *output_data =
        reinterpret_cast<cufft_real_t *>(device_padded_real);

    check_cufft(traits::exec_c2r(c2r_plan_, input_data, output_data),
                "cufftExecC2R/Z2D");
  }

  /**
   * @brief Copy memory between buffers (device-to-device, host-to-device,
   * etc.).
   *
   * Uses cudaMemcpy with cudaMemcpyDefault to automatically determine the
   * appropriate copy direction based on pointer locations.
   *
   * @param dest Destination pointer (device or host)
   * @param src Source pointer (device or host)
   * @param bytes Number of bytes to copy
   * @throws std::runtime_error if the CUDA memcpy operation fails
   */
  void memcpy(void *dest, const void *src, size_t bytes) const {
    check_cuda(cudaMemcpyAsync(dest, src, bytes, cudaMemcpyDefault, stream_),
               "cudaMemcpyAsync");
  }

  /// Use manual packing by default; opt-in to MPI_Alltoallw via
  /// PARAFAFT_GPU_ALLTOALLW
  static constexpr bool use_alltoallw =
      static_cast<bool>(PARAFAFT_GPU_ALLTOALLW);
  /// cuFFT does not handle distributed MPI communication (see CuFFTMpBackend)
  static constexpr bool handles_distributed = false;

  /**
   * @brief 2D strided async device memory copy on the backend stream.
   */
  void memcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                size_t width, size_t height) const {
    check_cuda(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                 cudaMemcpyDefault, stream_),
               "cudaMemcpy2DAsync");
  }

  /**
   * @brief Synchronize the backend stream. Blocks until all enqueued
   *        operations (FFTs, memcpy) complete.
   */
  void sync() const {
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

  /**
   * @brief No-op: device allocations have no host NUMA first-touch semantics.
   *
   * Present so ParaFaFT can call it unconditionally; see the FFTW backend for
   * the CPU implementation.
   */
  void first_touch(void * /*data*/, std::size_t /*bytes*/) const {}

  // ========== P2P / IPC Methods ==========

  static constexpr bool use_p2p = true;

  static int device_id() {
    int dev;
    check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
    return dev;
  }

  static bool can_access_peer(int src_dev, int dst_dev) {
    int can = 0;
    cudaDeviceCanAccessPeer(&can, src_dev, dst_dev);
    return can != 0;
  }

  static void enable_peer_access(int peer_dev) {
    cudaError_t err = cudaDeviceEnablePeerAccess(peer_dev, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
      check_cuda(err, "cudaDeviceEnablePeerAccess");
  }

  /**
   * @brief Whether a peer link is a top-tier, full-duplex interconnect (NVLink).
   *
   * Decides whether P2P reads must be serialized into direction-separated
   * phases. That serialization exists only to dodge the throughput collapse
   * seen when two GPUs read from each other simultaneously across a shared
   * PCIe switch; a full-duplex NVLink/NVSwitch fabric has no such pathology,
   * so phasing there merely halves the achievable bandwidth.
   *
   * The CUDA runtime P2P attributes cannot answer this reliably:
   * cudaDevP2PAttrPerformanceRank is a *relative* ranking, so on a uniform-PCIe
   * box the PCIe link is the best available and reports rank 0 — indistinguishable
   * from NVLink. We instead query NVML's NVLink port state directly, via dlopen
   * so no NVML header or link dependency is added (libnvidia-ml.so.1 ships with
   * the driver). We return true only when device @p src_dev has a live NVLink
   * port whose remote endpoint is device @p dst_dev.
   *
   * Fails safe: any dlopen/dlsym/NVML/lookup failure, or a PCIe-only fabric,
   * yields false and keeps the conservative phased schedule. NVSwitch systems
   * also report false here (a GPU's NVLink remote endpoint is the switch, not
   * the peer GPU), which over-phases safely; PARAFAFT_GPU_UNPHASED_P2P recovers
   * that bandwidth. The same device (oversubscribed single-GPU) needs no phasing.
   */
  static bool peer_link_is_top_tier(int src_dev, int dst_dev) {
    if (src_dev == dst_dev)
      return true;

    // PCI addresses from CUDA, to match against NVML devices by bus id (robust
    // to CUDA_VISIBLE_DEVICES reordering the CUDA ordinals).
    cudaDeviceProp prop_src{}, prop_dst{};
    if (cudaGetDeviceProperties(&prop_src, src_dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop_dst, dst_dev) != cudaSuccess) {
      cudaGetLastError(); // clear sticky error; absence of data is not fatal
      return false;
    }

    void *lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib)
      return false; // NVML unavailable → assume PCIe (safe default)

    // NVML signatures resolved by name to avoid a header/link dependency.
    using InitFn = unsigned int (*)();
    using ShutdownFn = unsigned int (*)();
    using HandleByPciFn = unsigned int (*)(const char *, void **);
    using NvLinkStateFn = unsigned int (*)(void *, unsigned int, unsigned int *);
    using NvLinkRemotePciFn = unsigned int (*)(void *, unsigned int, void *);
    using GetPciInfoFn = unsigned int (*)(void *, void *);

    auto fn_init = reinterpret_cast<InitFn>(dlsym(lib, "nvmlInit_v2"));
    auto fn_shutdown = reinterpret_cast<ShutdownFn>(dlsym(lib, "nvmlShutdown"));
    auto fn_handle_by_pci =
        reinterpret_cast<HandleByPciFn>(dlsym(lib, "nvmlDeviceGetHandleByPciBusId_v2"));
    auto fn_nvlink_state =
        reinterpret_cast<NvLinkStateFn>(dlsym(lib, "nvmlDeviceGetNvLinkState"));
    auto fn_nvlink_remote_pci =
        reinterpret_cast<NvLinkRemotePciFn>(dlsym(lib, "nvmlDeviceGetNvLinkRemotePciInfo_v2"));
    auto fn_get_pci_info =
        reinterpret_cast<GetPciInfoFn>(dlsym(lib, "nvmlDeviceGetPciInfo_v3"));

    if (!fn_init || !fn_shutdown || !fn_handle_by_pci || !fn_nvlink_state ||
        !fn_nvlink_remote_pci || !fn_get_pci_info) {
      dlclose(lib);
      return false;
    }

    // Mirror the full nvmlPciInfo_t layout. NVML writes the whole struct, so a
    // truncated mirror (only the fields we read) would let it overflow the
    // stack — we size it fully and read just domain/bus/device.
    struct NvmlPciInfo {
      char busIdLegacy[16];
      unsigned int domain, bus, device;
      unsigned int pciDeviceId;
      unsigned int pciSubSystemId;
      char busId[32];
    };

    bool found = false;
    if (fn_init() == 0) { // NVML_SUCCESS == 0
      char bus_id_src[32], bus_id_dst[32];
      std::snprintf(bus_id_src, sizeof(bus_id_src), "%08x:%02x:%02x.0",
                    prop_src.pciDomainID, prop_src.pciBusID, prop_src.pciDeviceID);
      std::snprintf(bus_id_dst, sizeof(bus_id_dst), "%08x:%02x:%02x.0",
                    prop_dst.pciDomainID, prop_dst.pciBusID, prop_dst.pciDeviceID);

      void *dev_src = nullptr, *dev_dst = nullptr;
      if (fn_handle_by_pci(bus_id_src, &dev_src) == 0 &&
          fn_handle_by_pci(bus_id_dst, &dev_dst) == 0) {
        NvmlPciInfo pci_dst{};
        fn_get_pci_info(dev_dst, &pci_dst);

        // Enumerate NVLink ports on the source device (up to 18 on recent HW).
        for (unsigned int link = 0; link < 18 && !found; ++link) {
          unsigned int state = 0; // nvmlEnableState_t; NVML_FEATURE_ENABLED == 1
          if (fn_nvlink_state(dev_src, link, &state) != 0 || state != 1)
            continue;
          NvmlPciInfo remote_pci{};
          if (fn_nvlink_remote_pci(dev_src, link, &remote_pci) != 0)
            continue;
          if (remote_pci.domain == pci_dst.domain &&
              remote_pci.bus == pci_dst.bus &&
              remote_pci.device == pci_dst.device)
            found = true;
        }
      }
      fn_shutdown();
    }
    dlclose(lib);
    return found;
  }

  static constexpr size_t ipc_handle_size = sizeof(cudaIpcMemHandle_t);

  static void ipc_get_handle(void *devptr, void *handle) {
    check_cuda(cudaIpcGetMemHandle(
                   reinterpret_cast<cudaIpcMemHandle_t *>(handle), devptr),
               "cudaIpcGetMemHandle");
  }

  static void *ipc_open_handle(const void *handle) {
    void *ptr;
    check_cuda(cudaIpcOpenMemHandle(
                   &ptr, *reinterpret_cast<const cudaIpcMemHandle_t *>(handle),
                   cudaIpcMemLazyEnablePeerAccess),
               "cudaIpcOpenMemHandle");
    return ptr;
  }

  static void ipc_close_handle(void *ptr) { cudaIpcCloseMemHandle(ptr); }

private:
  template <typename T>
  static int narrow_plan_arg(T value, const char *name) {
    using Limit = std::numeric_limits<int>;
    bool out_of_range;
    if constexpr (std::is_signed_v<T>) {
      out_of_range = value < static_cast<T>(Limit::min()) ||
                     value > static_cast<T>(Limit::max());
    } else {
      out_of_range = value > static_cast<T>(Limit::max());
    }
    if (out_of_range) {
      throw std::runtime_error(std::string("cuFFT plan parameter '") + name +
                               "' exceeds int range; cufftPlanMany is int-limited — "
                               "use cufftXtMakePlanMany for 64-bit batches.");
    }
    return static_cast<int>(value);
  }

  int num_stages_; ///< Number of FFT stages

  // C2C plans (for general complex-to-complex transforms)
  std::vector<cufftHandle>
      forward_plans_; ///< Forward C2C plans (one per stage)
  std::vector<cufftHandle>
      backward_plans_; ///< Backward C2C plans (one per stage)

  // R2C/C2R plans
  cufftHandle r2c_plan_ = 0; ///< In-place R2C plan
  cufftHandle c2r_plan_ = 0; ///< In-place C2R plan

  // R2C plan metadata for execution
  int r2c_length_ = 0; ///< R2C transform length (stored for potential use)
  int r2c_batch_ = 0;  ///< R2C batch size (stored for potential use)
  int r2c_dist_ =
      0; ///< R2C distance between batches (stored for potential use)

  cudaStream_t stream_ = nullptr; ///< CUDA stream for async FFT and memcpy ops

  /**
   * @brief Check cuFFT result and throw on error.
   *
   * @param result cuFFT result code to check
   * @param operation Description of the operation for error messages
   * @throws std::runtime_error if result is not CUFFT_SUCCESS
   */
  // Helper: Check cuFFT result and throw on error
  static void check_cufft(cufftResult result, const char *operation) {
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error(std::string("cuFFT error in ") + operation +
                               ": code " + std::to_string(result));
    }
  }
};

} // namespace parafaft

#endif // defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__NVCC__)

#endif // PARAFAFT_BACKEND_CUFFT_HPP
