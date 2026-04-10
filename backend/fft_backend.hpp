/**
 * @file fft_backend.hpp
 * @brief FFT backend interface definition and concept documentation for ParaFaFT.
 *
 * This header defines the FFTDirection enum and documents the interface requirements
 * for FFT backends used by the ParaFaFT library. Backends must conform to this
 * documented concept to be usable with ParaFaFT and ParaFaFT_R2C.
 */

#ifndef PARAFAFT_BACKEND_HPP
#define PARAFAFT_BACKEND_HPP

#include <complex>

namespace parafaft
{

  /**
   * @enum FFTDirection
   * @brief Enum specifying the direction of an FFT transform.
   *
   * Abstracts the underlying library's direction constants (e.g., FFTW_FORWARD/FFTW_BACKWARD,
   * CUFFT_FORWARD/CUFFT_INVERSE) for portability.
   */
  enum class FFTDirection {
    Forward, ///< Forward FFT (time/space domain → frequency domain)
    Backward ///< Backward/inverse FFT (frequency domain → time/space domain)
  };

  /**
   * @enum FFTPlanFlag
   * @brief Specifies the FFT planning strategy.
   *
   * Controls how much time the FFT library spends during plan creation to find
   * optimal algorithms. Higher effort flags yield faster execution but slower
   * initialization. For FFTW, maps to FFTW_ESTIMATE/FFTW_MEASURE/FFTW_PATIENT.
   * GPU backends (cuFFT, hipFFT) ignore this flag.
   */
  enum class FFTPlanFlag {
    Estimate, ///< Fastest planning, reasonable performance
    Measure,  ///< Benchmarks several algorithms, picks fastest (2-5x speedup typical)
    Patient   ///< Exhaustive benchmarking, best performance
  };

  /**
   * @page backend_concept FFT Backend Concept
   *
   * @section overview Overview
   * A valid FFT backend must be a class that provides the types and methods
   * documented below. This concept enables ParaFaFT to work with different
   * FFT libraries (e.g., FFTW, cuFFT) through a common interface.
   *
   * @section types Required Type Definitions
   *
   * @code
   * using Complex = ...;        // Complex number type (e.g., std::complex<double>)
   * using Buffer = ...;         // Real buffer type (e.g., std::vector<double>)
   * using ComplexBuffer = ...;  // Complex buffer type (e.g., std::vector<Complex>)
   * @endcode
   *
   * @section c2c_methods C2C Transform Methods (Required)
   *
    * @subsection constructor Constructor
    * @code
    * Backend(int num_stages, FFTPlanFlag plan_flag = FFTPlanFlag::Estimate);
    * Backend(int num_stages, MPI_Comm comm, FFTPlanFlag plan_flag = FFTPlanFlag::Estimate);
    * @endcode
    * Construct a backend with storage for the given number of FFT stages.
    * The optional MPI_Comm parameter allows the backend to compute optimal thread count
    * based on available hardware concurrency divided by MPI task count.
    * The plan_flag parameter controls the planning strategy: Estimate for fast planning,
    * Measure or Patient for optimized execution. GPU backends may ignore this flag.
   *
   * @subsection create_stage_plan Plan Creation
   * @code
   * void create_stage_plan(
   *     int stage,                   // Stage index (0 to num_stages-1)
   *     int length,                  // FFT size (number of complex elements)
   *     int batch,                   // Number of 1D transforms
   *     std::complex<double>* data,  // Persistent data pointer for planning
   *     int stride,                  // Element spacing within each FFT
   *     int dist                     // Distance between consecutive FFTs
   * );
   * @endcode
   *
   * @subsection execute_stage Execution
   * @code
   * void execute_stage(int stage, FFTDirection direction, std::complex<double>* data);
   * @endcode
   * Execute the pre-created plan for the specified stage and direction.
   *
   * @subsection destructor Destructor
   * @code
   * ~Backend();
   * @endcode
   * Clean up all allocated plans and resources.
   *
   * @subsection memcpy Memory Copy
   * @code
   * static void memcpy(void* dest, const void* src, size_t bytes);
   * @endcode
   * Copy memory (may be hardware-specific, e.g., cudaMemcpy for GPU backends).
   *
   * @section design Design Principle
   * Plans are created once during initialization with representative data pointers.
   * Execution uses the new-array interface (e.g., fftw_execute_dft) which allows
   * applying the plan to different data arrays with the same layout, enabling
   * efficient reuse for operations on different memory slices.
   *
   * @section r2c_extension R2C Backend Extension
   * For R2C/C2R transforms (required by ParaFaFT_R2C), backends must also provide:
   *
   * @subsection r2c_plan R2C In-Place Plan Creation
   * @code
   * void create_r2c_inplace_plan(
   *     int length,          // Real-space FFT length N
   *     int batch,           // Number of 1D transforms
   *     double* padded_real, // Padded real buffer (size 2*(N/2+1) per batch)
   *     int stride,          // Element stride (typically 1)
   *     int dist             // Distance between batches in doubles
   * );
   * @endcode
   *
   * @subsection r2c_exec R2C Execution
   * @code
   * void execute_r2c_inplace(double* padded_real);
   * @endcode
   *
   * @subsection c2r_plan C2R In-Place Plan Creation
   * @code
   * void create_c2r_inplace_plan(
   *     int length,          // Real-space output length N
   *     int batch,           // Number of transforms
   *     double* padded_real, // Padded real buffer
   *     int stride,          // Element stride
   *     int dist             // Distance between batches
   * );
   * @endcode
   *
   * @subsection c2r_exec C2R Execution
   * @code
   * void execute_c2r_inplace(double* padded_real);
   * @endcode
   *
   * @note R2C transforms produce N/2+1 complex values from N real values.
   *       C2R transforms produce N real values from N/2+1 complex values.
   *       In-place transforms require padded buffers with 2*(N/2+1) doubles per row.
   *
   * @section distributed Distributed Transform Extension (Optional)
   *
   * Backends may optionally handle MPI communication and multi-GPU decomposition
   * internally (e.g., cuFFTMp with NVSHMEM, rocFFT with MPI). When a backend
   * sets `handles_distributed = true`, ParaFaFT delegates the entire distributed
   * transform to the backend instead of managing per-stage FFTs and exchanges.
   *
   * @subsection distributed_flag Static Flag
   * @code
   * static constexpr bool handles_distributed = false;  // default
   * @endcode
   *
   * When `true`, the backend must additionally provide:
   * @code
   * struct DistributedInfo {
   *   int local_shape[D];    // local input array dimensions
   *   int global_start[D];   // global offset of local block
   *   int output_shape[D];   // local output dimensions (may differ for R2C)
   *   int output_start[D];   // global offset of output block
   *   int required_size;     // minimum buffer size in elements
   * };
   *
   * void setup_distributed(const int global_shape[D], MPI_Comm comm);
   * DistributedInfo get_distributed_info() const;
   *
   * // Buffer management — backend allocates optimized memory (e.g. NVSHMEM)
   * Complex* get_buffer();         // device pointer to internal buffer
   * double* get_real_buffer();     // for R2C: same buffer as double*
   *
   * // In-place transforms on internal buffer (zero-copy)
   * void forward();               // C2C forward
   * void backward();              // C2C backward
   * void forward_r2c();           // R2C forward
   * void backward_c2r();          // C2R backward
   * @endcode
   *
   * @note Distributed backends are typically templated on dimension D and may
   *       static_assert to restrict supported dimensions (e.g., 2D/3D only).
   *
   * @section constraints Constraints
   * Backend implementations must be:
   * - Move-constructible (for storage in containers)
   * - Non-copyable (FFT plans typically cannot be safely copied)
   */

} // namespace parafaft

// Include specific backend implementations

#include "./fftw3/fft_backend_fftw.hpp"
#include "./cufft/fft_backend_cufft.hpp"
#include "./cufft/fft_backend_cufftmp.hpp"
#include "./hipfft/fft_backend_hipfft.hpp"

#endif // PARAFAFT_BACKEND_HPP
