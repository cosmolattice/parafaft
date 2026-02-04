#ifndef PARAFAFT_R2C_HPP
#define PARAFAFT_R2C_HPP

// ============================================================================
// Real-to-Complex (R2C) / Complex-to-Real (C2R) Parallel FFT Implementation
// ============================================================================
//
// This implementation extends the pencil decomposition approach from
// parafaft_generic.hpp to handle real-valued data with full roundtrip support:
//   - forward(): R2C transform (real input → complex output)
//   - backward(): C2R transform (complex input → real output)
//   - forward_in_place(): In-place R2C (buffer: real → complex)
//   - backward_in_place(): In-place C2R (buffer: complex → real)
//
// Key differences from C2C:
//   - Input type: real (double) instead of complex
//   - Output shape: [..., N/2+1] instead of [..., N] on last axis
//   - First stage uses fftw_plan_many_dft_r2c (forward) / c2r (backward)
//   - Subsequent stages use standard C2C FFT on reduced-size complex arrays
//
// In-Place API
// -------------
// The forward_in_place() and backward_in_place() methods operate fully in-place
// on a single buffer. The buffer is reinterpreted between real and
// complex views:
//
//   double* buffer = fftw_alloc_real(fft.get_required_output_size());
//
//   // Initialize real data (first N elements per row)
//   // ...
//
//   fft.forward_in_place(buffer);  // Real → Complex (in-place)
//   // Complex result: reinterpret_cast<Complex*>(buffer)
//
//   fft.backward_in_place(buffer); // Complex → Real (in-place)
//   // Real result: first N elements per row
//
//   fftw_free(buffer);
//
// Memory Optimization
// -------------------
// Uses ping-pong buffering with the user's padded buffer and one internal
// complex buffer. This reduces memory usage compared to separate I/O buffers.
//
// Buffer Alignment Recommendation
// -------------------------------
// For best performance with FFTW's new-array execution, allocate user buffers
// using FFTW-aligned memory:
//
//   double* real_data = fftw_alloc_real(fft.get_local_real_size());
//   fftw_complex* complex_data = fftw_alloc_complex(fft.get_local_complex_size());
//   // ... use with forward()/backward() ...
//   fftw_free(real_data);
//   fftw_free(complex_data);
//
// Standard std::vector or malloc'd buffers will work correctly but may have
// slightly reduced performance on some architectures if alignment differs
// from the internal planning buffers.
//
// ============================================================================

#include <mpi.h>
#include "backend/fftw3/fft_backend_fftw.hpp"
#include <vector>
#include <complex>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace parafaft
{
  // ============================================================================
  // Utility Functions
  // ============================================================================

  /**
   * @brief Balanced block-contiguous decomposition using Barry Smith's formula.
   *
   * Distributes N elements across M processors ensuring load balance:
   * processors either get q or q+1 elements where q = N/M.
   *
   * @param N Total number of elements to distribute
   * @param M Number of processors
   * @param p Processor rank (0 <= p < M)
   * @param[out] n Number of elements assigned to processor p
   * @param[out] s Starting index for processor p in global array
   */
  inline void r2c_decompose(int N, int M, int p, int &n, int &s)
  {
    int q = N / M;
    int r = N % M;
    n = (r > p) ? (q + 1) : q;
    s = (r > p) ? (n * p) : (q * p + r);
  }

  /**
   * @brief Create MPI subarray datatypes for partitioning along an axis.
   *
   * Creates MPI subarray datatypes that describe how to partition a D-dimensional
   * array along a specific axis. This is key to avoiding local transposes by letting
   * MPI handle the complex index mapping during redistribution.
   *
   * @param datatype Base MPI datatype (e.g., MPI_C_DOUBLE_COMPLEX)
   * @param ndims Number of dimensions
   * @param sizes Size of the array in each dimension
   * @param axis Axis along which to partition
   * @param nparts Number of partitions (typically number of processors)
   * @param[out] subarrays Array of nparts MPI datatypes (must be pre-allocated)
   */
  inline void r2c_subarray(MPI_Datatype datatype, int ndims, const int sizes[], int axis, int nparts,
                           MPI_Datatype subarrays[])
  {
    std::vector<int> subsizes(ndims), substarts(ndims);
    for (int i = 0; i < ndims; i++) {
      subsizes[i] = sizes[i];
      substarts[i] = 0;
    }
    for (int p = 0; p < nparts; p++) {
      int n, s;
      r2c_decompose(sizes[axis], nparts, p, n, s);
      subsizes[axis] = n;
      substarts[axis] = s;

      MPI_Type_create_subarray(ndims, sizes, subsizes.data(), substarts.data(), MPI_ORDER_C, datatype, &subarrays[p]);
      MPI_Type_commit(&subarrays[p]);
    }
  }

  /**
   * @brief Global data redistribution using MPI_Alltoallw.
   *
   * Redistributes data between two pencil decomposition stages, changing which
   * axis is local vs distributed without any local transposes.
   *
   * @param comm MPI subcommunicator for the redistribution
   * @param datatype Base MPI datatype
   * @param ndims Number of dimensions
   * @param sizesA Array dimensions before redistribution
   * @param arrayA Input array (distributed along axisA)
   * @param axisA Axis that is distributed in the input
   * @param sizesB Array dimensions after redistribution
   * @param[out] arrayB Output array (will be distributed along axisB)
   * @param axisB Axis that will be distributed in the output
   */
  inline void r2c_exchange(MPI_Comm comm, MPI_Datatype datatype, int ndims, const int sizesA[], void *arrayA, int axisA,
                           const int sizesB[], void *arrayB, int axisB)
  {
    int nparts;
    MPI_Comm_size(comm, &nparts);

    std::vector<MPI_Datatype> subarraysA(nparts), subarraysB(nparts);
    r2c_subarray(datatype, ndims, sizesA, axisA, nparts, subarraysA.data());
    r2c_subarray(datatype, ndims, sizesB, axisB, nparts, subarraysB.data());

    std::vector<int> counts(nparts, 1), displs(nparts, 0);

    MPI_Barrier(comm);

    MPI_Alltoallw(arrayA, counts.data(), displs.data(), subarraysA.data(), arrayB, counts.data(), displs.data(),
                  subarraysB.data(), comm);

    MPI_Barrier(comm);

    for (int p = 0; p < nparts; p++) {
      MPI_Type_free(&subarraysA[p]);
      MPI_Type_free(&subarraysB[p]);
    }
  }

  // ============================================================================
  // ParaFaFT_R2C Class
  // ============================================================================

  /**
   * @brief Parallel Real-to-Complex (R2C) and Complex-to-Real (C2R) FFT using pencil decomposition.
   *
   * This class implements distributed-memory parallel FFT for real-valued data using
   * MPI pencil decomposition. It supports arbitrary dimensions D and provides both
   * out-of-place and in-place transform interfaces.
   *
   * Key features:
   * - Forward R2C: real input → complex output with reduced last dimension (N/2+1)
   * - Backward C2R: complex input → real output (inverse of forward)
   * - In-place variants that operate on a single buffer
   * - Ping-pong buffering with minimal memory overhead
   *
   * @tparam D Number of dimensions (must be >= 2)
   * @tparam Backend FFT backend type (default: FFTWBackend)
   */
  template <int D, typename Backend = FFTWBackend> class ParaFaFT_R2C
  {
  public:
    using Complex = typename Backend::Complex;
    using Buffer = typename Backend::Buffer;
    using ComplexBuffer = typename Backend::ComplexBuffer;

    /**
     * @brief Construct a parallel R2C FFT object.
     *
     * Sets up the processor grid, pencil decomposition, and creates FFT plans
     * for all stages of the transform.
     *
     * @param global_shape Array of D integers specifying the global real array dimensions
     * @param comm MPI communicator (default: MPI_COMM_WORLD)
     */
    ParaFaFT_R2C(const int global_shape[D], MPI_Comm comm = MPI_COMM_WORLD) : comm_world_(comm), backend_(D)
    {
      MPI_Comm_rank(comm_world_, &rank_);
      MPI_Comm_size(comm_world_, &size_);

      // Store global real array dimensions
      for (int i = 0; i < D; ++i) {
        global_real_shape_[i] = global_shape[i];
      }

      // Create (D-1)-dimensional Cartesian processor grid
      std::vector<int> dims(D - 1, 0);
      MPI_Dims_create(size_, D - 1, dims.data());

      for (int i = 0; i < D - 1; ++i) {
        dims_[i] = dims[i];
      }

      // Create Cartesian communicator (non-periodic)
      std::vector<int> periods(D - 1, 0);
      MPI_Cart_create(comm_world_, D - 1, dims.data(), periods.data(), 1, &comm_cart_);

      // Get this processor's coordinates in the grid
      std::vector<int> coords(D - 1);
      MPI_Cart_coords(comm_cart_, rank_, D - 1, coords.data());
      for (int i = 0; i < D - 1; ++i) {
        coords_[i] = coords[i];
      }

      // Create (D-1) subcommunicators
      subcomms_.resize(D - 1);
      for (int i = 0; i < D - 1; ++i) {
        std::vector<int> remain_dims(D - 1, 0);
        remain_dims[i] = 1;
        MPI_Cart_sub(comm_cart_, remain_dims.data(), &subcomms_[i]);
      }

      // Compute local array shapes for each stage
      setup_stage_shapes();

      // Compute maximum complex buffer size across all stages
      int max_complex_size = 0;
      for (int stage = 0; stage < D; ++stage) {
        int size = 1;
        for (int i = 0; i < D; ++i) {
          size *= stage_output_shapes_[stage][i];
        }
        max_complex_size = std::max(max_complex_size, size);
      }

      // Allocate ping-pong buffer sized to handle any stage
      // (scratch_a_ was removed - only used during planning, now uses local temp)
      scratch_b_.resize(max_complex_size);

      // Create FFT plans
      create_backend_plans();
    }

    /**
     * @brief Destructor. Frees MPI communicators if MPI is not finalized.
     */
    ~ParaFaFT_R2C()
    {
      int finalized;
      MPI_Finalized(&finalized);
      if (!finalized) {
        for (auto &comm : subcomms_) {
          if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
          }
        }
        if (comm_cart_ != MPI_COMM_NULL) {
          MPI_Comm_free(&comm_cart_);
        }
      }
    }

    /**
     * @brief Forward R2C transform (out-of-place).
     *
     * Computes the forward Real-to-Complex FFT, transforming real-space data
     * to Fourier-space complex coefficients.
     *
     * @param real_input Input real array of size get_local_real_size().
     *                   Not modified; user retains ownership.
     * @param complex_output Output complex array. Must be at least
     *                       get_required_output_size()/2 complex elements
     *                       to accommodate intermediate MPI redistributions.
     *                       Final result uses only get_local_complex_size() elements.
     *
     * @note The output buffer is used as working space during the transform.
     *       Intermediate stages may require more space than the final output
     *       due to MPI redistribution changing local array sizes.
     */
    void forward(const double *real_input, Complex *complex_output)
    {
      std::cout << "Rank " << rank_ << ": Padding real input data" << std::endl;
      // Pad the data and move it into the destination buffer
      copy_real_to_padded(real_input, reinterpret_cast<double *>(complex_output));

      std::cout << "Rank " << rank_ << ": Starting forward_in_place()" << std::endl;
      // In-place forward transform
      forward_in_place(reinterpret_cast<double *>(complex_output));
    }

    /**
     * @brief Forward R2C transform (fully in-place).
     *
     * Computes the forward Real-to-Complex FFT entirely in-place on a single buffer.
     * The buffer is reinterpreted between real and complex views during computation.
     *
     * @param padded_buffer In-place real/complex buffer.
     *                      Size: get_required_output_size() doubles.
     *                      Input: Padded real data with 2*(N/2+1) doubles per row,
     *                             where the first N elements contain the real values.
     *                      Output: Complex data accessible via reinterpret_cast<Complex*>(buffer).
     *
     * @note After completion, access the complex output via:
     *       `reinterpret_cast<Complex*>(buffer)`
     *       The complex array has get_local_complex_size() valid elements.
     *
     * @warning Input real data is destroyed during computation!
     */
    void forward_in_place(double *padded_buffer)
    {
      // Reinterpret padded real buffer as complex for ping-pong
      Complex *padded_as_complex = reinterpret_cast<Complex *>(padded_buffer);

      // Stage 0: In-place R2C FFT on padded buffer
      backend_.execute_r2c_inplace(padded_buffer);

      // Determine starting buffer based on parity of swaps
      // D-1 iterations means D-1 swaps
      // Even swaps: end where started; Odd swaps: end in opposite buffer
      Complex *src;
      Complex *dst;
      if ((D - 1) % 2 == 0) {
        // Even swaps: start at padded, end at padded
        src = padded_as_complex;
        dst = scratch_b_.data();
      } else {
        // Odd swaps: start at scratch_b, end at padded
        // Need to copy R2C output to scratch_b_ first
        int stage0_complex_size = 1;
        for (int i = 0; i < D; ++i) {
          stage0_complex_size *= stage_output_shapes_[0][i];
        }
        backend_.memcpy(scratch_b_.data(), padded_as_complex, stage0_complex_size * sizeof(Complex));
        src = scratch_b_.data();
        dst = padded_as_complex;
      }

      // Stages 1 to D-1: MPI redistribute + C2C FFT
      for (int stage = 1; stage < D; ++stage) {
        int axis = D - 1 - stage;
        int prev_axis = axis + 1;

        // MPI exchange: src → dst
        r2c_exchange(subcomms_[axis], MPI_C_DOUBLE_COMPLEX, D, stage_output_shapes_[stage - 1].data(), src, prev_axis,
                     stage_shapes_[stage].data(), dst, axis);

        // C2C FFT in-place on dst
        perform_fft_on_buffer(stage, axis, FFTDirection::Forward, dst);

        // Swap for next iteration
        std::swap(src, dst);
      }

      // After D-1 swaps, src points to padded_as_complex
      // Result is already in place - no copy needed
    }

    /**
     * @brief Backward C2R transform (out-of-place).
     *
     * Computes the backward Complex-to-Real FFT, transforming Fourier-space
     * complex coefficients back to real-space data.
     *
     * @param complex_input Input complex array of size get_local_complex_size().
     *                      Not modified; user retains ownership.
     * @param real_output Output real array of size get_local_real_size().
     *                    Will contain reconstructed real data.
     *
     * @note Internally uses a working buffer of size get_required_output_size()/2
     *       complex elements to handle intermediate MPI redistributions.
     */
    void backward(const Complex *complex_input, double *real_output)
    {
      // In-place backward transform
      backward_in_place(reinterpret_cast<double *>(complex_input));

      // Copy real output from padded buffer
      copy_padded_to_real(reinterpret_cast<double *>(complex_input), real_output);
    }

    /**
     * @brief Backward C2R transform (fully in-place).
     *
     * Computes the backward Complex-to-Real FFT entirely in-place on a single buffer.
     * The buffer is reinterpreted between complex and real views during computation.
     *
     * @param padded_buffer In-place complex/real buffer.
     *                      Size: get_required_output_size() doubles.
     *                      Input: Complex data (from forward_in_place or user).
     *                      Output: Real data in first N elements per row.
     *
     * @note Input complex data is interpreted as `reinterpret_cast<Complex*>(buffer)`.
     *       After completion, the buffer contains real data.
     *
     * @warning Input complex data is destroyed during computation!
     */
    void backward_in_place(double *padded_buffer)
    {
      // Reinterpret padded real buffer as complex for ping-pong
      Complex *padded_as_complex = reinterpret_cast<Complex *>(padded_buffer);

      // Determine starting buffer based on parity
      // We need src to point to padded_as_complex after D-1 swaps for in-place C2R
      Complex *src;
      Complex *dst;
      if ((D - 1) % 2 == 0) {
        // Even swaps: start at padded, end at padded
        // Input is already in padded_as_complex - no copy needed
        src = padded_as_complex;
        dst = scratch_b_.data();
      } else {
        // Odd swaps: start at scratch_b, end at padded
        // Must copy input from padded to scratch_b to start there
        int input_size = 1;
        for (int i = 0; i < D; ++i) {
          input_size *= stage_shapes_[D - 1][i];
        }
        backend_.memcpy(scratch_b_.data(), padded_as_complex, input_size * sizeof(Complex));
        src = scratch_b_.data();
        dst = padded_as_complex;
      }

      // Stages D-1 down to 1: C2C backward FFT + MPI redistribute
      for (int stage = D - 1; stage >= 1; --stage) {
        int axis = D - 1 - stage;

        // C2C backward FFT in-place on src
        perform_fft_on_buffer(stage, axis, FFTDirection::Backward, src);

        // MPI exchange: src → dst
        int prev_axis = axis + 1;
        r2c_exchange(subcomms_[axis], MPI_C_DOUBLE_COMPLEX, D, stage_shapes_[stage].data(), src, axis,
                     stage_output_shapes_[stage - 1].data(), dst, prev_axis);

        // Swap for next iteration
        std::swap(src, dst);
      }

      // After D-1 swaps, src points to padded_as_complex
      // Stage 0: In-place C2R FFT on padded buffer
      backend_.execute_c2r_inplace(padded_buffer);
    }

    /**
     * @brief Get the local real array size.
     *
     * @return Total number of real elements in the local input array.
     */
    int get_local_real_size() const
    {
      int size = 1;
      for (int i = 0; i < D; ++i) {
        size *= stage_shapes_[0][i];
      }
      return size;
    }

    /**
     * @brief Get the output memory size (in doubles) required for transforms.
     *
     * Returns the maximum buffer size needed across all stages, in doubles.
     * This must be large enough for:
     *   1. Stage 0 padded real input: batch × 2×(N/2+1) doubles
     *   2. All intermediate complex stages after MPI redistribution
     *
     * The redistribution can change local sizes, so the maximum is returned.
     *
     * @return Buffer size in doubles (divide by 2 for complex element count).
     */
    int get_required_output_size() const
    {
      // Stage 0 padded real buffer size
      int stage0_padded_size = 1;
      for (int i = 0; i < D - 1; ++i) {
        stage0_padded_size *= stage_shapes_[0][i];
      }
      stage0_padded_size *= 2 * (global_real_shape_[D - 1] / 2 + 1); // 2 doubles per complex

      // Maximum complex size across all stages (including intermediate redistributions)
      int max_complex_size = 0;
      for (int stage = 0; stage < D; ++stage) {
        int size = 1;
        for (int i = 0; i < D; ++i) {
          size *= stage_output_shapes_[stage][i];
        }
        max_complex_size = std::max(max_complex_size, size);
      }

      // Return max of padded real size and 2× max complex size (complex = 2 doubles)
      return std::max(stage0_padded_size, 2 * max_complex_size);
    }

    /**
     * @brief Get the local real array shape.
     *
     * @param[out] shape Array of D integers to receive the local shape.
     */
    void get_local_real_shape(int shape[D]) const
    {
      for (int i = 0; i < D; ++i) {
        shape[i] = stage_shapes_[0][i];
      }
    }

    /**
     * @brief Get the starting indices of the local real array in the global array.
     *
     * @param[out] start Array of D integers to receive the starting indices.
     */
    void get_real_global_start(int start[D]) const
    {
      for (int i = 0; i < D; ++i) {
        start[i] = real_global_start_[i];
      }
    }

    /**
     * @brief Get the local complex array size (final output).
     *
     * @return Total number of complex elements in the local output array.
     */
    int get_local_complex_size() const
    {
      int size = 1;
      for (int i = 0; i < D; ++i) {
        size *= stage_shapes_[D - 1][i];
      }
      return size;
    }

    /**
     * @brief Get the local complex array shape (final output).
     *
     * @param[out] shape Array of D integers to receive the local complex shape.
     */
    void get_local_complex_shape(int shape[D]) const
    {
      for (int i = 0; i < D; ++i) {
        shape[i] = stage_shapes_[D - 1][i];
      }
    }

    /**
     * @brief Get the starting indices of the local complex array in the global array.
     *
     * @param[out] start Array of D integers to receive the starting indices.
     */
    void get_complex_global_start(int start[D]) const
    {
      for (int i = 0; i < D; ++i) {
        start[i] = complex_global_start_[i];
      }
    }

    /**
     * @brief Print shape information for debugging.
     *
     * Outputs global shapes and per-stage local shapes to stdout.
     * Only rank 0 prints to avoid duplicate output.
     */
    void print_shapes() const
    {
      if (rank_ == 0) {
        std::cout << "Global real shape: [";
        for (int i = 0; i < D; ++i) {
          std::cout << global_real_shape_[i];
          if (i < D - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "Global complex shape: [";
        for (int i = 0; i < D; ++i) {
          std::cout << global_complex_shape_[i];
          if (i < D - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        for (int stage = 0; stage < D; ++stage) {
          std::cout << "Stage " << stage << " input shape: [";
          for (int i = 0; i < D; ++i) {
            std::cout << stage_shapes_[stage][i];
            if (i < D - 1) std::cout << ", ";
          }
          std::cout << "]";
          if (stage == 0) {
            std::cout << " (real)";
          }
          std::cout << "\n";

          std::cout << "Stage " << stage << " output shape: [";
          for (int i = 0; i < D; ++i) {
            std::cout << stage_output_shapes_[stage][i];
            if (i < D - 1) std::cout << ", ";
          }
          std::cout << "] (complex)\n";
        }
      }
    }

  private:
    /**
     * @brief Copy non-padded real array to FFTW-compatible padded format.
     *
     * Converts from contiguous real storage (N elements per row) to padded
     * storage (2*(N/2+1) doubles per row) required for in-place R2C transforms.
     *
     * @param real_input Source array with contiguous real data.
     * @param[out] padded_output Destination array with padded layout.
     */
    void copy_real_to_padded(const double *real_input, double *padded_output) const
    {
      const int last_dim = global_real_shape_[D - 1];
      const int complex_last_dim = global_real_shape_[D - 1] / 2 + 1;
      const int padded_stride = 2 * complex_last_dim;

      // Compute batch size (product of all dimensions except last)
      int batch = 1;
      for (int i = 0; i < D - 1; ++i) {
        batch *= stage_shapes_[0][i];
      }

      // Copy each row, leaving padding uninitialized
      for (int b = 0; b < batch; ++b) {
        backend_.memcpy(padded_output + b * padded_stride, real_input + b * last_dim, last_dim * sizeof(double));
      }
    }

    /**
     * @brief Copy padded real array to contiguous non-padded format.
     *
     * Converts from padded storage (2*(N/2+1) doubles per row) back to
     * contiguous real storage (N elements per row).
     *
     * @param padded_input Source array with padded layout.
     * @param[out] real_output Destination array with contiguous real data.
     */
    void copy_padded_to_real(const double *padded_input, double *real_output) const
    {
      const int last_dim = global_real_shape_[D - 1];
      const int complex_last_dim = global_real_shape_[D - 1] / 2 + 1;
      const int padded_stride = 2 * complex_last_dim;

      // Compute batch size (product of all dimensions except last)
      int batch = 1;
      for (int i = 0; i < D - 1; ++i) {
        batch *= stage_shapes_[0][i];
      }

      // Copy each row, skipping padding
      for (int b = 0; b < batch; ++b) {
        backend_.memcpy(real_output + b * last_dim, padded_input + b * padded_stride, last_dim * sizeof(double));
      }
    }

    /**
     * @brief Initialize stage shapes for the pencil decomposition.
     *
     * Computes local array shapes for each stage of the FFT pipeline,
     * accounting for MPI redistribution between stages.
     */
    void setup_stage_shapes()
    {
      // Store both real and complex global shapes
      for (int i = 0; i < D; ++i) {
        global_complex_shape_[i] = global_real_shape_[i];
      }
      // Last axis is reduced in complex space: N -> N/2+1
      global_complex_shape_[D - 1] = global_real_shape_[D - 1] / 2 + 1;

      stage_shapes_.resize(D);
      stage_output_shapes_.resize(D);
      for (int stage = 0; stage < D; ++stage) {
        stage_shapes_[stage].resize(D);
        stage_output_shapes_[stage].resize(D);
      }

      // Compute local sizes for processor grid
      std::vector<int> local_n(D - 1);
      std::vector<int> local_s(D - 1);
      for (int i = 0; i < D - 1; ++i) {
        r2c_decompose(global_real_shape_[i], dims_[i], coords_[i], local_n[i], local_s[i]);
      }

      // Store real starting positions
      for (int i = 0; i < D - 1; ++i) {
        real_global_start_[i] = local_s[i];
      }
      real_global_start_[D - 1] = 0; // Last axis fully local

      // Stage 0 INPUT (REAL): axes [0..D-2] distributed, axis D-1 fully local
      for (int axis = 0; axis < D - 1; ++axis) {
        stage_shapes_[0][axis] = local_n[axis];
      }
      stage_shapes_[0][D - 1] = global_real_shape_[D - 1]; // Full real size

      // Stage 0 OUTPUT (COMPLEX): same distribution, but last axis is N/2+1
      for (int axis = 0; axis < D - 1; ++axis) {
        stage_output_shapes_[0][axis] = local_n[axis];
      }
      stage_output_shapes_[0][D - 1] = global_complex_shape_[D - 1]; // Reduced complex size

      // Stages 1 to D-1 (COMPLEX): use global_complex_shape_ for sizes
      for (int stage = 1; stage < D; ++stage) {
        int num_processed = stage;

        for (int axis = 0; axis < D; ++axis) {
          if (axis < D - 1 - num_processed) {
            // Not yet processed: original distribution
            stage_shapes_[stage][axis] = local_n[axis];
          } else if (axis == D - 1 - stage) {
            // Current FFT axis: fully local
            stage_shapes_[stage][axis] = global_complex_shape_[axis];
          } else if (axis > D - 1 - stage) {
            // Already processed: redistributed
            int num_still_original = D - 1 - num_processed;
            int grid_dim = num_still_original + (axis - (D - stage));

            if (grid_dim >= 0 && grid_dim < D - 1) {
              // Decompose using COMPLEX shape
              int n, s;
              r2c_decompose(global_complex_shape_[axis], dims_[grid_dim], coords_[grid_dim], n, s);
              stage_shapes_[stage][axis] = n;
            } else {
              stage_shapes_[stage][axis] = global_complex_shape_[axis];
            }
          }
        }
        // For stages 1+, input and output shapes are the same
        for (int axis = 0; axis < D; ++axis) {
          stage_output_shapes_[stage][axis] = stage_shapes_[stage][axis];
        }
      }

      // Compute complex output starting positions (final stage distribution)
      complex_global_start_[0] = 0; // Axis 0 not distributed in final stage
      for (int i = 1; i < D; ++i) {
        int n, s;
        r2c_decompose(global_complex_shape_[i], dims_[i - 1], coords_[i - 1], n, s);
        complex_global_start_[i] = s;
      }
    }

    /**
     * @brief Create FFT plans for all stages using the backend.
     *
     * Creates R2C/C2R plans for stage 0 and C2C plans for subsequent stages.
     */
    void create_backend_plans()
    {
      // Stage 0: R2C plan
      int batch0 = 1;
      for (int i = 0; i < D - 1; ++i) {
        batch0 *= stage_shapes_[0][i];
      }
      int real_length = global_real_shape_[D - 1];
      int complex_length = global_complex_shape_[D - 1];

      // In-place R2C and C2R plans for padded memory optimization
      // Create temporary padded buffer for planning
      Buffer temp_padded(batch0 * 2 * complex_length);
      int padded_dist = 2 * complex_length; // 2*(N/2+1) doubles per transform
      backend_.create_r2c_inplace_plan(real_length, batch0, temp_padded.data(), 1, padded_dist);
      backend_.create_c2r_inplace_plan(real_length, batch0, temp_padded.data(), 1, padded_dist);

      // Stages 1 to D-1: C2C plans (use local temp buffer for planning only)
      // Note: For stages 1+, axis ranges from D-2 down to 0, never equal to D-1
      ComplexBuffer plan_temp(scratch_b_.size()); // Temporary for planning
      for (int stage = 1; stage < D; ++stage) {
        int axis = D - 1 - stage; // axis ∈ [0, D-2] for stage ∈ [1, D-1]
        int length = global_complex_shape_[axis];

        if (axis == 0) {
          // First axis: strided FFTs
          int batch = 1;
          for (int i = 1; i < D; ++i) {
            batch *= stage_shapes_[stage][i];
          }
          int stride = batch;
          backend_.create_stage_plan(stage, length, batch, plan_temp.data(), stride, 1);
        } else {
          // Middle axes: need to handle via loops in perform_fft_on_buffer
          int trailing_size = 1;
          for (int i = axis + 1; i < D; ++i) {
            trailing_size *= stage_shapes_[stage][i];
          }
          backend_.create_stage_plan(stage, length, trailing_size, plan_temp.data(), trailing_size, 1);
        }
      }
    }

    /**
     * @brief Execute C2C FFT on a buffer for a specific stage.
     *
     * Handles the complexity of strided FFTs for different axes,
     * looping over leading dimensions when necessary.
     *
     * @param stage Stage index (1 to D-1).
     * @param axis Axis along which to perform the FFT.
     * @param direction Forward or Backward FFT direction.
     * @param buffer Buffer containing the complex data.
     */
    void perform_fft_on_buffer(int stage, int axis, FFTDirection direction, Complex *buffer)
    {
      if (axis == 0) {
        // First axis: single batched call handles all transforms
        backend_.execute_stage(stage, direction, buffer);
      } else {
        // Middle axes: loop over leading dimensions
        // Each iteration processes transforms along 'axis' for one slice of leading dims
        int leading_size = 1;
        for (int i = 0; i < axis; ++i) {
          leading_size *= stage_shapes_[stage][i];
        }

        int trailing_size = 1;
        for (int i = axis + 1; i < D; ++i) {
          trailing_size *= stage_shapes_[stage][i];
        }

        for (int i = 0; i < leading_size; ++i) {
          Complex *base_ptr = buffer + i * global_complex_shape_[axis] * trailing_size;
          backend_.execute_stage(stage, direction, base_ptr);
        }
      }
    }

    // =========================================================================
    // Member Variables
    // =========================================================================

    MPI_Comm comm_world_;            ///< World communicator
    MPI_Comm comm_cart_;             ///< Cartesian communicator for processor grid
    std::vector<MPI_Comm> subcomms_; ///< Subcommunicators for each axis redistribution

    int rank_;          ///< This processor's rank in comm_world_
    int size_;          ///< Total number of processors
    int dims_[D - 1];   ///< Processor grid dimensions
    int coords_[D - 1]; ///< This processor's coordinates in grid

    int global_real_shape_[D];    ///< Original real array shape: [N₀, ..., N_{D-1}]
    int global_complex_shape_[D]; ///< Reduced complex shape: [N₀, ..., N_{D-1}/2+1]
    int real_global_start_[D];    ///< Start indices for local real input
    int complex_global_start_[D]; ///< Start indices for local complex output

    /// Stage shapes: stage_shapes_[0] is REAL input, rest are COMPLEX
    std::vector<std::vector<int>> stage_shapes_;
    /// Stage output shapes: stage_output_shapes_[0] is COMPLEX after R2C
    std::vector<std::vector<int>> stage_output_shapes_;

    ComplexBuffer scratch_b_; ///< Ping-pong buffer for intermediate stages

    Backend backend_; ///< FFT backend (FFTW, cuFFT, etc.)
  };

} // namespace parafaft

#endif // PARAFAFT_R2C_HPP
