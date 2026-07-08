#ifndef PARAFAFT_R2C_HPP
#define PARAFAFT_R2C_HPP

// ============================================================================
// Real-to-Complex (R2C) / Complex-to-Real (C2R) Parallel FFT Implementation
// ============================================================================
//
// This implementation extends the pencil decomposition approach from
// parafaft_c2c.hpp to handle real-valued data with full roundtrip support:
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
//   fftw_complex* complex_data =
//   fftw_alloc_complex(fft.get_local_complex_size());
//   // ... use with forward()/backward() ...
//   fftw_free(real_data);
//   fftw_free(complex_data);
//
// Standard std::vector or malloc'd buffers will work correctly but may have
// slightly reduced performance on some architectures if alignment differs
// from the internal planning buffers.
//
// ============================================================================

#include "./parafaft_common.hpp"
#include <algorithm>
#include <complex>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace parafaft {

// ============================================================================
// ParaFaFT_R2C Class
// ============================================================================

/**
 * @brief Parallel Real-to-Complex (R2C) and Complex-to-Real (C2R) FFT using
 * pencil decomposition.
 *
 * This class implements distributed-memory parallel FFT for real-valued data
 * using MPI pencil decomposition. It supports arbitrary dimensions D and
 * provides both out-of-place and in-place transform interfaces.
 *
 * Key features:
 * - Forward R2C: real input → complex output with reduced last dimension
 * (N/2+1)
 * - Backward C2R: complex input → real output (inverse of forward)
 * - In-place variants that operate on a single buffer
 * - Ping-pong buffering with minimal memory overhead
 *
 * @tparam D Number of dimensions (must be >= 2)
 * @tparam Backend FFT backend type (default: FFTWBackend<>). The scalar
 *         precision is selected via the backend (e.g. `FFTWBackend<float>`).
 */
template <int D, typename Backend = FFTWBackend<>>
class ParaFaFT_R2C {
public:
  using FloatType = typename Backend::FloatType;
  using Complex = typename Backend::Complex;
  using Buffer = typename Backend::Buffer;
  using ComplexBuffer = typename Backend::ComplexBuffer;

  /**
   * @brief Construct a parallel R2C FFT object.
   *
   * Sets up the processor grid, pencil decomposition, and creates FFT plans
   * for all stages of the transform.
   *
   * @param global_shape Array of D integers specifying the global real array
   * dimensions
   * @param comm MPI communicator (default: MPI_COMM_WORLD)
   * @param plan_flag FFTW planning strategy (default: Estimate). Use Measure or
   *                  Patient for faster FFT execution at the cost of longer
   *                  initialization. Only affects the FFTW backend.
   */
  ParaFaFT_R2C(const int global_shape[D], MPI_Comm comm = MPI_COMM_WORLD,
               FFTPlanFlag plan_flag = FFTPlanFlag::Estimate)
      : comm_world_(comm), backend_(D, comm, plan_flag) {
    MPI_Comm_rank(comm_world_, &rank_);
    MPI_Comm_size(comm_world_, &size_);

    // If D=1, we cannot distribute across multiple processors. Handle this case
    // separately.
    if (D < 2) {
      if (size_ > 1) {
        if (rank_ == 0) {
          std::cerr << "ParaFaFT Error: D=1 case cannot be run with multiple "
                       "processes."
                    << std::endl;
        }
        throw std::invalid_argument(
            "ParaFaFT D must be >= 2 for distributed FFT.");
      }
    }

    // Store global real array dimensions
    for (int i = 0; i < D; ++i) {
      global_real_shape_[i] = global_shape[i];
    }

    if constexpr (Backend::handles_distributed) {
      // Backend handles MPI decomposition, communication, and FFT internally
      backend_.setup_distributed(global_shape, comm);
      auto info = backend_.get_distributed_info();
      // Store shapes for public API queries. get_required_output_size() loops
      // from 0..D over stage_output_shapes_, so allocate D stages and
      // replicate the backend-reported shape into each slot.
      stage_shapes_.resize(D);
      stage_output_shapes_.resize(D);
      for (int s = 0; s < D; ++s) {
        stage_shapes_[s].resize(D);
        stage_output_shapes_[s].resize(D);
        for (int i = 0; i < D; ++i) {
          stage_shapes_[s][i] = info.local_shape[i];
          stage_output_shapes_[s][i] = info.output_shape[i];
        }
      }
      for (int i = 0; i < D; ++i) {
        real_global_start_[i] = info.global_start[i];
        complex_global_start_[i] = info.output_start[i];
      }
      // Store complex global shape for R2C (last axis N/2+1)
      for (int i = 0; i < D; ++i) {
        global_complex_shape_[i] = global_real_shape_[i];
      }
      global_complex_shape_[D - 1] = global_real_shape_[D - 1] / 2 + 1;
      // required_size is stored via stage_output_shapes_ for
      // get_required_output_size()
    } else {
      // Standard ParaFaFT: manual pencil decomposition + MPI exchange

      // Create (D-1)-dimensional Cartesian processor grid
      int dims[D - 1];
      for (int i = 0; i < D - 1; ++i)
        dims[i] = 0;
      MPI_Dims_create(size_, D - 1, dims);

      for (int i = 0; i < D - 1; ++i) {
        dims_[i] = dims[i];
      }

      // Create Cartesian communicator (non-periodic)
      int periods[D - 1];
      for (int i = 0; i < D - 1; ++i)
        periods[i] = 0;
      MPI_Cart_create(comm_world_, D - 1, dims, periods, 1, &comm_cart_);

      // Get this processor's coordinates in the grid
      MPI_Cart_coords(comm_cart_, rank_, D - 1, coords_);

      // Create (D-1) subcommunicators
      for (int i = 0; i < D - 1; ++i) {
        int remain_dims[D - 1];
        for (int j = 0; j < D - 1; ++j)
          remain_dims[j] = 0;
        remain_dims[i] = 1;
        MPI_Cart_sub(comm_cart_, remain_dims, &subcomms_[i]);
      }

      // Compute local array shapes for each stage
      setup_stage_shapes();

      // Cache MPI subarray datatypes for all stage transitions.
      // For each of (D-1) transitions between stage t and stage t+1:
      //   fwd_send_types_[t]: send types (subarray of stage_output_shapes_[t]
      //   along axis D-1-t) fwd_recv_types_[t]: recv types (subarray of
      //   stage_shapes_[t+1] along axis D-2-t)
      // Backward uses these in reverse: send=fwd_recv, recv=fwd_send.
      cache_exchange_types();

      // Compute maximum complex buffer size across all stages
      std::size_t max_complex_size = 0;
      for (int stage = 0; stage < D; ++stage) {
        std::size_t size = 1;
        for (int i = 0; i < D; ++i) {
          size *= static_cast<std::size_t>(stage_output_shapes_[stage][i]);
        }
        max_complex_size = std::max(max_complex_size, size);
      }

      // Allocate ping-pong buffer sized to handle any stage
      scratch_b_.resize(max_complex_size);

      // Allocate pack buffer for manual packing exchange (GPU backends)
      if constexpr (!Backend::use_alltoallw) {
        pack_buffer_.resize(max_complex_size);
      }

      // Create FFT plans
      create_backend_plans();

      // Set up P2P IPC for same-node GPU exchanges
      setup_p2p();
    }
  }

  /**
   * @brief Destructor. Frees MPI communicators and cached datatypes if MPI is
   * not finalized.
   */
  ~ParaFaFT_R2C() {
    // Close IPC handles before pack_buffer_ is freed
    if constexpr (Backend::use_p2p) {
      for (auto &info : p2p_info_) {
        if (!info.any_enabled)
          continue;
        for (int p = 0; p < static_cast<int>(info.remote_pack_ptrs.size());
             ++p) {
          if (info.peer_enabled[p] && info.remote_pack_ptrs[p] &&
              info.remote_pack_ptrs[p] != pack_buffer_.data())
            Backend::ipc_close_handle(info.remote_pack_ptrs[p]);
        }
      }
    }

    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      if constexpr (!Backend::handles_distributed) {
        // Free cached MPI subarray datatypes (only created when using
        // alltoallw)
        if constexpr (Backend::use_alltoallw) {
          for (auto &types : fwd_send_types_) {
            for (auto &t : types) {
              MPI_Type_free(&t);
            }
          }
          for (auto &types : fwd_recv_types_) {
            for (auto &t : types) {
              MPI_Type_free(&t);
            }
          }
        }

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
   *                       Final result uses only get_local_complex_size()
   * elements.
   *
   * @note The output buffer is used as working space during the transform.
   *       Intermediate stages may require more space than the final output
   *       due to MPI redistribution changing local array sizes.
   */
  void forward(const FloatType *real_input, Complex *complex_output) {
    // Pad the data and move it into the destination buffer
    copy_real_to_padded(real_input, reinterpret_cast<FloatType *>(complex_output));

    // In-place forward transform
    forward_in_place(reinterpret_cast<FloatType *>(complex_output));
  }

  /**
   * @brief Forward R2C transform (fully in-place).
   *
   * Computes the forward Real-to-Complex FFT entirely in-place on a single
   * buffer. The buffer is reinterpreted between real and complex views during
   * computation.
   *
   * @param padded_buffer In-place real/complex buffer.
   *                      Size: get_required_output_size() doubles.
   *                      Input: Padded real data with 2*(N/2+1) doubles per
   * row, where the first N elements contain the real values. Output: Complex
   * data accessible via reinterpret_cast<Complex*>(buffer).
   *
   * @note After completion, access the complex output via:
   *       `reinterpret_cast<Complex*>(buffer)`
   *       The complex array has get_local_complex_size() valid elements.
   *
   * @warning Input real data is destroyed during computation!
   */
  void forward_in_place(FloatType *padded_buffer) {
    if constexpr (Backend::handles_distributed) {
      backend_.forward_r2c();
      return;
    }
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
      std::size_t stage0_complex_size = 1;
      for (int i = 0; i < D; ++i) {
        stage0_complex_size *= static_cast<std::size_t>(stage_output_shapes_[0][i]);
      }
      backend_.memcpy(scratch_b_.data(), padded_as_complex,
                      stage0_complex_size * sizeof(Complex));
      src = scratch_b_.data();
      dst = padded_as_complex;
    }

    // Stages 1 to D-1: MPI redistribute + C2C FFT
    for (int stage = 1; stage < D; ++stage) {
      int axis = D - 1 - stage;

      // MPI exchange: src → dst
      int trans = stage - 1; // transition index
      if constexpr (Backend::use_alltoallw) {
        exchange(subcomms_[axis], nparts_[trans], src,
                 fwd_send_types_[trans].data(), dst,
                 fwd_recv_types_[trans].data(), exchange_counts_.data(),
                 exchange_displs_.data());
      } else if (nparts_[trans] == 1) {
        // Single-rank subcommunicator: local reshape, no MPI needed
        exchange_local<Backend>(backend_, src, dst, pack_buffer_.data(),
                                fwd_exchange_geom_[trans]);
      } else if (p2p_info_[trans].any_enabled) {
        exchange_hybrid<Backend>(
            backend_, subcomms_[axis], nparts_[trans], src, dst,
            pack_buffer_.data(), p2p_info_[trans].remote_pack_ptrs.data(),
            p2p_info_[trans].peer_enabled.data(), fwd_exchange_geom_[trans]);
      } else {
        exchange_packed<Backend>(backend_, subcomms_[axis], nparts_[trans], src,
                                 dst, pack_buffer_.data(),
                                 fwd_exchange_geom_[trans]);
      }

      // C2C FFT in-place on dst
      perform_fft_on_buffer(stage, axis, FFTDirection::Forward, dst);

      // Swap for next iteration
      std::swap(src, dst);
    }

    // After D-1 swaps, src points to padded_as_complex
    // Result is already in place - no copy needed
    backend_.sync();
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
  void backward(Complex *complex_input, FloatType *real_output) {
    // In-place backward transform
    backward_in_place(reinterpret_cast<FloatType *>(complex_input));

    // Copy real output from padded buffer
    copy_padded_to_real(reinterpret_cast<FloatType *>(complex_input), real_output);
    backend_.sync();
  }

  /**
   * @brief Backward C2R transform (fully in-place).
   *
   * Computes the backward Complex-to-Real FFT entirely in-place on a single
   * buffer. The buffer is reinterpreted between complex and real views during
   * computation.
   *
   * @param padded_buffer In-place complex/real buffer.
   *                      Size: get_required_output_size() doubles.
   *                      Input: Complex data (from forward_in_place or user).
   *                      Output: Real data in first N elements per row.
   *
   * @note Input complex data is interpreted as
   * `reinterpret_cast<Complex*>(buffer)`. After completion, the buffer contains
   * real data.
   *
   * @warning Input complex data is destroyed during computation!
   */
  void backward_in_place(FloatType *padded_buffer) {
    if constexpr (Backend::handles_distributed) {
      backend_.backward_c2r();
      return;
    }
    // Reinterpret padded real buffer as complex for ping-pong
    Complex *padded_as_complex = reinterpret_cast<Complex *>(padded_buffer);

    // Determine starting buffer based on parity
    // We need src to point to padded_as_complex after D-1 swaps for in-place
    // C2R
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
      std::size_t input_size = 1;
      for (int i = 0; i < D; ++i) {
        input_size *= static_cast<std::size_t>(stage_shapes_[D - 1][i]);
      }
      backend_.memcpy(scratch_b_.data(), padded_as_complex,
                      input_size * sizeof(Complex));
      src = scratch_b_.data();
      dst = padded_as_complex;
    }

    // Stages D-1 down to 1: C2C backward FFT + MPI redistribute
    for (int stage = D - 1; stage >= 1; --stage) {
      int axis = D - 1 - stage;

      // C2C backward FFT in-place on src
      perform_fft_on_buffer(stage, axis, FFTDirection::Backward, src);

      // MPI exchange: src → dst (backward uses swapped send/recv)
      int trans = stage - 1; // transition index
      if constexpr (Backend::use_alltoallw) {
        exchange(subcomms_[axis], nparts_[trans], src,
                 fwd_recv_types_[trans].data(), dst,
                 fwd_send_types_[trans].data(), exchange_counts_.data(),
                 exchange_displs_.data());
      } else if (nparts_[trans] == 1) {
        // Single-rank subcommunicator: local reshape, no MPI needed
        exchange_local<Backend>(backend_, src, dst, pack_buffer_.data(),
                                bwd_exchange_geom_[trans]);
      } else if (p2p_info_[trans].any_enabled) {
        exchange_hybrid<Backend>(
            backend_, subcomms_[axis], nparts_[trans], src, dst,
            pack_buffer_.data(), p2p_info_[trans].remote_pack_ptrs.data(),
            p2p_info_[trans].peer_enabled.data(), bwd_exchange_geom_[trans]);
      } else {
        exchange_packed<Backend>(backend_, subcomms_[axis], nparts_[trans], src,
                                 dst, pack_buffer_.data(),
                                 bwd_exchange_geom_[trans]);
      }

      // Swap for next iteration
      std::swap(src, dst);
    }

    // After D-1 swaps, src points to padded_as_complex
    // Stage 0: In-place C2R FFT on padded buffer
    backend_.execute_c2r_inplace(padded_buffer);
    backend_.sync();
  }

  /**
   * @brief Get the local real array size. This is the logical size without any
   * padding, i.e. only appropriate for out-of-place transformations. For
   * in-place transformations, use get_required_output_size() to determine the
   * necessary buffer size.
   *
   * @return Total number of real elements in the local input array.
   */
  std::size_t get_local_real_size() const {
    std::size_t size = 1;
    for (int i = 0; i < D; ++i) {
      size *= static_cast<std::size_t>(stage_shapes_[0][i]);
    }
    return size;
  }

  /**
   * @brief Get the output memory size (in doubles) required for in-place
   * transforms.
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
  std::size_t get_required_output_size() const {
    // Stage 0 padded real buffer size
    std::size_t stage0_padded_size = 1;
    for (int i = 0; i < D - 1; ++i) {
      stage0_padded_size *= static_cast<std::size_t>(stage_shapes_[0][i]);
    }
    stage0_padded_size *= static_cast<std::size_t>(
        2 * (global_real_shape_[D - 1] / 2 + 1)); // 2 doubles per complex

    // Maximum complex size across all stages (including intermediate
    // redistributions)
    std::size_t max_complex_size = 0;
    for (int stage = 0; stage < D; ++stage) {
      std::size_t size = 1;
      for (int i = 0; i < D; ++i) {
        size *= static_cast<std::size_t>(stage_output_shapes_[stage][i]);
      }
      max_complex_size = std::max(max_complex_size, size);
    }

    // Return max of padded real size and 2× max complex size (complex = 2
    // doubles)
    return std::max(stage0_padded_size, 2 * max_complex_size);
  }

  /**
   * @brief Get the local real array shape, i.e. the logical sizes without any
   * padding!
   *
   * @param[out] shape Array of D integers to receive the local shape.
   */
  void get_local_real_shape(int shape[D]) const {
    for (int i = 0; i < D; ++i) {
      shape[i] = stage_shapes_[0][i];
    }
  }

  /**
   * @brief Get the starting indices of the local real array in the global
   * array.
   *
   * @param[out] start Array of D integers to receive the starting indices.
   */
  void get_real_global_start(int start[D]) const {
    for (int i = 0; i < D; ++i) {
      start[i] = real_global_start_[i];
    }
  }

  /**
   * @brief Get the local complex array size (final output).
   *
   * @return Total number of complex elements in the local output array.
   */
  std::size_t get_local_complex_size() const {
    std::size_t size = 1;
    for (int i = 0; i < D; ++i) {
      size *= static_cast<std::size_t>(stage_shapes_[D - 1][i]);
    }
    return size;
  }

  /**
   * @brief Get the local complex array shape (final output).
   *
   * @param[out] shape Array of D integers to receive the local complex shape.
   */
  void get_local_complex_shape(int shape[D]) const {
    for (int i = 0; i < D; ++i) {
      shape[i] = stage_shapes_[D - 1][i];
    }
  }

  /**
   * @brief Get the starting indices of the local complex array in the global
   * array.
   *
   * @param[out] start Array of D integers to receive the starting indices.
   */
  void get_complex_global_start(int start[D]) const {
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
  void print_shapes() const {
    if (rank_ == 0) {
      std::cout << "Global real shape: [";
      for (int i = 0; i < D; ++i) {
        std::cout << global_real_shape_[i];
        if (i < D - 1)
          std::cout << ", ";
      }
      std::cout << "]\n";

      std::cout << "Global complex shape: [";
      for (int i = 0; i < D; ++i) {
        std::cout << global_complex_shape_[i];
        if (i < D - 1)
          std::cout << ", ";
      }
      std::cout << "]\n";

      for (int stage = 0; stage < D; ++stage) {
        std::cout << "Stage " << stage << " input shape: [";
        for (int i = 0; i < D; ++i) {
          std::cout << stage_shapes_[stage][i];
          if (i < D - 1)
            std::cout << ", ";
        }
        std::cout << "]";
        if (stage == 0) {
          std::cout << " (real)";
        }
        std::cout << "\n";

        std::cout << "Stage " << stage << " output shape: [";
        for (int i = 0; i < D; ++i) {
          std::cout << stage_output_shapes_[stage][i];
          if (i < D - 1)
            std::cout << ", ";
        }
        std::cout << "] (complex)\n";
      }
    }
  }

  /**
   * @brief Get the domain decomposition used by this FFT object.
   *
   * @param[out] decomposition Array of D integers to receive the domain
   * decomposition.
   */
  void get_domain_decomposition(int decomposition[D]) const {
    for (int i = 0; i < D - 1; ++i) {
      decomposition[i] = dims_[i];
    }
    decomposition[D - 1] = 1; // Last dimension not distributed
  }

  /**
   * @brief Get the backend-managed buffer for distributed transforms.
   *
   * When the backend handles distributed transforms (e.g., cuFFTMp), the
   * buffer is allocated using optimized memory (e.g., NVSHMEM). Users should
   * write padded real data into the FloatType* buffer before calling
   * forward_in_place(), and read results after backward_in_place().
   *
   * @return Device pointer to the internal buffer as FloatType*, or nullptr
   *         if the backend does not manage its own buffer.
   */
  FloatType *get_real_buffer() {
    if constexpr (Backend::handles_distributed) {
      return backend_.get_real_buffer();
    } else {
      return nullptr;
    }
  }

  /**
   * @brief Get the backend-managed buffer as complex pointer.
   *
   * @return Device pointer to the internal buffer as Complex*, or nullptr.
   */
  Complex *get_buffer() {
    if constexpr (Backend::handles_distributed) {
      return backend_.get_buffer();
    } else {
      return nullptr;
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
  void copy_real_to_padded(const FloatType *real_input,
                           FloatType *padded_output) const {
    const int last_dim = global_real_shape_[D - 1];
    const int complex_last_dim = global_real_shape_[D - 1] / 2 + 1;
    const int padded_stride = 2 * complex_last_dim;

    // Compute batch size (product of all dimensions except last)
    std::size_t batch = 1;
    for (int i = 0; i < D - 1; ++i) {
      batch *= static_cast<std::size_t>(stage_shapes_[0][i]);
    }

    // Single 2D copy: contiguous real rows → padded rows
    backend_.memcpy2d(padded_output, padded_stride * sizeof(FloatType), real_input,
                      last_dim * sizeof(FloatType), last_dim * sizeof(FloatType),
                      batch);
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
  void copy_padded_to_real(const FloatType *padded_input,
                           FloatType *real_output) const {
    const int last_dim = global_real_shape_[D - 1];
    const int complex_last_dim = global_real_shape_[D - 1] / 2 + 1;
    const int padded_stride = 2 * complex_last_dim;

    // Compute batch size (product of all dimensions except last)
    std::size_t batch = 1;
    for (int i = 0; i < D - 1; ++i) {
      batch *= static_cast<std::size_t>(stage_shapes_[0][i]);
    }

    // Single 2D copy: padded rows → contiguous real rows
    backend_.memcpy2d(real_output, last_dim * sizeof(FloatType), padded_input,
                      padded_stride * sizeof(FloatType), last_dim * sizeof(FloatType),
                      batch);
  }

  /**
   * @brief Cache MPI subarray datatypes for all stage transitions.
   *
   * Pre-computes send/receive MPI subarray datatypes for each of the (D-1)
   * redistributions. These types are reused on every forward/backward call,
   * avoiding repeated MPI_Type_create_subarray + MPI_Type_commit +
   * MPI_Type_free.
   *
   * For transition t (between stage t and stage t+1):
   *   Forward: send from stage_output_shapes_[t] along axis D-1-t,
   *            recv into stage_shapes_[t+1] along axis D-2-t
   *   Backward uses the same types swapped (send=fwd_recv, recv=fwd_send)
   */
  void cache_exchange_types() {
    fwd_send_types_.resize(D - 1);
    fwd_recv_types_.resize(D - 1);
    nparts_.resize(D - 1);

    // Find maximum nparts across all subcommunicators for counts/displs arrays
    int max_nparts = 0;
    for (int t = 0; t < D - 1; ++t) {
      int comm_idx = D - 2 - t; // subcomms_[axis] where axis = D-2-t
      MPI_Comm_size(subcomms_[comm_idx], &nparts_[t]);
      max_nparts = std::max(max_nparts, nparts_[t]);
    }
    if constexpr (Backend::use_alltoallw) {
      exchange_counts_.assign(max_nparts, 1);
      exchange_displs_.assign(max_nparts, 0);

      for (int t = 0; t < D - 1; ++t) {
        int send_axis =
            D - 1 - t; // axis distributed in stage_output_shapes_[t]
        int recv_axis = D - 2 - t; // axis distributed in stage_shapes_[t+1]

        fwd_send_types_[t].resize(nparts_[t]);
        fwd_recv_types_[t].resize(nparts_[t]);

        const MPI_Datatype mpi_complex = mpi_complex_type<FloatType>();
        subarray(mpi_complex, D, stage_output_shapes_[t].data(),
                 send_axis, nparts_[t], fwd_send_types_[t].data());
        subarray(mpi_complex, D, stage_shapes_[t + 1].data(),
                 recv_axis, nparts_[t], fwd_recv_types_[t].data());
      }
    }

    // Pre-compute exchange geometry for packed/hybrid paths (GPU backends)
    if constexpr (!Backend::use_alltoallw) {
      fwd_exchange_geom_.resize(D - 1);
      bwd_exchange_geom_.resize(D - 1);
      for (int t = 0; t < D - 1; ++t) {
        int send_axis = D - 1 - t;
        int recv_axis = D - 2 - t;
        // R2C forward: src = stage_output_shapes_[t], dst = stage_shapes_[t+1]
        init_exchange_geometry(fwd_exchange_geom_[t], nparts_[t], D,
                               stage_output_shapes_[t].data(), send_axis,
                               stage_shapes_[t + 1].data(), recv_axis);
        // Backward is forward with src/dst swapped
        init_exchange_geometry(bwd_exchange_geom_[t], nparts_[t], D,
                               stage_shapes_[t + 1].data(), recv_axis,
                               stage_output_shapes_[t].data(), send_axis);
        // Exchange per-peer send displacements for the P2P/IPC read path.
        // Same subcomm the fwd/bwd exchange of transition t runs over.
        MPI_Comm geom_comm = subcomms_[D - 2 - t];
        init_remote_send_displs(fwd_exchange_geom_[t], geom_comm);
        init_remote_send_displs(bwd_exchange_geom_[t], geom_comm);
      }
    }
  }

  /**
   * @brief Set up CUDA/HIP IPC for P2P GPU-to-GPU exchange.
   */
  void setup_p2p() {
    if constexpr (Backend::use_p2p) {
      p2p_info_.resize(D - 1);
      int my_device = Backend::device_id();

      for (int t = 0; t < D - 1; ++t) {
        int nparts = nparts_[t];
        if (nparts <= 1)
          continue;

        MPI_Comm comm = subcomms_[D - 2 - t];

        std::vector<int> devices(nparts);
        MPI_Allgather(&my_device, 1, MPI_INT, devices.data(), 1, MPI_INT, comm);

        // Get this rank's subrank within the subcommunicator
        int my_subrank;
        MPI_Comm_rank(comm, &my_subrank);

        // Classify subcomm ranks by physical node. CUDA IPC handles are
        // strictly intra-node, so inter-node peers must fall through to the
        // non-P2P MPI path regardless of device ordinal or peer-access query.
        MPI_Comm shared_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                            &shared_comm);
        int node_leader = my_subrank;
        MPI_Bcast(&node_leader, 1, MPI_INT, 0, shared_comm);
        std::vector<int> node_id(nparts);
        MPI_Allgather(&node_leader, 1, MPI_INT, node_id.data(), 1, MPI_INT,
                      comm);
        MPI_Comm_free(&shared_comm);

        // Check P2P capability per neighbour
        p2p_info_[t].peer_enabled.resize(nparts, false);
        for (int p = 0; p < nparts; ++p) {
          if (p == my_subrank) {
            p2p_info_[t].peer_enabled[p] = true; // self is always "p2p"
          } else if (node_id[p] == node_id[my_subrank] &&
                     (devices[p] == my_device ||
                      Backend::can_access_peer(my_device, devices[p]))) {
            p2p_info_[t].peer_enabled[p] = true;
          }
        }

        // Check if any remote neighbour supports P2P
        bool any_p2p = false;
        for (int p = 0; p < nparts; ++p) {
          if (p != my_subrank && p2p_info_[t].peer_enabled[p]) {
            any_p2p = true;
            break;
          }
        }
        if (!any_p2p)
          continue;

        // Enable peer access only for capable neighbours
        for (int p = 0; p < nparts; ++p) {
          if (p2p_info_[t].peer_enabled[p] && devices[p] != my_device)
            Backend::enable_peer_access(devices[p]);
        }

        // Exchange IPC handles for pack_buffer_ (collective — all must call)
        using Handle = std::array<char, Backend::ipc_handle_size>;
        Handle my_handle{};
        Backend::ipc_get_handle(pack_buffer_.data(), my_handle.data());
        std::vector<Handle> handles(nparts);
        MPI_Allgather(my_handle.data(), sizeof(Handle), MPI_BYTE,
                      handles.data(), sizeof(Handle), MPI_BYTE, comm);

        // Open remote handles only for P2P-capable neighbours
        p2p_info_[t].any_enabled = true;
        p2p_info_[t].remote_pack_ptrs.resize(nparts, nullptr);
        for (int p = 0; p < nparts; ++p) {
          if (!p2p_info_[t].peer_enabled[p])
            continue;
          if (p == my_subrank) {
            p2p_info_[t].remote_pack_ptrs[p] = pack_buffer_.data();
          } else {
            p2p_info_[t].remote_pack_ptrs[p] =
                Backend::ipc_open_handle(handles[p].data());
          }
        }
      }
    }
  }

  /**
   * @brief Initialize stage shapes for the pencil decomposition.
   *
   * Computes local array shapes for each stage of the FFT pipeline,
   * accounting for MPI redistribution between stages.
   */
  void setup_stage_shapes() {
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
    int local_n[D - 1];
    int local_s[D - 1];
    for (int i = 0; i < D - 1; ++i) {
      decompose(global_real_shape_[i], dims_[i], coords_[i], local_n[i],
                local_s[i]);
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
    stage_output_shapes_[0][D - 1] =
        global_complex_shape_[D - 1]; // Reduced complex size

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
            decompose(global_complex_shape_[axis], dims_[grid_dim],
                      coords_[grid_dim], n, s);
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
      decompose(global_complex_shape_[i], dims_[i - 1], coords_[i - 1], n, s);
      complex_global_start_[i] = s;
    }
  }

  /**
   * @brief Create FFT plans for all stages using the backend.
   *
   * Creates R2C/C2R plans for stage 0 and C2C plans for subsequent stages.
   */
  void create_backend_plans() {
    // Stage 0: R2C plan
    std::size_t batch0 = 1;
    for (int i = 0; i < D - 1; ++i) {
      batch0 *= static_cast<std::size_t>(stage_shapes_[0][i]);
    }
    int real_length = global_real_shape_[D - 1];
    int complex_length = global_complex_shape_[D - 1];

    // In-place R2C and C2R plans for padded memory optimization.
    // Reuse scratch_b_ (already allocated, large enough for all stages) as the
    // planning buffer. FFTW's new-array execute API allows plans created with
    // one buffer to be executed on any buffer with compatible alignment/layout.
    // Note: FFTW_MEASURE may write to the buffer during planning, but
    // scratch_b_ contains no useful data at construction time.
    std::ptrdiff_t padded_dist = 2 * complex_length; // 2*(N/2+1) scalars per transform
    FloatType *plan_buf = reinterpret_cast<FloatType *>(scratch_b_.data());
    backend_.create_r2c_inplace_plan(real_length, batch0, plan_buf, 1,
                                     padded_dist);
    backend_.create_c2r_inplace_plan(real_length, batch0, plan_buf, 1,
                                     padded_dist);

    // Stages 1 to D-1: C2C plans (reuse scratch_b_ as planning buffer)
    // Note: For stages 1+, axis ranges from D-2 down to 0, never equal to D-1
    for (int stage = 1; stage < D; ++stage) {
      int axis = D - 1 - stage; // axis ∈ [0, D-2] for stage ∈ [1, D-1]
      int length = global_complex_shape_[axis];

      if (axis == 0) {
        // First axis: strided FFTs
        std::size_t batch = 1;
        for (int i = 1; i < D; ++i) {
          batch *= static_cast<std::size_t>(stage_shapes_[stage][i]);
        }
        std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(batch);
        backend_.create_stage_plan(stage, length, batch, scratch_b_.data(),
                                   stride, 1);
      } else {
        // Middle axes: need to handle via loops in perform_fft_on_buffer
        std::size_t trailing_size = 1;
        for (int i = axis + 1; i < D; ++i) {
          trailing_size *= static_cast<std::size_t>(stage_shapes_[stage][i]);
        }
        std::ptrdiff_t trailing_dist = static_cast<std::ptrdiff_t>(trailing_size);
        backend_.create_stage_plan(stage, length, trailing_size,
                                   scratch_b_.data(), trailing_dist, 1);
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
  void perform_fft_on_buffer(int stage, int axis, FFTDirection direction,
                             Complex *buffer) {
    if (axis == 0) {
      // First axis: single batched call handles all transforms
      backend_.execute_stage(stage, direction, buffer);
    } else {
      // Middle axes: loop over leading dimensions
      // Each iteration processes transforms along 'axis' for one slice of
      // leading dims
      std::size_t leading_size = 1;
      for (int i = 0; i < axis; ++i) {
        leading_size *= static_cast<std::size_t>(stage_shapes_[stage][i]);
      }

      std::size_t trailing_size = 1;
      for (int i = axis + 1; i < D; ++i) {
        trailing_size *= static_cast<std::size_t>(stage_shapes_[stage][i]);
      }

      const std::size_t slice_stride =
          static_cast<std::size_t>(global_complex_shape_[axis]) * trailing_size;
      for (std::size_t i = 0; i < leading_size; ++i) {
        Complex *base_ptr = buffer + i * slice_stride;
        backend_.execute_stage(stage, direction, base_ptr);
      }
    }
  }

  // =========================================================================
  // Member Variables
  // =========================================================================

  MPI_Comm comm_world_;      ///< World communicator
  MPI_Comm comm_cart_;       ///< Cartesian communicator for processor grid
  MPI_Comm subcomms_[D - 1]; ///< Subcommunicators for each axis redistribution

  int rank_;          ///< This processor's rank in comm_world_
  int size_;          ///< Total number of processors
  int dims_[D - 1];   ///< Processor grid dimensions
  int coords_[D - 1]; ///< This processor's coordinates in grid

  int global_real_shape_[D]; ///< Original real array shape: [N₀, ..., N_{D-1}]
  int global_complex_shape_[D]; ///< Reduced complex shape: [N₀, ...,
                                ///< N_{D-1}/2+1]
  int real_global_start_[D];    ///< Start indices for local real input
  int complex_global_start_[D]; ///< Start indices for local complex output

  /// Stage shapes: stage_shapes_[0] is REAL input, rest are COMPLEX
  std::vector<std::vector<int>> stage_shapes_;
  /// Stage output shapes: stage_output_shapes_[0] is COMPLEX after R2C
  std::vector<std::vector<int>> stage_output_shapes_;

  /// Cached MPI subarray datatypes for forward send (D-1 transitions)
  std::vector<std::vector<MPI_Datatype>> fwd_send_types_;
  /// Cached MPI subarray datatypes for forward recv (D-1 transitions)
  std::vector<std::vector<MPI_Datatype>> fwd_recv_types_;
  /// Number of partitions per subcommunicator for each transition
  std::vector<int> nparts_;
  /// Pre-allocated counts array for MPI_Alltoallw (all 1s)
  std::vector<int> exchange_counts_;
  /// Pre-allocated displacements array for MPI_Alltoallw (all 0s)
  std::vector<int> exchange_displs_;

  ComplexBuffer scratch_b_;   ///< Ping-pong buffer for intermediate stages
  ComplexBuffer pack_buffer_; ///< Pack buffer for contiguous MPI exchange (GPU
                              ///< backends only)

  /// Pre-computed exchange geometry for packed/hybrid paths (GPU backends)
  std::vector<ExchangeGeometry> fwd_exchange_geom_;
  std::vector<ExchangeGeometry> bwd_exchange_geom_;

  /// Per-subcommunicator P2P exchange info (GPU backends only)
  struct P2PInfo {
    bool any_enabled = false;       ///< True if at least one neighbour uses P2P
    std::vector<char> peer_enabled; ///< Per-neighbour: true = P2P, false = MPI
    std::vector<void *> remote_pack_ptrs;
  };
  std::vector<P2PInfo> p2p_info_;

  Backend backend_; ///< FFT backend (FFTW, cuFFT, etc.)
};

} // namespace parafaft

#endif // PARAFAFT_R2C_HPP
