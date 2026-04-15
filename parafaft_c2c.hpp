#ifndef PARAFAFT_C2C_HPP
#define PARAFAFT_C2C_HPP

// ============================================================================
// Generic Dimension-Agnostic Parallel FFT Implementation
// ============================================================================
//
// Reference: Dalcín, L., Mortensen, M., & Keyes, D. E. (2019).
//           Fast parallel multidimensional FFT using advanced MPI.
//           Journal of Parallel and Distributed Computing, 128, 137-150.
//
// This implementation extends the algorithm from the paper to support
// arbitrary dimensions D using C++ templates.
//
// Key Innovation (Section 1):
//   Uses MPI derived datatypes (MPI_Type_create_subarray + MPI_Alltoallw)
//   to perform global redistribution WITHOUT local transpose operations.
//   This eliminates costly data rearrangements compared to traditional methods.
//
// ============================================================================

#include "./parafaft_common.hpp"
#include <algorithm>
#include <array>
#include <complex>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace parafaft {

// ============================================================================
// ParaFaFT_C2C Class
// ============================================================================

/**
 * @brief Generic D-dimensional parallel FFT using pencil decomposition.
 *
 * This class implements distributed-memory parallel FFT for complex-valued data
 * using MPI pencil decomposition. It supports arbitrary dimensions D and
 * provides both forward and backward transforms.
 *
 * Reference: Dalcín, L., Mortensen, M., & Keyes, D. E. (2019).
 *            Fast parallel multidimensional FFT using advanced MPI.
 *            Journal of Parallel and Distributed Computing, 128, 137-150.
 *
 * Pencil Decomposition Strategy (Paper Section 3):
 *
 * For D dimensions, we use a (D-1)-dimensional processor grid.
 * Example for 3D with 2D processor grid [P0, P1]:
 *
 *   Stage A: axes [0,1] distributed, axis 2 local → FFT along axis 2
 *   Stage B: axes [0,2] distributed, axis 1 local → FFT along axis 1
 *   Stage C: axes [1,2] distributed, axis 0 local → FFT along axis 0
 *
 * Total: D FFT operations + (D-1) global redistributions
 *
 * Memory Layout (Paper Section 2.1):
 *   Data is stored in C/row-major order (last index varies fastest).
 *   Each processor holds a contiguous block of the D-dimensional array.
 *
 * @tparam D Number of dimensions (must be >= 2)
 * @tparam Backend FFT backend type (default: FFTWBackend)
 */
template <int D, typename Backend = FFTWBackend> class ParaFaFT_C2C {
public:
  using Complex = typename Backend::Complex;
  using Buffer = typename Backend::Buffer;
  using ComplexBuffer = typename Backend::ComplexBuffer;

  /**
   * @brief Construct a parallel FFT object.
   *
   * Sets up the processor grid, pencil decomposition, allocates working
   * buffers, and creates FFT plans for all stages of the transform.
   *
   * Reference: Paper Algorithm 1, lines 1-3 and Section 2.2 (Cartesian
   * Topology)
   *
   * For D dimensions, creates a (D-1)-dimensional Cartesian processor grid.
   * Example for 3D with 4 processors: MPI_Dims_create(4, 2, dims) → dims = [2,
   * 2]
   *
   * @param global_shape Array of D integers specifying the global array
   * dimensions
   * @param comm MPI communicator (default: MPI_COMM_WORLD)
   * @param plan_flag FFTW planning strategy (default: Estimate). Use Measure or
   *                  Patient for faster FFT execution at the cost of longer
   *                  initialization. Only affects the FFTW backend.
   */
  ParaFaFT_C2C(const int global_shape[D], MPI_Comm comm = MPI_COMM_WORLD,
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

    // Store global array dimensions
    for (int i = 0; i < D; ++i) {
      global_shape_[i] = global_shape[i];
    }

    if constexpr (Backend::handles_distributed) {
      // Backend handles MPI decomposition, communication, and FFT internally
      // (e.g., cuFFTMp with NVSHMEM, rocFFT with MPI)
      backend_.setup_distributed(global_shape, comm);
      auto info = backend_.get_distributed_info();
      // Store shapes for public API queries — use stage_shapes_[0] for input,
      // stage_shapes_[D-1] for output (reusing existing member storage)
      stage_shapes_.resize(D);
      for (int s = 0; s < D; ++s)
        stage_shapes_[s].resize(D);
      for (int i = 0; i < D; ++i) {
        stage_shapes_[0][i] = info.local_shape[i];
        stage_shapes_[D - 1][i] = info.output_shape[i];
        global_start_[i] = info.global_start[i];
        output_start_[i] = info.output_start[i];
      }
      max_stage_size_ = info.required_size;
    } else {
      // Standard ParaFaFT: manual pencil decomposition + MPI exchange

      // Create (D-1)-dimensional Cartesian processor grid
      // Paper Section 2.2: This topology enables efficient subcommunicators
      int dims[D - 1];
      for (int i = 0; i < D - 1; ++i)
        dims[i] = 0;
      MPI_Dims_create(size_, D - 1, dims); // Balanced grid dimensions

      for (int i = 0; i < D - 1; ++i) {
        dims_[i] = dims[i];
      }

      // Create Cartesian communicator (non-periodic)
      int periods[D - 1];
      for (int i = 0; i < D - 1; ++i)
        periods[i] = 0;
      MPI_Cart_create(comm_world_, D - 1, dims, periods, 1, &comm_cart_);

      // Get this processor's coordinates in the grid
      // These coordinates determine which data partition this processor owns
      MPI_Cart_coords(comm_cart_, rank_, D - 1, coords_);

      // Create (D-1) subcommunicators, one for each grid dimension
      // Paper Section 2.2: These enable efficient all-to-all communication
      // along specific dimensions during redistribution
      //
      // subcomms_[i] contains all processors that differ only in coordinate i
      // This is used for redistributing along grid dimension i
      for (int i = 0; i < D - 1; ++i) {
        int remain_dims[D - 1];
        for (int j = 0; j < D - 1; ++j)
          remain_dims[j] = 0;
        remain_dims[i] = 1; // Keep only dimension i
        MPI_Cart_sub(comm_cart_, remain_dims, &subcomms_[i]);
      }

      // Compute local array shapes for each stage
      // This determines data distribution at each stage of the algorithm
      setup_stage_shapes();

      // Cache MPI subarray datatypes for all stage transitions.
      // For each of (D-1) transitions between stage t and stage t+1:
      //   fwd_send_types_[t]: send types (subarray of stage_shapes_[t] along
      //   axis D-1-t) fwd_recv_types_[t]: recv types (subarray of
      //   stage_shapes_[t+1] along axis D-2-t)
      // Backward uses these in reverse: send=fwd_recv, recv=fwd_send.
      cache_exchange_types();

      // Compute maximum buffer size across all stages for ping-pong allocation
      max_stage_size_ = 0;
      for (int stage = 0; stage < D; ++stage) {
        int size = 1;
        for (int i = 0; i < D; ++i) {
          size *= stage_shapes_[stage][i];
        }
        max_stage_size_ = std::max(max_stage_size_, size);
      }

      // Allocate single scratch buffer for ping-pong with user's data buffer.
      // The user's buffer (sized to get_required_output_size()) serves as one
      // ping-pong buffer; this scratch buffer serves as the other.
      scratch_buffer_.resize(max_stage_size_);

      // Allocate pack buffer for manual packing exchange (GPU backends)
      if constexpr (!Backend::use_alltoallw) {
        pack_buffer_.resize(max_stage_size_);
      }

      // Create FFT plans for all stages
      create_backend_plans();

      // Set up P2P IPC for same-node GPU exchanges
      setup_p2p();
    }
  }

  /**
   * @brief Create FFT plans for all stages using the backend.
   *
   * Analyzes the memory layout for each stage and creates appropriate plans
   * with correct stride and distance parameters.
   *
   * Reuses scratch_buffer_ (already allocated, large enough for all stages) as
   * the planning buffer. FFTW's new-array execute API allows plans created with
   * one buffer to be executed on any buffer with compatible alignment/layout.
   */
  void create_backend_plans() {
    for (int stage = 0; stage < D; ++stage) {
      int axis = D - 1 - stage;
      int length = global_shape_[axis];

      // Compute batch count and stride/dist based on axis position
      if (axis == D - 1) {
        // Last axis: contiguous FFTs
        int batch = 1;
        for (int i = 0; i < D - 1; ++i) {
          batch *= stage_shapes_[stage][i];
        }
        backend_.create_stage_plan(stage, length, batch, scratch_buffer_.data(),
                                   1, length);

      } else if (axis == 0) {
        // First axis: strided FFTs
        int batch = 1;
        for (int i = 1; i < D; ++i) {
          batch *= stage_shapes_[stage][i];
        }
        int stride = batch;
        backend_.create_stage_plan(stage, length, batch, scratch_buffer_.data(),
                                   stride, 1);

      } else {
        // Middle axis: need to handle via loops in perform_fft_on_buffer
        // Create plan for trailing batch only
        int trailing_size = 1;
        for (int i = axis + 1; i < D; ++i) {
          trailing_size *= stage_shapes_[stage][i];
        }
        // Plan will be called multiple times for leading dimensions
        backend_.create_stage_plan(stage, length, trailing_size,
                                   scratch_buffer_.data(), trailing_size, 1);
      }
    }
  }

  /**
   * @brief Destructor. Frees MPI communicators and cached datatypes if MPI is
   * not finalized.
   */
  ~ParaFaFT_C2C() {
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
   * @brief Get the local array size (stage 0 distribution).
   *
   * Returns the logical size of local data at the initial distribution.
   * For the minimum buffer size required by forward()/backward(), use
   * get_required_output_size() instead.
   *
   * @return Total number of complex elements in the local array.
   */
  int get_local_size() const {
    int size = 1;
    for (int i = 0; i < D; ++i) {
      size *= stage_shapes_[0][i];
    }
    return size;
  }

  /**
   * @brief Get the minimum buffer size required for forward()/backward().
   *
   * Returns the maximum local array size across all internal stages of the
   * transform. The user's data buffer must be at least this many complex
   * elements, because intermediate MPI redistributions may temporarily
   * increase the local array size due to uneven decomposition.
   *
   * @return Minimum number of complex elements the data buffer must hold.
   */
  int get_required_output_size() const { return max_stage_size_; }

  /**
   * @brief Get the local array shape (stage 0 distribution).
   *
   * @param[out] shape Array of D integers to receive the local shape.
   */
  void get_local_shape(int shape[D]) const {
    for (int i = 0; i < D; ++i) {
      shape[i] = stage_shapes_[0][i];
    }
  }

  /**
   * @brief Get the starting indices of the local array in the global array
   * (stage 0).
   *
   * @param[out] start Array of D integers to receive the starting indices.
   */
  void get_global_start(int start[D]) const {
    for (int i = 0; i < D; ++i) {
      start[i] = global_start_[i];
    }
  }

  /**
   * @brief Get the local array shape after forward FFT (stage D-1
   * distribution).
   *
   * @param[out] shape Array of D integers to receive the final local shape.
   */
  void get_final_shape(int shape[D]) const {
    for (int i = 0; i < D; ++i) {
      shape[i] = stage_shapes_[D - 1][i];
    }
  }

  /**
   * @brief Get the starting indices after forward FFT (stage D-1 distribution).
   *
   * @param[out] start Array of D integers to receive the final starting
   * indices.
   */
  void get_final_start(int start[D]) const {
    if constexpr (Backend::handles_distributed) {
      for (int i = 0; i < D; ++i)
        start[i] = output_start_[i];
    } else {
      // For final stage, need to compute based on which axes are distributed
      // Stage D-1: axes [D-1, ..., 1] have been processed, axis 0 is fully
      // local Axes [1, 2, ..., D-1] are distributed using grid dimensions [0,
      // 1, ..., D-2]
      start[0] = 0; // Axis 0 is not distributed in final stage
      for (int i = 1; i < D; ++i) {
        // Axis i is distributed on grid dimension i-1
        int n, s;
        decompose(global_shape_[i], dims_[i - 1], coords_[i - 1], n, s);
        start[i] = s;
      }
    }
  }

  /**
   * @brief Get the backend-managed buffer for distributed transforms.
   *
   * When the backend handles distributed transforms (e.g., cuFFTMp), the
   * buffer is allocated by the backend using optimized memory (e.g., NVSHMEM).
   * Users should write data into this buffer before calling forward() and
   * read results from it after. The memory is usable like normal device memory.
   *
   * @return Device pointer to the internal buffer, or nullptr if the backend
   *         does not manage its own buffer.
   */
  Complex *get_buffer() {
    if constexpr (Backend::handles_distributed) {
      return backend_.get_buffer();
    } else {
      return nullptr;
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
   * @brief Print stage shapes for debugging.
   *
   * Outputs local array shapes at each stage to stdout.
   * Only rank 0 prints to avoid duplicate output.
   */
  void print_stage_shapes() const {
    if (rank_ == 0) {
      for (int stage = 0; stage < D; ++stage) {
        std::cout << "Stage " << stage << ": [";
        for (int i = 0; i < D; ++i) {
          std::cout << stage_shapes_[stage][i];
          if (i < D - 1)
            std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }
    }
  }

  /**
   * @brief Forward FFT transform (C2C).
   *
   * Computes the forward Complex-to-Complex FFT, transforming spatial-domain
   * data to frequency-domain coefficients.
   *
   * Reference: Paper Algorithm 1 (Main Loop)
   *
   * Algorithm overview:
   *   for each stage k from 0 to D-1:
   *       1. FFT along the local axis (axis D-1-k)
   *       2. If not last stage: redistribute to make next axis local
   *
   * Example (3D case from paper Section 3.1):
   *   Stage 0: FFT along Z (local), redistribute Z→Y
   *   Stage 1: FFT along Y (local), redistribute Y→X
   *   Stage 2: FFT along X (local), done
   *
   * @param data Input/output complex array.
   *             Must be at least get_required_output_size() elements.
   *             Input: spatial-domain data (stage 0 distribution)
   *             Output: frequency-domain data (stage D-1 distribution)
   *
   * @note User must normalize by 1/N after calling for correct FFT
   * normalization.
   */
  void forward(Complex *data) {
    if constexpr (Backend::handles_distributed) {
      backend_.forward();
      return;
    }
    // Determine starting buffer based on parity of swaps.
    // D-1 exchanges → D-1 swaps. After all swaps, src holds the final result.
    // We want the result to end up in data (the user's buffer).
    Complex *src;
    Complex *dst;
    if ((D - 1) % 2 == 0) {
      // Even swaps: src ends where it started → start with data
      src = data;
      dst = scratch_buffer_.data();
    } else {
      // Odd swaps: src ends at opposite → start with scratch so result is in
      // data
      int size0 = 1;
      for (int i = 0; i < D; ++i)
        size0 *= stage_shapes_[0][i];
      backend_.memcpy(scratch_buffer_.data(), data, size0 * sizeof(Complex));
      src = scratch_buffer_.data();
      dst = data;
    }

    // Paper Algorithm 1, lines 4-9: Main loop over dimensions
    // Perform D 1D-FFTs with (D-1) global redistributions
    for (int stage = 0; stage < D; ++stage) {
      int axis = D - 1 - stage; // Process axes from last to first

      // Paper Algorithm 1, line 5: Local FFT in-place on src
      perform_fft_on_buffer(stage, axis, FFTDirection::Forward, src);

      // Paper Algorithm 1, lines 6-8: Global redistribution
      // Make the next axis local for the next FFT stage
      if (stage < D - 1) {
        int next_axis = D - 2 - stage; // Next axis to become local
        if constexpr (Backend::use_alltoallw) {
          exchange(subcomms_[next_axis], nparts_[stage], src,
                   fwd_send_types_[stage].data(), dst,
                   fwd_recv_types_[stage].data(), exchange_counts_.data(),
                   exchange_displs_.data());
        } else if (nparts_[stage] == 1) {
          // Single-rank subcommunicator: local reshape, no MPI needed
          exchange_local<Backend>(backend_, src, dst, pack_buffer_.data(),
                                  fwd_exchange_geom_[stage]);
        } else if (p2p_info_[stage].any_enabled) {
          exchange_hybrid<Backend>(
              backend_, subcomms_[next_axis], nparts_[stage], src, dst,
              pack_buffer_.data(), p2p_info_[stage].remote_pack_ptrs.data(),
              p2p_info_[stage].peer_enabled.data(), fwd_exchange_geom_[stage]);
        } else {
          exchange_packed<Backend>(
              backend_, subcomms_[next_axis], nparts_[stage], src, dst,
              pack_buffer_.data(), fwd_exchange_geom_[stage]);
        }
        std::swap(src, dst);
      }
    }

    // After D-1 swaps, src points to data — result is already in user's buffer
    backend_.sync();
  }

  /**
   * @brief Backward (inverse) FFT transform (C2C).
   *
   * Computes the backward Complex-to-Complex FFT, transforming frequency-domain
   * coefficients back to spatial-domain data.
   *
   * Reference: Paper Algorithm 1 (Reversed)
   *
   * Algorithm overview:
   *   for each stage k from D-1 down to 0:
   *       1. FFT along the currently local axis (using FFTW_BACKWARD)
   *       2. If not first stage: redistribute to make previous axis local
   *
   * Example (3D case):
   *   Stage 2: FFT along X (local), redistribute X→Y
   *   Stage 1: FFT along Y (local), redistribute Y→Z
   *   Stage 0: FFT along Z (local), done
   *
   * @param data Input/output complex array.
   *             Must be at least get_required_output_size() elements.
   *             Input: frequency-domain data (stage D-1 distribution)
   *             Output: spatial-domain data (stage 0 distribution)
   *
   * @note User must normalize by 1/N after calling for correct IFFT
   * normalization.
   * @note forward() followed by backward() recovers original data (up to
   * scaling).
   */
  void backward(Complex *data) {
    if constexpr (Backend::handles_distributed) {
      backend_.backward();
      return;
    }
    // Determine starting buffer based on parity of swaps.
    // D-1 exchanges → D-1 swaps. We want the result to end up in data.
    Complex *src;
    Complex *dst;
    if ((D - 1) % 2 == 0) {
      // Even swaps: src ends where it started → start with data
      src = data;
      dst = scratch_buffer_.data();
    } else {
      // Odd swaps: src ends at opposite → start with scratch so result is in
      // data
      int sizeD = 1;
      for (int i = 0; i < D; ++i)
        sizeD *= stage_shapes_[D - 1][i];
      backend_.memcpy(scratch_buffer_.data(), data, sizeD * sizeof(Complex));
      src = scratch_buffer_.data();
      dst = data;
    }

    // Reverse of forward algorithm: process stages from D-1 down to 0
    for (int stage = D - 1; stage >= 0; --stage) {
      int axis = D - 1 - stage; // Same axis order as forward

      // Local inverse FFT along currently local axis
      perform_fft_on_buffer(stage, axis, FFTDirection::Backward, src);

      // Redistribute to previous stage (reverse of forward redistribution)
      if (stage > 0) {
        // Backward exchange: send with fwd_recv types, recv with fwd_send types
        int trans = stage - 1;        // transition index
        int comm_idx = D - 1 - stage; // = axis
        if constexpr (Backend::use_alltoallw) {
          exchange(subcomms_[comm_idx], nparts_[trans], src,
                   fwd_recv_types_[trans].data(), dst,
                   fwd_send_types_[trans].data(), exchange_counts_.data(),
                   exchange_displs_.data());
        } else if (nparts_[trans] == 1) {
          // Single-rank subcommunicator: local reshape, no MPI needed
          exchange_local<Backend>(backend_, src, dst, pack_buffer_.data(),
                                  bwd_exchange_geom_[trans]);
        } else if (p2p_info_[trans].any_enabled) {
          exchange_hybrid<Backend>(
              backend_, subcomms_[comm_idx], nparts_[trans], src, dst,
              pack_buffer_.data(), p2p_info_[trans].remote_pack_ptrs.data(),
              p2p_info_[trans].peer_enabled.data(), bwd_exchange_geom_[trans]);
        } else {
          exchange_packed<Backend>(
              backend_, subcomms_[comm_idx], nparts_[trans], src, dst,
              pack_buffer_.data(), bwd_exchange_geom_[trans]);
        }
        std::swap(src, dst);
      }
    }

    // After D-1 swaps, src points to data — result is already in user's buffer
    backend_.sync();
  }

private:
  /**
   * @brief Cache MPI subarray datatypes for all stage transitions.
   *
   * Pre-computes send/receive MPI subarray datatypes for each of the (D-1)
   * redistributions. These types are reused on every forward/backward call,
   * avoiding repeated MPI_Type_create_subarray + MPI_Type_commit +
   * MPI_Type_free.
   *
   * For transition t (between stage t and stage t+1):
   *   Forward: send from stage_shapes_[t] along axis D-1-t,
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
      MPI_Comm_size(subcomms_[D - 2 - t], &nparts_[t]);
      max_nparts = std::max(max_nparts, nparts_[t]);
    }

    if constexpr (Backend::use_alltoallw) {
      exchange_counts_.assign(max_nparts, 1);
      exchange_displs_.assign(max_nparts, 0);

      for (int t = 0; t < D - 1; ++t) {
        int send_axis = D - 1 - t; // axis distributed in stage t
        int recv_axis = D - 2 - t; // axis distributed in stage t+1

        fwd_send_types_[t].resize(nparts_[t]);
        fwd_recv_types_[t].resize(nparts_[t]);

        subarray(MPI_C_DOUBLE_COMPLEX, D, stage_shapes_[t].data(), send_axis,
                 nparts_[t], fwd_send_types_[t].data());
        subarray(MPI_C_DOUBLE_COMPLEX, D, stage_shapes_[t + 1].data(),
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
        init_exchange_geometry(fwd_exchange_geom_[t], nparts_[t], D,
                               stage_shapes_[t].data(), send_axis,
                               stage_shapes_[t + 1].data(), recv_axis);
        // Backward is forward with src/dst swapped
        init_exchange_geometry(bwd_exchange_geom_[t], nparts_[t], D,
                               stage_shapes_[t + 1].data(), recv_axis,
                               stage_shapes_[t].data(), send_axis);
      }
    }
  }

  /**
   * @brief Compute stage shapes (data distribution at each FFT stage).
   *
   * Computes the local array dimensions at each stage of the FFT algorithm.
   * This determines which axes are distributed vs local.
   *
   * Reference: Paper Section 3 (Pencil Decomposition)
   *
   * Key concept: At each stage, one axis is "local" (processor has all elements
   * along that axis), while other axes are distributed across processors.
   *
   * Example (3D with processor grid [2,2]):
   *   Global shape: [32, 32, 32]
   *   Stage 0: shape=[16, 16, 32]  (axes 0,1 distributed; axis 2 local)
   *   Stage 1: shape=[16, 32, 16]  (axes 0,2 distributed; axis 1 local)
   *   Stage 2: shape=[32, 16, 16]  (axes 1,2 distributed; axis 0 local)
   */
  void setup_stage_shapes() {
    stage_shapes_.resize(D);
    for (int stage = 0; stage < D; ++stage) {
      stage_shapes_[stage].resize(D);
    }

    // Compute local sizes for each processor grid dimension using Equation (9)
    // local_n[i] = number of elements this processor owns along grid dim i
    // local_s[i] = starting index in global array along grid dim i
    int local_n[D - 1];
    int local_s[D - 1];
    for (int i = 0; i < D - 1; ++i) {
      decompose(global_shape_[i], dims_[i], coords_[i], local_n[i], local_s[i]);
    }

    // Store global starting position for stage 0 (initial distribution)
    // Used by user to initialize data with correct global coordinates
    for (int i = 0; i < D - 1; ++i) {
      global_start_[i] = local_s[i];
    }
    global_start_[D - 1] = 0; // Last axis is fully local in stage 0

    // Compute stage shapes following the pencil decomposition pattern:
    //
    // Stage 0: axes [0, ..., D-2] distributed on grid, axis D-1 local
    // Stage 1: axes [0, ..., D-3] keep original distribution,
    //          axis D-2 becomes local, axis D-1 gets redistributed
    // ...
    // Stage k: axis D-1-k is local for FFT,
    //          axes D-k,...,D-1 have been redistributed to new grid dimensions
    //
    for (int stage = 0; stage < D; ++stage) {
      int num_processed = stage; // Number of axes that have been FFT'd

      for (int axis = 0; axis < D; ++axis) {
        if (axis < D - 1 - num_processed) {
          // Axis not yet processed: still has original distribution
          stage_shapes_[stage][axis] = local_n[axis];

        } else if (axis == D - 1 - stage) {
          // This is THE axis for FFT at this stage: fully local
          stage_shapes_[stage][axis] = global_shape_[axis];

        } else if (axis > D - 1 - stage) {
          // Axis was processed in earlier stage: redistributed to new grid dim
          // Map redistributed axes to grid dimensions not used by original axes
          int num_still_original = D - 1 - num_processed;
          int grid_dim = num_still_original + (axis - (D - stage));

          if (grid_dim >= 0 && grid_dim < D - 1) {
            stage_shapes_[stage][axis] = local_n[grid_dim];
          } else {
            // Edge case: grid dimension doesn't exist (e.g., size-1 dimension)
            // Use full global size (effectively not distributed)
            stage_shapes_[stage][axis] = global_shape_[axis];
          }
        } else {
          // Between current and processed: fully local (transition state)
          stage_shapes_[stage][axis] = global_shape_[axis];
        }
      }
    }
  }

  /**
   * @brief Execute C2C FFT on a buffer for a specific stage.
   *
   * Handles the complexity of strided FFTs for different axes,
   * looping over leading dimensions when necessary.
   *
   * Reference: Paper Section 2.1 (Local FFT Operations)
   *
   * At each stage, one axis is fully local (processor has complete data).
   * We perform 1D FFT along this local axis for all pencils.
   *
   * Memory layout considerations:
   *   - For axis D-1 (last): data is contiguous → single backend call
   *   - For axis 0 (first): data is strided → single backend call
   *   - For middle axes: loop over leading dimensions, call backend per slice
   *
   * @param stage Stage index (0 to D-1).
   * @param axis Axis along which to perform the FFT.
   * @param direction Forward or Backward FFT direction.
   * @param buffer Buffer containing the complex data.
   */
  void perform_fft_on_buffer(int stage, int axis, FFTDirection direction,
                             Complex *buffer) {
    // CASE 1: FFT along last axis (axis D-1) or first axis (axis 0)
    // Backend plan handles these with a single call
    if (axis == D - 1 || axis == 0) {
      backend_.execute_stage(stage, direction, buffer);

      // CASE 2: FFT along middle axis (0 < axis < D-1)
      // Need to loop over leading dimensions
    } else {
      // Number of independent "batches" (loop over these)
      int leading_size = 1;
      for (int i = 0; i < axis; ++i) {
        leading_size *= stage_shapes_[stage][i];
      }

      // Number of FFTs per batch (can do simultaneously)
      int trailing_size = 1;
      for (int i = axis + 1; i < D; ++i) {
        trailing_size *= stage_shapes_[stage][i];
      }

      // Execute plan for each leading slice
      for (int i = 0; i < leading_size; ++i) {
        // Pointer to start of this batch
        Complex *base_ptr = buffer + i * global_shape_[axis] * trailing_size;
        backend_.execute_stage(stage, direction, base_ptr);
      }
    }
  }

  /**
   * @brief Set up CUDA/HIP IPC for P2P GPU-to-GPU exchange.
   *
   * For each subcommunicator, checks if all ranks are on same-node GPUs with
   * P2P access. If so, exports pack_buffer_ via IPC and opens remote handles,
   * enabling direct GPU-to-GPU copies that bypass MPI.
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

        // Gather device IDs in this subcommunicator
        std::vector<int> devices(nparts);
        MPI_Allgather(&my_device, 1, MPI_INT, devices.data(), 1, MPI_INT, comm);

        // Get this rank's subrank in the subcommunicator
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

  // =========================================================================
  // Member Variables
  // =========================================================================

  MPI_Comm comm_world_; ///< Original communicator (typically MPI_COMM_WORLD)
  MPI_Comm comm_cart_;  ///< (D-1)-dimensional Cartesian topology
  MPI_Comm subcomms_[D - 1]; ///< Subcommunicators for each grid dimension

  int rank_;          ///< This processor's rank
  int size_;          ///< Total number of processors
  int dims_[D - 1];   ///< Processor grid dimensions (from MPI_Dims_create)
  int coords_[D - 1]; ///< This processor's coordinates in the grid

  int global_shape_[D]; ///< Global array dimensions
  int global_start_[D]; ///< Starting indices for this processor (stage 0)
  int output_start_[D]; ///< Starting indices for output (stage D-1), used by
                        ///< distributed backends

  /// Local array shape at each stage: stage_shapes_[stage][axis]
  std::vector<std::vector<int>> stage_shapes_;
  /// Maximum buffer size across all stages (for ping-pong allocation)
  int max_stage_size_;
  /// Scratch buffer for ping-pong with the user's data buffer
  ComplexBuffer scratch_buffer_;
  /// Pack buffer for contiguous MPI exchange (GPU backends only)
  ComplexBuffer pack_buffer_;

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

  Backend backend_; ///< FFT backend (stateful, stores pre-created plans)
};

} // namespace parafaft

#endif // PARAFAFT_C2C_HPP
