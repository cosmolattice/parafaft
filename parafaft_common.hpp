#ifndef PARAFAFT_COMMON_HPP
#define PARAFAFT_COMMON_HPP

// ============================================================================
// Common Utility Functions for ParaFaFT
// ============================================================================
//
// Reference: Dalcín, L., Mortensen, M., & Keyes, D. E. (2019).
//           Fast parallel multidimensional FFT using advanced MPI.
//           Journal of Parallel and Distributed Computing, 128, 137-150.
//
// This header contains decomposition, subarray datatype creation, and
// exchange routines shared by both C2C and R2C parallel FFT implementations.
//
// Key Innovation (Section 1):
//   Uses MPI derived datatypes (MPI_Type_create_subarray + MPI_Alltoallw)
//   to perform global redistribution WITHOUT local transpose operations.
//   This eliminates costly data rearrangements compared to traditional methods.
//
// ============================================================================

#include "./backend/fft_backend.hpp"
#include <cstring>
#include <mpi.h>
#include <vector>

namespace parafaft {
// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Balanced block-contiguous decomposition using Barry Smith's formula.
 *
 * Distributes N elements across M processors ensuring load balance:
 * processors either get q or q+1 elements where q = N/M.
 *
 * Reference: Paper Equation (9) on page 140
 *
 * @param N Total number of elements to distribute
 * @param M Number of processors
 * @param p Processor rank (0 <= p < M)
 * @param[out] n Number of elements assigned to processor p
 * @param[out] s Starting index for processor p in global array
 */
inline void decompose(int N, int M, int p, int &n, int &s) {
  int q = N / M; // Base number of elements per processor
  int r = N % M; // Remainder elements to distribute

  // Processors 0..r-1 get q+1 elements, processors r..M-1 get q elements
  n = (r > p) ? (q + 1) : q;
  s = (r > p) ? (n * p) : (q * p + r);
}

/**
 * @brief Create MPI subarray datatypes for partitioning along an axis.
 *
 * Creates MPI subarray datatypes that describe how to partition a D-dimensional
 * array along a specific axis. This is the key to avoiding local transposes by
 * letting MPI handle the complex index mapping during redistribution.
 *
 * Reference: Paper Section 2.3 and Algorithm 1
 *
 * Traditional methods require:
 *   1. Local transpose to make data contiguous
 *   2. MPI_Alltoall with contiguous buffers
 *   3. Local transpose again
 *
 * This method (Algorithm 1):
 *   1. Create MPI_Type_create_subarray describing discontiguous layout
 *   2. MPI_Alltoallw handles the complexity in MPI layer
 *   3. NO local transposes needed!
 *
 * @param datatype Base MPI datatype (e.g., MPI_C_DOUBLE_COMPLEX)
 * @param ndims Number of dimensions
 * @param sizes Size of the array in each dimension
 * @param axis Axis along which to partition
 * @param nparts Number of partitions (typically number of processors)
 * @param[out] subarrays Array of nparts MPI datatypes (must be pre-allocated)
 */
inline void subarray(MPI_Datatype datatype, int ndims, const int sizes[],
                     int axis, int nparts, MPI_Datatype subarrays[]) {
  // Use a fixed-size stack buffer for subsizes/substarts
  std::vector<int> subsizes(ndims), substarts(ndims);

  // Initialize: full size in all dimensions, starting at origin
  for (int i = 0; i < ndims; i++) {
    subsizes[i] = sizes[i];
    substarts[i] = 0;
  }

  // Create one subarray datatype for each partition
  for (int p = 0; p < nparts; p++) {
    int n, s;
    decompose(sizes[axis], nparts, p, n, s); // Use Equation (9)
    subsizes[axis] = n;                      // Size of partition p along axis
    substarts[axis] = s;                     // Starting position of partition p

    // MPI_Type_create_subarray describes the discontiguous memory layout
    // This is the key innovation - no need to pack/unpack data manually
    MPI_Type_create_subarray(ndims, sizes, subsizes.data(), substarts.data(),
                             MPI_ORDER_C, datatype, &subarrays[p]);
    MPI_Type_commit(&subarrays[p]);
  }
}

/**
 * @brief Global data redistribution using MPI_Alltoallw with pre-cached
 * datatypes.
 *
 * Redistributes data between two pencil decomposition stages, changing which
 * axis is local vs distributed without any local transposes.
 *
 * Reference: Paper Algorithm 1 and Section 2.3
 *
 * This implements the "global redistribution" step between FFT stages.
 * MPI_Alltoallw with subarray datatypes handles the complex index mapping
 * automatically. No manual packing/unpacking!
 *
 * @param comm MPI subcommunicator for the redistribution
 * @param nparts Number of partitions in the subcommunicator
 * @param arrayA Input array (distributed along axisA)
 * @param subarraysA Pre-created send subarray datatypes
 * @param[out] arrayB Output array (will be distributed along axisB)
 * @param subarraysB Pre-created receive subarray datatypes
 * @param counts Pre-allocated counts array (all 1s, size nparts)
 * @param displs Pre-allocated displacements array (all 0s, size nparts)
 */
inline void exchange(MPI_Comm comm, int nparts, void *arrayA,
                     const MPI_Datatype subarraysA[], void *arrayB,
                     const MPI_Datatype subarraysB[], const int counts[],
                     const int displs[]) {
  // This is THE key operation (Paper Algorithm 1, line 7):
  // All-to-all exchange using derived datatypes
  // Handles all the complex index calculations internally
  MPI_Alltoallw(arrayA, counts, displs, subarraysA, arrayB, counts, displs,
                subarraysB, comm);
}

/**
 * @brief Packed exchange: manual pack → MPI_Alltoallv → unpack.
 *
 * Replaces MPI_Alltoallw with derived subarray types by explicitly packing
 * non-contiguous data into contiguous GPU buffers, using MPI_Alltoallv
 * (which supports GPU-direct for contiguous data), then unpacking.
 *
 * This avoids the slow host-staging fallback that most GPU-aware MPI
 * implementations use for non-contiguous derived datatypes on device memory.
 *
 * Memory optimization: MPI receives into src_array (whose original data was
 * already packed out), so only one extra buffer (pack_buf) is needed.
 *
 * @tparam BackendT FFT backend type (must provide memcpy2d)
 * @param comm MPI subcommunicator for the redistribution
 * @param nparts Number of partitions in the subcommunicator
 * @param ndims Number of array dimensions
 * @param src_array Source array (data to send, will be overwritten with
 *                  received packed data as temporary storage)
 * @param src_sizes Full array dimensions at the source stage
 * @param src_axis Axis along which source data is partitioned for sending
 * @param dst_array Destination array (will contain unpacked received data)
 * @param dst_sizes Full array dimensions at the destination stage
 * @param dst_axis Axis along which destination data is partitioned for
 *                 receiving
 * @param pack_buf Temporary buffer for packing (size >= total elements)
 */
template <typename BackendT>
inline void exchange_packed(
    const BackendT &backend,
    MPI_Comm comm, int nparts, int ndims,
    typename BackendT::Complex *src_array, const int src_sizes[], int src_axis,
    typename BackendT::Complex *dst_array, const int dst_sizes[], int dst_axis,
    typename BackendT::Complex *pack_buf)
{
  using Complex = typename BackendT::Complex;
  size_t elem_size = sizeof(Complex);

  // Compute send-side geometry
  size_t src_leading = 1, src_trailing = 1;
  for (int i = 0; i < src_axis; ++i) src_leading *= src_sizes[i];
  for (int i = src_axis + 1; i < ndims; ++i) src_trailing *= src_sizes[i];

  // Compute recv-side geometry
  size_t dst_leading = 1, dst_trailing = 1;
  for (int i = 0; i < dst_axis; ++i) dst_leading *= dst_sizes[i];
  for (int i = dst_axis + 1; i < ndims; ++i) dst_trailing *= dst_sizes[i];

  // When leading == 1, the partition axis is the slowest-varying and each
  // partition is already a single contiguous block in memory — identical to
  // the packed layout. We can skip the pack or unpack step entirely.
  bool src_contiguous = (src_leading == 1);
  bool dst_contiguous = (dst_leading == 1);

  std::vector<int> send_counts(nparts), send_displs(nparts);
  std::vector<int> recv_counts(nparts), recv_displs(nparts);

  // Compute send counts/displacements
  size_t send_offset = 0;
  for (int p = 0; p < nparts; ++p) {
    int n, s;
    decompose(src_sizes[src_axis], nparts, p, n, s);
    size_t count = src_leading * n * src_trailing;
    send_counts[p] = static_cast<int>(count);
    send_displs[p] = static_cast<int>(send_offset);
    send_offset += count;
  }

  // Compute recv counts/displacements
  size_t recv_offset = 0;
  for (int p = 0; p < nparts; ++p) {
    int n, s;
    decompose(dst_sizes[dst_axis], nparts, p, n, s);
    size_t count = dst_leading * n * dst_trailing;
    recv_counts[p] = static_cast<int>(count);
    recv_displs[p] = static_cast<int>(recv_offset);
    recv_offset += count;
  }

  // Pack src_array into pack_buf (skip when src is already contiguous)
  // Pack is async on the backend stream, ordered after any preceding FFT.
  if (!src_contiguous) {
    for (int p = 0; p < nparts; ++p) {
      int n, s;
      decompose(src_sizes[src_axis], nparts, p, n, s);
      size_t row_bytes = static_cast<size_t>(n) * src_trailing * elem_size;
      backend.memcpy2d(
          reinterpret_cast<char *>(pack_buf) + static_cast<size_t>(send_displs[p]) * elem_size,
          row_bytes,
          reinterpret_cast<char *>(src_array) + static_cast<size_t>(s) * src_trailing * elem_size,
          static_cast<size_t>(src_sizes[src_axis]) * src_trailing * elem_size,
          row_bytes,
          src_leading);
    }
  }

  // Choose send/recv buffer pointers:
  //   send_from: pack_buf if we packed, src_array if src is contiguous
  //   recv_into: dst_array if dst is contiguous, pack_buf if src was
  //              contiguous (pack_buf is free), src_array otherwise
  //              (src_array is free after packing)
  Complex *send_from = src_contiguous ? src_array : pack_buf;
  Complex *recv_into;
  if (dst_contiguous) {
    recv_into = dst_array;
  } else if (src_contiguous) {
    recv_into = pack_buf; // pack_buf unused since we skipped packing
  } else {
    recv_into = src_array; // src_array free since we packed its data out
  }

  // Sync stream: ensure all async pack/FFT operations are committed
  // before MPI reads from the buffer (MPI is not stream-aware).
  backend.sync();

  // Non-blocking MPI: post receives first, then sends, overlap self-copy
  // with network transfers. Uses point-to-point (not collectives) because
  // collective accelerators (e.g. HCOLL) may not support GPU device pointers.
  int myrank;
  MPI_Comm_rank(comm, &myrank);
  std::vector<MPI_Request> requests;
  requests.reserve(2 * (nparts - 1));

  for (int p = 0; p < nparts; ++p) {
    if (p == myrank) continue;
    size_t recv_byte_off = static_cast<size_t>(recv_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Irecv(reinterpret_cast<char *>(recv_into) + recv_byte_off,
              recv_counts[p], MPI_C_DOUBLE_COMPLEX, p, 0, comm, &req);
    requests.push_back(req);
  }
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank) continue;
    size_t send_byte_off = static_cast<size_t>(send_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Isend(reinterpret_cast<char *>(send_from) + send_byte_off,
              send_counts[p], MPI_C_DOUBLE_COMPLEX, p, 0, comm, &req);
    requests.push_back(req);
  }

  // Self-copy: async on GPU stream, overlaps with network transfers
  {
    size_t send_byte_off = static_cast<size_t>(send_displs[myrank]) * elem_size;
    size_t recv_byte_off = static_cast<size_t>(recv_displs[myrank]) * elem_size;
    backend.memcpy(
        reinterpret_cast<char *>(recv_into) + recv_byte_off,
        reinterpret_cast<char *>(send_from) + send_byte_off,
        static_cast<size_t>(send_counts[myrank]) * elem_size);
  }

  // Wait for all remote transfers to complete
  if (!requests.empty()) {
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                MPI_STATUSES_IGNORE);
  }

  // Unpack recv_into into dst_array (skip when dst is already contiguous)
  // Async on stream, enqueued after MPI_Waitall guarantees data is valid.
  if (!dst_contiguous) {
    for (int p = 0; p < nparts; ++p) {
      int n, s;
      decompose(dst_sizes[dst_axis], nparts, p, n, s);
      size_t row_bytes = static_cast<size_t>(n) * dst_trailing * elem_size;
      backend.memcpy2d(
          reinterpret_cast<char *>(dst_array) + static_cast<size_t>(s) * dst_trailing * elem_size,
          static_cast<size_t>(dst_sizes[dst_axis]) * dst_trailing * elem_size,
          reinterpret_cast<char *>(recv_into) + static_cast<size_t>(recv_displs[p]) * elem_size,
          row_bytes,
          row_bytes,
          dst_leading);
    }
  }
}

/**
 * @brief Hybrid exchange: mixes P2P direct GPU copies with MPI point-to-point.
 *
 * For neighbours where P2P (IPC) is available, uses direct GPU-to-GPU memory
 * copies via IPC-mapped pointers. For non-P2P neighbours, falls back to MPI
 * Isend/Irecv. Two MPI_Barrier calls synchronize the P2P pack/read phases;
 * MPI transfers overlap with the P2P reads between the barriers.
 *
 * When all neighbours are P2P-capable, this degenerates to the pure-P2P path
 * (no MPI sends/recvs). When none are, this function should not be called —
 * use exchange_packed() instead.
 *
 * @tparam BackendT FFT backend type
 * @param backend Backend instance for async memcpy
 * @param comm MPI subcommunicator
 * @param nparts Number of partitions
 * @param ndims Number of array dimensions
 * @param src_array Source array
 * @param src_sizes Source dimensions
 * @param src_axis Source partition axis
 * @param dst_array Destination array
 * @param dst_sizes Destination dimensions
 * @param dst_axis Destination partition axis
 * @param pack_buf Local pack buffer (IPC-exported)
 * @param remote_pack_ptrs IPC-mapped pointers (non-null for P2P neighbours)
 * @param peer_enabled Per-neighbour flags: nonzero = P2P, zero = MPI
 */
template <typename BackendT>
inline void exchange_hybrid(
    const BackendT &backend,
    MPI_Comm comm, int nparts, int ndims,
    typename BackendT::Complex *src_array, const int src_sizes[], int src_axis,
    typename BackendT::Complex *dst_array, const int dst_sizes[], int dst_axis,
    typename BackendT::Complex *pack_buf,
    void *const *remote_pack_ptrs,
    const char *peer_enabled)
{
  using Complex = typename BackendT::Complex;
  size_t elem_size = sizeof(Complex);

  // Compute send-side geometry
  size_t src_leading = 1, src_trailing = 1;
  for (int i = 0; i < src_axis; ++i) src_leading *= src_sizes[i];
  for (int i = src_axis + 1; i < ndims; ++i) src_trailing *= src_sizes[i];

  // Compute recv-side geometry
  size_t dst_leading = 1, dst_trailing = 1;
  for (int i = 0; i < dst_axis; ++i) dst_leading *= dst_sizes[i];
  for (int i = dst_axis + 1; i < ndims; ++i) dst_trailing *= dst_sizes[i];

  bool dst_contiguous = (dst_leading == 1);

  // Compute send counts/displacements
  std::vector<int> send_counts(nparts), send_displs(nparts);
  size_t send_offset = 0;
  for (int p = 0; p < nparts; ++p) {
    int n, s;
    decompose(src_sizes[src_axis], nparts, p, n, s);
    size_t count = src_leading * n * src_trailing;
    send_counts[p] = static_cast<int>(count);
    send_displs[p] = static_cast<int>(send_offset);
    send_offset += count;
  }

  // Compute recv counts/displacements
  std::vector<int> recv_counts(nparts), recv_displs(nparts);
  size_t recv_offset = 0;
  for (int p = 0; p < nparts; ++p) {
    int n, s;
    decompose(dst_sizes[dst_axis], nparts, p, n, s);
    size_t count = dst_leading * n * dst_trailing;
    recv_counts[p] = static_cast<int>(count);
    recv_displs[p] = static_cast<int>(recv_offset);
    recv_offset += count;
  }

  // Always pack into pack_buf: P2P neighbours read via IPC handle,
  // MPI neighbours send from pack_buf via Isend.
  for (int p = 0; p < nparts; ++p) {
    int n, s;
    decompose(src_sizes[src_axis], nparts, p, n, s);
    size_t row_bytes = static_cast<size_t>(n) * src_trailing * elem_size;
    backend.memcpy2d(
        reinterpret_cast<char *>(pack_buf) + static_cast<size_t>(send_displs[p]) * elem_size,
        row_bytes,
        reinterpret_cast<char *>(src_array) + static_cast<size_t>(s) * src_trailing * elem_size,
        static_cast<size_t>(src_sizes[src_axis]) * src_trailing * elem_size,
        row_bytes,
        src_leading);
  }

  // Sync: ensure pack data is committed before MPI reads or P2P reads
  backend.sync();

  // Barrier: all ranks must finish packing before P2P reads begin
  MPI_Barrier(comm);

  // Choose recv buffer
  Complex *recv_into = dst_contiguous ? dst_array : src_array;

  int myrank;
  MPI_Comm_rank(comm, &myrank);

  // Post MPI Irecv/Isend for non-P2P neighbours
  std::vector<MPI_Request> requests;
  requests.reserve(2 * nparts);
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank || peer_enabled[p]) continue;
    size_t recv_byte_off = static_cast<size_t>(recv_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Irecv(reinterpret_cast<char *>(recv_into) + recv_byte_off,
              recv_counts[p], MPI_C_DOUBLE_COMPLEX, p, 0, comm, &req);
    requests.push_back(req);
  }
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank || peer_enabled[p]) continue;
    size_t send_byte_off = static_cast<size_t>(send_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Isend(reinterpret_cast<char *>(pack_buf) + send_byte_off,
              send_counts[p], MPI_C_DOUBLE_COMPLEX, p, 0, comm, &req);
    requests.push_back(req);
  }

  // P2P direct GPU-to-GPU copy for capable neighbours
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank || !peer_enabled[p]) continue;
    size_t recv_byte_off = static_cast<size_t>(recv_displs[p]) * elem_size;
    size_t remote_off = static_cast<size_t>(send_displs[myrank]) * elem_size;
    char *remote_buf = reinterpret_cast<char *>(remote_pack_ptrs[p]);
    backend.memcpy(
        reinterpret_cast<char *>(recv_into) + recv_byte_off,
        remote_buf + remote_off,
        static_cast<size_t>(recv_counts[p]) * elem_size);
  }

  // Self-copy (local rank)
  {
    size_t send_byte_off = static_cast<size_t>(send_displs[myrank]) * elem_size;
    size_t recv_byte_off = static_cast<size_t>(recv_displs[myrank]) * elem_size;
    backend.memcpy(
        reinterpret_cast<char *>(recv_into) + recv_byte_off,
        reinterpret_cast<char *>(pack_buf) + send_byte_off,
        static_cast<size_t>(send_counts[myrank]) * elem_size);
  }

  // Sync stream: ensure P2P reads and self-copy complete
  backend.sync();

  // Wait for MPI transfers to complete
  if (!requests.empty()) {
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                MPI_STATUSES_IGNORE);
  }

  // Barrier: all ranks must finish reading before pack_buf can be reused
  MPI_Barrier(comm);

  // Unpack recv_into into dst_array (skip when dst is already contiguous)
  if (!dst_contiguous) {
    for (int p = 0; p < nparts; ++p) {
      int n, s;
      decompose(dst_sizes[dst_axis], nparts, p, n, s);
      size_t row_bytes = static_cast<size_t>(n) * dst_trailing * elem_size;
      backend.memcpy2d(
          reinterpret_cast<char *>(dst_array) + static_cast<size_t>(s) * dst_trailing * elem_size,
          static_cast<size_t>(dst_sizes[dst_axis]) * dst_trailing * elem_size,
          reinterpret_cast<char *>(recv_into) + static_cast<size_t>(recv_displs[p]) * elem_size,
          row_bytes,
          row_bytes,
          dst_leading);
    }
  }
}

} // namespace parafaft

#endif // PARAFAFT_COMMON_HPP
