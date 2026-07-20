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
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace parafaft {
// ============================================================================
// Precision → MPI datatype mapping
// ============================================================================

/**
 * @brief Map a C++ floating-point type to the corresponding MPI complex datatype.
 *
 * Specialized for double → MPI_C_DOUBLE_COMPLEX and float → MPI_C_FLOAT_COMPLEX.
 * Used by exchange helpers and core classes to select the right MPI datatype
 * based on the active FloatType precision.
 */
template <typename T> inline MPI_Datatype mpi_complex_type();
template <> inline MPI_Datatype mpi_complex_type<double>() { return MPI_C_DOUBLE_COMPLEX; }
template <> inline MPI_Datatype mpi_complex_type<float>() { return MPI_C_FLOAT_COMPLEX; }

/**
 * @brief Map a C++ floating-point type to the corresponding MPI real datatype.
 */
template <typename T> inline MPI_Datatype mpi_real_type();
template <> inline MPI_Datatype mpi_real_type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_real_type<float>() { return MPI_FLOAT; }

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
  // Preconditions for MPI_Type_create_subarray. MPI rejects with MPI_ERR_ARG
  // when any dimension's size is zero, or when the partition axis can't be
  // split evenly enough for every partition to get >= 1 element. Both
  // happen when the global grid is too small for the chosen pencil
  // decomposition (e.g. nGrid=4 across 5 ranks: some rank ends up owning
  // zero rows along the split axis). Failing here gives a much clearer
  // diagnostic than the deferred MPI abort.
  for (int i = 0; i < ndims; ++i) {
    if (sizes[i] <= 0) {
      std::ostringstream msg;
      msg << "ParaFaFT: cannot create subarray with zero extent on axis " << i
          << " (sizes[" << i << "] = " << sizes[i]
          << "). The global grid is too small for this pencil decomposition; "
             "increase the grid size on this axis or reduce the number of MPI "
             "ranks.";
      throw std::runtime_error(msg.str());
    }
  }
  if (sizes[axis] < nparts) {
    std::ostringstream msg;
    msg << "ParaFaFT: cannot decompose axis " << axis << " of extent "
        << sizes[axis] << " across " << nparts
        << " partitions (need extent >= nparts so every partition gets >= 1 "
           "element). Increase the global grid size on this axis or reduce "
           "the number of MPI ranks.";
    throw std::runtime_error(msg.str());
  }

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

// ============================================================================
// Processor grid topology
// ============================================================================

/**
 * @brief Choose processor grid dimensions that respect node boundaries.
 *
 * MPI_Dims_create alone returns a balanced factorization in non-increasing
 * order, with no idea where node boundaries fall. MPI_Cart_create then ranks
 * row-major, so the *last* grid dimension is the rank-contiguous one. When
 * that trailing dimension does not match the number of ranks per node, a grid
 * dimension straddles nodes and the subcommunicator for that transition mixes
 * intra-node and inter-node peers — the worst case for the exchange, which
 * then pays intra-node synchronization costs across the network.
 *
 * Concretely, 8 ranks on 4-rank nodes gave dims {4,2}: the size-4 dimension
 * has stride 2 and so spans both nodes. Pinning the trailing dimension to the
 * node size instead yields {2,4}, where the size-4 subcommunicator is exactly
 * one node and the size-2 subcommunicator is purely inter-node.
 *
 * Only applied when the job spans more than one node and the ranks divide
 * evenly among nodes. A single-node job is topologically uniform, so its grid
 * is left exactly as MPI_Dims_create chose it.
 *
 * @param nranks Total ranks to arrange
 * @param ndims Number of grid dimensions (D-1)
 * @param ppn Ranks per node; pass 1 to disable topology awareness
 * @param[out] dims Grid dimensions, dims[ndims-1] being rank-contiguous
 */
inline void choose_dims(int nranks, int ndims, int ppn, int dims[])
{
  for (int i = 0; i < ndims; ++i)
    dims[i] = 0;

  const bool topology_helps = ndims >= 2 && ppn > 1 && nranks > ppn &&
                              (nranks % ppn) == 0;
  if (!topology_helps) {
    MPI_Dims_create(nranks, ndims, dims);
    return;
  }

  // Reserve the rank-contiguous dimension for one full node, then balance the
  // remaining ranks (one entry per node) across the leading dimensions.
  std::vector<int> leading(ndims - 1, 0);
  MPI_Dims_create(nranks / ppn, ndims - 1, leading.data());
  for (int i = 0; i < ndims - 1; ++i)
    dims[i] = leading[i];
  dims[ndims - 1] = ppn;
}

/**
 * @brief Determine how many ranks of @p comm share a physical node.
 *
 * Collective over @p comm. Returns 1 — which makes choose_dims() a no-op —
 * when the allocation is heterogeneous (nodes hosting different rank counts,
 * where no single trailing dimension can match every node) or when the
 * PARAFAFT_DISABLE_TOPO_GRID environment variable is set. The escape hatch
 * exists so a bad detection can be worked around at run time rather than
 * needing a rebuild.
 *
 * PARAFAFT_RANKS_PER_NODE overrides the detected value. That makes the
 * multi-node grid reproducible on a single machine — running 8 oversubscribed
 * ranks with the variable set to 4 builds the same grid a two-node job would
 * — which is the only way to exercise the multi-node layout without a cluster.
 * Values below 1 are ignored.
 *
 * @param comm Communicator to inspect
 * @return Ranks per node, or 1 if unknown, uniform-free, or disabled
 */
inline int detect_ranks_per_node(MPI_Comm comm)
{
  const char *disable = std::getenv("PARAFAFT_DISABLE_TOPO_GRID");
  if (disable != nullptr && disable[0] != '\0' && disable[0] != '0')
    return 1;

  const char *forced = std::getenv("PARAFAFT_RANKS_PER_NODE");
  if (forced != nullptr && forced[0] != '\0') {
    const int value = std::atoi(forced);
    if (value >= 1)
      return value;
  }

  MPI_Comm shared_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shared_comm);
  int local_size = 0;
  MPI_Comm_size(shared_comm, &local_size);
  MPI_Comm_free(&shared_comm);

  // Every rank must agree, otherwise no single trailing dimension fits.
  int min_size = 0, max_size = 0;
  MPI_Allreduce(&local_size, &min_size, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&local_size, &max_size, 1, MPI_INT, MPI_MAX, comm);
  if (min_size != max_size)
    return 1;

  return min_size;
}

// ============================================================================
// Pre-computed Exchange Geometry
// ============================================================================

/**
 * @brief Pre-computed geometry for MPI exchange operations.
 *
 * Caches all decomposition results, counts, displacements, and layout
 * metadata for a single stage transition. Populated once at construction
 * time and reused on every forward/backward call, eliminating per-call
 * heap allocations and redundant decompose() arithmetic.
 */
struct ExchangeGeometry {
  // Element counts and displacements are size_t, not int: displacements
  // accumulate to the full local element count, which passes 2^31 for large
  // grids (3D C2C double at N=1626 and up). Narrowing here produced negative
  // displacements and silent corruption rather than a diagnosable failure.
  // The only places that must be int are the MPI count arguments, which are
  // narrowed through checked_mpi_count() at the call site.
  std::vector<std::size_t> send_counts;  ///< Per-peer send element counts
  std::vector<std::size_t> send_displs;  ///< Per-peer send element displacements
  std::vector<std::size_t> recv_counts;  ///< Per-peer recv element counts
  std::vector<std::size_t> recv_displs;  ///< Per-peer recv element displacements
  /// Per-peer offset of *our* block inside peer p's send/pack buffer, i.e.
  /// peer p's send_displs[myrank]. Needed by the P2P/IPC path, which reads
  /// directly out of a peer's pack buffer: send_displs is rank-dependent
  /// whenever a trailing axis is split unevenly (differing src_trailing), so
  /// the local send_displs[myrank] would index the peer's buffer wrongly.
  /// Populated by init_remote_send_displs() (collective). Empty otherwise.
  std::vector<std::size_t> remote_send_displs;
  std::vector<int> src_n;        ///< Per-peer: decompose n for src axis
  std::vector<int> src_s;        ///< Per-peer: decompose s (start) for src axis
  std::vector<int> dst_n;        ///< Per-peer: decompose n for dst axis
  std::vector<int> dst_s;        ///< Per-peer: decompose s (start) for dst axis
  size_t src_leading;            ///< Product of dims before src axis
  size_t src_trailing;           ///< Product of dims after src axis
  size_t dst_leading;            ///< Product of dims before dst axis
  size_t dst_trailing;           ///< Product of dims after dst axis
  int src_axis_extent;           ///< Full extent of src partition axis
  int dst_axis_extent;           ///< Full extent of dst partition axis
  bool src_contiguous;           ///< True when src_leading == 1 (no pack needed)
  bool dst_contiguous;           ///< True when dst_leading == 1 (no unpack needed)
  mutable std::vector<MPI_Request> requests;  ///< Pre-allocated MPI request buffer
};

/**
 * @brief Narrow an element count to the int taken by MPI point-to-point calls.
 *
 * The geometry carries counts as size_t, but MPI_Isend/MPI_Irecv take int.
 * A single message above 2^31 elements cannot be expressed, so fail loudly
 * here rather than wrap to a negative count and corrupt the transfer.
 *
 * @param count Element count to narrow
 * @param what Label naming the count, used in the exception message
 * @return @p count as int
 * @throws std::runtime_error when @p count exceeds INT_MAX
 */
inline int checked_mpi_count(std::size_t count, const char *what)
{
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    std::ostringstream msg;
    msg << "ParaFaFT: " << what << " of " << count
        << " elements exceeds the int count MPI point-to-point accepts ("
        << std::numeric_limits<int>::max()
        << "). Use more ranks so each message is smaller.";
    throw std::runtime_error(msg.str());
  }
  return static_cast<int>(count);
}

/**
 * @brief Initialize exchange geometry for a single stage transition.
 *
 * Pre-computes all per-peer decomposition results, counts, displacements,
 * and layout metadata. Called once per transition at construction time.
 *
 * @param geom Geometry struct to populate
 * @param nparts Number of partitions (peers in subcommunicator)
 * @param ndims Number of array dimensions
 * @param src_sizes Source array dimensions
 * @param src_axis Axis partitioned in source layout
 * @param dst_sizes Destination array dimensions
 * @param dst_axis Axis partitioned in destination layout
 */
inline void init_exchange_geometry(
    ExchangeGeometry &geom, int nparts, int ndims,
    const int src_sizes[], int src_axis,
    const int dst_sizes[], int dst_axis)
{
  geom.src_leading = 1;
  geom.src_trailing = 1;
  for (int i = 0; i < src_axis; ++i) geom.src_leading *= src_sizes[i];
  for (int i = src_axis + 1; i < ndims; ++i) geom.src_trailing *= src_sizes[i];

  geom.dst_leading = 1;
  geom.dst_trailing = 1;
  for (int i = 0; i < dst_axis; ++i) geom.dst_leading *= dst_sizes[i];
  for (int i = dst_axis + 1; i < ndims; ++i) geom.dst_trailing *= dst_sizes[i];

  geom.src_contiguous = (geom.src_leading == 1);
  geom.dst_contiguous = (geom.dst_leading == 1);
  geom.src_axis_extent = src_sizes[src_axis];
  geom.dst_axis_extent = dst_sizes[dst_axis];

  geom.src_n.resize(nparts);
  geom.src_s.resize(nparts);
  geom.dst_n.resize(nparts);
  geom.dst_s.resize(nparts);
  geom.send_counts.resize(nparts);
  geom.send_displs.resize(nparts);
  geom.recv_counts.resize(nparts);
  geom.recv_displs.resize(nparts);

  size_t send_offset = 0;
  for (int p = 0; p < nparts; ++p) {
    decompose(src_sizes[src_axis], nparts, p, geom.src_n[p], geom.src_s[p]);
    size_t count = geom.src_leading * geom.src_n[p] * geom.src_trailing;
    geom.send_counts[p] = count;
    geom.send_displs[p] = send_offset;
    send_offset += count;
  }

  size_t recv_offset = 0;
  for (int p = 0; p < nparts; ++p) {
    decompose(dst_sizes[dst_axis], nparts, p, geom.dst_n[p], geom.dst_s[p]);
    size_t count = geom.dst_leading * geom.dst_n[p] * geom.dst_trailing;
    geom.recv_counts[p] = count;
    geom.recv_displs[p] = recv_offset;
    recv_offset += count;
  }

  geom.requests.reserve(2 * (nparts > 0 ? nparts - 1 : 0));
}

/**
 * @brief Exchange per-peer send displacements so each rank learns, for every
 * peer p, the offset of its own block inside peer p's pack buffer.
 *
 * The P2P/IPC exchange reads a peer's pack buffer directly and therefore needs
 * that peer's layout, not the local one. An MPI_Alltoall of send_displs yields
 * exactly remote_send_displs[p] = (peer p's send_displs)[myrank], because
 * MPI_Alltoall delivers into slot p whatever peer p placed in slot myrank.
 *
 * Collective over @p comm; must be called by all ranks in the subcommunicator.
 * A no-op for the packed/alltoallw paths, which never read remote buffers.
 *
 * @param geom Geometry whose send_displs are exchanged; remote_send_displs set
 * @param comm Subcommunicator matching this transition's exchange
 */
inline void init_remote_send_displs(ExchangeGeometry &geom, MPI_Comm comm)
{
  static_assert(sizeof(std::size_t) == 8,
                "remote_send_displs is exchanged as MPI_UINT64_T");
  const int nparts = static_cast<int>(geom.send_displs.size());
  geom.remote_send_displs.resize(nparts);
  MPI_Alltoall(geom.send_displs.data(), 1, MPI_UINT64_T,
               geom.remote_send_displs.data(), 1, MPI_UINT64_T, comm);
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
 * @brief Local exchange for single-rank subcommunicators (nparts == 1).
 *
 * When a subcommunicator has only one rank, no MPI communication is needed.
 * This performs a direct src → dst data movement (possibly with layout
 * transformation) without any synchronization overhead.
 *
 * @tparam BackendT FFT backend type (must provide memcpy, memcpy2d)
 * @param backend Backend instance for async memcpy
 * @param src_array Source array
 * @param dst_array Destination array
 * @param pack_buf Temporary buffer (used only when both src and dst are strided)
 * @param geom Pre-computed exchange geometry for this transition
 */
template <typename BackendT>
inline void exchange_local(
    const BackendT &backend,
    typename BackendT::Complex *src_array,
    typename BackendT::Complex *dst_array,
    typename BackendT::Complex *pack_buf,
    const ExchangeGeometry &geom)
{
  using Complex = typename BackendT::Complex;
  size_t elem_size = sizeof(Complex);
  size_t total_bytes = static_cast<size_t>(geom.send_counts[0]) * elem_size;

  if (geom.src_contiguous && geom.dst_contiguous) {
    // Both contiguous: single memcpy
    backend.memcpy(dst_array, src_array, total_bytes);
  } else if (geom.dst_contiguous) {
    // Strided src → contiguous dst: single memcpy2d
    size_t src_spitch = static_cast<size_t>(geom.src_axis_extent) * geom.src_trailing * elem_size;
    size_t row_bytes = static_cast<size_t>(geom.src_n[0]) * geom.src_trailing * elem_size;
    // See the identity note below: at nparts == 1 the source stride vanishes
    // and this is a linear copy, which must not go through the 2D path.
    if (row_bytes == src_spitch) {
      if (dst_array != src_array) {
        backend.memcpy(dst_array, src_array,
                       row_bytes * static_cast<size_t>(geom.src_leading));
      }
      return;
    }
    backend.memcpy2d(dst_array, row_bytes, src_array, src_spitch,
                     row_bytes, geom.src_leading);
  } else if (geom.src_contiguous) {
    // Contiguous src → strided dst: single memcpy2d
    size_t dst_dpitch = static_cast<size_t>(geom.dst_axis_extent) * geom.dst_trailing * elem_size;
    size_t row_bytes = static_cast<size_t>(geom.dst_n[0]) * geom.dst_trailing * elem_size;
    if (row_bytes == dst_dpitch) {
      if (dst_array != src_array) {
        backend.memcpy(dst_array, src_array,
                       row_bytes * static_cast<size_t>(geom.dst_leading));
      }
      return;
    }
    backend.memcpy2d(dst_array, dst_dpitch, src_array, row_bytes,
                     row_bytes, geom.dst_leading);
  } else {
    // Both non-contiguous: pack src → pack_buf, then unpack pack_buf → dst
    // (No sync needed between — both are on the same stream)
    size_t src_spitch = static_cast<size_t>(geom.src_axis_extent) * geom.src_trailing * elem_size;
    size_t src_row_bytes = static_cast<size_t>(geom.src_n[0]) * geom.src_trailing * elem_size;
    size_t dst_dpitch = static_cast<size_t>(geom.dst_axis_extent) * geom.dst_trailing * elem_size;
    size_t dst_row_bytes = static_cast<size_t>(geom.dst_n[0]) * geom.dst_trailing * elem_size;

    // Identity fast path. The sole partition owns the whole axis, so
    // src_n[0] == src_axis_extent (and likewise for dst) and both memcpy2d
    // calls degenerate to width == pitch, i.e. linear copies. Packing and
    // unpacking then just moves the array to pack_buf and straight back:
    // element order is unchanged, so the pair reduces to one flat copy.
    //
    // Skipping them is not only faster — it is required for large grids.
    // A degenerate 2D copy spanning the full array exceeds the driver's 2D
    // transfer limit once the array passes 4 GiB (3D R2C double at N=1024 is
    // ~8.6 GB), which the driver rejects with "invalid argument".
    size_t src_total = src_row_bytes * static_cast<size_t>(geom.src_leading);
    size_t dst_total = dst_row_bytes * static_cast<size_t>(geom.dst_leading);
    if (src_row_bytes == src_spitch && dst_row_bytes == dst_dpitch &&
        src_total == dst_total) {
      if (dst_array != src_array) {
        backend.memcpy(dst_array, src_array, src_total);
      }
      return;
    }

    backend.memcpy2d(pack_buf, src_row_bytes, src_array, src_spitch,
                     src_row_bytes, geom.src_leading);
    backend.memcpy2d(dst_array, dst_dpitch, pack_buf, dst_row_bytes,
                     dst_row_bytes, geom.dst_leading);
  }
}

/**
 * @brief Packed exchange using pre-computed geometry.
 *
 * Replaces MPI_Alltoallw with derived subarray types by explicitly packing
 * non-contiguous data into contiguous GPU buffers, using MPI point-to-point
 * (which supports GPU-direct for contiguous data), then unpacking.
 *
 * Uses pre-computed ExchangeGeometry to avoid per-call heap allocations
 * and redundant decompose() calls.
 *
 * @tparam BackendT FFT backend type (must provide memcpy2d)
 * @param backend Backend instance for async memcpy
 * @param comm MPI subcommunicator for the redistribution
 * @param nparts Number of partitions in the subcommunicator
 * @param src_array Source array (data to send)
 * @param dst_array Destination array (will contain received data)
 * @param pack_buf Temporary buffer for packing (size >= total elements)
 * @param geom Pre-computed exchange geometry for this transition
 */
template <typename BackendT>
inline void exchange_packed(
    const BackendT &backend,
    MPI_Comm comm, int nparts,
    typename BackendT::Complex *src_array,
    typename BackendT::Complex *dst_array,
    typename BackendT::Complex *pack_buf,
    ExchangeGeometry &geom)
{
  using Complex = typename BackendT::Complex;
  size_t elem_size = sizeof(Complex);

  // Pack src_array into pack_buf (skip when src is already contiguous)
  // Pack is async on the backend stream, ordered after any preceding FFT.
  if (!geom.src_contiguous) {
    size_t src_spitch = static_cast<size_t>(geom.src_axis_extent) * geom.src_trailing * elem_size;
    for (int p = 0; p < nparts; ++p) {
      size_t row_bytes = static_cast<size_t>(geom.src_n[p]) * geom.src_trailing * elem_size;
      backend.memcpy2d(
          reinterpret_cast<char *>(pack_buf) + static_cast<size_t>(geom.send_displs[p]) * elem_size,
          row_bytes,
          reinterpret_cast<char *>(src_array) + static_cast<size_t>(geom.src_s[p]) * geom.src_trailing * elem_size,
          src_spitch,
          row_bytes,
          geom.src_leading);
    }
  }

  // Choose send/recv buffer pointers:
  //   send_from: pack_buf if we packed, src_array if src is contiguous
  //   recv_into: dst_array if dst is contiguous, pack_buf if src was
  //              contiguous (pack_buf is free), src_array otherwise
  //              (src_array is free after packing)
  Complex *send_from = geom.src_contiguous ? src_array : pack_buf;
  Complex *recv_into;
  if (geom.dst_contiguous) {
    recv_into = dst_array;
  } else if (geom.src_contiguous) {
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
  geom.requests.clear();

  const MPI_Datatype mpi_dtype = mpi_complex_type<typename BackendT::FloatType>();
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank) continue;
    size_t recv_byte_off = static_cast<size_t>(geom.recv_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Irecv(reinterpret_cast<char *>(recv_into) + recv_byte_off,
              checked_mpi_count(geom.recv_counts[p], "recv count"),
              mpi_dtype, p, 0, comm, &req);
    geom.requests.push_back(req);
  }
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank) continue;
    size_t send_byte_off = static_cast<size_t>(geom.send_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Isend(reinterpret_cast<char *>(send_from) + send_byte_off,
              checked_mpi_count(geom.send_counts[p], "send count"),
              mpi_dtype, p, 0, comm, &req);
    geom.requests.push_back(req);
  }

  // Self-copy: async on GPU stream, overlaps with network transfers
  {
    size_t send_byte_off = static_cast<size_t>(geom.send_displs[myrank]) * elem_size;
    size_t recv_byte_off = static_cast<size_t>(geom.recv_displs[myrank]) * elem_size;
    backend.memcpy(
        reinterpret_cast<char *>(recv_into) + recv_byte_off,
        reinterpret_cast<char *>(send_from) + send_byte_off,
        static_cast<size_t>(geom.send_counts[myrank]) * elem_size);
  }

  // Wait for all remote transfers to complete
  if (!geom.requests.empty()) {
    MPI_Waitall(static_cast<int>(geom.requests.size()), geom.requests.data(),
                MPI_STATUSES_IGNORE);
  }

  // Unpack recv_into into dst_array (skip when dst is already contiguous)
  // Async on stream, enqueued after MPI_Waitall guarantees data is valid.
  if (!geom.dst_contiguous) {
    size_t dst_dpitch = static_cast<size_t>(geom.dst_axis_extent) * geom.dst_trailing * elem_size;
    for (int p = 0; p < nparts; ++p) {
      size_t row_bytes = static_cast<size_t>(geom.dst_n[p]) * geom.dst_trailing * elem_size;
      backend.memcpy2d(
          reinterpret_cast<char *>(dst_array) + static_cast<size_t>(geom.dst_s[p]) * geom.dst_trailing * elem_size,
          dst_dpitch,
          reinterpret_cast<char *>(recv_into) + static_cast<size_t>(geom.recv_displs[p]) * elem_size,
          row_bytes,
          row_bytes,
          geom.dst_leading);
    }
  }
}

/**
 * @brief Hybrid exchange using pre-computed geometry.
 *
 * Mixes P2P direct GPU copies with MPI point-to-point, using pre-computed
 * ExchangeGeometry to avoid per-call allocations.
 *
 * For neighbours where P2P (IPC) is available, uses direct GPU-to-GPU memory
 * copies via IPC-mapped pointers. For non-P2P neighbours, falls back to MPI
 * Isend/Irecv. Two MPI_Barrier calls synchronize the P2P pack/read phases;
 * MPI transfers overlap with the P2P reads between the barriers.
 *
 * Optimization: when src data is already contiguous (src_leading == 1),
 * uses a single memcpy into pack_buf instead of nparts strided 2D copies.
 *
 * @tparam BackendT FFT backend type
 * @param backend Backend instance for async memcpy
 * @param comm MPI subcommunicator
 * @param nparts Number of partitions
 * @param src_array Source array
 * @param dst_array Destination array
 * @param pack_buf Local pack buffer (IPC-exported)
 * @param remote_pack_ptrs IPC-mapped pointers (non-null for P2P neighbours)
 * @param peer_enabled Per-neighbour flags: nonzero = P2P, zero = MPI
 * @param geom Pre-computed exchange geometry for this transition
 */
template <typename BackendT>
inline void exchange_hybrid(
    const BackendT &backend,
    MPI_Comm comm, int nparts,
    typename BackendT::Complex *src_array,
    typename BackendT::Complex *dst_array,
    typename BackendT::Complex *pack_buf,
    void *const *remote_pack_ptrs,
    const char *peer_enabled,
    bool phased,
    ExchangeGeometry &geom)
{
  using Complex = typename BackendT::Complex;
  size_t elem_size = sizeof(Complex);

  // Pack into pack_buf: P2P neighbours read via IPC handle,
  // MPI neighbours send from pack_buf via Isend.
  if (!geom.src_contiguous) {
    size_t src_spitch = static_cast<size_t>(geom.src_axis_extent) * geom.src_trailing * elem_size;
    for (int p = 0; p < nparts; ++p) {
      size_t row_bytes = static_cast<size_t>(geom.src_n[p]) * geom.src_trailing * elem_size;
      backend.memcpy2d(
          reinterpret_cast<char *>(pack_buf) + static_cast<size_t>(geom.send_displs[p]) * elem_size,
          row_bytes,
          reinterpret_cast<char *>(src_array) + static_cast<size_t>(geom.src_s[p]) * geom.src_trailing * elem_size,
          src_spitch,
          row_bytes,
          geom.src_leading);
    }
  } else {
    // src is contiguous — single memcpy into pack_buf
    size_t total_bytes = (static_cast<size_t>(geom.send_displs[nparts - 1]) +
                          geom.send_counts[nparts - 1]) * elem_size;
    backend.memcpy(pack_buf, src_array, total_bytes);
  }

  // Sync: ensure pack data is committed before MPI reads or P2P reads
  backend.sync();

  // Barrier: all ranks must finish packing before P2P reads begin
  MPI_Barrier(comm);

  // Choose recv buffer
  Complex *recv_into = geom.dst_contiguous ? dst_array : src_array;

  int myrank;
  MPI_Comm_rank(comm, &myrank);

  // Post MPI Irecv/Isend for non-P2P neighbours
  const MPI_Datatype mpi_dtype = mpi_complex_type<typename BackendT::FloatType>();
  geom.requests.clear();
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank || peer_enabled[p]) continue;
    size_t recv_byte_off = static_cast<size_t>(geom.recv_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Irecv(reinterpret_cast<char *>(recv_into) + recv_byte_off,
              checked_mpi_count(geom.recv_counts[p], "recv count"),
              mpi_dtype, p, 0, comm, &req);
    geom.requests.push_back(req);
  }
  for (int p = 0; p < nparts; ++p) {
    if (p == myrank || peer_enabled[p]) continue;
    size_t send_byte_off = static_cast<size_t>(geom.send_displs[p]) * elem_size;
    MPI_Request req;
    MPI_Isend(reinterpret_cast<char *>(pack_buf) + send_byte_off,
              checked_mpi_count(geom.send_counts[p], "send count"),
              mpi_dtype, p, 0, comm, &req);
    geom.requests.push_back(req);
  }

  // Self-copy from local pack buffer (no PCIe traffic, all ranks)
  {
    size_t send_byte_off = static_cast<size_t>(geom.send_displs[myrank]) * elem_size;
    size_t recv_byte_off = static_cast<size_t>(geom.recv_displs[myrank]) * elem_size;
    backend.memcpy(
        reinterpret_cast<char *>(recv_into) + recv_byte_off,
        reinterpret_cast<char *>(pack_buf) + send_byte_off,
        static_cast<size_t>(geom.send_counts[myrank]) * elem_size);
  }

  // P2P read schedule.
  //
  // On a shared PCIe switch, simultaneous bidirectional reads between two
  // GPUs degrade throughput by 10x+ non-deterministically.  Where that
  // applies (phased == true) the directed reads are split into two phases
  // such that no bidirectional pair (i reads j AND j reads i) appears in the
  // same phase.
  //
  // Assignment rule: for each pair {a,b} with a < b, the reader in
  // phase 0 is the lower rank when (a+b) is even, the higher rank
  // when (a+b) is odd.  Phase 1 gets the reverse direction.  This
  // balances the load: each rank does ceil(npeers/2) reads per phase.
  //
  // Cost: 2 sequential phases of ceil((nparts-1)/2) × copy_time each,
  // plus one sync+barrier between phases.  For N=8 GPUs with 12 ms
  // per copy this is 2 × 4 × 12 ms = 96 ms, vs 672 ms fully serial.
  //
  // On a full-duplex fabric (NVLink/NVSwitch) that contention does not
  // exist, so phased == false issues every read in one pass and skips the
  // intervening fence: the reads then overlap instead of running as two
  // dependent halves, and one sync + one collective barrier disappear from
  // the critical path of every exchange.
  const int nphases = phased ? 2 : 1;
  for (int phase = 0; phase < nphases; ++phase) {
    for (int p = 0; p < nparts; ++p) {
      if (p == myrank || !peer_enabled[p]) continue;
      if (phased) {
        int lo = (myrank < p) ? myrank : p;
        int hi = (myrank < p) ? p : myrank;
        // Lower rank reads first when (lo+hi) is even; higher reads first
        // when odd.  XOR with phase to swap directions in phase 1.
        bool lower_reads = (((lo + hi) % 2 == 0) != (phase == 1));
        bool i_read = (myrank == lo) ? lower_reads : !lower_reads;
        if (!i_read) continue;
      }

      size_t recv_byte_off = static_cast<size_t>(geom.recv_displs[p]) * elem_size;
      // Index the *peer's* pack buffer with the peer's own displacement of our
      // block (remote_send_displs[p] == peer p's send_displs[myrank]). Using the
      // local send_displs[myrank] is wrong when src_trailing differs per rank.
      size_t remote_off = static_cast<size_t>(geom.remote_send_displs[p]) * elem_size;
      char *remote_buf = reinterpret_cast<char *>(remote_pack_ptrs[p]);
      backend.memcpy(
          reinterpret_cast<char *>(recv_into) + recv_byte_off,
          remote_buf + remote_off,
          static_cast<size_t>(geom.recv_counts[p]) * elem_size);
    }
    // Sync + barrier between phases so phase-0 reads complete before
    // phase-1 reads begin (avoids bidirectional traffic on the switch).
    // After the last phase the outer sync + barrier handles completion.
    if (phased && phase == 0) {
      backend.sync();
      MPI_Barrier(comm);
    }
  }

  // Sync stream: ensure P2P reads and self-copy complete
  backend.sync();

  // Wait for MPI transfers to complete
  if (!geom.requests.empty()) {
    MPI_Waitall(static_cast<int>(geom.requests.size()), geom.requests.data(),
                MPI_STATUSES_IGNORE);
  }

  // Barrier: all ranks must finish reading before pack_buf can be reused
  MPI_Barrier(comm);

  // Unpack recv_into into dst_array (skip when dst is already contiguous)
  if (!geom.dst_contiguous) {
    size_t dst_dpitch = static_cast<size_t>(geom.dst_axis_extent) * geom.dst_trailing * elem_size;
    for (int p = 0; p < nparts; ++p) {
      size_t row_bytes = static_cast<size_t>(geom.dst_n[p]) * geom.dst_trailing * elem_size;
      backend.memcpy2d(
          reinterpret_cast<char *>(dst_array) + static_cast<size_t>(geom.dst_s[p]) * geom.dst_trailing * elem_size,
          dst_dpitch,
          reinterpret_cast<char *>(recv_into) + static_cast<size_t>(geom.recv_displs[p]) * elem_size,
          row_bytes,
          row_bytes,
          geom.dst_leading);
    }
  }
}

} // namespace parafaft

#endif // PARAFAFT_COMMON_HPP
