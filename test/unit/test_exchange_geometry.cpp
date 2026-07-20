// Unit tests for exchange geometry arithmetic in parafaft_common.hpp.
//
// These are host-only and need no GPU and no live communicator: they cover the
// pure arithmetic that is otherwise only reachable at grid sizes far too large
// to run in a test (the 64-bit displacement paths need >2^31 elements per
// rank, which is tens of GB of real data).

#include "../../parafaft_common.hpp"

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::cout << "  " << what << (ok ? " [PASS]" : " [FAIL]") << std::endl;
  if (!ok)
    ++failures;
}

// checked_mpi_count() must pass small counts through untouched and refuse
// anything MPI's int count cannot represent, rather than wrapping negative.
void test_checked_mpi_count() {
  std::cout << "checked_mpi_count:" << std::endl;

  check(parafaft::checked_mpi_count(0, "zero") == 0, "zero passes through");
  check(parafaft::checked_mpi_count(12345, "small") == 12345,
        "small count passes through");

  const std::size_t int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  check(parafaft::checked_mpi_count(int_max, "max") ==
            std::numeric_limits<int>::max(),
        "INT_MAX is accepted");

  bool threw = false;
  try {
    parafaft::checked_mpi_count(int_max + 1, "overflow");
  } catch (const std::runtime_error &) {
    threw = true;
  }
  check(threw, "INT_MAX + 1 throws instead of wrapping");
}

// The regression this guards: send_displs accumulates to the full local
// element count. Stored as int it wrapped negative past 2^31, silently
// corrupting every offset derived from it.
void test_large_geometry_displacements() {
  std::cout << "init_exchange_geometry (>2^31 elements):" << std::endl;

  // 3D, split along the last axis across 2 parts. Per-part element count is
  // 2048 * 2048 * 1024 = 2^32, i.e. twice what an int can hold.
  const int sizes[3] = {2048, 2048, 2048};
  const int nparts = 2;
  parafaft::ExchangeGeometry geom;
  parafaft::init_exchange_geometry(geom, nparts, 3, sizes, 2, sizes, 1);

  const std::size_t expected_count =
      static_cast<std::size_t>(2048) * 2048 * 1024;
  const std::size_t int_max =
      static_cast<std::size_t>(std::numeric_limits<int>::max());

  check(geom.send_counts[0] > int_max, "send count exceeds INT_MAX (precondition)");
  check(geom.send_counts[0] == expected_count, "send count is exact");
  check(geom.send_displs[0] == 0, "first send displacement is zero");
  check(geom.send_displs[1] == expected_count,
        "second send displacement did not wrap");
  check(geom.recv_displs[1] > int_max, "recv displacement exceeds INT_MAX");

  // A count this large cannot be sent as a single MPI message; the guard must
  // say so rather than let a negative count reach MPI.
  bool threw = false;
  try {
    parafaft::checked_mpi_count(geom.send_counts[0], "send count");
  } catch (const std::runtime_error &) {
    threw = true;
  }
  check(threw, "oversized send count is rejected at the MPI boundary");
}

// exchange_local's identity fast path triggers when the single partition spans
// the whole axis, which is exactly what decompose() yields at nparts == 1.
// Verify that precondition holds so the fast path is reachable.
void test_single_partition_is_identity() {
  std::cout << "exchange_local identity precondition (nparts == 1):" << std::endl;

  const int sizes[3] = {8, 16, 32};
  parafaft::ExchangeGeometry geom;
  parafaft::init_exchange_geometry(geom, 1, 3, sizes, 2, sizes, 1);

  check(geom.src_n[0] == geom.src_axis_extent,
        "sole partition spans the whole src axis");
  check(geom.dst_n[0] == geom.dst_axis_extent,
        "sole partition spans the whole dst axis");

  // Both copies degenerate to width == pitch, so the pack/unpack pair is a
  // linear round trip and the totals must agree.
  const std::size_t src_total = static_cast<std::size_t>(geom.src_n[0]) *
                                geom.src_trailing * geom.src_leading;
  const std::size_t dst_total = static_cast<std::size_t>(geom.dst_n[0]) *
                                geom.dst_trailing * geom.dst_leading;
  check(src_total == dst_total, "src and dst totals agree");
}

// choose_dims must align the rank-contiguous (last) grid dimension with the
// node size whenever the job spans several nodes, and must leave
// MPI_Dims_create's choice alone otherwise.
void test_choose_dims() {
  std::cout << "choose_dims:" << std::endl;

  auto dims_for = [](int nranks, int ndims, int ppn) {
    std::vector<int> dims(ndims, 0);
    parafaft::choose_dims(nranks, ndims, ppn, dims.data());
    return dims;
  };

  // The motivating case: 8 ranks on 4-rank nodes. MPI_Dims_create alone gives
  // {4,2}, whose size-4 dimension straddles both nodes.
  std::vector<int> d = dims_for(8, 2, 4);
  check(d[0] == 2 && d[1] == 4, "8 ranks / 4 per node -> {2,4}");

  d = dims_for(16, 2, 4);
  check(d[0] == 4 && d[1] == 4, "16 ranks / 4 per node -> {4,4}");

  d = dims_for(32, 2, 4);
  check(d[0] == 8 && d[1] == 4, "32 ranks / 4 per node -> {8,4}");

  // Pure permutation could not reach this one; the trailing dimension has to
  // be constructed as the node size, not picked from Dims_create's factors.
  d = dims_for(64, 2, 4);
  check(d[0] == 16 && d[1] == 4, "64 ranks / 4 per node -> {16,4}");

  // 4D: leading dimensions are balanced across nodes, trailing is the node.
  d = dims_for(16, 3, 4);
  check(d[0] * d[1] == 4 && d[2] == 4, "16 ranks / 4 per node, 3 grid dims");

  // Single-node jobs are topologically uniform — leave them untouched.
  std::vector<int> single = dims_for(4, 2, 4);
  std::vector<int> reference(2, 0);
  MPI_Dims_create(4, 2, reference.data());
  check(single == reference, "single-node job keeps MPI_Dims_create layout");

  // Disabled / unknown topology falls back to MPI_Dims_create.
  std::vector<int> disabled = dims_for(8, 2, 1);
  reference.assign(2, 0);
  MPI_Dims_create(8, 2, reference.data());
  check(disabled == reference, "ppn == 1 falls back to MPI_Dims_create");

  // Ranks not divisible by node size: no trailing dimension fits every node.
  std::vector<int> ragged = dims_for(6, 2, 4);
  reference.assign(2, 0);
  MPI_Dims_create(6, 2, reference.data());
  check(ragged == reference, "indivisible rank count falls back");

  // Whatever is chosen must still be a factorization of the rank count.
  bool all_product_ok = true;
  const int ppns[] = {1, 2, 4, 8};
  for (int ndims = 2; ndims <= 4; ++ndims) {
    for (int nranks = 1; nranks <= 64; ++nranks) {
      for (int ppn : ppns) {
        std::vector<int> g = dims_for(nranks, ndims, ppn);
        int product = 1;
        for (int v : g)
          product *= v;
        if (product != nranks)
          all_product_ok = false;
      }
    }
  }
  check(all_product_ok, "dims always multiply to the rank count (1..64)");
}

} // namespace

int main(int argc, char **argv) {
  std::cout << "########################################" << std::endl;
  std::cout << "# TEST: unit/exchange_geometry" << std::endl;
  std::cout << "########################################" << std::endl;

  MPI_Init(&argc, &argv);

  test_checked_mpi_count();
  test_large_geometry_displacements();
  test_single_partition_is_identity();
  test_choose_dims();

  std::cout << "Total failures: " << failures << std::endl;

  MPI_Finalize();
  return failures;
}
