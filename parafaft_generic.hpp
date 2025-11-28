#ifndef PARAFAFT_GENERIC_HPP
#define PARAFAFT_GENERIC_HPP

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

#include <mpi.h>
#include "backend/fftw3/fft_backend_fftw.hpp"
#include <vector>
#include <complex>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace parafaft {

// ============================================================================
// Balanced Block-Contiguous Decomposition
// Reference: Paper Equation (9) and Section 2.2
// ============================================================================
//
// Distributes N elements across M processors using Barry Smith's formula.
// Ensures load balance: processors either get q or q+1 elements where q = N/M.
//
// Parameters:
//   N - Total number of elements to distribute
//   M - Number of processors
//   p - Processor rank (0 <= p < M)
//   n - [OUTPUT] Number of elements assigned to processor p
//   s - [OUTPUT] Starting index for processor p in global array
//
// Paper Reference: Equation (9) on page 140
//
inline void decompose(int N, int M, int p, int& n, int& s) {
    int q = N / M;  // Base number of elements per processor
    int r = N % M;  // Remainder elements to distribute

    // Processors 0..r-1 get q+1 elements, processors r..M-1 get q elements
    n = (r > p) ? (q + 1) : q;
    s = (r > p) ? (n * p) : (q * p + r);
}

// ============================================================================
// MPI Subarray Datatype Creation
// Reference: Paper Section 2.3 and Algorithm 1
// ============================================================================
//
// Creates MPI subarray datatypes for partitioning a D-dimensional array
// along a specific axis. This is the key to avoiding local transposes.
//
// Paper Context (Section 2.3):
//   Traditional methods:
//     1. Local transpose to make data contiguous
//     2. MPI_Alltoall with contiguous buffers
//     3. Local transpose again
//
//   This method (Algorithm 1):
//     1. Create MPI_Type_create_subarray describing discontiguous layout
//     2. MPI_Alltoallw handles the complexity in MPI layer
//     3. NO local transposes needed!
//
// Parameters:
//   datatype - Base MPI datatype (e.g., MPI_C_DOUBLE_COMPLEX)
//   ndims    - Number of dimensions
//   sizes    - Size of the array in each dimension
//   axis     - Axis along which to partition
//   nparts   - Number of partitions (typically number of processors)
//   subarrays - [OUTPUT] Array of nparts MPI datatypes
//
inline void subarray(MPI_Datatype datatype, int ndims, const int sizes[], int axis,
                     int nparts, MPI_Datatype subarrays[]) {
    std::vector<int> subsizes(ndims), substarts(ndims);

    // Initialize: full size in all dimensions, starting at origin
    for (int i = 0; i < ndims; i++) {
        subsizes[i] = sizes[i];
        substarts[i] = 0;
    }

    // Create one subarray datatype for each partition
    for (int p = 0; p < nparts; p++) {
        int n, s;
        decompose(sizes[axis], nparts, p, n, s);  // Use Equation (9)
        subsizes[axis] = n;    // Size of partition p along axis
        substarts[axis] = s;   // Starting position of partition p

        // MPI_Type_create_subarray describes the discontiguous memory layout
        // This is the key innovation - no need to pack/unpack data manually
        MPI_Type_create_subarray(ndims, sizes, subsizes.data(), substarts.data(),
                                MPI_ORDER_C, datatype, &subarrays[p]);
        MPI_Type_commit(&subarrays[p]);
    }
}

// ============================================================================
// Global Data Redistribution (Exchange Operation)
// Reference: Paper Algorithm 1 and Section 2.3
// ============================================================================
//
// Redistributes data between two pencil decomposition stages.
// Changes which axis is local vs distributed without any local transposes.
//
// Paper Context (Algorithm 1, lines 6-8):
//   This implements the "global redistribution" step between FFT stages.
//   In stage i: data is distributed along certain axes
//   In stage i+1: data is distributed along different axes
//   This function performs the all-to-all communication to rearrange data.
//
// Example (3D case from paper Section 3.1, Figure 1):
//   Before: Array distributed in X,Y; local in Z (pencils along Z)
//   After:  Array distributed in X,Z; local in Y (pencils along Y)
//
// The key insight: MPI_Alltoallw with subarray datatypes handles the
// complex index mapping automatically. No manual packing/unpacking!
//
// Parameters:
//   comm     - MPI subcommunicator for the redistribution
//   datatype - Base MPI datatype
//   ndims    - Number of dimensions
//   sizesA   - Array dimensions before redistribution
//   arrayA   - Input array (distributed along axisA)
//   axisA    - Axis that is distributed in the input
//   sizesB   - Array dimensions after redistribution
//   arrayB   - Output array (will be distributed along axisB)
//   axisB    - Axis that will be distributed in the output
//
inline void exchange(MPI_Comm comm, MPI_Datatype datatype, int ndims,
                     const int sizesA[], void* arrayA, int axisA,
                     const int sizesB[], void* arrayB, int axisB) {
    int nparts;
    MPI_Comm_size(comm, &nparts);

    // Create subarray datatypes for both source and destination layouts
    std::vector<MPI_Datatype> subarraysA(nparts), subarraysB(nparts);
    subarray(datatype, ndims, sizesA, axisA, nparts, subarraysA.data());
    subarray(datatype, ndims, sizesB, axisB, nparts, subarraysB.data());

    // MPI_Alltoallw parameters: each process sends/receives one subarray
    std::vector<int> counts(nparts, 1), displs(nparts, 0);

    // This is THE key operation (Paper Algorithm 1, line 7):
    // All-to-all exchange using derived datatypes
    // Handles all the complex index calculations internally
    MPI_Alltoallw(arrayA, counts.data(), displs.data(), subarraysA.data(),
                  arrayB, counts.data(), displs.data(), subarraysB.data(), comm);

    // Clean up datatypes
    for (int p = 0; p < nparts; p++) {
        MPI_Type_free(&subarraysA[p]);
        MPI_Type_free(&subarraysB[p]);
    }
}

// ============================================================================
// Generic D-dimensional Pencil FFT Class
// Reference: Paper Algorithm 1 and Section 3 (Pencil Decomposition)
// ============================================================================
//
// PENCIL DECOMPOSITION STRATEGY (Paper Section 3):
//
// For D dimensions, we use a (D-1)-dimensional processor grid.
// Example for 3D with 2D processor grid [P0, P1]:
//
//   Stage A: axes [0,1] distributed, axis 2 local
//            → FFT along axis 2
//            → Redistribute: axis 2 becomes distributed, axis 1 becomes local
//
//   Stage B: axes [0,2] distributed, axis 1 local
//            → FFT along axis 1
//            → Redistribute: axis 1 becomes distributed, axis 0 becomes local
//
//   Stage C: axes [1,2] distributed, axis 0 local
//            → FFT along axis 0
//            → Done! (no redistribution needed after last FFT)
//
// Total: D FFT operations + (D-1) global redistributions
//
// GENERALIZATION TO ARBITRARY D:
//
// This implementation extends the 3D/4D cases from the paper to work for
// any dimension D. The stage progression follows the pattern:
//   - Process axes from last to first: [D-1, D-2, ..., 1, 0]
//   - At stage k, axis (D-1-k) becomes fully local for FFT
//   - After FFT, redistribute to make next axis local
//
// MEMORY LAYOUT (Paper Section 2.1):
//
// Data is stored in C/row-major order (last index varies fastest).
// Each processor holds a contiguous block of the D-dimensional array.
// No explicit pencil data structure - just a flat complex array with
// implicit D-dimensional indexing via stage_shapes_.
//
// ============================================================================
template<int D, typename Backend = FFTWBackend>
class ParaFaFT {
public:
    using Complex = std::complex<double>;

    // ========================================================================
    // Constructor: Setup Processor Grid and Pencil Decomposition
    // Reference: Paper Section 2.2 (Cartesian Topology) and Section 3
    // ========================================================================
    //
    // Paper Algorithm 1, lines 1-3: Initialize processor topology
    //
    // For D dimensions, we create a (D-1)-dimensional Cartesian processor grid.
    // This allows us to distribute (D-1) axes while keeping one axis fully local.
    //
    // Example for 3D with 4 processors:
    //   MPI_Dims_create(4, 2, dims) might give dims = [2, 2]
    //   Creates a 2×2 grid of processors
    //
    // Example for 4D with 8 processors:
    //   MPI_Dims_create(8, 3, dims) might give dims = [2, 2, 2]
    //   Creates a 2×2×2 grid of processors
    //
    ParaFaFT(const int global_shape[D], MPI_Comm comm = MPI_COMM_WORLD)
        : comm_world_(comm), backend_(D) {

        MPI_Comm_rank(comm_world_, &rank_);
        MPI_Comm_size(comm_world_, &size_);

        // Store global array dimensions
        for (int i = 0; i < D; ++i) {
            global_shape_[i] = global_shape[i];
        }

        // Create (D-1)-dimensional Cartesian processor grid
        // Paper Section 2.2: This topology enables efficient subcommunicators
        std::vector<int> dims(D - 1, 0);
        MPI_Dims_create(size_, D - 1, dims.data());  // Balanced grid dimensions

        for (int i = 0; i < D - 1; ++i) {
            dims_[i] = dims[i];
        }

        // Create Cartesian communicator (non-periodic)
        std::vector<int> periods(D - 1, 0);
        MPI_Cart_create(comm_world_, D - 1, dims.data(), periods.data(), 1, &comm_cart_);

        // Get this processor's coordinates in the grid
        // These coordinates determine which data partition this processor owns
        std::vector<int> coords(D - 1);
        MPI_Cart_coords(comm_cart_, rank_, D - 1, coords.data());
        for (int i = 0; i < D - 1; ++i) {
            coords_[i] = coords[i];
        }

        // Create (D-1) subcommunicators, one for each grid dimension
        // Paper Section 2.2: These enable efficient all-to-all communication
        // along specific dimensions during redistribution
        //
        // subcomms_[i] contains all processors that differ only in coordinate i
        // This is used for redistributing along grid dimension i
        subcomms_.resize(D - 1);
        for (int i = 0; i < D - 1; ++i) {
            std::vector<int> remain_dims(D - 1, 0);
            remain_dims[i] = 1;  // Keep only dimension i
            MPI_Cart_sub(comm_cart_, remain_dims.data(), &subcomms_[i]);
        }

        // Compute local array shapes for each stage
        // This determines data distribution at each stage of the algorithm
        setup_stage_shapes();

        // Allocate working arrays for each stage
        // We need D working buffers, one for each FFT stage
        // This avoids in-place transformations and simplifies implementation
        stage_arrays_.resize(D);
        for (int stage = 0; stage < D; ++stage) {
            int size = 1;
            for (int i = 0; i < D; ++i) {
                size *= stage_shapes_[stage][i];
            }
            stage_arrays_[stage].resize(size);
        }

        // Create FFT plans for all stages (after stage arrays are allocated)
        create_backend_plans();
    }

    // ========================================================================
    // Create Backend Plans
    // ========================================================================
    //
    // Create FFT plans for all stages. Must be called after stage arrays
    // are allocated, as plans are bound to data pointers.
    //
    // This function analyzes the memory layout for each stage and creates
    // appropriate plans with correct stride and distance parameters.
    //
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
                backend_.create_stage_plan(stage, length, batch,
                                          stage_arrays_[stage].data(), 1, length);

            } else if (axis == 0) {
                // First axis: strided FFTs
                int batch = 1;
                for (int i = 1; i < D; ++i) {
                    batch *= stage_shapes_[stage][i];
                }
                int stride = batch;
                backend_.create_stage_plan(stage, length, batch,
                                          stage_arrays_[stage].data(), stride, 1);

            } else {
                // Middle axis: need to handle via loops in perform_fft
                // Create plan for trailing batch only
                int trailing_size = 1;
                for (int i = axis + 1; i < D; ++i) {
                    trailing_size *= stage_shapes_[stage][i];
                }
                // Plan will be called multiple times for leading dimensions
                backend_.create_stage_plan(stage, length, trailing_size,
                                          stage_arrays_[stage].data(), trailing_size, 1);
            }
        }
    }

    ~ParaFaFT() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            for (auto& comm : subcomms_) {
                if (comm != MPI_COMM_NULL) {
                    MPI_Comm_free(&comm);
                }
            }
            if (comm_cart_ != MPI_COMM_NULL) {
                MPI_Comm_free(&comm_cart_);
            }
        }
    }

    int get_local_size() const {
        int size = 1;
        for (int i = 0; i < D; ++i) {
            size *= stage_shapes_[0][i];
        }
        return size;
    }

    void get_local_shape(int shape[D]) const {
        for (int i = 0; i < D; ++i) {
            shape[i] = stage_shapes_[0][i];
        }
    }

    void get_global_start(int start[D]) const {
        for (int i = 0; i < D; ++i) {
            start[i] = global_start_[i];
        }
    }

    // Get final shape (after forward FFT, stage D-1 distribution)
    void get_final_shape(int shape[D]) const {
        for (int i = 0; i < D; ++i) {
            shape[i] = stage_shapes_[D - 1][i];
        }
    }

    // Get final global starting indices (after forward FFT, stage D-1 distribution)
    void get_final_start(int start[D]) const {
        // For final stage, need to compute based on which axes are distributed
        // Stage D-1: axes [D-1, ..., 1] have been processed, axis 0 is fully local
        // Axes [1, 2, ..., D-1] are distributed using grid dimensions [0, 1, ..., D-2]
        start[0] = 0;  // Axis 0 is not distributed in final stage
        for (int i = 1; i < D; ++i) {
            // Axis i is distributed on grid dimension i-1
            int n, s;
            decompose(global_shape_[i], dims_[i-1], coords_[i-1], n, s);
            start[i] = s;
        }
    }

    // Debug: print stage shapes
    void print_stage_shapes() const {
        if (rank_ == 0) {
            for (int stage = 0; stage < D; ++stage) {
                std::cout << "Stage " << stage << ": [";
                for (int i = 0; i < D; ++i) {
                    std::cout << stage_shapes_[stage][i];
                    if (i < D - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }
    }

    // ========================================================================
    // Forward FFT Transform
    // Reference: Paper Algorithm 1 (Main Loop)
    // ========================================================================
    //
    // Implements Algorithm 1 from the paper, lines 4-9.
    //
    // ALGORITHM OVERVIEW:
    // for each stage k from 0 to D-1:
    //     1. FFT along the local axis (axis D-1-k)
    //     2. If not last stage: redistribute to make next axis local
    //
    // EXAMPLE (3D case from paper Section 3.1):
    //   Stage 0: FFT along Z (local), redistribute Z→Y
    //   Stage 1: FFT along Y (local), redistribute Y→X
    //   Stage 2: FFT along X (local), done
    //
    // DATA FLOW:
    //   Input:  data in stage 0 distribution (axes [0..D-2] distributed, axis D-1 local)
    //   Output: data in stage D-1 distribution (axes [1..D-1] distributed, axis 0 local)
    //
    // NOTE: User must normalize by 1/N after calling this function for correct FFT
    //
    void forward(Complex* data) {
        // Initialize working arrays (avoid contamination from previous runs)
        for (auto& arr : stage_arrays_) {
            std::fill(arr.begin(), arr.end(), Complex(0.0, 0.0));
        }

        // Copy input data to stage 0 working array
        int size0 = 1;
        for (int i = 0; i < D; ++i) size0 *= stage_shapes_[0][i];
        std::memcpy(stage_arrays_[0].data(), data, size0 * sizeof(Complex));

        // Paper Algorithm 1, lines 4-9: Main loop over dimensions
        // Perform D 1D-FFTs with (D-1) global redistributions
        for (int stage = 0; stage < D; ++stage) {
            int axis = D - 1 - stage;  // Process axes from last to first

            // Paper Algorithm 1, line 5: Local FFT
            // This is a 1D FFT along the axis that is currently fully local
            perform_fft(stage, axis, FFTDirection::Forward);

            // Paper Algorithm 1, lines 6-8: Global redistribution
            // Make the next axis local for the next FFT stage
            if (stage < D - 1) {
                int next_axis = D - 2 - stage;  // Next axis to become local
                exchange(subcomms_[next_axis], MPI_C_DOUBLE_COMPLEX, D,
                        stage_shapes_[stage].data(), stage_arrays_[stage].data(), axis,
                        stage_shapes_[stage + 1].data(), stage_arrays_[stage + 1].data(), next_axis);
            }
        }

        // Copy result from final stage back to user's array
        int sizeD = 1;
        for (int i = 0; i < D; ++i) sizeD *= stage_shapes_[D - 1][i];
        std::memcpy(data, stage_arrays_[D - 1].data(), sizeD * sizeof(Complex));
    }

    // ========================================================================
    // Backward (Inverse) FFT Transform
    // Reference: Paper Algorithm 1 (Reversed)
    // ========================================================================
    //
    // Performs inverse FFT by reversing the forward algorithm.
    //
    // ALGORITHM OVERVIEW:
    // for each stage k from D-1 down to 0:
    //     1. FFT along the currently local axis (using FFTW_BACKWARD)
    //     2. If not first stage: redistribute to make previous axis local
    //
    // EXAMPLE (3D case):
    //   Stage 2: FFT along X (local), redistribute X→Y
    //   Stage 1: FFT along Y (local), redistribute Y→Z
    //   Stage 0: FFT along Z (local), done
    //
    // DATA FLOW:
    //   Input:  data in stage D-1 distribution (output from forward())
    //   Output: data in stage 0 distribution (original distribution)
    //
    // This exactly reverses the forward transform, enabling perfect roundtrip:
    //   forward(data) followed by backward(data) recovers original (up to scaling)
    //
    // NOTE: User must normalize by 1/N after calling this function for correct IFFT
    //
    void backward(Complex* data) {
        // Initialize working arrays (avoid contamination from previous runs)
        for (auto& arr : stage_arrays_) {
            std::fill(arr.begin(), arr.end(), Complex(0.0, 0.0));
        }

        // Copy input data to final stage working array
        int sizeD = 1;
        for (int i = 0; i < D; ++i) sizeD *= stage_shapes_[D - 1][i];
        std::memcpy(stage_arrays_[D - 1].data(), data, sizeD * sizeof(Complex));

        // Reverse of forward algorithm: process stages from D-1 down to 0
        for (int stage = D - 1; stage >= 0; --stage) {
            int axis = D - 1 - stage;  // Same axis order as forward

            // Local inverse FFT along currently local axis
            perform_fft(stage, axis, FFTDirection::Backward);

            // Redistribute to previous stage (reverse of forward redistribution)
            if (stage > 0) {
                // The axis that will be local in the previous stage
                int next_axis = axis + 1;
                // Use the subcommunicator for the current axis
                int comm_idx = axis;
                exchange(subcomms_[comm_idx], MPI_C_DOUBLE_COMPLEX, D,
                        stage_shapes_[stage].data(), stage_arrays_[stage].data(), axis,
                        stage_shapes_[stage - 1].data(), stage_arrays_[stage - 1].data(), next_axis);
            }
        }

        // Copy result back
        int size0 = 1;
        for (int i = 0; i < D; ++i) size0 *= stage_shapes_[0][i];
        std::memcpy(data, stage_arrays_[0].data(), size0 * sizeof(Complex));
    }

private:
    // ========================================================================
    // Compute Stage Shapes (Data Distribution at Each FFT Stage)
    // Reference: Paper Section 3 (Pencil Decomposition)
    // ========================================================================
    //
    // This function computes the local array dimensions at each stage of the
    // FFT algorithm. This determines which axes are distributed vs local.
    //
    // KEY CONCEPT FROM PAPER (Section 3):
    //   At each stage, one axis is "local" (processor has all elements along
    //   that axis), while other axes are distributed across processors.
    //
    // EXAMPLE (3D with processor grid [2,2]):
    //   Global shape: [32, 32, 32]
    //   Processor grid: [2, 2]
    //
    //   Stage 0: shape=[16, 16, 32]  (axes 0,1 distributed; axis 2 local)
    //   Stage 1: shape=[16, 32, 16]  (axes 0,2 distributed; axis 1 local)
    //   Stage 2: shape=[32, 16, 16]  (axes 1,2 distributed; axis 0 local)
    //
    // GENERALIZATION TO D DIMENSIONS:
    //   Uses balanced decomposition (Equation 9) for each distributed axis.
    //   The pattern ensures each axis becomes fully local exactly once.
    //
    void setup_stage_shapes() {
        stage_shapes_.resize(D);
        for (int stage = 0; stage < D; ++stage) {
            stage_shapes_[stage].resize(D);
        }

        // Compute local sizes for each processor grid dimension using Equation (9)
        // local_n[i] = number of elements this processor owns along grid dim i
        // local_s[i] = starting index in global array along grid dim i
        std::vector<int> local_n(D - 1);
        std::vector<int> local_s(D - 1);
        for (int i = 0; i < D - 1; ++i) {
            decompose(global_shape_[i], dims_[i], coords_[i], local_n[i], local_s[i]);
        }

        // Store global starting position for stage 0 (initial distribution)
        // Used by user to initialize data with correct global coordinates
        for (int i = 0; i < D - 1; ++i) {
            global_start_[i] = local_s[i];
        }
        global_start_[D - 1] = 0;  // Last axis is fully local in stage 0

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
            int num_processed = stage;  // Number of axes that have been FFT'd

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

    // ========================================================================
    // Perform 1D FFT Along a Specified Axis
    // Reference: Paper Section 2.1 (Local FFT Operations)
    // ========================================================================
    //
    // Executes 1D FFT along the given axis using the backend abstraction.
    // The backend handles all FFTW details and provides a clean interface.
    //
    // PAPER CONTEXT (Section 2.1):
    //   At each stage, one axis is fully local (processor has complete data).
    //   We perform 1D FFT along this local axis for all pencils.
    //
    // EXAMPLE (3D, FFT along Z):
    //   If array shape is [Nx, Ny, Nz] and Z is local,
    //   we perform Nx×Ny independent 1D FFTs, each of length Nz.
    //
    // MEMORY LAYOUT:
    //   C/row-major order: last index varies fastest
    //   For axis D-1 (last): data is contiguous → single backend call
    //   For axis 0 (first):  data is strided → single backend call
    //   For middle axes:     loop over leading dimensions, call backend per slice
    //
    void perform_fft(int stage, int axis, FFTDirection direction) {
        // CASE 1: FFT along last axis (axis D-1) or first axis (axis 0)
        // Backend plan handles these with a single call
        if (axis == D - 1 || axis == 0) {
            backend_.execute_stage(stage, direction, stage_arrays_[stage].data());

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
                Complex* base_ptr = stage_arrays_[stage].data() +
                                   i * global_shape_[axis] * trailing_size;
                backend_.execute_stage(stage, direction, base_ptr);
            }
        }
    }

    // ========================================================================
    // Member Variables
    // ========================================================================

    // MPI communicators
    MPI_Comm comm_world_;              // Original communicator (typically MPI_COMM_WORLD)
    MPI_Comm comm_cart_;               // (D-1)-dimensional Cartesian topology
    std::vector<MPI_Comm> subcomms_;   // Subcommunicators for each grid dimension

    // Processor information
    int rank_, size_;                  // Rank and total number of processors
    int dims_[D - 1];                  // Processor grid dimensions (from MPI_Dims_create)
    int coords_[D - 1];                // This processor's coordinates in the grid

    // Array dimensions
    int global_shape_[D];              // Global array dimensions
    int global_start_[D];              // Starting indices for this processor (stage 0)

    // Stage information (computed by setup_stage_shapes)
    std::vector<std::vector<int>> stage_shapes_;   // [stage][axis] = local size per stage
    std::vector<std::vector<Complex>> stage_arrays_;  // Working buffers for each stage

    // FFT backend (stateful, stores pre-created plans)
    Backend backend_;
};

} // namespace parafaft

#endif // PARAFAFT_GENERIC_HPP
