#ifndef MPIFFT_R2C_HPP
#define MPIFFT_R2C_HPP

// ============================================================================
// Real-to-Complex (R2C) / Complex-to-Real (C2R) Parallel FFT Implementation
// ============================================================================
//
// This implementation extends the pencil decomposition approach from
// mpifft_generic.hpp to handle real-valued data with full roundtrip support:
//   - forward(): R2C transform (real input → complex output)
//   - backward(): C2R transform (complex input → real output)
//
// Key differences from C2C:
//   - Input type: real (double) instead of complex
//   - Output shape: [..., N/2+1] instead of [..., N] on last axis
//   - First stage uses fftw_plan_many_dft_r2c (forward) / c2r (backward)
//   - Subsequent stages use standard C2C FFT on reduced-size complex arrays
//
// Memory Optimization
// -------------------
// Uses ping-pong buffering with two internal complex buffers instead of D+1
// separate buffers. This reduces internal memory usage by 50-67% depending
// on dimensionality (3D: 50%, 4D: 60%, 5D: 67%).
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

namespace mpifft {

// ============================================================================
// Utility Functions (shared with mpifft_generic.hpp)
// ============================================================================

// Balanced block-contiguous decomposition using Barry Smith's formula
inline void r2c_decompose(int N, int M, int p, int& n, int& s) {
    int q = N / M;
    int r = N % M;
    n = (r > p) ? (q + 1) : q;
    s = (r > p) ? (n * p) : (q * p + r);
}

// Create MPI subarray datatypes for partitioning along an axis
inline void r2c_subarray(MPI_Datatype datatype, int ndims, const int sizes[], int axis,
                         int nparts, MPI_Datatype subarrays[]) {
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
        MPI_Type_create_subarray(ndims, sizes, subsizes.data(), substarts.data(),
                                MPI_ORDER_C, datatype, &subarrays[p]);
        MPI_Type_commit(&subarrays[p]);
    }
}

// Global data redistribution using MPI_Alltoallw
inline void r2c_exchange(MPI_Comm comm, MPI_Datatype datatype, int ndims,
                         const int sizesA[], void* arrayA, int axisA,
                         const int sizesB[], void* arrayB, int axisB) {
    int nparts;
    MPI_Comm_size(comm, &nparts);

    std::vector<MPI_Datatype> subarraysA(nparts), subarraysB(nparts);
    r2c_subarray(datatype, ndims, sizesA, axisA, nparts, subarraysA.data());
    r2c_subarray(datatype, ndims, sizesB, axisB, nparts, subarraysB.data());

    std::vector<int> counts(nparts, 1), displs(nparts, 0);

    MPI_Alltoallw(arrayA, counts.data(), displs.data(), subarraysA.data(),
                  arrayB, counts.data(), displs.data(), subarraysB.data(), comm);

    for (int p = 0; p < nparts; p++) {
        MPI_Type_free(&subarraysA[p]);
        MPI_Type_free(&subarraysB[p]);
    }
}

template<int D, typename Backend = FFTWBackend>
class PencilFFT_R2C {
public:
    using Complex = std::complex<double>;

    // ========================================================================
    // Constructor: Setup Processor Grid and Pencil Decomposition for R2C
    // ========================================================================
    PencilFFT_R2C(const int global_shape[D], MPI_Comm comm = MPI_COMM_WORLD)
        : comm_world_(comm), backend_(D) {

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

        // Allocate two ping-pong buffers, each sized to handle any stage
        // This replaces D+1 separate buffers with just 2, saving 50-67% memory
        scratch_a_.resize(max_complex_size);
        scratch_b_.resize(max_complex_size);

        // Create FFT plans
        create_backend_plans();
    }

    ~PencilFFT_R2C() {
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

    // ========================================================================
    // Forward R2C Transform
    //
    // @param real_input   Input real array (not modified; user retains ownership)
    // @param complex_output  Output complex array (will contain Fourier coefficients)
    //
    // The input array is read directly without copying. For best performance,
    // allocate with fftw_alloc_real().
    // ========================================================================
    void forward(const double* real_input, Complex* complex_output) {
        // No zero initialization needed - all buffers are fully overwritten

        // Stage 0: R2C FFT from user's real buffer directly to scratch_a
        // Safety: FFTW R2C with FFTW_ESTIMATE preserves input array (unlike C2R).
        // The const_cast is safe because fftw_execute_dft_r2c does not modify
        // the input for out-of-place transforms planned with FFTW_ESTIMATE.
        backend_.execute_r2c(const_cast<double*>(real_input), scratch_a_.data());

        // Ping-pong between internal buffers
        Complex* src = scratch_a_.data();
        Complex* dst = scratch_b_.data();

        // Stages 1 to D-1: MPI redistribute + C2C FFT
        for (int stage = 1; stage < D; ++stage) {
            int axis = D - 1 - stage;
            int prev_axis = axis + 1;

            // MPI exchange: src → dst
            r2c_exchange(subcomms_[axis], MPI_C_DOUBLE_COMPLEX, D,
                    stage_output_shapes_[stage - 1].data(), src, prev_axis,
                    stage_shapes_[stage].data(), dst, axis);

            // C2C FFT in-place on dst
            perform_fft_on_buffer(stage, axis, FFTDirection::Forward, dst);

            // Swap for next iteration
            std::swap(src, dst);
        }

        // After D-1 swaps, src points to result buffer
        // Parity: D=3 → src=scratch_a, D=4 → src=scratch_b, etc.
        int complex_size = 1;
        for (int i = 0; i < D; ++i) {
            complex_size *= stage_shapes_[D - 1][i];
        }
        std::memcpy(complex_output, src, complex_size * sizeof(Complex));
    }

    // ========================================================================
    // Backward C2R Transform
    //
    // @param complex_input  Input complex array (not modified; user retains ownership)
    // @param real_output    Output real array (will contain reconstructed real data)
    //
    // The output array is written directly without copying. For best performance,
    // allocate with fftw_alloc_real().
    // ========================================================================
    void backward(const Complex* complex_input, double* real_output) {
        // No zero initialization needed - all buffers are fully overwritten

        // Copy input to internal buffer to start ping-pong
        // (Cannot read user's complex_input directly because C2R may destroy its input)
        int input_size = 1;
        for (int i = 0; i < D; ++i) {
            input_size *= stage_shapes_[D - 1][i];
        }
        std::memcpy(scratch_a_.data(), complex_input, input_size * sizeof(Complex));

        // Ping-pong between internal buffers
        Complex* src = scratch_a_.data();
        Complex* dst = scratch_b_.data();

        // Stages D-1 down to 1: C2C backward FFT + MPI redistribute
        for (int stage = D - 1; stage >= 1; --stage) {
            int axis = D - 1 - stage;

            // C2C backward FFT in-place on src
            perform_fft_on_buffer(stage, axis, FFTDirection::Backward, src);

            // MPI exchange: src → dst
            int prev_axis = axis + 1;
            r2c_exchange(subcomms_[axis], MPI_C_DOUBLE_COMPLEX, D,
                    stage_shapes_[stage].data(), src, axis,
                    stage_output_shapes_[stage - 1].data(), dst, prev_axis);

            // Swap for next iteration
            std::swap(src, dst);
        }

        // After D-1 swaps, src points to buffer with stage 0 data
        // Parity: D=3 → src=scratch_a, D=4 → src=scratch_b, etc.
        // Stage 0: C2R FFT directly to user's real buffer
        backend_.execute_c2r(src, real_output);
    }

    // ========================================================================
    // Query Functions for Input (Real Space)
    // ========================================================================
    int get_local_real_size() const {
        int size = 1;
        for (int i = 0; i < D; ++i) {
            size *= stage_shapes_[0][i];
        }
        return size;
    }

    void get_local_real_shape(int shape[D]) const {
        for (int i = 0; i < D; ++i) {
            shape[i] = stage_shapes_[0][i];
        }
    }

    void get_real_global_start(int start[D]) const {
        for (int i = 0; i < D; ++i) {
            start[i] = real_global_start_[i];
        }
    }

    // ========================================================================
    // Query Functions for Output (Complex/Fourier Space)
    // ========================================================================
    int get_local_complex_size() const {
        int size = 1;
        for (int i = 0; i < D; ++i) {
            size *= stage_shapes_[D - 1][i];
        }
        return size;
    }

    void get_local_complex_shape(int shape[D]) const {
        for (int i = 0; i < D; ++i) {
            shape[i] = stage_shapes_[D - 1][i];
        }
    }

    void get_complex_global_start(int start[D]) const {
        for (int i = 0; i < D; ++i) {
            start[i] = complex_global_start_[i];
        }
    }

    // ========================================================================
    // Debug: Print Shapes
    // ========================================================================
    void print_shapes() const {
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
    // ========================================================================
    // Setup Stage Shapes for R2C
    // ========================================================================
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
        std::vector<int> local_n(D - 1);
        std::vector<int> local_s(D - 1);
        for (int i = 0; i < D - 1; ++i) {
            r2c_decompose(global_real_shape_[i], dims_[i], coords_[i], local_n[i], local_s[i]);
        }

        // Store real starting positions
        for (int i = 0; i < D - 1; ++i) {
            real_global_start_[i] = local_s[i];
        }
        real_global_start_[D - 1] = 0;  // Last axis fully local

        // Stage 0 INPUT (REAL): axes [0..D-2] distributed, axis D-1 fully local
        for (int axis = 0; axis < D - 1; ++axis) {
            stage_shapes_[0][axis] = local_n[axis];
        }
        stage_shapes_[0][D - 1] = global_real_shape_[D - 1];  // Full real size

        // Stage 0 OUTPUT (COMPLEX): same distribution, but last axis is N/2+1
        for (int axis = 0; axis < D - 1; ++axis) {
            stage_output_shapes_[0][axis] = local_n[axis];
        }
        stage_output_shapes_[0][D - 1] = global_complex_shape_[D - 1];  // Reduced complex size

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
                        r2c_decompose(global_complex_shape_[axis], dims_[grid_dim],
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
        complex_global_start_[0] = 0;  // Axis 0 not distributed in final stage
        for (int i = 1; i < D; ++i) {
            int n, s;
            r2c_decompose(global_complex_shape_[i], dims_[i - 1], coords_[i - 1], n, s);
            complex_global_start_[i] = s;
        }
    }

    // ========================================================================
    // Create Backend Plans
    // ========================================================================
    void create_backend_plans() {
        // Stage 0: R2C plan
        int batch0 = 1;
        for (int i = 0; i < D - 1; ++i) {
            batch0 *= stage_shapes_[0][i];
        }
        int real_length = global_real_shape_[D - 1];
        int complex_length = global_complex_shape_[D - 1];

        // Allocate temporary real buffer for plan creation only.
        // FFTW's new-array execute requires arrays with same alignment as planning arrays.
        // std::vector provides suitable alignment for FFTW with FFTW_ESTIMATE.
        // Note: User buffers should ideally use fftw_alloc_real() for guaranteed alignment.
        std::vector<double> temp_real(batch0 * real_length);

        // R2C: real input → complex output (scratch_a_)
        // Plan with temp_real, execute with user's real buffer at runtime
        backend_.create_r2c_plan(
            real_length, batch0,
            temp_real.data(), scratch_a_.data(),
            1, real_length,
            1, complex_length
        );

        // C2R plan: complex input → real output
        // Plan with scratch_a_ and temp_real, execute with scratch buffer and user's real buffer
        backend_.create_c2r_plan(
            real_length, batch0,
            scratch_a_.data(), temp_real.data(),
            1, complex_length,
            1, real_length
        );

        // Stages 1 to D-1: C2C plans (use scratch_a_ for planning)
        // Note: For stages 1+, axis ranges from D-2 down to 0, never equal to D-1
        for (int stage = 1; stage < D; ++stage) {
            int axis = D - 1 - stage;  // axis ∈ [0, D-2] for stage ∈ [1, D-1]
            int length = global_complex_shape_[axis];

            if (axis == 0) {
                // First axis: strided FFTs
                int batch = 1;
                for (int i = 1; i < D; ++i) {
                    batch *= stage_shapes_[stage][i];
                }
                int stride = batch;
                backend_.create_stage_plan(stage, length, batch,
                                          scratch_a_.data(), stride, 1);
            } else {
                // Middle axes: need to handle via loops in perform_fft_on_buffer
                int trailing_size = 1;
                for (int i = axis + 1; i < D; ++i) {
                    trailing_size *= stage_shapes_[stage][i];
                }
                backend_.create_stage_plan(stage, length, trailing_size,
                                          scratch_a_.data(), trailing_size, 1);
            }
        }
    }

    // ========================================================================
    // Perform C2C FFT on specified buffer (for ping-pong buffering)
    // ========================================================================
    void perform_fft_on_buffer(int stage, int axis, FFTDirection direction, Complex* buffer) {
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
                Complex* base_ptr = buffer + i * global_complex_shape_[axis] * trailing_size;
                backend_.execute_stage(stage, direction, base_ptr);
            }
        }
    }

    // ========================================================================
    // Member Variables
    // ========================================================================

    // MPI communicators
    MPI_Comm comm_world_;
    MPI_Comm comm_cart_;
    std::vector<MPI_Comm> subcomms_;

    // Processor information
    int rank_, size_;
    int dims_[D - 1];
    int coords_[D - 1];

    // Shape tracking - KEY DIFFERENCE from C2C
    int global_real_shape_[D];       // Original real array: [N₀, ..., N_{D-1}]
    int global_complex_shape_[D];    // Reduced complex: [N₀, ..., N_{D-1}/2+1]
    int real_global_start_[D];       // Start indices for real input
    int complex_global_start_[D];    // Start indices for complex output

    // Stage shapes
    // stage_shapes_[0] is the REAL input shape
    // stage_output_shapes_[0] is the COMPLEX output shape after R2C
    // For stages 1+, stage_shapes_ and stage_output_shapes_ are the same
    std::vector<std::vector<int>> stage_shapes_;
    std::vector<std::vector<int>> stage_output_shapes_;

    // Working arrays - two ping-pong buffers for all stages
    // Memory optimization: Only 2 buffers instead of D+1, reducing memory by 50-67%
    std::vector<Complex> scratch_a_;  // Ping-pong buffer A
    std::vector<Complex> scratch_b_;  // Ping-pong buffer B

    // FFT backend
    Backend backend_;
};

} // namespace mpifft

#endif // MPIFFT_R2C_HPP
