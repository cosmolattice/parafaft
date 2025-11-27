#ifndef MPIFFT_BACKEND_HPP
#define MPIFFT_BACKEND_HPP

#include <complex>

namespace mpifft {

// Direction enum (abstracts FFTW_FORWARD/BACKWARD)
enum class FFTDirection {
    Forward,
    Backward
};

// Backend Concept (documented requirements):
// A valid FFT backend must be a class that provides:
//
// 1. Constructor taking number of stages:
//    Backend(int num_stages)
//
// 2. Plan creation method (called during initialization):
//    void create_stage_plan(
//        int stage,                   // Stage index (0 to num_stages-1)
//        int length,                  // FFT size
//        int batch,                   // Number of transforms
//        std::complex<double>* data,  // Persistent data pointer for this stage
//        int stride,                  // Element spacing within FFT
//        int dist                     // Spacing between FFTs
//    );
//
// 3. Execution method (reuses pre-created plans):
//    void execute_stage(int stage, FFTDirection direction, std::complex<double>* data);
//
// 4. Destructor that cleans up all plans:
//    ~Backend()
//
// Design principle: Plans are created once during initialization with
// representative data pointers. Execution uses fftw_execute_dft which
// allows applying the plan to different data arrays with the same layout.
// This enables efficient reuse for operations on different memory slices.
//
// R2C Backend Extension (required for PencilFFT_R2C):
// In addition to the above methods, backends supporting R2C transforms must provide:
//
// 5. R2C plan creation (forward: real → complex):
//    void create_r2c_plan(
//        int length,                         // Real-space FFT length N
//        int batch,                          // Number of 1D transforms
//        double* real_in,                    // Real input array (for planning)
//        std::complex<double>* complex_out,  // Complex output array (for planning)
//        int istride,                        // Input: element stride
//        int idist,                          // Input: distance between batches
//        int ostride,                        // Output: element stride
//        int odist                           // Output: distance between batches
//    );
//
// 6. R2C execution:
//    void execute_r2c(double* real_in, std::complex<double>* complex_out);
//
// 7. C2R plan creation (backward: complex → real):
//    void create_c2r_plan(
//        int length,                         // Real-space output length N
//        int batch,                          // Number of 1D transforms
//        std::complex<double>* complex_in,   // Complex input array (for planning)
//        double* real_out,                   // Real output array (for planning)
//        int istride,                        // Input: element stride
//        int idist,                          // Input: distance between batches
//        int ostride,                        // Output: element stride
//        int odist                           // Output: distance between batches
//    );
//
// 8. C2R execution:
//    void execute_c2r(std::complex<double>* complex_in, double* real_out);
//
// Note: R2C transforms produce N/2+1 complex values from N real values.
//       C2R transforms produce N real values from N/2+1 complex values.
//
// Backend implementations must be:
// - Move-constructible (for storage in containers)
// - Non-copyable (plans cannot be safely copied)

} // namespace mpifft

#endif // MPIFFT_BACKEND_HPP
