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
// Backend implementations must be:
// - Move-constructible (for storage in containers)
// - Non-copyable (plans cannot be safely copied)

} // namespace mpifft

#endif // MPIFFT_BACKEND_HPP
