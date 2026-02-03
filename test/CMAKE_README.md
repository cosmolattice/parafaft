# CMake Build Instructions

This directory contains both a traditional Makefile and a CMake alternative (`CMakeLists.txt`) for building the parallel FFT tests.

## Using CMake (Recommended)

### Prerequisites
- CMake 3.10 or higher
- MPI (OpenMPI, Intel MPI, or similar)
- FFTW3 library
- pkg-config (for finding FFTW3)

### Basic Build Process

1. Create a build directory (out-of-source build recommended):
   ```bash
   mkdir build
   cd build
   ```

2. Configure the project:
   ```bash
   cmake ..
   ```

3. Build all targets:
   ```bash
   cmake --build .
   ```
   
   Or if using Make generator:
   ```bash
   make
   ```

### Available Targets

#### Build Targets
- `cmake --build .` - Build all tests
- `cmake --build . --target clean` - Remove all built files

#### Run Targets
- `cmake --build . --target run-unit` - Run backend unit tests
- `cmake --build . --target run-reference` - Run serial reference tests
- `cmake --build . --target run-c2c` - Run C2C MPI tests (4 processes)
- `cmake --build . --target run-r2c` - Run R2C MPI tests (4 processes)
- `cmake --build . --target run-all` - Run all tests
- `cmake --build . --target compare` - Compare MPI vs serial output
- `cmake --build . --target show-help` - Display help information

#### Using Make Syntax (if using Make generator)
If you configured CMake to use the Make generator, you can also use traditional make commands:
```bash
make all
make run-unit
make run-c2c
make run-r2c
make run-reference
make run-all
make compare
make show-help
make clean
```

### CTest Integration

CMake also provides CTest integration for automated testing:

```bash
# Run all tests through CTest
ctest

# Run tests with verbose output
ctest -V

# Run specific test
ctest -R unit_tests

# Run tests in parallel
ctest -j4
```

### Project Structure

The CMake build creates the following directory structure in the build folder:
```
build/
├── c2c/           # C2C MPI test executables
├── r2c/           # R2C MPI test executables  
├── unit/          # Unit test executables
├── reference/     # Reference serial test executables
└── ...           # Other CMake generated files
```

### Customizing the Build

You can customize the build using CMake variables:

```bash
# Use a specific MPI implementation
cmake -DMPI_CXX_COMPILER=mpic++ ..

# Set custom FFTW3 paths
cmake -DFFTW3_ROOT=/path/to/fftw3 ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Comparison with Makefile

### Advantages of CMake:
- Better dependency management
- Cross-platform support
- Integration with IDEs
- CTest integration for automated testing  
- Out-of-source builds (keeps source directory clean)
- Better handling of library discovery
- Parallel builds by default

### Makefile Equivalent Commands:
| Makefile Command     | CMake Equivalent                         |
| -------------------- | ---------------------------------------- |
| `make all`           | `cmake --build .`                        |
| `make clean`         | `cmake --build . --target clean`         |
| `make run-c2c`       | `cmake --build . --target run-c2c`       |
| `make run-r2c`       | `cmake --build . --target run-r2c`       |
| `make run-unit`      | `cmake --build . --target run-unit`      |
| `make run-reference` | `cmake --build . --target run-reference` |
| `make run-all`       | `cmake --build . --target run-all`       |
| `make compare`       | `cmake --build . --target compare`       |
| `make help`          | `cmake --build . --target show-help`     |

### Note on Dependencies

The CMake configuration automatically handles:
- Finding MPI compiler and libraries
- Locating FFTW3 using pkg-config
- Setting up proper include and library paths
- Managing compiler flags for MPI

If you encounter issues with dependency detection, you may need to:
- Install pkg-config if not available
- Set `PKG_CONFIG_PATH` to include FFTW3's .pc files
- Manually specify library paths using CMake variables

## Troubleshooting

### Common Issues:

1. **MPI not found**: Ensure MPI is installed and `mpicxx` is in your PATH
2. **FFTW3 not found**: Install pkg-config and ensure FFTW3 .pc files are accessible
3. **Build fails**: Check that all header files exist in the expected locations (`../parafaft_generic.hpp`, etc.)

### Getting Help:

Run the help target for available commands:
```bash
cmake --build . --target show-help
```
