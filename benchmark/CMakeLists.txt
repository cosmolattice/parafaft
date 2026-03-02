add_executable(bench_c2c bench_c2c.cpp)
target_link_libraries(bench_c2c parafaft::parafaft)
set_target_properties(bench_c2c PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bench)

add_executable(bench_r2c bench_r2c.cpp)
target_link_libraries(bench_r2c parafaft::parafaft)
set_target_properties(bench_r2c PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bench)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bench)
