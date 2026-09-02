cmake_minimum_required(VERSION 3.10)
project(vector-db)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# must precede add_executable: only applies to targets created after it
add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/arch:AVX2>)

add_executable(main src/main.cpp src/hnsw.cpp)
add_executable(benchmark src/benchmark.cpp src/hnsw.cpp)

# Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(hnsw_module python/bindings.cpp src/hnsw.cpp)