cmake_minimum_required(VERSION 3.10)
project(vector-db)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# must precede add_executable: only applies to targets created after it
if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/arch:AVX2>)
else()
    add_compile_options($<$<CONFIG:Release>:-O2> $<$<CONFIG:Release>:-march=native>)
endif()

# On to isolate whether NEON itself causes recall/self-query instability on ARM:
# cmake -B build -DFORCE_SCALAR_DISTANCE=ON builds with l2_distance's scalar path
# forced even on a NEON-capable target, so the same benchmark can be run twice
# (NEON active vs. forced scalar) and directly compared.
option(FORCE_SCALAR_DISTANCE "Force HNSW::l2_distance to use its scalar path even when __ARM_NEON is defined" OFF)
if(FORCE_SCALAR_DISTANCE)
    add_compile_definitions(FORCE_SCALAR_DISTANCE)
endif()

add_executable(main src/main.cpp src/hnsw.cpp)
add_executable(benchmark src/benchmark.cpp src/hnsw.cpp)
add_executable(test_distance src/test_distance.cpp src/hnsw.cpp)

# Python bindings
find_package(pybind11 REQUIRED)
pybind11_add_module(hnsw_module python/bindings.cpp src/hnsw.cpp)
install(TARGETS hnsw_module DESTINATION vector_db)