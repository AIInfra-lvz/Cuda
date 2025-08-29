cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
cmake_policy(SET CMP0112 NEW)

project(Cutlass_Examples VERSION 0.1.0 LANGUAGES CUDA CXX C)

set(CMAKE_CUDA_ARCHITECTURES 120) 

# using gdb
set(CMAKE_BUILD_TYPE "Debug")

if(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    set(NVCC_FLAGS -O3 -Wno-deprecated-gpu-targets)
    message(STATUS "${PROJECT_NAME} compile cuda code in release mode")
else()
    set(NVCC_FLAGS -g -G -O0 -Wno-deprecated-gpu-targets)
    message(STATUS "${PROJECT_NAME} compile cuda code in debug mode")
endif()

set(CUDA_NVCC_FLAGS "-g -G -O0")

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -ftemplate-backtrace-limit=0")
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(fmha_blackwell_sm120)
