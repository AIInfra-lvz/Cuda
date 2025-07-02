cmake_minimum_required(VERSION 3.17.0)
project(CudaCore VERSION 0.1.0 LANGUAGES CUDA CXX C)

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

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(0)
    add_executable(CudaReduce CudaReduce.cu)
    target_link_libraries(CudaReduce PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(CudaReduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
    target_compile_options(CudaReduce PRIVATE -lineinfo)
endif()

if(0)
    add_executable(GEMM GEMM.cu)
    target_link_libraries(GEMM PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(GEMM PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
    target_compile_options(GEMM PRIVATE -lineinfo)
endif()

if(0)
    add_executable(mergeSort mergeSort.cu)
    target_link_libraries(mergeSort PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(mergeSort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
    target_compile_options(mergeSort PRIVATE -lineinfo)
endif()

if(1)
    add_executable(convolutionSeparable convolutionSeparable.cu)
    target_link_libraries(convolutionSeparable PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(convolutionSeparable PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
    target_compile_options(convolutionSeparable PRIVATE -lineinfo)
endif()